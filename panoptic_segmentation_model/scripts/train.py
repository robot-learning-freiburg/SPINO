# pylint: disable=wrong-import-position

import argparse
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import distributed
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).absolute().parent.parent))

from eval.meters import AverageMeter, ConfusionMatrixMeter, ConstantMeter
from io_utils import io_utils, logging
from io_utils.visualizations import gen_visualizations, plot_confusion_matrix
from misc import train_utils
from networks.model_setup import gen_models

parser = argparse.ArgumentParser(description="Pretrain SPINO on a given dataset")
parser.add_argument("--run_name", required=True, type=str,
                    help="Name of the run")
parser.add_argument("--project_root_dir", required=True, type=str,
                    help="The root directory of the project")
parser.add_argument("--mode", required=True, type=str,
                    help="'train' the model or 'test' the model?")
parser.add_argument("--eval", action="store_true",
                    help="Do a single validation run")
parser.add_argument("--resume", metavar="FILE", type=str,
                    help="Resume training from given file")
parser.add_argument("--pre_train", type=str, nargs="+",
                    help="Start from the given pre-trained snapshots, overwriting each with the "
                         "next one in the list. Snapshots can be given in the format "
                         "'{module_name}:{path}', where '{module_name} is one of "
                         "'backbone_panoptic', 'semantic_head', 'instance_head', or "
                         "'semantic_head'. In that case only that part of the network will be"
                         " loaded from the snapshot")
parser.add_argument("--freeze_modules", nargs="+", default=[],
                    help="The modules to freeze. Default is empty")
parser.add_argument("--filename_defaults_config", required=True, type=str,
                    help="Path to defaults configuration file")
parser.add_argument("--filename_config", required=True, type=str,
                    help="Path to configuration file")
parser.add_argument("--comment", type=str,
                    help="Comment to add to WandB")
parser.add_argument("--seed", type=int, default=20,
                    help="Seed to initialize 'torch', 'random', and 'numpy'")
parser.add_argument("--debug", type=bool, default=False,
                    help="Should the program run in 'debug' mode?")


def train(model, optimizer, scheduler, dataloader, meters, config, epoch, device,
          wandb_summary, global_step, debug) -> int:
    model.train()

    loss_weights = config.losses.weights
    log_step_interval = config.logging.log_step_interval
    num_epochs = config.train.nof_epochs
    log_train_samples = config.logging.log_train_samples
    if config.dataset.normalization.active:
        rgb_mean = config.dataset.normalization.rgb_mean
        rgb_std = config.dataset.normalization.rgb_std
    else:
        rgb_mean, rgb_std = (0., 0., 0.), (0., 0., 0.)
    img_scale = config.visualization.scale

    time_meters = {
        "data_time": AverageMeter((), meters["losses"]["loss"].momentum),
        "batch_time": AverageMeter((), meters["losses"]["loss"].momentum)
    }

    # Main training loop:
    data_time = time.time()
    for batch_idx, sample in enumerate(dataloader):
        global_step += 1

        # Move input data to cuda
        sample_cuda = train_utils.dict_to_cuda(sample, device)

        # Log the data loading time
        time_meters["data_time"].update(torch.tensor(time.time() - data_time))
        batch_time = time.time()

        # Run network
        optimizer.zero_grad()
        losses, results, stats = model(sample_cuda, "train")
        if not debug:
            distributed.barrier()  # Wait for all processes

        losses = OrderedDict((k, v.mean()) for k, v in losses.items() if v is not None)
        losses["loss"] = sum(loss_weights[loss_name] * loss
                             for loss_name, loss in losses.items())

        # Increment the optimizer and backpropagate the gradients
        losses["loss"].backward()
        optimizer.step()

        # Log the time taken to execute the batch
        time_meters["batch_time"].update(torch.tensor(time.time() - batch_time))

        with torch.no_grad():
            for loss_name, loss_value in losses.items():
                if loss_value is not None:
                    meters["losses"][loss_name].update(loss_value.cpu())
            for stat_name, stat_value in stats.items():
                if stat_value is not None:
                    meters["metrics"][stat_name].update(stat_value.cpu())

        # Clean-up
        del losses, stats, sample, sample_cuda

        # Log
        if (batch_idx + 1) % log_step_interval == 0:
            if wandb_summary is not None:
                logging.log_iter(meters["losses"], None, epoch, num_epochs, batch_idx + 1,
                                 len(dataloader), scheduler.get_last_lr()[0], batch=True)
                logging.log_wandb(wandb_summary,
                                  "train",
                                  meters["losses"],
                                  meters["metrics"],
                                  time_meters,
                                  batch=True,
                                  epoch=epoch,
                                  learning_rate=scheduler.get_last_lr()[0],
                                  global_step=global_step)

        data_time = time.time()

    # This only needs to be done once every epoch
    scheduler.step()

    if wandb_summary is not None:
        logging.log_wandb(wandb_summary,
                          "train",
                          meters["losses"],
                          meters["metrics"],
                          time_meters,
                          batch=False,
                          epoch=epoch,
                          learning_rate=scheduler.get_last_lr()[0],
                          global_step=global_step)

    if log_train_samples:
        logging.log_info("Log training samples.", debug=debug)
        model.eval()
        wandb_vis_dict = {}
        max_vis_count = 5

        log_dataloader = DataLoader(dataloader.dataset,
                                    batch_size=1,
                                    pin_memory=False,
                                    num_workers=1)
        for batch_idx, sample in enumerate(log_dataloader):
            with torch.no_grad():
                sample_cuda = train_utils.dict_to_cuda(sample, device)
                del sample

                do_panoptic_fusion = config.model.make_semantic and config.model.make_instance
                _, results, _ = model(sample_cuda, "eval", do_panoptic_fusion=do_panoptic_fusion)

                wandb_vis_dict, max_vis_count = gen_visualizations(sample_cuda, results,
                                                                   wandb_vis_dict, img_scale,
                                                                   rgb_mean, rgb_std, max_vis_count,
                                                                   config.dataset.remove_classes,
                                                                   config.dataset.label_mode)

                del results
                if max_vis_count == 0:
                    break

        if wandb_summary is not None:
            logging.log_wandb_images("train", wandb_vis_dict, wandb_summary, global_step)

        model.train()
        del wandb_vis_dict

    # Wait for the GPU that did the wandb logging
    if not debug:
        distributed.barrier()

    torch.cuda.empty_cache()
    return global_step


def validate(model, dataloader, config, epoch, wandb_summary, device, global_step,
             wandb_panel="val", compute_loss=True, print_results=False, debug=False):
    model.eval()

    ignore_classes = config.eval.semantic.ignore_classes
    loss_weights = config.losses.weights
    do_panoptic_fusion = config.model.make_semantic and config.model.make_instance
    log_step_interval = config.logging.log_step_interval
    num_epochs = getattr(config.train, "nof_epochs", 0)
    if config.dataset.normalization.active:
        rgb_mean = config.dataset.normalization.rgb_mean
        rgb_std = config.dataset.normalization.rgb_std
    else:
        rgb_mean, rgb_std = (0., 0., 0.), (0., 0., 0.)
    img_scale = config.visualization.scale

    # Get evaluators
    if not debug:
        model_acc = model.module
    else:
        model_acc = model

    if model_acc.semantic_algo is not None:
        sem_eval = model_acc.semantic_algo.evaluator
        num_classes = sem_eval.num_classes
    else:
        sem_eval, num_classes = None, 0
    if model_acc.instance_algo is not None and model_acc.semantic_algo is not None:
        panoptic_eval = model_acc.instance_algo.evaluator
        panoptic_eval.reset()
    else:
        panoptic_eval = None

    val_meters = {
        "losses": {
            "loss": AverageMeter(()),
            "semantic": AverageMeter(()),
            "center": AverageMeter(()),
            "offset": AverageMeter(()),
        },
        "metrics": {
            # Semantic
            "sem_conf": ConfusionMatrixMeter(num_classes),
            "sem_miou": ConstantMeter(()),
            "sem_miou_pixels": ConstantMeter(()),

            # Panoptic
            "p_pq": ConstantMeter(()),
            "p_sq": ConstantMeter(()),
            "p_rq": ConstantMeter(()),
            "p_stuff_pq": ConstantMeter(()),
            "p_stuff_sq": ConstantMeter(()),
            "p_stuff_rq": ConstantMeter(()),
            "p_things_pq": ConstantMeter(()),
            "p_things_sq": ConstantMeter(()),
            "p_things_rq": ConstantMeter(())
        }
    }

    wandb_vis_dict = {}
    max_vis_count = 5

    for it, sample in tqdm(enumerate(dataloader), disable=not print_results, total=len(dataloader),
                           desc="Evaluating"):
        with torch.no_grad():
            sample_cuda = train_utils.dict_to_cuda(sample, device)

            # Run network
            losses, results, stats = model(sample_cuda, "train",
                                           do_panoptic_fusion=do_panoptic_fusion,
                                           sem_ignore_classes=ignore_classes)
            if not debug:
                distributed.barrier()

            if compute_loss:
                losses = OrderedDict((k, v.mean()) for k, v in losses.items() if v is not None)
                losses["loss"] = sum(loss_weights[loss_name] * loss
                                     for loss_name, loss in losses.items())
            else:
                losses = {}

            if not debug:
                for _, val in stats.items():
                    if val is not None:
                        val = val / distributed.get_world_size()
                        distributed.all_reduce(val, distributed.ReduceOp.SUM)

            for loss_name, loss_value in losses.items():
                if loss_value is not None:
                    val_meters["losses"][loss_name].update(loss_value.cpu())
            for stat_name, stat_value in stats.items():
                if stat_value is not None and stat_name in val_meters["metrics"]:
                    val_meters["metrics"][stat_name].update(stat_value.cpu())

            # Update panoptic metrics
            if do_panoptic_fusion:
                sample_cuda_semantic = sample_cuda["semantic"].clone()
                for ignore_class in ignore_classes:
                    sample_cuda_semantic[sample_cuda["semantic"] == ignore_class] = 255
                pan_img_gt, _ = model_acc.instance_algo.panoptic_fusion(sample_cuda_semantic,
                                                                        sample_cuda["center"],
                                                                        sample_cuda["offset"])
                panoptic_eval.update(pan_img_gt, results["panoptic"])

            # Prepare the wandb visualization
            if (it + 1) % log_step_interval == 0 and wandb_summary is not None:
                wandb_vis_dict, max_vis_count = gen_visualizations(sample_cuda, results,
                                                                   wandb_vis_dict, img_scale,
                                                                   rgb_mean, rgb_std, max_vis_count,
                                                                   config.dataset.remove_classes,
                                                                   config.dataset.label_mode)
            # Log batch
            if (it + 1) % log_step_interval == 0:
                if wandb_summary is not None and not print_results:
                    logging.log_iter(val_meters["metrics"], None, epoch, num_epochs, it + 1,
                                     len(dataloader), batch=True)

            del losses, results, stats, sample, sample_cuda

    # Semantics computation
    # Reduce confusion matrix to non-ignored classes only
    if sem_eval is not None:
        sem_conf_mat_filtered = sem_eval.filter_sem_conf_mat(
            val_meters["metrics"]["sem_conf"].sum, device, debug)
        # Classes that are not covered in the ground truth should not be considered
        indices_with_gt = sem_conf_mat_filtered.sum(dim=1) != 0
        sem_miou_score = sem_eval.compute_sem_miou(sem_conf_mat_filtered)[indices_with_gt].mean()
        sem_miou_score_pixels = sem_eval.compute_sem_miou(sem_conf_mat_filtered, True)
        val_meters["metrics"]["sem_miou"].update(sem_miou_score)
        val_meters["metrics"]["sem_miou_pixels"].update(sem_miou_score_pixels)
        # Add the visualization of the confusion matrix of the entire dataset.
        wandb_vis_dict["conf_mat"] = plot_confusion_matrix(sem_conf_mat_filtered,
                                                           config.dataset.remove_classes,
                                                           config.dataset.label_mode)

    # Panoptics computation
    if panoptic_eval is not None:
        panoptic_scores = panoptic_eval.evaluate()
        val_meters["metrics"]["p_pq"].update(torch.tensor(panoptic_scores["All"]["pq"]))
        val_meters["metrics"]["p_sq"].update(torch.tensor(panoptic_scores["All"]["sq"]))
        val_meters["metrics"]["p_rq"].update(torch.tensor(panoptic_scores["All"]["rq"]))
        val_meters["metrics"]["p_stuff_pq"].update(torch.tensor(panoptic_scores["Stuff"]["pq"]))
        val_meters["metrics"]["p_stuff_sq"].update(torch.tensor(panoptic_scores["Stuff"]["sq"]))
        val_meters["metrics"]["p_stuff_rq"].update(torch.tensor(panoptic_scores["Stuff"]["rq"]))
        val_meters["metrics"]["p_things_pq"].update(torch.tensor(panoptic_scores["Things"]["pq"]))
        val_meters["metrics"]["p_things_sq"].update(torch.tensor(panoptic_scores["Things"]["sq"]))
        val_meters["metrics"]["p_things_rq"].update(torch.tensor(panoptic_scores["Things"]["rq"]))
    else:
        panoptic_scores = None

    # Log the accumulated wandb images and the confusion_matrix
    if wandb_summary is not None:
        logging.log_wandb_images(wandb_panel, wandb_vis_dict, wandb_summary, global_step)
        logging.log_wandb(wandb_summary,
                          wandb_panel,
                          val_meters["losses"],
                          val_meters["metrics"],
                          time_meters=None,
                          batch=False,
                          epoch=epoch,
                          learning_rate=None,
                          global_step=global_step)

        if panoptic_scores is not None:
            logging.log_wandb_table_panoptic(wandb_panel, panoptic_scores, wandb_summary,
                                             global_step, config.dataset.remove_classes,
                                             config.dataset.label_mode)

    if print_results:
        print("-" * 20)
        for k, v in val_meters["metrics"].items():
            if "_c" in k:
                continue
            v_ = v.iou.mean().item() if isinstance(v, ConfusionMatrixMeter) else v.mean.item()
            print(f"{k:12} = {v_:>6.3f}")
        print("-" * 20)

    torch.cuda.empty_cache()


def main(args):
    # Set the random number seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Initialize devices
    if not args.debug:
        device_id, device, rank, world_size = train_utils.init_device()
    else:
        print("\033[91m" + "ACTIVE DEBUG MODE" + "\033[0m")
        rank, world_size = 0, 1
        device_id, device = rank, torch.device(rank + 6)

    # Create directories
    if not args.debug:  # and not args.eval:
        log_dir, run_dir, saved_models_dir = io_utils.create_run_directories(args, rank)
    else:
        log_dir, run_dir, saved_models_dir = None, None, None

    # Load configuration
    config = io_utils.gen_config(args, cfg_type="train")

    # Initialize logging
    if not args.debug:  # and not args.eval:
        wandb_summary = train_utils.init_logging(args, log_dir=log_dir, run_dir=run_dir,
                                                 config=config, init_wandb=rank == 0)
    else:
        wandb_summary = None

    # Create dataloaders
    train_dataloader, val_dataloader = train_utils.gen_dataloader(args, config, rank, world_size)

    # Create model
    model = gen_models(config,
                       device=device,
                       stuff_classes=val_dataloader.dataset.stuff_classes,
                       thing_classes=val_dataloader.dataset.thing_classes,
                       ignore_classes=val_dataloader.dataset.ignore_classes +
                                      config.eval.semantic.ignore_classes)

    # Freeze modules based on the argument inputs
    model = train_utils.freeze_modules(args.freeze_modules, model)

    # Resume or use pretrained checkpoint
    assert not (args.resume and args.pre_train), "resume and pre_train are mutually exclusive"
    modules = io_utils.make_modules_list(config)
    if args.resume:
        # Load all modules and the optimizer (see below)
        logging.log_info("Loading checkpoint from %s", args.resume, debug=args.debug)
        checkpoint = io_utils.resume_from_checkpoint(model, args.resume, modules)
    elif args.pre_train:
        # Only load certain modules
        logging.log_info("Loading pre-trained model from %s", args.pre_train, debug=args.debug)
        io_utils.pretrained_from_checkpoints(model, args.pre_train, modules, rank)
        checkpoint = None
    else:
        assert not args.eval, "--resume is needed in eval mode"
        checkpoint = None

    # Initialize GPU stuff
    model = train_utils.model_to_cuda(config, args, model, device, device_id)
    torch.multiprocessing.set_sharing_strategy("file_system")

    # Only evaluate the model from a given checkpoint
    if args.eval:
        validate(model=model,
                 dataloader=val_dataloader,
                 config=config,
                 epoch=0,
                 wandb_summary=wandb_summary,
                 device=device,
                 global_step=0,
                 print_results=True,
                 debug=args.debug)
        logging.log_info("End of evaluation script!", debug=args.debug)
        if wandb_summary is not None:
            wandb_summary.finish()
        return

    # Create optimizer and scheduler
    optimizer = train_utils.gen_optimizer(config.train.optimizer, model)
    scheduler = train_utils.gen_lr_scheduler(config, optimizer)

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["state_dict"]["optimizer"])
        scheduler.load_state_dict(checkpoint["state_dict"]["scheduler"])
        torch.set_rng_state(checkpoint["state_dict"]["torch_rng"])
        np.random.set_state(checkpoint["state_dict"]["numpy_rng"])

    # Meters used for training
    momentum = 1. - 1. / (len(train_dataloader) + 1e-7)  # Compute a running mean
    train_meters = {
        "losses": {
            "loss": AverageMeter((), momentum),
            "semantic": AverageMeter((), momentum),
            "center": AverageMeter((), momentum),
            "offset": AverageMeter((), momentum),
        },
        "metrics": {
            # Semantic
            "sem_conf": ConfusionMatrixMeter(train_dataloader.dataset.num_classes),
        }
    }

    if args.resume:
        starting_epoch = checkpoint["training_meta"]["epoch"] + 1
        global_step = checkpoint["training_meta"]["global_step"]
        for type_, meter_type in train_meters.items():
            for name, meter in meter_type.items():
                meter.load_state_dict(checkpoint["state_dict"][f"{type_}_{name}_meter"])
        del checkpoint
    else:
        starting_epoch = 0
        global_step = 0

    for epoch in range(starting_epoch, config.train.nof_epochs):
        logging.log_info("Starting epoch %d", epoch, debug=args.debug)

        # Run training epoch
        global_step = train(model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            dataloader=train_dataloader,
                            meters=train_meters,
                            config=config,
                            epoch=epoch,
                            wandb_summary=wandb_summary,
                            device=device,
                            global_step=global_step,
                            debug=args.debug)

        # Save snapshot (only on rank 0)
        if not args.debug and rank == 0:
            snapshot_file = os.path.join(saved_models_dir, f"epoch_{epoch:04d}.pth")
            logging.log_info("Saving snapshot to %s", snapshot_file, debug=args.debug)
            meters_out_dict = {}
            for type_, meter_type in train_meters.items():
                for name, meter in meter_type.items():
                    meters_out_dict[f"{type_}_{name}_meter"] = meter.state_dict()
            io_utils.save_checkpoint(snapshot_file,
                                     config,
                                     epoch,
                                     global_step,
                                     model.module.get_state_dict(),
                                     optimizer=optimizer.state_dict(),
                                     scheduler=scheduler.state_dict(),
                                     torch_rng=torch.get_rng_state(),
                                     numpy_rng=np.random.get_state(),
                                     **meters_out_dict)

        if (epoch + 1) % config.logging.val_epoch_interval == 0:
            logging.log_info("Validating epoch %d", epoch, debug=args.debug)
            validate(model=model,
                     dataloader=val_dataloader,
                     config=config,
                     epoch=epoch,
                     wandb_summary=wandb_summary,
                     device=device,
                     global_step=global_step,
                     debug=args.debug)

    logging.log_info("End of training script!", debug=args.debug)
    if wandb_summary is not None:
        wandb_summary.finish()


if __name__ == "__main__":
    main(parser.parse_args())
