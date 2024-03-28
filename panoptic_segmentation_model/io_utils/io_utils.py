import os
import shutil
from typing import Any, Dict, List

import torch
from cfg.default_config import get_cfg_defaults
from yacs.config import CfgNode as CN

# -------------------------------------------------------- #
# Saving / loading checkpoints


def make_modules_list(config: CN) -> List[str]:
    modules = set()
    modules.add("backbone_panoptic")
    if config.model.make_semantic:
        modules.add("semantic_head")
    if config.model.make_instance:
        modules.add("instance_head")
    return list(modules)


def resume_from_checkpoint(model, checkpoint: str, modules: List[str]):
    checkpoint = torch.load(checkpoint, map_location="cpu")
    state_dict = checkpoint["state_dict"]

    for module in modules:
        if module in state_dict and state_dict[module] is not None:
            model_module = getattr(model, module)
            if model_module is not None:
                _load_pretraining_dict(model_module, state_dict[module])
        else:
            raise KeyError(f"The given checkpoint does not contain a state_dict for module "
                           f"{module}")

    return checkpoint


def pretrained_from_checkpoints(model, checkpoints: List[str], modules: List[str], rank: int):
    for checkpoint in checkpoints:
        if ":" in checkpoint:
            module, checkpoint = checkpoint.split(":")
        else:
            module = None

        checkpoint = torch.load(checkpoint, map_location="cpu")

        if module is None:
            _load_pretraining_dict(getattr(model, "body"), checkpoint)
        else:
            if module in modules:
                state_dict = checkpoint["state_dict"]
                if module not in state_dict or state_dict[module] is None:
                    raise KeyError(f"The given checkpoint does not contain a state_dict for module "
                                   f"{module}")
                if rank == 0:
                    print(f"Loading {module} layers...")
                _load_pretraining_dict(getattr(model, module), state_dict[module])
            else:
                raise ValueError(f"Unrecognized network module {module}")


def save_checkpoint(file: str, config: CN, epoch: int, global_step: int,
                    model_state_dict: Dict[str, Any], **kwargs):
    state_dict = dict(kwargs)
    state_dict.update(model_state_dict)  # Some modules might be set to None
    data = {
        "config": config.dump(),  # as YAML string
        "state_dict": state_dict,
        "training_meta": {
            "epoch": epoch,
            "global_step": global_step
        }
    }
    torch.save(data, file)


def _load_pretraining_dict(model, state_dict):
    """Load state dictionary from a pre-training checkpoint

    This is an even less strict version of `model.load_state_dict(..., False)`, which also
    ignores  parameters from `state_dict` that don"t have the same shapes as the corresponding
    ones in `model`. This is useful when loading from pre-trained models that are trained on
    different datasets.

    Parameters
    ----------
    model : torch.nn.Model
        Target model
    state_dict : dict
        Dictionary of model parameters
    """
    model_sd = model.state_dict()

    for k, v in model_sd.items():
        if k in state_dict:
            if v.shape != state_dict[k].shape:
                print(f"The shape of the layer does not match: {k} - "
                      f"{state_dict[k].shape} vs {v.shape}")
                assert False

    model.load_state_dict(state_dict, False)


# -------------------------------------------------------- #
# Logging


def log_train(writer, total_loss, losses, images_dict, max_imgs, batch_idx, samples_per_sec,
              total_time, epoch, num_total_steps, step):
    _log_images(writer, images_dict, max_imgs, step)
    _log_losses(writer, losses, step)
    _log_cmd(batch_idx, samples_per_sec, total_time, total_loss, epoch, num_total_steps,
             step)


def log_val(writer, rgb, step):
    # Log imgs
    writer.add_images("rgb", rgb.data, step)


def _log_losses(writer, losses, step):
    for loss_type, loss in losses.items():
        writer.add_scalar(f"{loss_type}", loss, step)


def _log_images(writer, images_dict, max_imgs, step):
    for identifier, batch_imgs in images_dict.items():
        writer.add_images(identifier, batch_imgs[:max_imgs].data, step)


# Adapted from: https://github.com/nianticlabs/monodepth2/blob/master/trainer.py
def _log_cmd(batch_idx, samples_per_sec, total_time, total_loss, epoch, num_total_steps,
             step):
    time_left = (num_total_steps / step - 1.0) * total_time if step > 0 else 0

    print_string = "Training epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                   " | total loss (weighted) {:.5f} | time elapsed: {} | time left: {}"
    print(
        print_string.format(epoch, batch_idx, samples_per_sec, total_loss,
                            sec_to_hm_str(total_time), sec_to_hm_str(time_left)))


def sec_to_hm_str(t):
    """
    Source: https://github.com/nianticlabs/monodepth2/blob/master/utils.py
    Convert time in seconds to a nice string
    e.g. 10239 -> "02h50m39s"
    """
    h, m, s = sec_to_hm(t)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def sec_to_hm(t):
    """
    Source: https://github.com/nianticlabs/monodepth2/blob/master/utils.py
    Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


# -------------------------------------------------------- #


def gen_config(args, cfg_type="train"):
    if cfg_type == "train":
        cfg = get_cfg_defaults()
    else:
        raise NotImplementedError(
            "You can only choose between 'pretrain' and 'pred' cfg types")
    if args.filename_config is not None:
        path_cfg = os.path.join(args.project_root_dir, "panoptic_segmentation_model/cfg",
                                args.filename_config)
        cfg.merge_from_file(path_cfg)
    cfg.freeze()
    return cfg


def create_run_directories(args, rank):
    root_dir = args.project_root_dir
    experiment_dir = os.path.join(root_dir, "experiments")
    if args.mode == "train":
        run_dir = os.path.join(experiment_dir, f"train_{args.run_name}")
    elif args.mode == "train_extra":
        run_dir = os.path.join(experiment_dir, f"train_extra_{args.run_name}")
    elif args.mode == "test":
        run_dir = os.path.join(experiment_dir, f"test_{args.run_name}")
    else:
        raise RuntimeError("Invalid choice. --mode must be either 'train' or 'test'")
    saved_models_dir = os.path.join(run_dir, "saved_models")
    log_dir = os.path.join(run_dir, "logs")
    config_dir = os.path.join(run_dir, "config")

    path_save_config_file = os.path.join(run_dir, args.filename_config)
    if args.filename_defaults_config is not None:
        path_save_defaults_config_file = os.path.join(run_dir,
                                                      f"defaults_{args.filename_defaults_config}")
    else:
        path_save_defaults_config_file = None

    # Create the directory
    if rank == 0 and (not os.path.exists(experiment_dir)):
        os.mkdir(experiment_dir)
    if rank == 0:
        print("run_dir", run_dir)
        assert not os.path.exists(run_dir), \
            f"Run folder '{run_dir}' already found! Delete it to reuse the run name."

    if rank == 0:
        os.mkdir(run_dir)
        os.mkdir(saved_models_dir)
        os.mkdir(log_dir)
        os.mkdir(config_dir)

    path_config = os.path.join(args.project_root_dir, "panoptic_segmentation_model/cfg",
                               args.filename_config)
    path_default_config = os.path.join(args.project_root_dir, "panoptic_segmentation_model/cfg",
                                       args.filename_defaults_config)
    if rank == 0:
        shutil.copyfile(path_config, path_save_config_file)
        shutil.copyfile(path_default_config, path_save_defaults_config_file)

    return log_dir, run_dir, saved_models_dir


def create_results_directories(args, rank):
    root_dir = args.project_root_dir
    experiment_dir = os.path.join(root_dir, "model_output")
    if args.mode == "train":
        run_dir = os.path.join(experiment_dir, f"train_{args.run_name}")
    elif args.mode == "train_extra":
        run_dir = os.path.join(experiment_dir, f"train_extra_{args.run_name}")
    elif args.mode == "test":
        run_dir = os.path.join(experiment_dir, f"test_{args.run_name}")
    else:
        raise RuntimeError("Invalid choice. --mode must be either 'train' or 'test'")
    preds_dir = os.path.join(run_dir, "predictions")
    visu_dir = os.path.join(run_dir, "visualizations")

    # Create the directory
    if rank == 0 and (not os.path.exists(experiment_dir)):
        os.mkdir(experiment_dir)
    if rank == 0:
        print("run_dir", run_dir)
        assert not os.path.exists(run_dir), \
            f"Results folder '{run_dir}' already found! Delete it to reuse the run name."

    if rank == 0:
        os.mkdir(run_dir)
        os.mkdir(preds_dir)
        os.mkdir(visu_dir)

    return preds_dir, visu_dir
