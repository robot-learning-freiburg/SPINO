import os
from collections.abc import Mapping
from functools import partial

import torch
import wandb
from datasets import Cityscapes, DistributedMixedBatchSampler, Kitti360
from io_utils import logging
from misc.solver import WarmupPolyLR
from misc.utils import convert_to_dict
from mmcv.parallel.data_container import DataContainer
from torch import distributed
from torch.nn.parallel import DistributedDataParallel
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataloader import default_collate


def dict_to_cuda(sample, device):
    sample_cuda = {}
    for key, data_i in sample.items():
        if isinstance(data_i, list):
            # The nested list is used for metrics computation on the cpu (meta). Due to this,
            # we do not want to move it to cuda
            if isinstance(data_i[0], list):
                continue
            sample_cuda[key] = [m.cuda(device=device, non_blocking=True) for m in data_i]
        if isinstance(data_i, dict):
            sample_cuda[key] = dict_to_cuda(data_i, device)
        else:
            sample_cuda[key] = sample[key].cuda(device=device, non_blocking=True)

    return sample_cuda


def init_device():
    # Initialize multi-processing
    distributed.init_process_group(backend="nccl", init_method="env://")
    device_id, device = int(os.environ["LOCAL_RANK"]), torch.device(int(os.environ["LOCAL_RANK"]))
    rank, world_size = distributed.get_rank(), distributed.get_world_size()
    torch.cuda.set_device(device_id)

    return device_id, device, rank, world_size


def model_to_cuda(cfg, args, model, device, device_id):
    if not args.debug:
        torch.backends.cudnn.benchmark = cfg.general.cudnn_benchmark
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            model)  # Convert all instances of batch norm to SyncBatchNorm
        model = DistributedDataParallel(model.cuda(device),
                                        device_ids=[device_id],
                                        output_device=device_id,
                                        find_unused_parameters=True)
    else:
        model = model.cuda(device)

    return model


def init_logging(args, log_dir, run_dir, config, init_wandb=False):
    config_dict = convert_to_dict(config)
    if init_wandb:
        wandb_summary = wandb.init(project="spino",
                                   entity="label_efficient_ps",
                                   dir=run_dir,
                                   name=f"{args.run_name}",
                                   job_type=args.mode,
                                   notes=args.comment,
                                   config=config_dict)
        logging.init(log_dir, "train" if args.mode == "train" else "test")
    else:
        wandb_summary = None
        logging.init(log_dir, "train" if args.mode == "train" else "test", file_logging=False)

    return wandb_summary


def collate_batch(items, samples_per_gpu):
    """Collate function to handle lists of non-Tensor data"""
    if isinstance(items[0], DataContainer):
        stacked = []
        if isinstance(items[0], torch.Tensor):
            for i in range(0, len(items), samples_per_gpu):
                assert isinstance(items[i].data, torch.Tensor)
                stacked.append(
                    default_collate([
                        sample.data for sample in items[i:i + samples_per_gpu]]))
                return DataContainer(stacked, items[0].stack, items[0].padding_value)

        return DataContainer(stacked, items[0].stack, items[0].padding_value)

    if isinstance(items[0], list):
        stacked = []
        for i in range(0, len(items), samples_per_gpu):
            stacked.extend(sample for sample in items[i:i + samples_per_gpu])
        return stacked

    if isinstance(items[0], Mapping):
        out = {}
        for key in items[0]:
            if isinstance(items[0][key], list):
                tmp = []
                for d in items:
                    for e in d[key]:
                        tmp.append(e)
                out[key] = collate_batch(tmp, samples_per_gpu)
            else:
                out[key] = collate_batch([d[key] for d in items], samples_per_gpu)
        return out

    return default_collate(items)


def gen_dataloader(args, cfg, rank, world_size):
    collate_fn_train = partial(collate_batch, samples_per_gpu=cfg.train.batch_size_per_gpu)

    if args.eval:
        train_dl = None
    else:
        # Create train dataloader
        logging.log_info("Creating train dataloader...", debug=args.debug)
        if cfg.dataset.name == "kitti_360":
            dataset_train = Kitti360(cfg.dataset.train_split, cfg.dataset,
                                     label_mode=cfg.dataset.label_mode,
                                     sequences=cfg.dataset.train_sequences,
                                     sequence_reference_mode="semantic")
        elif cfg.dataset.name == "cityscapes":
            dataset_train = Cityscapes(cfg.dataset.train_split, cfg.dataset,
                                       label_mode=cfg.dataset.label_mode,
                                       return_only_rgb=cfg.dataset.return_only_rgb)
        else:
            raise NotImplementedError(f"Dataset {cfg.dataset.name} is not yet implemented")

        if not args.debug:
            indices_gt = getattr(cfg.dataset, "indices_gt", None)
            if indices_gt:
                train_sampler = (
                    DistributedMixedBatchSampler(dataset_train, world_size, rank, shuffle=True,
                                                 indices_gt=cfg.dataset.indices_gt,
                                                 batch_size=cfg.train.batch_size_per_gpu))
            else:
                train_sampler = DistributedSampler(dataset_train, world_size, rank, shuffle=True)

            train_dl = DataLoader(dataset_train,
                                  sampler=train_sampler,
                                  batch_size=cfg.train.batch_size_per_gpu,
                                  collate_fn=collate_fn_train,
                                  pin_memory=False,
                                  num_workers=cfg.train.nof_workers_per_gpu)
        else:
            train_dl = DataLoader(dataset_train,
                                  batch_size=cfg.train.batch_size_per_gpu,
                                  collate_fn=collate_fn_train,
                                  pin_memory=False,
                                  num_workers=cfg.train.nof_workers_per_gpu)

    # Create validation dataloader
    logging.log_info("Creating val dataloader...", debug=args.debug)

    if cfg.dataset.name == "kitti_360":
        dataset_val = Kitti360(cfg.dataset.val_split, cfg.dataset,
                               label_mode=cfg.dataset.label_mode,
                               sequences=cfg.dataset.val_sequences,
                               sequence_reference_mode="semantic")
    elif cfg.dataset.name == "cityscapes":
        dataset_val = Cityscapes(cfg.dataset.val_split, cfg.dataset,
                                 label_mode=cfg.dataset.label_mode,
                                 return_only_rgb=cfg.dataset.return_only_rgb)
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset.name} is not yet implemented")

    collate_fn_val = partial(collate_batch, samples_per_gpu=cfg.val.batch_size_per_gpu)
    if not args.debug:
        val_sampler = DistributedSampler(dataset_val, world_size, rank, shuffle=True)
        val_dl = DataLoader(dataset_val,
                            sampler=val_sampler,
                            batch_size=cfg.val.batch_size_per_gpu,
                            collate_fn=collate_fn_val,
                            pin_memory=False,
                            num_workers=cfg.val.nof_workers_per_gpu)
    else:
        val_dl = DataLoader(dataset_val,
                            batch_size=cfg.val.batch_size_per_gpu,
                            collate_fn=collate_fn_val,
                            pin_memory=False,
                            num_workers=cfg.val.nof_workers_per_gpu)

    return train_dl, val_dl


def gen_optimizer(cfg, model):
    assert isinstance(cfg.type, str)
    # You can implement other rules here...
    if cfg.type == "Adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    if cfg.type == "None":
        return None
    raise NotImplementedError(f"The optimizer ({cfg.type}) is not yet implemented.")


def gen_lr_scheduler(cfg, optimizer):
    assert isinstance(
        cfg.train.scheduler.type,
        str), f"The option cfg.train.scheduler.type has to be a string. Current type: " \
              f" {cfg.train.scheduler.type.type()}"
    # You can implement other rules here...
    if cfg.train.scheduler.type == "StepLR":
        return lr_scheduler.StepLR(optimizer,
                                   step_size=cfg.train.scheduler.step_lr.step_size,
                                   gamma=cfg.train.scheduler.step_lr.gamma)
    if cfg.train.scheduler.type == "WarmupPolyLR":
        return WarmupPolyLR(optimizer,
                            cfg.train.scheduler.warmup.max_iters,
                            cfg.train.scheduler.warmup.factor,
                            cfg.train.scheduler.warmup.iters,
                            cfg.train.scheduler.warmup.method,
                            -1,
                            cfg.train.scheduler.warmup.power,
                            cfg.train.scheduler.warmup.constant_ending)
    if cfg.train.scheduler.type == "None":
        return None
    raise NotImplementedError(f"The lr scheduler ({cfg.train.scheduler.type})  is not yet "
                              f"implemented.")


def freeze_modules(modules, model):
    for module in modules:
        print(f"Freezing module: {module}")
        for name, param in model.named_parameters():
            if name.startswith(module):
                param.requires_grad = False
    return model
