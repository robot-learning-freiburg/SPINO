from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from datasets import get_labels
from torch import Tensor


def gen_visualizations(
        sample,
        results: Dict[str, Any],
        wandb_vis_dict: Dict[str, Any],
        scale: float,
        rgb_mean: Tuple[float, float, float],
        rgb_std: Tuple[float, float, float],
        max_vis_count: int,
        remove_classes: List[int],
        label_mode: str,
) -> Tuple[Dict[str, Any], int]:
    vis_dict = {"overview": [], "panoptic": []}

    semantic_key = "semantic"
    instance_center_key = "center"
    instance_offset_key = "offset"
    panoptic_key = "panoptic"
    instance_key = "instance"

    batch_size = sample["rgb"].shape[0]
    for i in range(min(batch_size, max_vis_count)):
        overview = _add_rgb(sample["rgb"][i], scale, rgb_mean, rgb_std)
        if semantic_key in results.keys() and results[semantic_key] is not None:
            overview = _add_semantic_map(results[semantic_key][i], scale, remove_classes,
                                         label_mode, overview)
        vis_dict["overview"].append(overview)

        panoptic = _add_rgb(sample["rgb"][i], scale, rgb_mean, rgb_std)
        if semantic_key in sample.keys() and sample[semantic_key] is not None:
            panoptic = _add_semantic_map(sample[semantic_key][i], scale, remove_classes, label_mode,
                                         panoptic)
        if semantic_key in results.keys() and results[semantic_key] is not None:
            panoptic = _add_semantic_map(results[semantic_key][i], scale, remove_classes,
                                         label_mode, panoptic)
        if instance_center_key in results.keys() and results[instance_center_key] is not None:
            panoptic = _add_instance_center(results[instance_center_key][i], scale, panoptic)
        if instance_offset_key in results.keys() and results[instance_offset_key] is not None:
            panoptic = _add_instance_offset(results[instance_offset_key][i], scale, panoptic)
        if panoptic_key in results.keys() and instance_key in results.keys(
        ) and results[panoptic_key] is not None and results[instance_key] is not None:
            panoptic = _add_panoptic_map(results[panoptic_key][i],
                                         results[instance_key][i],
                                         scale,
                                         remove_classes,
                                         label_mode,
                                         panoptic,
                                         instance_only=True)
            panoptic = _add_panoptic_map(results[panoptic_key][i], results[instance_key][i], scale,
                                         remove_classes, label_mode, panoptic)
        vis_dict["panoptic"].append(panoptic)

    for key, value in vis_dict.items():
        if key in wandb_vis_dict.keys():
            wandb_vis_dict[key] += value
        else:
            wandb_vis_dict[key] = value

    # Return the remaining number of visualization samples
    return wandb_vis_dict, max_vis_count - len(vis_dict["overview"])


def plot_confusion_matrix(conf_mat: Tensor, remove_classes: List[int], label_mode: str):
    labels = get_labels(remove_classes, label_mode)
    ignore_classes = [255, -1]

    # Get the class names
    seen_ids = []
    class_labels = []
    for l in labels:
        if l.trainId in seen_ids or l.trainId in ignore_classes:
            continue
        seen_ids.append(l.trainId)
        class_labels.append(l.name)

    # Get the ratio. Row elts + Col elts - Diagonal elt (it is computed twice)
    # Small number added to avoid nan
    conf_mat_np = conf_mat / ((conf_mat.sum(dim=0) + conf_mat.sum(dim=1) - conf_mat.diag()) + 1e-8)
    conf_mat_np = conf_mat_np.detach().cpu().numpy()

    # Plot the confusion matrix
    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    conf_mat_plt = sns.heatmap(conf_mat_np * 100,
                               annot=True,
                               fmt=".2g",
                               vmin=0.0,
                               vmax=100.,
                               square=True,
                               xticklabels=class_labels,
                               yticklabels=class_labels,
                               annot_kws={"size": 7},
                               ax=ax)
    plt.close(fig)

    return conf_mat_plt


# -------------------------------------------------------- #

def _concat_images(image_0: Tensor, image_1: Tensor) -> Tensor:
    if image_0.shape[2] == image_1.shape[2]:
        return torch.cat((image_0, image_1), dim=1)

    max_width = max(image_0.shape[2], image_1.shape[2])
    height = image_0.shape[1] + image_1.shape[1]
    concat_image = torch.zeros((3, height, max_width), dtype=image_0.dtype,
                               device=image_0.device)
    concat_image[:, :image_0.shape[1], :image_0.shape[2]] = image_0
    concat_image[:, image_0.shape[1]:, :image_1.shape[2]] = image_1
    return concat_image


def _recover_image(img: Tensor, rgb_mean: Tuple[float, float, float],
                   rgb_std: Tuple[float, float, float]) -> Tensor:
    img = img * img.new(rgb_std).view(-1, 1, 1)
    img = img + img.new(rgb_mean).view(-1, 1, 1)
    return img


# -------------------------------------------------------- #

def _add_rgb(rgb: Union[Tensor, List[Tensor]],
             scale: float,
             rgb_mean: Optional[Tuple[float, float, float]] = None,
             rgb_std: Optional[Tuple[float, float, float]] = None,
             vis_image: Optional[Tensor] = None) -> Tensor:
    def __interp(img, scale_):
        img_scaled = F.interpolate(img.unsqueeze(0),
                                   scale_factor=scale_,
                                   recompute_scale_factor=False,
                                   mode="bilinear",
                                   align_corners=False).squeeze(0)
        if rgb_mean is not None:
            img_scaled = (_recover_image(img_scaled, rgb_mean, rgb_std) * 255).type(torch.int)
        else:
            img_scaled = (img_scaled * 255).type(torch.int)
        return img_scaled

    if isinstance(rgb, list):
        rgb_scaled = [__interp(rgb_, scale / len(rgb_)) for rgb_ in rgb]
        vis_rgb = torch.cat(rgb_scaled, dim=2)
    else:
        vis_rgb = __interp(rgb, scale)

    # Attach to existing visualization
    if vis_image is not None:
        vis_rgb = _concat_images(vis_image, vis_rgb)

    return vis_rgb


def _add_semantic_map(semantic_map: Tensor, scale: float, remove_classes: List[int],
                      label_mode: str, vis_image: Optional[Tensor] = None) -> Tensor:
    def __visualize(semantic, remove_classes_):
        dataset_labels = get_labels(remove_classes_, label_mode)
        colors_trainid = {label.trainId: label.color for label in dataset_labels}

        sem_vis = torch.zeros((3, semantic.shape[0], semantic.shape[1]),
                              dtype=torch.int32).to(semantic.device)

        # Color the map
        classes = torch.unique(semantic)
        for stuff_label in classes:
            if stuff_label == 255:
                continue
            sem_vis[:,
            (semantic == stuff_label).squeeze()] = torch.tensor(
                colors_trainid[stuff_label.item()],
                dtype=sem_vis.dtype,
                device=sem_vis.device).unsqueeze(1)

        return sem_vis

    semantic_map_scaled = F.interpolate(semantic_map.unsqueeze(0).unsqueeze(0).type(torch.float),
                                        scale_factor=scale,
                                        recompute_scale_factor=False,
                                        mode="nearest").squeeze().type(semantic_map.dtype)

    # Color the semantic labels
    vis_semantic = __visualize(semantic_map_scaled, remove_classes)

    # Attach to existing visualization
    if vis_image is not None:
        vis_semantic = _concat_images(vis_image, vis_semantic)

    return vis_semantic


def _add_instance_center(center_map: Tensor, scale: float,
                         vis_image: Optional[Tensor] = None) -> Tensor:
    center_map_scaled = F.interpolate(center_map.unsqueeze(0),
                                      scale_factor=scale,
                                      recompute_scale_factor=False,
                                      mode="bilinear",
                                      align_corners=False).squeeze()

    center_map_npy = center_map_scaled.detach().cpu().numpy()
    normalizer = mpl.colors.Normalize(vmin=0, vmax=center_map_npy.max())
    mapper = plt.cm.ScalarMappable(norm=normalizer, cmap="hot")
    colormapped_map = (mapper.to_rgba(center_map_npy)[:, :, :3] * 255).astype(np.uint8)
    vis_center = torch.from_numpy(colormapped_map).type(torch.int).permute(2, 0,
                                                                           1).to(center_map.device)

    # Attach to existing visualization
    if vis_image is not None:
        vis_center = _concat_images(vis_image, vis_center)

    return vis_center


def _add_instance_offset(offset_map: Tensor, scale: float,
                         vis_image: Optional[Tensor] = None) -> Tensor:
    def __visualize(offset):
        offset_npy = offset.detach().cpu().numpy()
        normalizer = mpl.colors.Normalize(vmin=offset_npy.min(), vmax=offset_npy.max())
        mapper = plt.cm.ScalarMappable(norm=normalizer, cmap="viridis")
        colormapped_map = (mapper.to_rgba(offset_npy)[:, :, :3] * 255).astype(np.uint8)
        colormapped_map[offset_npy == 0] = 0
        vis = torch.from_numpy(colormapped_map).type(torch.int).permute(2, 0, 1).to(offset.device)
        return vis

    offset_map_scaled = F.interpolate(offset_map.unsqueeze(0),
                                      scale_factor=scale / 2,
                                      recompute_scale_factor=False,
                                      mode="bilinear",
                                      align_corners=False).squeeze()
    vis_offset = [__visualize(offset_map_scaled[0]), __visualize(offset_map_scaled[1])]
    vis_offset = torch.cat(vis_offset, dim=2)

    # Attach to existing visualization
    if vis_image is not None:
        vis_offset = _concat_images(vis_image, vis_offset)

    return vis_offset


def _add_binary_mask(binary_mask_map: Tensor, scale: float,
                     vis_image: Optional[Tensor] = None) -> Tensor:
    binary_mask_map_scaled = F.interpolate(
        binary_mask_map.unsqueeze(0).unsqueeze(0).type(torch.float),
        scale_factor=scale,
        recompute_scale_factor=False,
        mode="nearest").squeeze().type(binary_mask_map.dtype)
    vis_binary_mask = binary_mask_map_scaled.repeat(3, 1, 1).type(vis_image.dtype) * 255

    # Attach to existing visualization
    if vis_image is not None:
        vis_binary_mask = _concat_images(vis_image, vis_binary_mask)

    return vis_binary_mask


def _add_instance_map(instance_map: Tensor, scale: float,
                      vis_image: Optional[Tensor] = None) -> Tensor:
    def __visualize(instance, rnd_generator_):
        inst_vis = torch.zeros((3, instance.shape[0], instance.shape[1]),
                               dtype=torch.int32).to(instance.device)

        instances = torch.unique(instance)
        for inst in instances:
            if inst == 0:
                continue  # Assigned to stuff area
            random_color = torch.randint(0,
                                         255, (3,),
                                         dtype=inst_vis.dtype,
                                         device=inst_vis.device,
                                         generator=rnd_generator_)
            inst_vis[:, (instance == inst)] = random_color.unsqueeze(1)

        return inst_vis

    instance_map_scaled = F.interpolate(instance_map.unsqueeze(0).unsqueeze(0).type(torch.float),
                                        scale_factor=scale,
                                        recompute_scale_factor=False,
                                        mode="nearest").squeeze().type(instance_map.dtype)

    # Same colors
    rnd_generator = torch.Generator(device=instance_map.device)
    rnd_generator.manual_seed(42)

    vis_instance = __visualize(instance_map_scaled, rnd_generator)

    # Attach to existing visualization
    if vis_image is not None:
        vis_instance = _concat_images(vis_image, vis_instance)

    return vis_instance


def _add_panoptic_map(panoptic_map: Tensor,
                      instance_map: Tensor,
                      scale: float,
                      remove_classes: List[int],
                      label_mode: str,
                      vis_image: Optional[Tensor] = None,
                      label_divisor: int = 1000,
                      instance_only: bool = False) -> Tensor:
    def __visualize(panoptic, instance, remove_classes_, label_divisor_, rnd_generator_,
                    instance_only_):
        dataset_labels = get_labels(remove_classes_, label_mode)
        colors_trainid = {label.trainId: label.color for label in dataset_labels}

        pan_vis = torch.zeros((3, panoptic.shape[0], panoptic.shape[1]),
                              dtype=torch.int32).to(panoptic.device)

        # Colorize the 'stuff' pixels
        if not instance_only_:
            # Restore semantic mask
            semantic = 255 * torch.ones_like(panoptic, dtype=torch.uint8)
            valid_mask = panoptic != -1
            semantic[valid_mask] = ((panoptic[valid_mask] - instance[valid_mask]) /
                                    label_divisor_).type(torch.uint8)

            stuff_mask = (instance == 0).logical_and(valid_mask)
            classes = torch.unique(semantic[stuff_mask])
            for stuff_label in classes:
                pan_vis[:,
                (semantic == stuff_label).squeeze()] = torch.tensor(
                    colors_trainid[stuff_label.item()],
                    dtype=pan_vis.dtype,
                    device=pan_vis.device).unsqueeze(1)

        # Colorize the 'thing' pixels with a separate color for each instance
        instances = torch.unique(instance)
        for inst in instances:
            if inst == 0:
                continue  # Assigned to stuff area
            random_color = torch.randint(0,
                                         255, (3,),
                                         dtype=pan_vis.dtype,
                                         device=pan_vis.device,
                                         generator=rnd_generator_)
            pan_vis[:, (instance == inst)] = random_color.unsqueeze(1)

        return pan_vis

    panoptic_map_scaled = F.interpolate(panoptic_map.unsqueeze(0).unsqueeze(0).type(torch.float),
                                        scale_factor=scale,
                                        recompute_scale_factor=False,
                                        mode="nearest").squeeze().type(panoptic_map.dtype)
    instance_map_scaled = F.interpolate(instance_map.unsqueeze(0).unsqueeze(0).type(torch.float),
                                        scale_factor=scale,
                                        recompute_scale_factor=False,
                                        mode="nearest").squeeze().type(instance_map.dtype)

    # Same colors
    rnd_generator = torch.Generator(device=panoptic_map.device)
    rnd_generator.manual_seed(42)

    vis_panoptic = __visualize(panoptic_map_scaled, instance_map_scaled, remove_classes,
                               label_divisor, rnd_generator, instance_only)

    # Attach to existing visualization
    if vis_image is not None:
        vis_panoptic = _concat_images(vis_image, vis_panoptic)

    return vis_panoptic
