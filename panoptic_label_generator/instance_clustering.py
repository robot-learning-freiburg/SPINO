import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import scipy.ndimage
import torch
from PIL import Image
from pytorch_lightning.cli import LightningCLI
from skimage.morphology import binary_erosion, remove_small_objects
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import Dataset
from utils.instance_colors import COLORS

# Ignore seome torch warnings
warnings.filterwarnings('ignore', '.*The default behavior for interpolate/upsample with float*')
warnings.filterwarnings(
    'ignore', '.*Default upsampling behavior when mode=bicubic is changed to align_corners=False*')
warnings.filterwarnings('ignore', '.*Only one label was provided to `remove_small_objects`*')


class InstanceCluster(pl.LightningModule):
    """Panoptic fusion module that uses the semantic and boundary model to cluster semantic blobs
    into instances.

    Parameters
    ----------
    semantic_model : pl.LightningModule
        Semantic segmentation model.
    semantic_model_ckpt : str
        Path to the semantic segmentation model checkpoint.
    boundary_model : pl.LightningModule
        Boundary detection model.
    boundary_model_ckpt : str
        Path to the boundary detection model checkpoint.
    structure_connectivity : List[List[int]]
        Connectivity matrix for the CCA (scipy.ndimage.label) function.
    instance_min_pixel : int
        Minimum number of pixels for an instance to be considered.
    erosion_structure : List[List[int]]
        Structure for the binary erosion (scipy.ndimage.binary_erosion) function.
    erosion_iterations : int
        Number of iterations for the binary erosion (scipy.ndimage.binary_erosion) function.
    output_size : Tuple[int, int]
        Output size of panoptic segmentation.
    ignore_index : int
        Ignore index for the semantic segmentation.
    test_plot : bool
        Whether to plot the predictions during testing.
    test_save_dir : str
        Directory to save the predictions during testing.
    test_save_vis : bool
        Whether to save the prediction visualization as images during testing.
    debug_plot : bool
        Whether to plot the intermediate predictions during testing.
    """

    def __init__(self, semantic_model: pl.LightningModule, semantic_model_ckpt: str,
                 boundary_model: pl.LightningModule, boundary_model_ckpt: str,
                 structure_connectivity: List[List[int]], instance_min_pixel: int,
                 erosion_structure: List[List[int]], erosion_iterations: int,
                 output_size: Tuple[int, int], ignore_index: int = 255,
                 test_plot: bool = False, test_save_dir: str = None, test_save_vis: bool = False,
                 debug_plot: bool = False):
        super().__init__()
        self.semantic_model = semantic_model
        semantic_model_ckpt_dict = torch.load(semantic_model_ckpt, map_location='cpu')
        self.semantic_model.load_state_dict(semantic_model_ckpt_dict['state_dict'])
        self.semantic_model.on_load_checkpoint(semantic_model_ckpt_dict)

        self.boundary_model = boundary_model
        boundary_model_ckpt_dict = torch.load(boundary_model_ckpt, map_location='cpu')
        self.boundary_model.load_state_dict(boundary_model_ckpt_dict['state_dict'])
        self.boundary_model.on_load_checkpoint(boundary_model_ckpt_dict)

        # share encoder if the same vit model is used
        if self.semantic_model.dinov2_vit_model == self.boundary_model.dinov2_vit_model:
            self.boundary_model.encoder = self.semantic_model.encoder

        for param in self.semantic_model.parameters():  # freeze
            param.requires_grad = False
        for param in self.boundary_model.parameters():  # freeze
            param.requires_grad = False

        self.structure_connectivity = np.array(structure_connectivity)
        self.instance_min_pixel = instance_min_pixel
        if erosion_iterations > 0:
            self.erosion_footprint = [(np.array(erosion_structure), erosion_iterations)]
        else:
            self.erosion_footprint = None
        self.output_size = output_size
        self.ignore_index = ignore_index

        self.test_plot = test_plot
        self.test_save_dir = test_save_dir
        self.test_save_vis = test_save_vis
        self.debug_plot = debug_plot

        id_color_array = COLORS  # (CLASSES, 3)
        np.random.seed(0)
        id_color_array = np.random.permutation(id_color_array)
        id_color_array[0] = [0, 0, 0]  # background
        self.id_color_array = (id_color_array * 255).astype(np.uint8)

    def get_dataset(self) -> Dataset:
        dataset = self.trainer.test_dataloaders[0].dataset
        return dataset

    def predict(self, rgb: torch.Tensor, rgb_original: torch.Tensor,
                ego_car_mask: Optional[torch.Tensor] = None) \
            -> Tuple[np.array, np.array, np.array]:
        pred_sem = self.semantic_model.predict(rgb, ego_car_mask)  # (B, H, W)
        pred_sem = pred_sem.cpu().numpy()  # (B, H, W)

        pred_boundary = self.boundary_model.predict(rgb)  # (B, H, W)
        pred_boundary = pred_boundary.cpu().numpy()  # (B, H, W)

        pred_instances_batch = []

        for rgb_i, rgb_original_i, pred_sem_i, pred_boundary_i in zip(rgb, rgb_original, pred_sem,
                                                                      pred_boundary):

            if self.debug_plot:
                self.semantic_model.plot(rgb_original_i, pred_sem_i)

            assert pred_sem_i.shape == tuple(self.output_size)
            pred_instances = np.zeros(self.output_size,
                                      dtype=int)  # to store the instance IDs (H, W)
            number_of_instances = 0

            thing_classes = self.get_dataset().thing_classes
            for semantic_class_id in thing_classes:
                semantic_class_mask = pred_sem_i == semantic_class_id  # (H, W)
                if np.sum(semantic_class_mask) == 0:  # skip if thing class is not present
                    continue

                # Cluster into semantic segments/blobs with CCA
                semantic_segments_mask = scipy.ndimage.label(semantic_class_mask,
                                                             structure=self.structure_connectivity)[
                    0]  # (H, W)

                if self.debug_plot:
                    self.plot_instances(rgb_original_i, semantic_segments_mask)

                for semantic_segment_id in np.unique(semantic_segments_mask):
                    if semantic_segment_id == 0:  # skip background
                        continue
                    semantic_segment_mask = (
                            semantic_segments_mask == semantic_segment_id)  # (H, W)
                    # skip small instances
                    if np.sum(semantic_segment_mask) < self.instance_min_pixel:
                        pred_sem_i[semantic_segment_mask] = self.ignore_index  # set to ignore class
                        continue

                    # Subtract boundary from semantic segment
                    instances_mask = np.logical_and(semantic_segment_mask,
                                                    pred_boundary_i)  # (H, W)

                    if self.erosion_footprint is not None:
                        instances_mask = binary_erosion(instances_mask,
                                                        footprint=self.erosion_footprint)  # (H, W)

                    # Cluster into instances with CCA
                    instances_mask = \
                        scipy.ndimage.label(instances_mask, structure=self.structure_connectivity)[
                            0]  # (H, W)
                    instances_mask = remove_small_objects(
                        instances_mask, min_size=self.instance_min_pixel)  # (H, W)

                    # if no large enough instance is found through the boundary estimation,
                    # use the whole semantic segment as one instance
                    if np.sum(instances_mask) == 0:
                        instances_mask[semantic_segment_mask] = 1

                    instances_ids = np.unique(instances_mask)
                    for i in range(1, len(instances_ids)):  # renumber instances to be consecutive
                        instances_mask[instances_mask == instances_ids[i]] = i

                    # if semantic has no instance, add them to the nearest instance with 1-NN
                    assert semantic_segment_mask.shape == instances_mask.shape
                    coordinates = np.indices(
                        (self.output_size[0], self.output_size[1])).reshape(2, -1).T  # (H * W, 2)
                    coordinates_sem_seg = coordinates[
                        semantic_segment_mask.reshape(-1) == 1]  # (M, 2)
                    coordinates_instances = coordinates[instances_mask.reshape(-1) != 0]  # (N, 2)

                    knn = KNeighborsClassifier(n_neighbors=1)
                    knn.fit(coordinates_instances,
                            instances_mask.reshape(-1)[instances_mask.reshape(-1) != 0])
                    instances_mask_shape = instances_mask.shape
                    instances_mask = instances_mask.reshape(-1)
                    instances_mask[semantic_segment_mask.reshape(-1) == 1] = \
                        knn.predict(coordinates_sem_seg)
                    instances_mask = instances_mask.reshape(instances_mask_shape)  # (H, W)

                    instances_mask += number_of_instances
                    instances_mask[instances_mask == number_of_instances] = 0

                    pred_instances += instances_mask
                    number_of_instances = np.max(pred_instances)

            pred_instances_batch.append(pred_instances)

        pred_instances = np.stack(pred_instances_batch, axis=0)  # (B, H, W)

        return pred_instances, pred_sem, pred_boundary

    def plot_instances(self, rgb: np.array, instances: np.array):
        plt.figure(figsize=(20, 6))
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(10, 10)

        rgb = rgb.transpose((1, 2, 0))  # (H, W, 3)
        instances_color = self.id_color_array[instances, :]  # (H, W, 3)

        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.grid(False)
        plt.imshow(rgb)

        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.grid(False)
        plt.imshow(rgb)
        plt.imshow(instances_color, cmap='jet', alpha=0.5, interpolation='nearest')
        plt.show()

    def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        rgb = batch['rgb']  # (B, 3, H, W)
        rgb_original = batch['rgb_original']  # (B, 3, H, W)
        rgb_original = rgb_original.cpu().numpy()  # (B, 3, H, W)
        ego_car_mask = batch.get('ego_car_mask', None)  # (B, H, W)

        pred_instances, pred_sem, pred_boundary = self.predict(rgb, rgb_original, ego_car_mask)

        # Assert that all pixel of the thing classes are assigned to an instance
        for pred_instances_i, pred_sem_i in zip(pred_instances, pred_sem):
            assert pred_sem_i.shape == pred_instances_i.shape
            thing_classes_mask = np.isin(pred_sem_i, self.get_dataset().thing_classes)
            assert np.all(thing_classes_mask == (pred_instances_i != 0))

        if self.test_plot:
            for rgb_original_i, pred_sem_i, pred_instances_i in zip(rgb_original, pred_sem,
                                                                    pred_instances):
                self.plot_instances(rgb_original_i, pred_instances_i)
                self.semantic_model.plot(rgb_original_i, pred_sem_i)

        if self.test_save_dir is not None:
            semantic_path = batch['semantic_path']
            instance_path = batch['instance_path']
            dataset = self.get_dataset()
            dataset_path_base = str(dataset.path_base)

            for pred_sem_i, pred_instances_i, pred_boundary_i, semantic_path_i, instance_path_i in \
                    zip(pred_sem, pred_instances, pred_boundary, semantic_path, instance_path):

                pred_sem_i_gt_format, pred_panoptic_i_gt_format = \
                    dataset.compute_panoptic_label_in_gt_format(pred_sem_i, pred_instances_i)

                pred_sem_i_path = semantic_path_i.replace(dataset_path_base, self.test_save_dir)
                if not os.path.exists(os.path.dirname(pred_sem_i_path)):
                    os.makedirs(os.path.dirname(pred_sem_i_path))
                pred_img = Image.fromarray(pred_sem_i_gt_format.astype(np.uint8))
                pred_img.save(pred_sem_i_path)

                pred_panoptic_i_path = instance_path_i.replace(dataset_path_base,
                                                               self.test_save_dir)
                if not os.path.exists(os.path.dirname(pred_panoptic_i_path)):
                    os.makedirs(os.path.dirname(pred_panoptic_i_path))
                pred_img = Image.fromarray(pred_panoptic_i_gt_format.astype(np.uint16))
                pred_img.save(pred_panoptic_i_path)

                if self.test_save_vis:
                    pred_sem_i_color = self.get_dataset().class_id_to_color()[pred_sem_i,
                                       :]  # (H, W, 3)
                    pred_ins_i_color = self.id_color_array[pred_instances_i, :]  # (H, W, 3)
                    pred_panop_i_color = np.zeros_like(pred_sem_i_color)
                    pred_panop_i_color[pred_instances_i == 0, :] = pred_sem_i_color[
                                                                   pred_instances_i == 0, :]
                    pred_panop_i_color[pred_instances_i != 0, :] = pred_ins_i_color[
                                                                   pred_instances_i != 0, :]

                    pred_img = Image.fromarray(pred_sem_i_color)
                    pred_sem_i_color_path = pred_sem_i_path.replace('.png', '_color.png')
                    pred_sem_i_color_path = pred_sem_i_color_path.replace('.npy', '_color.png')
                    pred_img.save(pred_sem_i_color_path)

                    pred_img = Image.fromarray(pred_panop_i_color)
                    pred_panop_i_color_path = pred_panoptic_i_path.replace('.png', '_color.png')
                    pred_panop_i_color_path = pred_panop_i_color_path.replace('.npy', '_color.png')
                    pred_img.save(pred_panop_i_color_path)

                    if pred_boundary_i is not None:
                        pred_boundary_i_path = pred_panoptic_i_path.replace('.png', '_boundary.png')
                        pred_boundary_i_path = pred_boundary_i_path.replace('.npy', '_boundary.png')
                        pred_img = Image.fromarray(pred_boundary_i.astype(np.uint8) * 255).convert(
                            'RGB')
                        pred_img.save(pred_boundary_i_path)


class InstanceClusterCLI(LightningCLI):

    def __init__(self):
        super().__init__(
            model_class=InstanceCluster,
            seed_everything_default=0,
            parser_kwargs={'parser_mode': 'omegaconf'},
            save_config_callback=None,
        )

    def add_arguments_to_parser(self, parser):
        # Dataset
        parser.add_argument('--data_params', type=Dict)


if __name__ == '__main__':
    cli = InstanceClusterCLI()
