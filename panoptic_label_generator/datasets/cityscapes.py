from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
from numpy.typing import ArrayLike
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from yacs.config import CfgNode as CN

from .dataset import Dataset


class Cityscapes(Dataset):
    CLASS_COLOR_19 = np.zeros((256, 3), dtype=np.uint8)
    CLASS_COLOR_19[:19, :] = np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ])

    CLASS_COLOR_27 = np.zeros((256, 3), dtype=np.uint8)
    CLASS_COLOR_27[:27, :] = np.array([
        [128, 64, 128],
        [244, 35, 232],
        [250, 170, 160],
        [230, 150, 140],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [180, 165, 180],
        [150, 100, 100],
        [150, 120, 90],
        [153, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 0, 90],
        [0, 0, 110],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ])

    def __init__(
            self,
            mode: str,
            cfg: CN,
            transform: List[Callable],
            return_only_rgb: bool = False,
            label_mode: str = "cityscapes_19",
            subset: List[int] = None,
    ):
        super().__init__("cityscapes", ["train", "val", "train_extra", "video"], mode, cfg,
                         transform, return_only_rgb,
                         label_mode)

        if mode in ["train", "val"]:
            self.frame_paths = self._get_frames()
        elif mode in ["train_extra", "video"]:  # no labels available
            self.frame_paths = self._get_frames_by_rgb()
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if subset is not None:
            self.frame_paths = [self.frame_paths[i] for i in subset]

    def _get_frames_by_rgb(self) -> List[Dict[str, Path]]:
        rgb_files = sorted(list((self.path_base / "leftImg8bit" / self.mode).glob("*/*.png")))
        frames = []
        for rgb in tqdm(rgb_files, desc=f"Collect Cityscapes frames [{self.mode}]"):
            file_base = rgb.stem.replace("_leftImg8bit", "")
            city = file_base.split("_")[0]
            # semantic and instance files do not need to exist, but we need the paths to save
            # the predictions
            semantic = self.path_base / "gtFine" / self.mode / city / \
                       f"{file_base}_gtFine_labelIds.png"
            instance = semantic.parent / semantic.name.replace("label", "instance")
            frames.append({
                "rgb": rgb,
                "semantic": semantic,
                "instance": instance,
            })
            assert rgb.exists(), f"RGB file does not exist: {rgb}"
        return frames

    def _get_frames(self) -> List[Dict[str, Path]]:
        """Gather the paths of the image, annotation, and camera intrinsics files
        Returns
        -------
        frames : list of dictionaries
            List containing the file paths of the RGB image, the semantic and instance annotations,
            and the camera intrinsics
        """
        semantic_files = sorted(
            list((self.path_base / "gtFine" / self.mode).glob("*/*_gtFine_labelIds.png")))
        frames = []
        for semantic in tqdm(semantic_files, desc=f"Collect Cityscapes frames [{self.mode}]"):
            file_base = semantic.stem.replace("_gtFine_labelIds", "")
            city = file_base.split("_")[0]
            rgb = self.path_base / "leftImg8bit_sequence" / self.mode / city / \
                  f"{file_base}_leftImg8bit.png"
            instance = semantic.parent / semantic.name.replace("label", "instance")
            frames.append({
                "rgb": rgb,
                "semantic": semantic,
                "instance": instance,
            })
            for path in frames[-1].values():
                if path is not None:
                    assert path.exists(), f"File does not exist: {path}"
        return frames

    def __getitem__(
            self,
            index: int,
            do_transform=True,
            return_only_rgb: bool = False
    ) -> Dict[str, Any]:
        """Collect all data for a single sample
        Parameters
        ----------
        index : int
            Will return the data sample with this index
        do_transform : bool
            If True, apply transform.
        return_only_rgb : bool
            If True, return only RGB image and no GT.
        Returns
        -------
        output : dict
            The output contains the following data:
            1) RGB image (3, H, W)
            2) filepath of RGB image
            3) index of the sample
            4) semantic and instance annotation (H, W)
            5) filepath of semantic and instance annotation
            6) loss weight for semantic prediction defined by instance size
            7) center heatmap of the instances (1, H, W)
            8) (x,y) offsets to the center of the instances (2, H, W)
            9) loss weights for the center heatmap and the (x,y) offsets (H, W)
            10) ego car mask (H, W)
        """

        # Read image
        image_path = self.frame_paths[index]["rgb"]
        image = Image.open(image_path).convert("RGB")
        # image_size = image.size
        image = self.resize(image)
        # height, width = self.image_size

        output = {
            "rgb": image,
            "rgb_path": str(image_path),
            "index": index,
        }

        if not (self.return_only_rgb or return_only_rgb):
            # Read semantic map
            semantic_path = self.frame_paths[index]["semantic"]
            semantic = cv2.imread(str(semantic_path), cv2.IMREAD_GRAYSCALE)  # 8-bit
            semantic = cv2.resize(semantic,
                                  list(reversed(self.image_size)),
                                  interpolation=cv2.INTER_NEAREST)

            # Read instance and convert to center heatmap and offset map
            instance_path = self.frame_paths[index]["instance"]
            instance = cv2.imread(str(instance_path), cv2.IMREAD_ANYDEPTH)  # 16-bit
            instance = cv2.resize(instance,
                                  list(reversed(self.image_size)),
                                  interpolation=cv2.INTER_NEAREST)

            ego_car_mask = np.logical_or(semantic == 1,
                                         semantic == 2)  # 1 = ego vehicle, 2 = rectification border

            # Convert to Cityscapes labels
            semantic_city = self._convert_semantics(semantic)

            # Compute instance IDs for thing classes in the Cityscapes domain.
            # For stuff, we set the ID to 0.
            class_instance = instance - semantic * 1000
            thing_mask = self._make_thing_mask(semantic_city, as_bool=True)
            instance_msk = thing_mask.copy()
            # Remove iscrowd instances
            instance_msk[instance < 1000] = False
            instance_city = np.zeros_like(instance, dtype=np.uint16)
            instance_city[instance_msk] = semantic_city[instance_msk] * 1000 + class_instance[
                instance_msk]

            # Generate semantic_weights map by instance mask size
            # semantic_weights = np.ones_like(instance_city, dtype=np.uint8)
            # semantic_weights[semantic_city == 255] = 0

            # Set the semantic weights by instance mask size
            # full_res_h, full_res_w = image_size[1], image_size[0]
            # small_instance_area = self.small_instance_area_full_res * (height / full_res_h) * (
            #         width / full_res_w)

            # inst_id, inst_area = np.unique(instance_city, return_counts=True)
            #
            # for instance_id, instance_area in zip(inst_id, inst_area):
            #     # Skip stuff pixels
            #     if instance_id == 0:
            #         continue
            #     if instance_area < small_instance_area:
            #         semantic_weights[instance_city == instance_id] = self.small_instance_weight
            #
            # # Compute center heatmap and (x,y) offsets to the center for each instance
            # offset, center = self.get_offset_center(instance_city, self.sigma, self.gaussian)
            #
            # # Generate pixel-wise loss weights
            # # Unlike Panoptic-DeepLab, we do not consider the is_crowd label. Following them, we
            # #  ignore stuff in the offset prediction.
            # center_weights = np.ones_like(center, dtype=np.uint8)
            # center_weights[0][semantic_city == 255] = 0
            # instance_msk_int = instance_msk.astype(np.uint8)
            # offset_weights = np.expand_dims(instance_msk_int, axis=0)

            output.update({
                "semantic": semantic_city,
                # "semantic_weights": semantic_weights,
                # "center": center,
                # "center_weights": center_weights,
                # "offset": offset,
                # "offset_weights": offset_weights,
                # "thing_mask": thing_mask.astype(np.uint8),
                "instance": instance_city.astype(np.int32),
                "ego_car_mask": ego_car_mask,
            })

        # For saving predictions in the Cityscapes format
        semantic_path = self.frame_paths[index]["semantic"]
        instance_path = self.frame_paths[index]["instance"]
        output.update({
            "semantic_path": str(semantic_path),
            "instance_path": str(instance_path),
        })

        if do_transform:
            output = self.transform(output)

        return output

    def _convert_semantics(self, semantic: ArrayLike) -> ArrayLike:
        if self.label_mode == "cityscapes_19":
            # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
            # Convert to Cityscapes labels and set non-existing labels to ignore, i.e., 255
            mapping_list = [
                (7, 0),  # road
                (8, 1),  # sidewalk
                (11, 2),  # building
                (12, 3),  # wall
                (13, 4),  # fence
                (17, 5),  # pole
                (19, 6),  # traffic light
                (20, 7),  # traffic sign
                (21, 8),  # vegetation
                (22, 9),  # terrain
                (23, 10),  # sky
                (24, 11),  # person
                (25, 12),  # rider
                (26, 13),  # car
                (27, 14),  # truck
                (28, 15),  # bus
                (31, 16),  # train
                (32, 17),  # motorcycle
                (33, 18),  # bicycle
            ]
        elif self.label_mode == "cityscapes_27":
            # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
            # Convert to Cityscapes labels and set non-existing labels to ignore, i.e., 255
            mapping_list = [
                (0, 255),  # unlabeled
                (1, 255),  # ego vehicle
                (2, 255),  # rect border
                (3, 255),  # out of roi
                (4, 255),  # static
                (5, 255),  # dynamic
                (6, 255),  # ground
                (7, 0),  # road
                (8, 1),  # sidewalk
                (9, 2),  # parking
                (10, 3),  # rail track
                (11, 4),  # building
                (12, 5),  # wall
                (13, 6),  # fence
                (14, 7),  # guard rail
                (15, 8),  # bridge
                (16, 9),  # tunnel
                (17, 10),  # pole
                (18, 11),  # polegroup
                (19, 12),  # traffic light
                (20, 13),  # traffic sign
                (21, 14),  # vegetation
                (22, 15),  # terrain
                (23, 16),  # sky
                (24, 17),  # person
                (25, 18),  # rider
                (26, 19),  # car
                (27, 20),  # truck
                (28, 21),  # bus
                (29, 22),  # caravan
                (30, 23),  # trailer
                (31, 24),  # train
                (32, 25),  # motorcycle
                (33, 26),  # bicycle
                (-1, 255),  # license plate
            ]
        else:
            raise ValueError(f"Unsupported label mode: {self.label_mode}")

        # Remove classes as specified in the config file
        mapping_list = self._rm_classes_mapping(self.remove_classes, mapping_list)

        semantic_out = 255 * np.ones_like(semantic, dtype=np.uint8)
        for mapping in mapping_list:
            semantic_out[semantic == mapping[0]] = mapping[1]
        return semantic_out

    def _convert_semantic_to_gt_format(self, semantic: np.array) -> np.array:
        if self.label_mode == "cityscapes_19":
            mapping_list = [
                (0, 7),  # road
                (1, 8),  # sidewalk
                (2, 11),  # building
                (3, 12),  # wall
                (4, 13),  # fence
                (5, 17),  # pole
                (6, 19),  # traffic light
                (7, 20),  # traffic sign
                (8, 21),  # vegetation
                (9, 22),  # terrain
                (10, 23),  # sky
                (11, 24),  # person
                (12, 25),  # rider
                (13, 26),  # car
                (14, 27),  # truck
                (15, 28),  # bus
                (16, 31),  # train
                (17, 32),  # motorcycle
                (18, 33),  # bicycle
                (255, 0),  # unlabeled
            ]
        elif self.label_mode == "cityscapes_27":
            mapping_list = [
                (0, 7),  # road
                (1, 8),  # sidewalk
                (2, 9),  # parking
                (3, 10),  # rail track
                (4, 11),  # building
                (5, 12),  # wall
                (6, 13),  # fence
                (7, 14),  # guard rail
                (8, 15),  # bridge
                (9, 16),  # tunnel
                (10, 17),  # pole
                (11, 18),  # polegroup
                (12, 19),  # traffic light
                (13, 20),  # traffic sign
                (14, 21),  # vegetation
                (15, 22),  # terrain
                (16, 23),  # sky
                (17, 24),  # person
                (18, 25),  # rider
                (19, 26),  # car
                (20, 27),  # truck
                (21, 28),  # bus
                (22, 29),  # caravan
                (23, 30),  # trailer
                (24, 31),  # train
                (25, 32),  # motorcycle
                (26, 33),  # bicycle
                (255, 0),  # unlabeled
            ]
        else:
            raise NotImplementedError(f"Unsupported label mode: {self.label_mode}")

        semantic_out = 255 * np.ones_like(semantic, dtype=np.uint8)
        for mapping in mapping_list:
            semantic_out[semantic == mapping[0]] = mapping[1]
        return semantic_out

    def class_id_to_color(self):
        if self.label_mode == "cityscapes_19":
            return self.CLASS_COLOR_19
        if self.label_mode == "cityscapes_27":
            return self.CLASS_COLOR_27
        raise NotImplementedError(f"Unsupported label mode: {self.label_mode}")

    def compute_panoptic_label_in_gt_format(self, pred_semantic: np.array,
                                            pred_instance: np.array) -> Tuple[np.array, np.array]:
        semantic = self._convert_semantic_to_gt_format(pred_semantic)

        instance_mask = pred_instance > 0
        instance_per_sem_class = np.zeros_like(pred_instance, dtype=np.uint16)
        for sem_class_id in np.unique(semantic):
            instance_ids = np.unique(pred_instance[semantic == sem_class_id])
            instance_ids = instance_ids[instance_ids > 0]
            total_instances_sem_class = 1
            for instance_id in instance_ids:
                instance_per_sem_class[pred_instance == instance_id] = total_instances_sem_class
                total_instances_sem_class += 1

        panoptic = semantic.copy().astype(np.uint16)
        panoptic[instance_mask] = semantic[instance_mask] * 1000 + instance_per_sem_class[
            instance_mask]

        return semantic, panoptic


class CityscapesDataModule(pl.LightningDataModule):

    def __init__(self, cfg_dataset: Dict[str, Any], num_classes: int, batch_size: int,
                 num_workers: int,
                 transform_train: List[Callable], transform_test: List[Callable], label_mode: str,
                 train_sample_indices: Optional[List[int]] = None,
                 test_sample_indices: Optional[List[int]] = None,
                 test_set: str = "train"):
        super().__init__()
        self.cfg_dataset = CN(init_dict=cfg_dataset)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.label_mode = label_mode
        self.train_sample_indices = train_sample_indices
        self.test_sample_indices = test_sample_indices
        self.test_set = test_set

        self.cityscapes_train: Optional[Cityscapes] = None
        self.cityscapes_test: Optional[Cityscapes] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.cityscapes_train = Cityscapes("train", self.cfg_dataset,
                                               transform=self.transform_train,
                                               return_only_rgb=False, label_mode=self.label_mode,
                                               subset=self.train_sample_indices)
            assert self.cityscapes_train.num_classes == self.num_classes
        if stage == "validate" or stage is None:
            pass

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            return_only_rgb = self.test_set in ["train_extra", "video"]
            self.cityscapes_test = Cityscapes(self.test_set, self.cfg_dataset,
                                              transform=self.transform_test,
                                              return_only_rgb=return_only_rgb,
                                              label_mode=self.label_mode,
                                              subset=self.test_sample_indices)
            assert self.cityscapes_test.num_classes == self.num_classes

        if stage == "predict" or stage is None:
            pass

    def train_dataloader(self):
        return DataLoader(self.cityscapes_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=False)

    def test_dataloader(self):
        return DataLoader(self.cityscapes_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=False, drop_last=False)
