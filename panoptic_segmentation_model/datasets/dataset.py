import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms
from yacs.config import CfgNode as CN


class Dataset(TorchDataset, ABC):
    """Torch dataset wrapper for custom datasets
    Parameters
    ----------
    assert_name : string
        Name of the custom dataset
    assert_modes : list of strings
        List of supported modes
    mode : string
        Which data to return, e.g., train vs val
    cfg : CfgNode
        The global configuration
    sigma_gaussian : int
        Sigma of the Gaussian used to compute the instance center heatmap
    return_only_rgb : bool
        If set to True, only the RGB images will be return resulting in more valid frames
    """

    def __init__(self,
                 assert_name: str,
                 assert_modes: List[str],
                 mode: str,
                 cfg: CN,
                 label_mode: str,
                 return_only_rgb: bool = False,
                 ):
        super().__init__()
        assert mode in assert_modes, f"Unsupported mode: {mode}"
        self.mode = mode
        self.return_only_rgb = return_only_rgb
        assert label_mode in ["kitti-360-14", "cityscapes-19", "cityscapes-27", "cityscapes"], \
            f"Unsupported label mode: {label_mode}"
        self.label_mode = label_mode

        # Parse configuration
        assert cfg.name == assert_name
        self.path_base = Path(cfg.path)

        assert os.path.exists(self.path_base), \
            f"The specified dataset path '{self.path_base}' does not exist."

        self.image_size = cfg.feed_img_size  # [H, W]
        self.sigma = cfg.center_heatmap_sigma
        self.small_instance_weight = cfg.small_instance_weight
        self.small_instance_area_full_res = cfg.small_instance_area_full_res
        self.augmentation_cfg = cfg.augmentation
        self.normalization_cfg = cfg.normalization
        self.remove_classes = cfg.remove_classes

        if self.mode == "val":
            self.augmentation_cfg = CN({"active": False})

        self.resize = transforms.Resize(self.image_size,
                                        interpolation=transforms.InterpolationMode.LANCZOS)

        # Compute the Gaussian kernel
        size = 6 * self.sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * self.sigma + 1, 3 * self.sigma + 1
        self.gaussian = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

        # This will be loaded in the specific datasets
        self.frame_paths = []

        # Dataset image features file
        self.features_file = self.path_base / "features.npy"
        self.features_file = self.features_file if self.features_file.exists() else None
        self.class_distribution_file = self.path_base / "class_distribution.pkl"
        self.class_distribution_file = self.class_distribution_file if \
            self.class_distribution_file.exists() else None

    # ----------------------------------------------------------------

    @abstractmethod
    def _get_frames(self) -> List[Dict[str, Path]]:
        raise NotImplementedError

    def _get_frames_only_rgb(self) -> List[Dict[str, Path]]:
        raise NotImplementedError

    # ----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.frame_paths)

    @abstractmethod
    def __getitem__(self, index: int, do_network_preparation: bool = True,
                    do_augmentation: bool = True, return_only_rgb: bool = False) -> Dict[str, Any]:
        raise NotImplementedError

    def _make_thing_mask(self, semantic: ArrayLike, as_bool: bool = False) -> ArrayLike:
        """
        Return an integer mask with 1 if pixel is a `thing`,
         otherwise return 0 for `stuff` and "ignore"
        Parameters
        ----------
        semantic : numpy array
            Semantic annotations
        as_bool : bool
            If set to true, the integer map (0, 1) is converted to boolean (False, True)
        Returns
        -------
        mask : numpy array
            Integer / boolean mask with 1 / True for thing pixels and 0 / False for stuff pixels
        """
        thing_ids = self.thing_classes
        mask = np.zeros_like(semantic, dtype=np.uint8)
        for thing in thing_ids:
            mask[semantic == thing] = 1
        if as_bool:
            return mask.astype(bool)
        return mask

    @staticmethod
    def get_offset_center(instance_city: ArrayLike, sigma: Optional[float] = 8,
                          gaussian: Optional[ArrayLike] = None) -> Tuple[ArrayLike, ArrayLike]:
        if gaussian is None:
            size = 6 * sigma + 3
            x = np.arange(0, size, 1, float)
            y = x[:, np.newaxis]
            x0, y0 = 3 * sigma + 1, 3 * sigma + 1
            gaussian = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        height, width = instance_city.shape
        # Compute center heatmap and (x,y) offsets to the center for each instance
        center = np.zeros((1, height, width), dtype=np.float32)
        offset = np.zeros((2, height, width), dtype=np.float32)
        x_coord = np.ones_like(instance_city, dtype=np.float32)
        y_coord = np.ones_like(instance_city, dtype=np.float32)
        x_coord = np.cumsum(x_coord, axis=1) - 1
        y_coord = np.cumsum(y_coord, axis=0) - 1
        inst_id, inst_area = np.unique(instance_city, return_counts=True)

        for instance_id, instance_area in zip(inst_id, inst_area):
            # Skip stuff and unlabeled pixels
            if instance_id == 0:
                continue

            # Find instance center
            mask_index = np.where(instance_city == instance_id)
            center_y, center_x = np.mean(mask_index[0]), np.mean(mask_index[1])

            # Generate heatmap
            y, x = int(np.round(center_y)), int(np.round(center_x))
            # Outside image boundary
            if not 0 <= x < width or not 0 <= y < height:
                continue
            # Upper left
            ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
            # Bottom right
            br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

            # Update the center heatmap
            c, d = max(0, -ul[0]), min(br[0], width) - ul[0]
            a, b = max(0, -ul[1]), min(br[1], height) - ul[1]
            cc, dd = max(0, ul[0]), min(br[0], width)
            aa, bb = max(0, ul[1]), min(br[1], height)
            center[0, aa:bb, cc:dd] = np.maximum(center[0, aa:bb, cc:dd], gaussian[a:b, c:d])

            # Generate the offset in x and y directions
            offset_y_index = (np.zeros_like(mask_index[0]), mask_index[0], mask_index[1])
            offset_x_index = (np.ones_like(mask_index[0]), mask_index[0], mask_index[1])
            offset[offset_y_index] = center_y - y_coord[mask_index]
            offset[offset_x_index] = center_x - x_coord[mask_index]

        return offset, center

    @staticmethod
    def _rm_classes_mapping(
            remove_classes: List[int],
            mapping_list: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        sub_list = dict.fromkeys(mapping_list, 0)
        for k_del in remove_classes:
            for idx, elem in enumerate(mapping_list):
                if elem[1] > k_del:
                    sub_list[elem] += 1
                elif elem[1] == k_del:
                    del sub_list[elem]
        adapted_list = [(k[0], (k[1] - v)) for k, v in sub_list.items()]
        return adapted_list

    # ----------------------------------------------------------------

    @property
    def stuff_classes(self) -> List[int]:
        if self.label_mode == "cityscapes-19":
            class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        elif self.label_mode == "cityscapes-27":
            class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        elif self.label_mode == "cityscapes":
            class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        elif self.label_mode =="kitti-360-14":
            class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        else:
            assert False, f"Unsupported label mode: {self.label_mode}"
        count = 0
        for cls in self.remove_classes:
            if cls in class_list:
                count += 1
        class_list = class_list[:-count] if count > 0 else class_list
        return class_list

    @property
    def thing_classes(self) -> List[int]:
        if self.label_mode == "cityscapes-19":
            class_list = [11, 12, 13, 14, 15, 16, 17, 18]
        elif self.label_mode == "cityscapes-27":
            class_list = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        elif self.label_mode == "cityscapes":
            class_list = [11, 12, 13, 14, 15, 16, 17, 18]
        elif self.label_mode =="kitti-360-14":
            class_list = [9, 10, 11, 12, 13]
        else:
            assert False, f"Unsupported label mode: {self.label_mode}"
        count_thing = 0
        count_stuff = 0
        for cls in self.remove_classes:
            if cls in class_list:
                count_thing += 1
            else:
                count_stuff += 1

        class_list = [elem - count_stuff for elem in class_list]
        class_list = class_list[:-count_thing] if count_thing > 0 else class_list
        return class_list

    @property
    def all_classes(self) -> List[int]:
        return self.stuff_classes + self.thing_classes

    @property
    def ignore_classes(self) -> List[int]:
        """Return the classes that are present in Cityscapes but not in the specific dataset"""
        return []

    @property
    def num_stuff(self) -> int:
        return len(self.stuff_classes)

    @property
    def num_things(self) -> int:
        return len(self.thing_classes)

    @property
    def num_classes(self) -> int:
        return len(self.all_classes)
