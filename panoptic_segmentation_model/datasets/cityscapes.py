from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from datasets.dataset import Dataset
from datasets.preprocessing import augment_data, prepare_for_network
from numpy.typing import ArrayLike
from PIL import Image
from tqdm import tqdm
from yacs.config import CfgNode as CN


class Cityscapes(Dataset):
    def __init__(
            self,
            mode: str,
            cfg: CN,
            label_mode: str,
            return_only_rgb: bool = False,
            mode_path: Optional[str] = None,
            is_gt: bool = True,
            subset: List[int] = None,
    ):
        super().__init__("cityscapes",
                         ["train", "trainval", "train_extra", "val", ],
                         mode, cfg, label_mode, return_only_rgb)

        if mode_path is not None:
            self.mode_path = mode_path
        else:
            self.mode_path = self.mode
        self.is_gt = is_gt
        self.frame_paths = self._get_frames()
        if self.return_only_rgb:
            self.frame_paths = self._get_frames_only_rgb()

        if subset is not None:
            self.frame_paths = [self.frame_paths[i] for i in subset]

    def _get_frames(self) -> List[Dict[str, Path]]:
        """Gather the paths of the image, annotation, and camera intrinsics files
        Returns
        -------
        frames : list of dictionaries
            List containing the file paths of the RGB image, the semantic and instance annotations,
            and the camera intrinsics
        """
        if self.mode == "trainval":
            self.mode = "train"
            frames = self._get_frames()
            self.mode = "val"
            frames += self._get_frames()
            self.mode = "trainval"
            return frames

        semantic_files = sorted(
            list((self.path_base / "gtFine" / self.mode_path).glob("*/*_gtFine_labelIds.png")))

        if not semantic_files:
            assert False, "No semantic annotations could be found in the specified path."

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

    def _get_frames_only_rgb(self) -> List[Dict[str, Path]]:
        """Gather the paths of the image files if only the RGB images should be returned
        For instance, when training depth only (unsupervised), we can exploit the full sequences
        instead of only the image tuples where there are semantic annotations for the center image.
        """
        frames = []
        with tqdm(desc="Collect Cityscapes RGB frames") as pbar:
            for frame in self.frame_paths:
                image = frame["rgb"]
                center_number = image.stem.split("_")[2]
                number_digits = len(center_number)
                sequence_files = []
                i = -1
                while True:
                    offset_number = int(center_number) + i
                    offset_frame_path = image.parent / image.name.replace(
                        center_number, str(offset_number).zfill(number_digits))
                    if offset_frame_path.exists():
                        sequence_files.append(offset_frame_path)
                        i -= 1
                    else:
                        break
                i = 1
                while True:
                    offset_number = int(center_number) + i
                    offset_frame_path = image.parent / image.name.replace(
                        center_number, str(offset_number).zfill(number_digits))
                    if offset_frame_path.exists():
                        sequence_files.append(offset_frame_path)
                        i += 1
                    else:
                        break
                sequence_files.sort()
                for file in sequence_files:
                    frames.append({"rgb": file, "camera": frame["camera"]})
                    pbar.update(1)
        return frames

    def __getitem__(self, index: int, do_network_preparation: bool = True,
                    do_augmentation: bool = True, return_only_rgb: bool = False) -> Dict[str, Any]:
        """Collect all data for a single sample
        Parameters
        ----------
        index : int
            Will return the data sample with this index
        Returns
        -------
        output : dict
            The output contains the following data:
            1) RGB images: center and offset images (3, H, W)
            2) semantic annotations (H, W)
            3) loss weight for semantic prediction defined by instance size
            4) center heatmap of the instances (1, H, W)
            5) (x,y) offsets to the center of the instances (2, H, W)
            6) loss weights for the center heatmap and the (x,y) offsets (H, W)
            7) camera intrinsics
        """

        # Read center and offset images
        image_path = self.frame_paths[index]["rgb"]
        image = Image.open(image_path).convert("RGB")
        image_size = image.size

        height, width = self.image_size

        output = {
            "rgb": self.resize(image),
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

            # Process instance map for later PQ computation
            # meta = self._get_meta_info(instance)

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
            semantic_weights = np.ones_like(instance_city, dtype=np.uint8)
            semantic_weights[semantic_city == 255] = 0

            # Set the semantic weights by instance mask size
            full_res_h, full_res_w = image_size[1], image_size[0]
            small_instance_area = self.small_instance_area_full_res * (height / full_res_h) * (
                    width / full_res_w)

            inst_id, inst_area = np.unique(instance_city, return_counts=True)

            for instance_id, instance_area in zip(inst_id, inst_area):
                # Skip stuff pixels
                if instance_id == 0:
                    continue
                if instance_area < small_instance_area:
                    semantic_weights[instance_city == instance_id] = self.small_instance_weight

            # Compute center heatmap and (x,y) offsets to the center for each instance
            offset, center = self.get_offset_center(instance_city, self.sigma, self.gaussian)

            # Generate pixel-wise loss weights
            # Unlike Panoptic-DeepLab, we do not consider the is_crowd label. Following them, we
            #  ignore stuff in the offset prediction.
            center_weights = np.ones_like(center, dtype=np.uint8)
            center_weights[0][semantic_city == 255] = 0
            instance_msk_int = instance_msk.astype(np.uint8)
            offset_weights = np.expand_dims(instance_msk_int, axis=0)

            output.update({
                "semantic": semantic_city,
                "semantic_weights": semantic_weights,
                "center": center,
                "center_weights": center_weights,
                "offset": offset,
                "offset_weights": offset_weights,
                "instance": instance_city.astype(np.int32),
            })

        if do_augmentation:
            augment_data(output, self.augmentation_cfg)

        if do_network_preparation:
            # Convert PIL image to torch.Tensor and normalize
            prepare_for_network(output, self.normalization_cfg)

        return output

    def _convert_semantics(self, semantic: ArrayLike) -> ArrayLike:
        if self.label_mode == "cityscapes":
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
            if not self.is_gt:
                # unlabeled (Remap unlabeled to road), for eval all pred must be labeled
                mapping_list.append((0, 0))
        elif self.label_mode == "cityscapes-27":
            # Convert to our labels and set non-existing labels to ignore, i.e., 255
            mapping_list = [
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
                # ----------------------------------------------------------------
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
            ]
            if not self.is_gt:
                # unlabeled (Remap unlabeled to road), for eval all pred must be labeled
                mapping_list.append((0, 0))
        elif self.label_mode == "cityscapes-19":
            # Convert to our labels and set non-existing labels to ignore, i.e., 255
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
                # ----------------------------------------------------------------
                (24, 11),  # person
                (25, 12),  # rider
                (26, 13),  # car
                (27, 14),  # truck
                (28, 15),  # bus
                (31, 16),  # train
                (32, 17),  # motorcycle
                (33, 18),  # bicycle
            ]
            if not self.is_gt:
                # unlabeled (Remap unlabeled to road), for eval all pred must be labeled
                mapping_list.append((0, 0))
        else:
            raise ValueError(f"Unsupported label mode: {self.label_mode}")

        # Remove classes as specified in the config file
        mapping_list = self._rm_classes_mapping(self.remove_classes, mapping_list)

        semantic_out = 255 * np.ones_like(semantic, dtype=np.uint8)
        for mapping in mapping_list:
            semantic_out[semantic == mapping[0]] = mapping[1]
        return semantic_out
