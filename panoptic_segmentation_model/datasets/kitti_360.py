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


class Kitti360(Dataset):
    def __init__(
            self,
            mode: str,
            cfg: CN,
            label_mode: str,
            return_only_rgb: bool = False,
            sequences: Optional[List[str]] = None,
            sequence_reference_mode: str = "rgb",
            mode_path: Optional[str] = None,
            is_gt: bool = True,
            subset: List[int] = None,
    ):
        super().__init__("kitti_360", ["train", "val", "sequence"], mode, cfg, label_mode,
                         return_only_rgb)
        if mode == "sequence":
            assert sequence_reference_mode in ["semantic", "rgb"], \
                f"Unsupported sequence reference mode: {sequence_reference_mode}"
            assert sequences is not None and len(sequences) > 0, \
                "In 'sequence' mode, sequences have to be given."
            for seq in sequences:
                assert seq in ["00", "02", "03", "04", "05", "06", "07", "09", "10"], \
                    f"Passed invalid sequence: {seq}"
        self.sequence_reference_mode = sequence_reference_mode
        # Boolean list, whether the data has been added. Using this we can reconstruct the indices
        #  between both RGB and semantic sequence modes.
        self.semantic_seq_mode_image_skipped = []

        if mode_path is not None:
            self.mode_path = mode_path
        else:
            self.mode_path = ""
        self.is_gt = is_gt

        self.sequences = sequences if self.mode == "sequence" else None
        self.frame_paths = self._get_frames()
        if self.return_only_rgb:
            assert self.mode != "sequence", "Not implemented"
            self.frame_paths = self._get_frames_only_rgb()
        self.lidar_to_rect, self.camera_matrix = None, None

        if subset is not None:
            self.frame_paths = [self.frame_paths[i] for i in subset]

    def _get_frames(self) -> List[Dict[str, Path]]:
        """Gather the paths of the image and annotation files
        Returns
        -------
        frames : list of dictionaries
            List containing the file paths of the RGB image, the semantic and instance annotations
        """
        frames = []
        if self.sequences is None:
            filename = self.path_base / "data_2d_semantics" / "train" / \
                       f"2013_05_28_drive_{self.mode}_frames.txt"
            with open(filename, "r", encoding="utf-8") as file:
                lines = file.read().splitlines()
            for line in tqdm(lines, desc=f"Collect KITTI-360 frames [{self.mode}]"):
                # Skip this file as there is no preceding file for the image triplet
                if self.mode == "val" and "0000004391.png" in line:
                    continue

                rgb = self.path_base / line.split(" ")[0]
                semantic = self.path_base / line.split(" ")[1]
                instance = semantic.parents[1] / "instance" / semantic.name
                frames.append({"rgb": rgb, "semantic": semantic, "instance": instance})
                for path in frames[-1].values():
                    if path is not None:
                        assert path.exists(), f"File does not exist: {path}"
                if self.mode_path != "":
                    raise NotImplementedError
        else:
            rgb_files = []
            for sequence in self.sequences:
                sequence_files = sorted(list((self.path_base / "data_2d_raw" /
                                              f"2013_05_28_drive_00{sequence}_sync" /
                                              "image_00" / "data_rect").glob("*.png")))

                rgb_files += sequence_files
            for rgb in tqdm(rgb_files, desc=f"Collect KITTI-360 frames: Seq. {self.sequences}"):
                sequence = rgb.parents[2].name
                sequence_labels = sequence + self.mode_path
                semantic = self.path_base / "data_2d_semantics" / "train" / sequence_labels / \
                           "image_00" / "semantic" / rgb.name
                instance = self.path_base / "data_2d_semantics" / "train" / sequence_labels / \
                           "image_00" / "instance" / rgb.name

                if self.sequence_reference_mode == "semantic":
                    if not semantic.exists() or not instance.exists():
                        self.semantic_seq_mode_image_skipped.append(True)
                        continue
                    self.semantic_seq_mode_image_skipped.append(False)
                else:  # 'rgb'
                    semantic = semantic if semantic.exists() else None
                    instance = instance if instance.exists() else None

                frames.append({"rgb": rgb, "semantic": semantic, "instance": instance})
                # if len(frames) == 100:
                #     break
        return frames

    def _get_frames_only_rgb(self) -> List[Dict[str, Path]]:
        """Gather the paths of the image files if only the RGB images should be returned
        For instance, when training depth only (unsupervised), we can exploit the full sequences
        instead of only the image tuples where there are semantic annotations for the center image.
        """
        frames = []
        # Iterate over the sequences
        sequences = sorted(list((self.path_base / "data_2d_raw").glob("*")))
        for sequence in tqdm(sequences, desc="Collect KITTI-360 RGB frames"):
            sequence_files = sorted(list(sequence.glob("image_00/data_rect/*.png")))
            # Remove frames such that the offset images will not be out of range
            for file in sequence_files:
                frames.append({"rgb": file})
        return frames

    def _read_lidar_to_rect(self) -> ArrayLike:
        # Prepare transform from Velodyne to rectified image
        cam_to_velo_path = self.path_base / "calibration" / "calib_cam_to_velo.txt"
        lastrow = np.array([0, 0, 0, 1]).reshape(1, 4)
        cam_to_velo = np.concatenate(
            (np.loadtxt(cam_to_velo_path).reshape(3, 4), lastrow))
        rect_path = self.path_base / "calibration" / "perspective.txt"
        with open(rect_path, "r", encoding="utf-8") as file:
            lines = file.read().splitlines()
        rect, K = np.eye(4), np.eye(3, 4)
        for line in lines:
            if line.split(" ")[0] == "R_rect_00:":
                rect[:3, :3] = np.fromstring(line.replace("R_rect_00: ", ""),
                                             dtype=float, sep=" ").reshape(3, 3)
            elif line.split(" ")[0] == "P_rect_00:":
                K = np.fromstring(line.replace("P_rect_00: ", ""), dtype=float,
                                  sep=" ").reshape(3, 4)
        velo_to_cam = np.linalg.inv(cam_to_velo)
        velo_to_rect = rect @ velo_to_cam
        return velo_to_rect, K

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
            3) center heatmap of the instances (1, H, W)
            4) (x,y) offsets to the center of the instances (2, H, W)
            5) loss weights for the center heatmap and the (x,y) offsets (H, W)
            6) camera intrinsics
        """

        # Read center and offset images
        image_path = self.frame_paths[index]["rgb"]
        image = Image.open(image_path).convert("RGB")
        image_size = image.size

        output = {
            "rgb": self.resize(image),
        }

        if not (self.return_only_rgb or return_only_rgb):
            # Read semantic map and convert to Cityscapes labels
            if not self.frame_paths[index]["semantic"] is None and not self.frame_paths[index][
                                                                           "instance"] is None:
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

                # Convert to Cityscapes labels
                semantic_city = self._convert_semantics(semantic)

                # Compute instance IDs for thing classes in the Cityscapes domain.
                # For stuff, we set the ID to 0.
                class_instance = instance - semantic * 1000
                thing_mask = self._make_thing_mask(semantic_city, as_bool=True)
                instance_city = np.zeros_like(instance, dtype=np.uint16)
                instance_city[
                    thing_mask] = semantic_city[thing_mask] * 1000 + class_instance[thing_mask]

                # Generate semantic_weights map by instance mask size
                semantic_weights = np.ones_like(instance_city, dtype=np.uint8)
                semantic_weights[semantic_city == 255] = 0

                # Semantic map used for evaluation (without very small instances)
                semantic_eval = semantic_city.copy()

                # Set the semantic weights by instance mask size
                height, width = self.image_size
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

                    # For evaluation, remove very small instances
                    if instance_area < small_instance_area * .1:
                        semantic_eval[instance_city == instance_id] = 255

                # Compute center heatmap and (x,y) offsets to the center for each instance
                offset, center = self.get_offset_center(instance_city, self.sigma, self.gaussian)

                # Generate pixel-wise loss weights
                center_weights = np.expand_dims(
                    self._make_thing_mask(semantic_city), axis=0)
                offset_weights = np.expand_dims(
                    self._make_thing_mask(semantic_city), axis=0)

                output.update({
                    "semantic": semantic_city,
                    "semantic_eval": semantic_eval,
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
            # https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/helpers/labels.py
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
                (34, 2),  # garage -> building
                (35, 4),  # gate -> fence
                (37, 5),  # smallpole -> pole
            ]
        elif self.label_mode == "kitti-360-14":
            # Convert to our labels and set non-existing labels to ignore, i.e., 255
            mapping_list = [
                (7, 0),  # road
                (8, 1),  # sidewalk
                (11, 2),  # building
                (34, 2),  # garage -> building
                (12, 2),  # wall -> building
                (13, 3),  # fence
                (35, 3),  # gate -> fence
                (17, 4),  # pole
                (37, 4),  # smallpole -> pole
                (20, 5),  # traffic sign
                (21, 6),  # vegetation
                (22, 7),  # terrain
                (23, 8),  # sky
                (24, 9),  # person
                (25, 10),  # rider
                (26, 11),  # car
                (27, 12),  # truck
                (32, 13),  # motorcycle -> two-wheeler
                (33, 13),  # bicycle -> two-wheeler
            ]
            if not self.is_gt:
                # unlabeled (Remap unlabeled to road), for eval all pred must be labeled
                mapping_list.append((0, 0))
                # unlabeled (Remap unlabeled to road), for eval all pred must be labeled
                mapping_list.append((255, 0))
        else:
            raise ValueError(f"Unsupported label mode: {self.label_mode}")

        # Remove classes as specified in the config file
        mapping_list = self._rm_classes_mapping(self.remove_classes, mapping_list)

        semantic_city = 255 * np.ones_like(semantic, dtype=np.uint8)
        for mapping in mapping_list:
            semantic_city[semantic == mapping[0]] = mapping[1]
        return semantic_city
