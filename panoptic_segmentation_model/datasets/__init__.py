import operator
from typing import List

from datasets.cityscapes import Cityscapes
from datasets.kitti_360 import Kitti360
from datasets.sampler import DistributedMixedBatchSampler
from datasets.spino_cityscapes_labels import Label
from datasets.spino_cityscapes_labels import \
    labels_19 as spino_cityscapes_labels_19
from datasets.spino_cityscapes_labels import \
    labels_27 as spino_cityscapes_labels_27
from datasets.spino_kitti_360_labels import \
    labels_14 as spino_kitti_360_labels_14

__all__ = ["Cityscapes", "Kitti360"]


def get_labels(remove_classes: List[int], mode: str) -> List[Label]:
    if mode == "cityscapes-19":
        labels = [label for label in spino_cityscapes_labels_19 if label.trainId not in [-1, 255]]
    elif mode == "cityscapes-27":
        labels = [label for label in spino_cityscapes_labels_27 if label.trainId not in [-1, 255]]
    elif mode == "kitti-360-14":
        labels = [label for label in spino_kitti_360_labels_14 if label.trainId not in [-1, 255]]
    else:
        raise ValueError(f"Unsupported label mode: {mode}")
    labels = sorted(labels, key=operator.attrgetter("trainId"))

    train_id = 0
    adapted_labels = []
    for label in labels:
        if label.trainId in remove_classes:
            continue
        label = label._replace(trainId=train_id)
        train_id += 1
        adapted_labels.append(label)

    return adapted_labels
