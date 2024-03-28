from typing import List, Optional

import numpy as np
import torch
from torch import Tensor, distributed


class SemanticEvaluator:
    """
    Evaluate semantic segmentation
    """

    def __init__(self, num_classes: int, ignore_classes: Optional[List[int]] = None,
                 ignore_index: int = 255):
        """
        Args:
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ignore_classes = ignore_classes if ignore_classes is not None else []

    def compute_sem_miou(self, sem_conf_mat, sum_pixels: bool = False):
        # If 'sum_pixels=True', a single value is returned where the IoU is computed as the mean
        #  over all classes weighted by pixels. Otherwise, the IoU of each class is returned.

        # Remove ignore_classes from the confusion matrix
        keep_classes = np.array(
            [i for i in range(self.num_classes) if i not in self.ignore_classes])
        filtered_sem_conf_mat = sem_conf_mat[keep_classes, :][:, keep_classes]

        # Computer mean IoU
        sem_intersection = filtered_sem_conf_mat.diag()
        sem_union = filtered_sem_conf_mat.sum(dim=1) + filtered_sem_conf_mat.sum(
            dim=0) - filtered_sem_conf_mat.diag() + 1e-8

        if sum_pixels:
            sem_miou = sem_intersection.sum() / sem_union.sum()
        else:
            sem_miou = sem_intersection / sem_union

        return sem_miou

    def filter_sem_conf_mat(self, sem_conf_mat, device, debug):
        # Remove all classes that shall be ignored
        sem_conf_mat = sem_conf_mat.to(device=device)
        if not debug:
            distributed.all_reduce(sem_conf_mat, distributed.ReduceOp.SUM)
        sem_conf_mat = sem_conf_mat.cpu()[:self.num_classes, :]

        return sem_conf_mat

    def compute_confusion_matrix(self, semantic_pred: Tensor, semantic_gt: Tensor) -> Tensor:
        conf_mat = semantic_pred[0].new_zeros(self.num_classes * self.num_classes, dtype=torch.int)
        for pred, target in zip(semantic_pred, semantic_gt):
            if self.ignore_classes:
                valid = torch.logical_and(target != self.ignore_index,
                                          ~sum(target == i for i in self.ignore_classes).bool())
            else:
                valid = target != self.ignore_index
            if valid.any():
                pred = pred[valid].int()
                target = target[valid].int()
                conf_mat.index_add_(0, target * self.num_classes + pred,
                                    conf_mat.new_ones(target.numel()))
        return conf_mat.view(self.num_classes, self.num_classes)
