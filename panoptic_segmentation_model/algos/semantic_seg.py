# pylint: disable=condition-evals-to-constant, unused-argument, unused-import

from typing import Dict, List, Optional, Tuple

import torch
from eval import SemanticEvaluator
from models import SemanticHead
from torch import Tensor, nn


class SemanticLoss:
    """Hard pixel mining with cross entropy loss, for semantic segmentation.
    Following DeepLab Cross Entropy loss
    https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/model/loss/criterion.py
    """

    def __init__(self,
                 device,
                 ignore_index: int = 255,
                 ignore_labels: Optional[List] = None,
                 top_k_percent_pixels: float = 1.0,
                 class_weights: Optional[Tuple[float, ...]] = None):
        if not 0. < top_k_percent_pixels <= 1.0:
            raise ValueError('top_k_percent_pixels must be within (0, 1]')
        self.device = device
        self.ignore_labels = ignore_labels
        self.top_k_percent_pixels = top_k_percent_pixels
        self.ignore_index = ignore_index
        weight = None
        if class_weights is not None:
            if ignore_labels is None:
                weight = torch.tensor(class_weights, device=device)
            else:
                weight = torch.tensor(
                    [w for label, w in enumerate(class_weights) if label not in ignore_labels],
                    device=device)
        self.ce_loss = nn.CrossEntropyLoss(weight=weight,
                                           ignore_index=ignore_index,
                                           reduction='none')

    def __call__(self, prediction_softmax: Tensor, target: Tensor, pixel_weights: Tensor,
                 return_per_pixel: bool = False) -> Tensor:
        if return_per_pixel:
            assert self.top_k_percent_pixels == 1.0, 'top_k must be 1.0 for return_per_pixel = True'

        if self.ignore_labels is not None:
            for ignore_label in self.ignore_labels:
                target[target == ignore_label] = self.ignore_index
            preserved_labels = [label for label in range(prediction_softmax.shape[1]) if
                                label not in self.ignore_labels]
            prediction_softmax = prediction_softmax[:, preserved_labels, ...]

        loss = self.ce_loss(prediction_softmax, target.long()) * pixel_weights

        if self.top_k_percent_pixels < 1.0:
            loss = loss.contiguous().view(-1)
            # Hard pixel mining
            top_k_pixels = int(self.top_k_percent_pixels * loss.numel())
            loss, _ = torch.topk(loss, top_k_pixels)

        if return_per_pixel:
            return loss
        return loss.mean()

# -------------------------------------------------------- #


class SemanticSegAlgo:

    def __init__(
            self,
            semantic_loss: SemanticLoss,
            evaluator: SemanticEvaluator,
    ):
        self.semantic_loss = semantic_loss
        self.evaluator = evaluator

    def training(
            self,
            feats: Tensor,
            semantic_head: SemanticHead,
            semantic_gt: Tensor,
            semantic_weights: Tensor,
            ignore_classes: Optional[List] = None,
            semantic_gt_eval: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        semantic_logits = semantic_head(feats)

        if ignore_classes is not None:
            semantic_logits_ignored = semantic_logits.detach().clone()
            for ignore_class in ignore_classes:
                semantic_logits_ignored[:, ignore_class, :, :] = -float('inf')
            semantic_pred = torch.argmax(semantic_logits_ignored, dim=1).type(torch.uint8)
        else:
            semantic_pred = torch.argmax(semantic_logits, dim=1).type(torch.uint8)
        if semantic_gt_eval is not None:
            confusion_matrix = self.evaluator.compute_confusion_matrix(semantic_pred,
                                                                       semantic_gt_eval)
        else:
            confusion_matrix = self.evaluator.compute_confusion_matrix(semantic_pred, semantic_gt)

        semantic_loss = self.semantic_loss(semantic_logits, semantic_gt, semantic_weights)

        return semantic_loss, confusion_matrix, semantic_pred

    def inference(self, feats: Tensor, semantic_head: SemanticHead) -> Tuple[Tensor, Tensor]:
        semantic_logits = semantic_head(feats)
        semantic_pred = torch.argmax(semantic_logits, dim=1).type(torch.uint8)
        return semantic_pred, semantic_logits

    def evaluation(self, feats: Tensor, semantic_head: SemanticHead, semantic_gt: Tensor) \
            -> Tuple[Tensor, Tensor, Tensor]:
        semantic_pred, semantic_logits = self.inference(feats, semantic_head)
        confusion_matrix = self.evaluator.compute_confusion_matrix(semantic_pred, semantic_gt)
        return confusion_matrix, semantic_pred, semantic_logits
