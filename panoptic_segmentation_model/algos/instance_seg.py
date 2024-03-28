from typing import Optional, Tuple, Union

import torch
from eval import PanopticEvaluator
from misc.post_processing_panoptic import get_panoptic_segmentation
from models import InstanceHead
from numpy.typing import ArrayLike
from torch import Tensor, nn


class CenterLoss:

    def __init__(self):
        self.mse_loss = nn.MSELoss(reduction="none")

    def __call__(self, prediction: Tensor, target: Tensor, pixel_weights: ArrayLike) -> Tensor:
        """
        Parameters
        ----------
        pixel_weights : torch.Tensor
            Ignore region with 0 is ignore and 1 is consider
        """
        loss = self.mse_loss(prediction, target)
        # loss = self.mse_loss(prediction, target) * pixel_weights
        # pixel_weights_sum = pixel_weights.sum(-1).sum(-1)
        # if (pixel_weights_sum > 0).all():
        #     loss = loss.sum(-1).sum(-1) / pixel_weights_sum  # per batch
        # else:
        #     loss_ = loss.sum(-1).sum(-1) / (pixel_weights_sum + 1e-10)  # per batch
        #     loss_[(pixel_weights_sum == 0).squeeze(), :] = loss[(pixel_weights.sum(-1).sum(
        #         -1) == 0).squeeze(), :].sum(-1).sum(-1) * 0
        #     loss = loss_
        return loss.mean()


class OffsetLoss:

    def __init__(self):
        self.l1_loss = nn.L1Loss(reduction="none")

    def __call__(self, prediction: Tensor, target: Tensor, pixel_weights: ArrayLike) -> Tensor:
        """
        Parameters
        ----------
        pixel_weights : torch.Tensor
            Ignore region with 0 is ignore and 1 is consider
        """
        loss = self.l1_loss(prediction, target)
        # This is to make sure that the weights are set wrong in any of the datasets.
        # loss = self.l1_loss(prediction, target) * pixel_weights
        # pixel_weights_sum = pixel_weights.sum(-1).sum(-1)
        # if (pixel_weights_sum > 0).all():
        #     loss = loss.sum(-1).sum(-1) / pixel_weights_sum  # per batch
        # else:
        #     loss_ = loss.sum(-1).sum(-1) / (pixel_weights_sum + 1e-10)  # per batch
        #     loss_[(pixel_weights_sum == 0).squeeze(), :] = loss[(pixel_weights.sum(-1).sum(
        #         -1) == 0).squeeze(), :].sum(-1).sum(-1) * 0
        #     loss = loss_
        return loss.mean()


class BinaryMaskLoss:

    def __init__(self, ignore_index: int = 255):
        self.ce_loss = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)

    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor:
        loss = self.ce_loss(prediction, target.long())
        return loss.mean()


# -------------------------------------------------------- #


class InstanceSegAlgo:
    """
    Parameters
    ----------
    thing_list :
        These IDs represent object instances ("things").
    """

    def __init__(
            self,
            center_loss: CenterLoss,
            offset_loss: OffsetLoss,
            evaluator: PanopticEvaluator,
            binary_mask_loss: Optional[BinaryMaskLoss] = None,
    ):
        self.center_loss = center_loss
        self.offset_loss = offset_loss
        self.binary_mask_loss = binary_mask_loss
        self.evaluator = evaluator

    def training(
            self,
            feats: Tensor,
            center: Tensor,
            offset: Tensor,
            center_weights: Tensor,
            offset_weights: Tensor,
            instance_head: InstanceHead
    ):
        """
        Parameters
        ----------
        feats : torch.Tensor
            Features from the multi-task encoder used as input to depth, segmentation, etc.
        center : torch.Tensor
            Ground truth heatmap of instance centers
        offset : torch.Tensor
            Ground truth offset (x,y) for each pixel to the closest instance center
        center_weights : torch.Tensor
            Pixel-wise loss weights for center loss
        offset_weights : torch.Tensor
            Pixel-wise loss weights for offset loss
        instance_head: InstanceHead
            Decoder for predicting instance centers and offsets
        """
        center_pred, offset_pred = instance_head(feats)

        # Compute losses
        center_loss = self.center_loss(center_pred, center, center_weights)
        offset_loss = self.offset_loss(offset_pred, offset, offset_weights)

        return center_loss, offset_loss, center_pred, offset_pred

    def inference(
            self,
            feats: Tensor,
            instance_head: InstanceHead,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """
        Parameters
        ----------
        feats : torch.Tensor
            Features from the multi-task encoder used as input to depth, segmentation, etc.
        instance_head: InstanceHead
            Decoder for predicting instance centers and offsets
        """
        center_pred, offset_pred = instance_head(feats)

        return center_pred, offset_pred

    def evaluation(self):
        raise NotImplementedError

    def panoptic_fusion(
            self,
            semantic: Tensor,
            center: Tensor,
            offset: Tensor,
            return_center: bool = False,
            threshold_center: float = None,
            do_merge_semantic_and_instance=True,
    ) -> Tuple[Optional[Tensor], Tensor]:
        """
        Note a change in the void label:
        - semantic map: 255
        - panoptic map: -1
        """

        batch_size = semantic.shape[0]
        if do_merge_semantic_and_instance:
            # Int16 since the largest expected number is 18 * 1000 < 32767
            panoptic = torch.empty_like(semantic, dtype=torch.int16)
        else:
            panoptic = None
        instance = torch.empty_like(semantic)

        center_pts = []

        for i in range(batch_size):
            thing_list = self.evaluator.thing_list
            label_divisor = 1000  # pan_ID = sem_class_id * label_divisor + inst_id
            stuff_area = 0
            void_label = 255
            threshold = .1 if threshold_center is None else threshold_center
            nms_kernel = 7
            top_k = 200
            foreground_mask = None

            semantic_ = semantic[i, :].unsqueeze(0)
            center_ = center[i, :].unsqueeze(0)
            offset_ = offset[i, :].unsqueeze(0)

            pan, cnt, inst = get_panoptic_segmentation(semantic_, center_, offset_, thing_list,
                                                       label_divisor, stuff_area, void_label,
                                                       threshold, nms_kernel, top_k,
                                                       foreground_mask,
                                                       do_merge_semantic_and_instance)
            center_pts.append(cnt)
            if do_merge_semantic_and_instance:
                panoptic[i] = pan
            instance[i] = inst
        if return_center:
            return panoptic, instance, center_pts
        return panoptic, instance
