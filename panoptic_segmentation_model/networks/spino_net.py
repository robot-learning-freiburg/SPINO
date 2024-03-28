from collections import OrderedDict
from typing import Any, Dict, List, Optional

from algos import InstanceSegAlgo, SemanticSegAlgo
from models import InstanceHead, ResnetEncoder, SemanticHead
from torch import Tensor, nn


class SpinoNet(nn.Module):

    def __init__(self, backbone_panoptic: ResnetEncoder, semantic_head: Optional[SemanticHead],
                 instance_head: Optional[InstanceHead], semantic_algo: Optional[SemanticSegAlgo],
                 instance_algo: Optional[InstanceSegAlgo]):
        super().__init__()
        if semantic_algo is not None:
            assert semantic_head is not None
        if instance_algo is not None:
            assert instance_head is not None

        self.backbone_panoptic = backbone_panoptic

        self.semantic_head = semantic_head
        self.instance_head = instance_head

        self.semantic_algo = semantic_algo
        self.instance_algo = instance_algo

    def forward(self,
                in_data: Dict[str, Tensor],
                mode: str = "infer",
                do_panoptic_fusion: bool = False,
                sem_ignore_classes: Optional[List[int]] = None):
        assert mode in ["train", "eval", "infer"], f"Unsupported mode: {mode}"
        # --------------------
        if do_panoptic_fusion:
            assert self.semantic_algo is not None
            assert self.instance_algo is not None

        # --------------------
        # In training mode, return the predictions, the statistics, and the loss(!)
        if mode == "train":
            panoptic_feats = [self.backbone_panoptic(in_data["rgb"])]

            # ----------
            # SEMANTICS TRAINING
            # ----------
            if self.semantic_algo is not None:
                semantic_gt_eval = in_data.get("semantic_eval")
                semantic_loss, confusion_matrix, semantic_pred = self.semantic_algo.training(
                    panoptic_feats[0], self.semantic_head, in_data["semantic"],
                    in_data["semantic_weights"], sem_ignore_classes, semantic_gt_eval)
            else:
                semantic_loss, confusion_matrix, semantic_pred = None, None, None

            # ----------
            # INSTANCE TRAINING
            # ----------
            if self.instance_algo is not None:
                center_loss, offset_loss, center_pred, offset_pred = \
                    self.instance_algo.training(panoptic_feats[0], center=in_data["center"],
                                                offset=in_data["offset"],
                                                center_weights=in_data["center_weights"],
                                                offset_weights=in_data["offset_weights"],
                                                instance_head=self.instance_head)

            else:
                center_loss, offset_loss, center_pred, offset_pred = None, None, None, None

        # In evaluation mode, return the predictions and the statistics. But not the loss.
        elif mode == "eval":
            # Get features for panoptics
            panoptic_feats = self.backbone_panoptic(in_data["rgb"])

            if self.semantic_algo is not None:
                semantic_gt = in_data.get("semantic")  # Not all samples have GT semantic
                if semantic_gt is None:
                    semantic_pred = self.semantic_algo.inference(
                        panoptic_feats, self.semantic_head)
                    confusion_matrix = None
                else:
                    semantic_gt = in_data.get("semantic_eval", semantic_gt)
                    confusion_matrix, semantic_pred, semantic_logits = \
                        self.semantic_algo.evaluation(panoptic_feats, self.semantic_head,
                                                      semantic_gt)
            else:
                confusion_matrix, semantic_pred = None, None
            semantic_loss = None

            if self.instance_algo is not None:
                center_pred, offset_pred = \
                    self.instance_algo.inference(panoptic_feats, self.instance_head)
            else:
                center_pred, offset_pred = None, None
            center_loss, offset_loss = None, None

        # In inference mode, only return the predictions
        elif mode == "infer":
            # Get features for panoptics
            panoptic_feats = self.backbone_panoptic(in_data["rgb"])

            if self.semantic_algo is not None:
                semantic_pred = self.semantic_algo.inference(panoptic_feats, self.semantic_head)
            else:
                semantic_pred = None, None
            semantic_loss, confusion_matrix = None, None

            if self.instance_algo is not None:
                center_pred, offset_pred = self.instance_algo.inference(
                    panoptic_feats,
                    self.instance_head)
            else:
                center_pred, offset_pred = None, None
            center_loss, offset_loss = None, None

        else:
            semantic_loss, confusion_matrix, semantic_pred, center_loss, offset_loss, center_pred, \
                offset_pred = None, None, None, None, None, None, None

        # --------------------
        if do_panoptic_fusion:
            panoptic_pred, instance_pred = self.instance_algo.panoptic_fusion(
                semantic_pred, center_pred, offset_pred)
        else:
            panoptic_pred, instance_pred = None, None

        # --------------------
        # Losses
        losses = OrderedDict()
        losses["semantic"] = semantic_loss
        losses["center"] = center_loss
        losses["offset"] = offset_loss

        # Predictions
        result = OrderedDict()
        result["semantic"] = semantic_pred
        result["center"] = center_pred
        result["offset"] = offset_pred
        result["panoptic"] = panoptic_pred
        result["instance"] = instance_pred

        # Statistics
        stats = OrderedDict()
        stats["sem_conf"] = confusion_matrix

        return losses, result, stats

    def get_state_dict(self) -> Dict[str, Any]:

        def _safe_state_dict(module):
            if module is None:
                return None
            return module.state_dict()

        state_dict = {
            "backbone_panoptic": _safe_state_dict(self.backbone_panoptic),
            "semantic_head": _safe_state_dict(self.semantic_head),
            "instance_head": _safe_state_dict(self.instance_head),
        }

        return state_dict
