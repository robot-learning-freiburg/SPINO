from typing import List

from algos import (
    BinaryMaskLoss,
    CenterLoss,
    InstanceSegAlgo,
    OffsetLoss,
    SemanticLoss,
    SemanticSegAlgo,
)
from eval import PanopticEvaluator, SemanticEvaluator
from models import (
    DinoV2,
    InstanceHead,
    ResnetEncoder,
    SemanticHead,
    ViTAdapter,
)
from networks import SpinoNet


def gen_models(cfg, device, stuff_classes: List[int], thing_classes: List[int],
               ignore_classes: List[int]):
    """Create the backbones, heads, losses, evaluators, and algorithms
    """
    # ----------
    # MULTI-TASK BACKBONE
    # ----------
    if cfg.model.backbone_panoptic.name == "resnet":
        backbone_panoptic = ResnetEncoder(cfg.model.backbone_panoptic.resnet.params.nof_layers,
                                          cfg.model.backbone_panoptic.resnet.params.weights_init
                                          == "pretrained")
    elif cfg.model.backbone_panoptic.name == "dino-vit":
        backbone_panoptic = DinoV2(
            cfg.dataset.feed_img_size,
            cfg.model.backbone_panoptic.dino_vit.params.type,
            cfg.model.backbone_panoptic.dino_vit.params.pretrained,
            cfg.model.backbone_panoptic.dino_vit.params.frozen,
            cfg.model.backbone_panoptic.dino_vit.params.drop_path_rate,
            cfg.model.backbone_panoptic.dino_vit.params.window_block_indexes,
            cfg.model.backbone_panoptic.dino_vit.params.window_size,
            cfg.model.backbone_panoptic.dino_vit.params.use_multi_scale_features)
    elif cfg.model.backbone_panoptic.name == "dino-vit-adapter":
        backbone_panoptic = ViTAdapter(
            cfg.model.backbone_panoptic.dino_vit_adapter.params.pretrain_size,
            cfg.model.backbone_panoptic.dino_vit_adapter.params.conv_inplane,
            cfg.model.backbone_panoptic.dino_vit_adapter.params.n_points,
            cfg.model.backbone_panoptic.dino_vit_adapter.params.deform_num_heads,
            cfg.model.backbone_panoptic.dino_vit_adapter.params.init_values,
            cfg.model.backbone_panoptic.dino_vit_adapter.params.interaction_indexes,
            cfg.model.backbone_panoptic.dino_vit_adapter.params.with_cffn,
            cfg.model.backbone_panoptic.dino_vit_adapter.params.cffn_ratio,
            cfg.model.backbone_panoptic.dino_vit_adapter.params.deform_ratio,
            cfg.model.backbone_panoptic.dino_vit_adapter.params.add_vit_feature,
            cfg.model.backbone_panoptic.dino_vit_adapter.params.use_extra_extractor,
            cfg.model.backbone_panoptic.dino_vit_adapter.params.with_cp,
            cfg.model.backbone_panoptic.dino_vit_adapter.params.vit_arch_name,
            cfg.model.backbone_panoptic.dino_vit_adapter.params.vit_kwargs,
            cfg.model.backbone_panoptic.dino_vit_adapter.params.vit_pretrained)
    else:
        raise ValueError(f"Unsupported network backbone: {cfg.model.backbone_panoptic.name}")

    # ----------
    # SEMANTICS SETUP
    # ----------
    if cfg.model.make_semantic:
        num_classes = len(stuff_classes) + len(thing_classes)
        if cfg.model.backbone_panoptic.name == "resnet":
            semantic_head = SemanticHead(backbone_panoptic.num_ch_enc,
                                         num_classes,
                                         cfg.dataset.feed_img_size,
                                         use_skips=True,
                                         use_guda_fusion=cfg.model.semantic_head.use_guda_fusion)
        elif cfg.model.backbone_panoptic.name == "dino-vit":
            semantic_head = SemanticHead([backbone_panoptic.feat_dim],
                                         num_classes,
                                         cfg.dataset.feed_img_size,
                                         use_skips=False,
                                         use_guda_fusion=cfg.model.semantic_head.use_guda_fusion,
                                         is_dino=True)
        elif cfg.model.backbone_panoptic.name == "dino-vit-adapter":
            semantic_head = SemanticHead([backbone_panoptic.num_features]*4,
                                         num_classes,
                                         cfg.dataset.feed_img_size,
                                         use_skips=True,
                                         use_guda_fusion=cfg.model.semantic_head.use_guda_fusion,
                                         is_dino=True)
        else:
            raise ValueError(f"The specified model {cfg.model.backbone_panoptic.name} "
                             f" is not supported")

        # Remove weights that belong to cfg.dataset.remove_classes
        class_weights = [wt for idx, wt in enumerate(cfg.semantics.class_weights) if idx not in
                         cfg.dataset.remove_classes]
        sem_loss = SemanticLoss(device=device, class_weights=class_weights,
                                top_k_percent_pixels=cfg.semantics.top_k,
                                ignore_labels=ignore_classes)

        sem_eval = SemanticEvaluator(num_classes=num_classes, ignore_classes=ignore_classes)
        sem_algo = SemanticSegAlgo(sem_loss, sem_eval)
    else:
        semantic_head, sem_algo = None, None

    # ----------
    # INSTANCE SETUP
    # ----------
    if cfg.model.make_instance:
        if cfg.model.backbone_panoptic.name == "resnet":
            instance_head = InstanceHead(backbone_panoptic.num_ch_enc,
                                         use_skips=True,
                                         feed_img_size=cfg.dataset.feed_img_size,
                                         is_dino=False)
        elif cfg.model.backbone_panoptic.name == "dino-vit":
            instance_head = InstanceHead([backbone_panoptic.feat_dim],
                                         use_skips=False,
                                         feed_img_size=cfg.dataset.feed_img_size,
                                         is_dino=True)
        elif cfg.model.backbone_panoptic.name == "dino-vit-adapter":
            instance_head = InstanceHead([backbone_panoptic.num_features]*4,
                                         use_skips=True,
                                         feed_img_size=cfg.dataset.feed_img_size,
                                         is_dino=True)
        else:
            raise ValueError(f"The specified model {cfg.model.backbone_panoptic.name}"
                             f" is not supported")

        instance_center_loss = CenterLoss()
        instance_offset_loss = OffsetLoss()
        binary_mask_loss = BinaryMaskLoss()
        panoptic_eval = PanopticEvaluator(stuff_list=stuff_classes,
                                          thing_list=thing_classes,
                                          label_divisor=1000, void_label=-1)
        instance_algo = InstanceSegAlgo(instance_center_loss, instance_offset_loss, panoptic_eval,
                                        binary_mask_loss)
    else:
        instance_head, instance_algo = None, None

    # ----------
    # OVERALL NETWORK
    # ----------
    spino_net = SpinoNet(backbone_panoptic=backbone_panoptic,
                         semantic_head=semantic_head,
                         instance_head=instance_head,
                         semantic_algo=sem_algo,
                         instance_algo=instance_algo)

    return spino_net
