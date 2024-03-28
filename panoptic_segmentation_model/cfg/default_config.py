# Reference point for all configurable options
from yacs.config import CfgNode as CN

# /----- Create a cfg node
cfg = CN()

# ********************************************************************
# /------ Training parameters
# ********************************************************************
cfg.train = CN()
cfg.train.nof_epochs = 20
cfg.train.nof_workers_per_gpu = 1
cfg.train.batch_size_per_gpu = 1

# /----- Optimizer parameters
cfg.train.optimizer = CN()
cfg.train.optimizer.type = 'Adam'
cfg.train.optimizer.learning_rate = 0.0001

# /----- Scheduler parameters
cfg.train.scheduler = CN()
cfg.train.scheduler.type = 'StepLR'  # 'StepLR', 'WarmupPolyLR'
# StepLR
cfg.train.scheduler.step_lr = CN()
cfg.train.scheduler.step_lr.step_size = 20
cfg.train.scheduler.step_lr.gamma = 0.1
# WarmupPolyLR
cfg.train.scheduler.warmup = CN()
cfg.train.scheduler.warmup.max_iters = 90000
cfg.train.scheduler.warmup.factor = 0.001
cfg.train.scheduler.warmup.iters = 1000
cfg.train.scheduler.warmup.method = 'linear'
cfg.train.scheduler.warmup.power = 0.9
cfg.train.scheduler.warmup.constant_ending = 0.

# ********************************************************************
# /------ Validation parameters
# ********************************************************************
cfg.val = CN()
cfg.val.batch_size_per_gpu = 1
cfg.val.nof_workers_per_gpu = 1

# ********************************************************************
# /----- Model parameters
# ********************************************************************
cfg.model = CN()

cfg.model.make_semantic = True
cfg.model.make_instance = True

cfg.model.backbone_panoptic = CN()
cfg.model.backbone_panoptic.name = ''

# --- ResNet
cfg.model.backbone_panoptic.resnet = CN()
cfg.model.backbone_panoptic.resnet.params = CN()
cfg.model.backbone_panoptic.resnet.params.nof_layers = 50
cfg.model.backbone_panoptic.resnet.params.weights_init = 'pretrained'

# --- DINOv2-ViT
cfg.model.backbone_panoptic.dino_vit = CN()
cfg.model.backbone_panoptic.dino_vit.params = CN()
cfg.model.backbone_panoptic.dino_vit.params.type = ''
cfg.model.backbone_panoptic.dino_vit.params.pretrained = True
cfg.model.backbone_panoptic.dino_vit.params.frozen = True
cfg.model.backbone_panoptic.dino_vit.params.drop_path_rate = 0.0
cfg.model.backbone_panoptic.dino_vit.params.use_multi_scale_features = False
cfg.model.backbone_panoptic.dino_vit.params.window_block_indexes = []
cfg.model.backbone_panoptic.dino_vit.params.window_size = 0

# --- DINOv2-ViT-Adapter
cfg.model.backbone_panoptic.dino_vit_adapter = CN()
cfg.model.backbone_panoptic.dino_vit_adapter.params = CN()
cfg.model.backbone_panoptic.dino_vit_adapter.params.pretrain_size = 518
cfg.model.backbone_panoptic.dino_vit_adapter.params.conv_inplane = 64
cfg.model.backbone_panoptic.dino_vit_adapter.params.n_points = 4
cfg.model.backbone_panoptic.dino_vit_adapter.params.deform_num_heads = 6
cfg.model.backbone_panoptic.dino_vit_adapter.params.init_values = 0.
cfg.model.backbone_panoptic.dino_vit_adapter.params.interaction_indexes = None
cfg.model.backbone_panoptic.dino_vit_adapter.params.with_cffn = True
cfg.model.backbone_panoptic.dino_vit_adapter.params.cffn_ratio = 0.25
cfg.model.backbone_panoptic.dino_vit_adapter.params.deform_ratio = 1.0
cfg.model.backbone_panoptic.dino_vit_adapter.params.add_vit_feature = False
cfg.model.backbone_panoptic.dino_vit_adapter.params.use_extra_extractor = True
cfg.model.backbone_panoptic.dino_vit_adapter.params.with_cp = False
cfg.model.backbone_panoptic.dino_vit_adapter.params.vit_arch_name = 'vit_base'
cfg.model.backbone_panoptic.dino_vit_adapter.params.vit_pretrained = True

cfg.model.backbone_panoptic.dino_vit_adapter.params.vit_kwargs = CN()
cfg.model.backbone_panoptic.dino_vit_adapter.params.vit_kwargs.img_size = 518
cfg.model.backbone_panoptic.dino_vit_adapter.params.vit_kwargs.patch_size = 14
cfg.model.backbone_panoptic.dino_vit_adapter.params.vit_kwargs.init_values = 1.0
cfg.model.backbone_panoptic.dino_vit_adapter.params.vit_kwargs.ffn_layer = 'mlp'
cfg.model.backbone_panoptic.dino_vit_adapter.params.vit_kwargs.block_chunks = 0
cfg.model.backbone_panoptic.dino_vit_adapter.params.vit_kwargs.embed_dim = 768
cfg.model.backbone_panoptic.dino_vit_adapter.params.vit_kwargs.depth = 12
cfg.model.backbone_panoptic.dino_vit_adapter.params.vit_kwargs.num_heads = 12
cfg.model.backbone_panoptic.dino_vit_adapter.params.vit_kwargs.mlp_ratio = 4

# --- Pose Net
cfg.model.pose_sflow_net = CN()
cfg.model.pose_sflow_net.input = 'pairs'
cfg.model.pose_sflow_net.params = CN()
cfg.model.pose_sflow_net.params.nof_layers = 18
cfg.model.pose_sflow_net.params.weights_init = 'pretrained'

# --- Semantic Head
cfg.model.semantic_head = CN()
cfg.model.semantic_head.use_guda_fusion = True

# ********************************************************************
# /----- Dataset parameters
# ********************************************************************
cfg.dataset = CN()
cfg.dataset.name = ''  # e.g., cityscapes or 'kitti_360'
cfg.dataset.path = ''  # e.g.
cfg.dataset.feed_img_size = []  # [height, width], e.g., [192, 640]
cfg.dataset.center_heatmap_sigma = 8
cfg.dataset.return_only_rgb = False
cfg.dataset.small_instance_area_full_res = 4096
cfg.dataset.small_instance_weight = 3
cfg.dataset.train_split = '' # 'train_extra'
cfg.dataset.train_sequences = []  # Only supported in 'sequence' split
cfg.dataset.val_split = 'val'
cfg.dataset.val_sequences = []  # Only supported in 'sequence' split
cfg.dataset.remove_classes = []
cfg.dataset.indices_gt = []
cfg.dataset.label_mode = '' # 'cityscapes', 'cityscapes-19', 'cityscapes-27'

# ********************************************************************
# /----- Preprocessing parameters
# ********************************************************************
cfg.dataset.augmentation = CN()
cfg.dataset.augmentation.active = True  # Whether to apply augmentation
cfg.dataset.augmentation.horizontal_flipping = True  # Randomly applied with prob=.5
cfg.dataset.augmentation.brightness_jitter = 0.2  # Or None
cfg.dataset.augmentation.contrast_jitter = 0.2  # Or None
cfg.dataset.augmentation.saturation_jitter = 0.2  # Or None
cfg.dataset.augmentation.hue_jitter = 0.1  # Or None

cfg.dataset.normalization = CN()
cfg.dataset.normalization.active = True
cfg.dataset.normalization.rgb_mean = (0.485, 0.456, 0.406)
cfg.dataset.normalization.rgb_std = (0.229, 0.224, 0.225)

# ********************************************************************
# /----- Evaluation parameters
# ********************************************************************
cfg.eval = CN()

cfg.eval.semantic = CN()
cfg.eval.semantic.ignore_classes = []

# ********************************************************************
# /----- Losses
# ********************************************************************
cfg.losses = CN()

cfg.losses.weights = CN()
cfg.losses.weights.semantic = 1.0

cfg.losses.weights.center = 1.0
cfg.losses.weights.offset = 1.0

# ********************************************************************
# /----- Semantics
# ********************************************************************
cfg.semantics = CN()
cfg.semantics.class_weights = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                               1.0, 1.0, 1.0, 1.0, 1.0)
cfg.semantics.top_k = 0.2

# ********************************************************************
# /----- Visualization
# *******************************************************************
cfg.visualization = CN()
cfg.visualization.scale = 1.  # Size of images on wandb

# ********************************************************************
# /----- Logging
# *******************************************************************
cfg.logging = CN()
cfg.logging.log_train_samples = True
# Number of epochs between validations
cfg.logging.val_epoch_interval = 1
# Number of steps before outputting a log entry
cfg.logging.log_step_interval = 10

# ********************************************************************
# /----- Additional parameters from PoBev
# ********************************************************************
cfg.general = CN()
cfg.general.cudnn_benchmark = False


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values"""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()
