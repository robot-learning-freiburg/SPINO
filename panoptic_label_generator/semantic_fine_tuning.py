import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from fine_tuning import FineTuner
from PIL import Image
from pytorch_lightning.cli import LightningCLI
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

# Ignore some torch warnings
warnings.filterwarnings('ignore', '.*The default behavior for interpolate/upsample with float*')
warnings.filterwarnings(
    'ignore', '.*Default upsampling behavior when mode=bicubic is changed to align_corners=False*')
warnings.filterwarnings('ignore', '.*Only one label was provided to `remove_small_objects`*')


class SemanticFineTuner(FineTuner):
    """Fine-tunes a small head on top of the DINOv2 model for semantic segmentation.

    Parameters
    ----------
    dinov2_vit_model : str
        ViT model name of DINOv2. One of ['vits14', 'vitl14', 'vitg14', 'vitb14'].
    num_classes : int
        Number of classes for semantic segmentation.
    train_output_size : Tuple[int, int]
        Output size [H, W] after head.
    blocks : List[int]
        List of block indices of ViT to use for feature extraction. If None, use only the last block.
    upsample_factor : float
        Upsample factor of the feature map after the ViT and before the head.
    head : str
        Head to use for semantic segmentation. One of ['linear', 'knn', 'cnn', 'mlp'].
    ignore_index : int
        Index to ignore in the loss.
    top_k_percent_pixels : float
        Percentage of hardest pixels to keep for the loss.
    test_output_size : Tuple[int, int]
        Final output size [H, W] of the model during prediction/testing.
    test_multi_scales : List[int]
        List of scales to use for multi-scale during prediction/testing e.g. [1, 2, 3].
    test_plot : bool
        Whether to plot the predictions during testing.
    test_save_dir : str
        Directory to save the predictions during testing.
    """

    def __init__(self, dinov2_vit_model: str, num_classes: int, train_output_size: Tuple[int, int],
                 blocks: Optional[List[int]] = None, upsample_factor: Optional[float] = None,
                 head: str = 'mlp',
                 ignore_index: int = -100, top_k_percent_pixels: float = 1.0,
                 test_output_size: Optional[Tuple[int, int]] = None,
                 test_multi_scales: Optional[List[int]] = None,
                 test_plot: bool = False, test_save_dir: Optional[str] = None):
        super().__init__(dinov2_vit_model=dinov2_vit_model, blocks=blocks,
                         upsample_factor=upsample_factor)
        self.num_classes = num_classes
        self.train_output_size = train_output_size
        self.ignore_index = ignore_index
        self.top_k_percent_pixels = top_k_percent_pixels
        self.test_output_size = test_output_size
        self.test_multi_scales = test_multi_scales
        self.test_plot = test_plot
        self.test_save_dir = test_save_dir

        head_input_dim = self.feat_dim * self.num_blocks
        if head == 'linear':
            self.head = nn.Conv2d(head_input_dim, num_classes, kernel_size=1, stride=1, padding=0)
        elif head == 'knn':
            self.head = KNeighborsClassifier(n_neighbors=5, leaf_size=10)
            self.knn_X = []
            self.knn_y = []
        elif head == 'cnn':
            self.head = nn.Sequential(
                nn.Conv2d(head_input_dim, 300, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(300, 300, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(300, 200, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(200, num_classes, kernel_size=3, stride=1, padding=1),
            )
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Conv2d(head_input_dim, 300, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(300, 300, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(300, 200, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(200, num_classes, kernel_size=1, stride=1, padding=0),
            )
        else:
            raise ValueError(f'Unknown head {head}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_encoder(x)  # (B, feat_dim, H, W)
        if isinstance(self.head, KNeighborsClassifier):
            if self.training:
                return x  # return only features during training
            feat_shape = x.shape
            x = x.permute(0, 2, 3, 1).reshape(-1, feat_shape[1])
            x = x.detach().cpu().numpy()
            x = self.head.predict_proba(x)  # (B * H * W, num_classes)
            x = torch.from_numpy(x).to(self.device)
            x = x.reshape(feat_shape[0], feat_shape[2], feat_shape[3], -1) \
                .permute(0, 3, 1, 2)  # (B, num_classes, H, W)
        else:
            x = self.head(x)  # (B, num_classes, H, W)
        x = nn.functional.interpolate(x, size=self.train_output_size, mode='bilinear',
                                      align_corners=False)
        return x

    def training_step(self, train_batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        rgb = train_batch['rgb']
        sem = train_batch['semantic'].long()

        if isinstance(self.head, KNeighborsClassifier):
            x = self(rgb)  # (B, feat_dim, H, W)
            feat_h, feat_w = x.shape[2:]
            x = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])  # (B * H * W, feat_dim)
            x = x.detach().cpu().numpy()
            self.knn_X.append(x)

            sem = TF.resize(sem, [feat_h, feat_w], interpolation=InterpolationMode.NEAREST)
            sem = sem.reshape(-1)
            sem = sem.detach().cpu().numpy()
            self.knn_y.append(sem)

            loss = torch.tensor([0.0], requires_grad=True).to(self.device)  # dummy loss
        else:
            sem = TF.resize(sem, self.train_output_size, interpolation=InterpolationMode.NEAREST)
            pred = self(rgb)
            loss = F.cross_entropy(pred, sem, ignore_index=self.ignore_index, reduction='none')

            if self.top_k_percent_pixels < 1.0:
                loss = loss.contiguous().view(-1)
                # Hard pixel mining
                top_k_pixels = int(self.top_k_percent_pixels * loss.numel())
                loss, _ = torch.topk(loss, top_k_pixels)
            loss = loss.mean()

        self.log('train_loss', loss)
        return loss

    def predict(self, rgb: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.test_multi_scales is None:
            pred = self(rgb)  # (B, num_classes, H, W)
            if not isinstance(self.head, KNeighborsClassifier):
                pred = torch.softmax(pred, dim=1)  # (B, num_classes, H, W)
        else:
            pred = self.multi_scale_test_augmentation(rgb)

        pred = nn.functional.interpolate(pred, size=self.test_output_size, mode='bilinear',
                                         align_corners=False)
        pred = pred.argmax(dim=1)  # (B, H, W)

        if mask is not None:
            pred[mask] = self.ignore_index
        return pred

    def multi_scale_test_augmentation(self, rgb: torch.Tensor) -> torch.Tensor:
        # Splitting and upscaling at multiple scales
        all_preds = []  # List to store predictions at all scales to fuse later
        batch_size = rgb.shape[0]
        img_h, img_w = rgb.shape[2:]

        for scale in self.test_multi_scales:
            image_h_split, image_w_split = img_h // scale, img_w // scale
            train_output_h_split, train_output_w_split = \
                self.train_output_size[0] // scale, self.train_output_size[1] // scale

            rgb_split = torch.split(rgb, image_h_split, dim=2)
            rgb_split = [torch.split(split, image_w_split, dim=3) for split in rgb_split]

            pred_scale = torch.zeros(
                (
                    batch_size, self.num_classes, self.train_output_size[0],
                    self.train_output_size[1]))
            for row, row_splits in enumerate(rgb_split):
                for col, rgb_split_i in enumerate(row_splits):
                    rgb_split_i_upscaled = T.functional.resize(
                        rgb_split_i, [img_h, img_w],
                        interpolation=InterpolationMode.BILINEAR)
                    pred = self(rgb_split_i_upscaled)  # (B, num_classes, H, W)
                    pred = T.functional.resize(
                        pred, [train_output_h_split, train_output_w_split],
                        interpolation=InterpolationMode.BILINEAR)  # (B, num_classes, H, W)
                    pred_scale[:, :, row * train_output_h_split:(row + 1) * train_output_h_split,
                    col * train_output_w_split:(col + 1) * train_output_w_split] \
                        = pred  # (B, num_classes, H, W)

            all_preds.append(pred_scale)

        # Concatenate and fuse scales
        pred = torch.stack(all_preds, dim=1)  # (B, S, num_classes, H, W)
        if not isinstance(self.head, KNeighborsClassifier):
            pred = torch.softmax(pred, dim=2)  # (B, S, num_classes, H, W)
        pred = pred.mean(dim=1)  # (B, num_classes, H, W)
        return pred

    def get_dataset(self) -> Dataset:
        dataset = self.trainer.test_dataloaders[0].dataset
        return dataset

    def plot(self, rgb: np.array, pred: np.array):
        plt.figure(figsize=(20, 6))
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(10, 10)

        rgb = rgb.transpose((1, 2, 0))  # (H, W, 3)
        dataset = self.get_dataset()
        pred_color = dataset.class_id_to_color()[pred, :]  # (H, W, 3)

        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.grid(False)
        plt.imshow(rgb)

        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.grid(False)
        plt.imshow(rgb)
        plt.imshow(pred_color, cmap='jet', alpha=0.5, interpolation='nearest')
        plt.show()

    def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
        rgb = batch['rgb']  # (B, 3, H, W)
        ego_car_mask = batch.get('ego_car_mask', None)  # (B, H, W)

        pred = self.predict(rgb, ego_car_mask)  # (B, H, W)

        pred = pred.cpu().numpy()  # (B, H, W)
        rgb_original = batch['rgb_original']  # (B, 3, H, W)
        rgb_original = rgb_original.cpu().numpy()  # (B, 3, H, W)

        if self.test_plot:
            for rgb_i, pred_i in zip(rgb_original, pred):
                self.plot(rgb_i, pred_i)

        if self.test_save_dir is not None:
            semantic_paths = batch['semantic_path']
            dataset = self.get_dataset()
            dataset_path_base = str(dataset.path_base)
            for pred_i, semantic_path in zip(pred, semantic_paths):
                pred_path = semantic_path.replace(dataset_path_base, self.test_save_dir)
                if not os.path.exists(os.path.dirname(pred_path)):
                    os.makedirs(os.path.dirname(pred_path))
                pred_img = Image.fromarray(pred_i.astype(np.uint8))
                pred_img.save(pred_path)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        if isinstance(self.head, KNeighborsClassifier):
            checkpoint['knn_X'] = self.knn_X
            checkpoint['knn_y'] = self.knn_y

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        if isinstance(self.head, KNeighborsClassifier):
            self.knn_X = np.concatenate(checkpoint['knn_X'], axis=0)
            self.knn_y = np.concatenate(checkpoint['knn_y'], axis=0)
            knn_X_valid = self.knn_X[self.knn_y != 255]
            knn_y_valid = self.knn_y[self.knn_y != 255]
            self.head = self.head.fit(knn_X_valid, knn_y_valid)


class SemanticFineTunerCLI(LightningCLI):

    def __init__(self):
        super().__init__(
            model_class=SemanticFineTuner,
            seed_everything_default=0,
            parser_kwargs={'parser_mode': 'omegaconf'},
            save_config_callback=None,
        )

    def add_arguments_to_parser(self, parser):
        # Dataset
        parser.add_argument('--data_params', type=Dict)


if __name__ == '__main__':
    cli = SemanticFineTunerCLI()
