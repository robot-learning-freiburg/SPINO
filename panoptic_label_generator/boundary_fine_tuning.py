import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from fine_tuning import FineTuner
from pytorch_lightning.cli import LightningCLI
from sklearn.neighbors import KNeighborsClassifier, radius_neighbors_graph
from torch import nn
from torchvision.transforms import InterpolationMode

# Ignore some torch warnings
warnings.filterwarnings('ignore', '.*The default behavior for interpolate/upsample with float*')
warnings.filterwarnings(
    'ignore', '.*Default upsampling behavior when mode=bicubic is changed to align_corners=False*')
warnings.filterwarnings('ignore', '.*Only one label was provided to `remove_small_objects`*')


class BoundaryFineTuner(FineTuner):
    """ Fine-tunes a small head on top of the DINOv2 model for boundary estimation.

    Parameters
    ----------
    dinov2_vit_model : str
        ViT model name of DINOv2. One of ['vits14', 'vitl14', 'vitg14', 'vitb14'].
    mode : str
        One of ['affinity', 'direct']. If 'affinity', predict affinity. If 'direct', predict
        boundary directly.
    upsample_factor : float
        Upsample factor of the feature map after the ViT and before the head.
    head : str
        Head to use for boundary estimation. One of ['linear', 'knn', 'cnn', 'mlp'].
    neighbor_radius : float
        Neighbors within this radius are considered to generate the GT boundary map.
    threshold_boundary : float
        Threshold to generate a binary boundary map from the softmax output.
    num_boundary_neighbors : int
        Number of neighbors with different instance label a pixel needs to be considered as a
        boundary pixel.
    test_output_size : Tuple[int, int]
        Final output size [H, W] of the model during prediction/testing.
    test_multi_scales : List[int]
        List of scales to use for multi-scale during prediction/testing e.g. [1, 2, 3].
    test_plot : bool
        Whether to plot the predictions during testing.
    """

    def __init__(self, dinov2_vit_model: str, mode: str = 'direct',
                 upsample_factor: Optional[float] = None, head: str = 'mlp',
                 neighbor_radius: float = 1.5, threshold_boundary: float = 0.95,
                 num_boundary_neighbors: int = 1,
                 test_output_size: Optional[Tuple[int, int]] = None,
                 test_multi_scales: Optional[List[int]] = None,
                 test_plot: bool = False):
        super().__init__(dinov2_vit_model=dinov2_vit_model, blocks=None,
                         upsample_factor=upsample_factor)
        assert mode in ['affinity', 'direct']
        self.mode = mode
        self.neighbor_radius = neighbor_radius
        self.threshold_boundary = threshold_boundary
        self.num_boundary_neighbors = num_boundary_neighbors
        self.test_output_size = test_output_size
        self.test_multi_scales = test_multi_scales
        self.test_plot = test_plot

        if self.mode == 'affinity':
            if self.head == 'mlp':
                self.head = nn.Sequential(
                    nn.Linear(2 * self.feat_dim, 600),
                    nn.ReLU(),
                    nn.Linear(600, 600),
                    nn.ReLU(),
                    nn.Linear(600, 400),
                    nn.ReLU(),
                    nn.Linear(400, 1),
                )
            else:
                raise NotImplementedError
        elif self.mode == 'direct':
            if head == 'linear':
                self.head = nn.Conv2d(self.feat_dim, 1, kernel_size=1, stride=1, padding=0)
            elif head == 'knn':
                self.head = KNeighborsClassifier(n_neighbors=5, leaf_size=10)
                self.knn_X = []
                self.knn_y = []
            elif head == 'cnn':
                self.head = nn.Sequential(
                    nn.Conv2d(self.feat_dim, 600, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(600, 600, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(600, 400, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(400, 1, kernel_size=3, stride=1, padding=1),
                )
            elif head == 'mlp':
                self.head = nn.Sequential(
                    nn.Conv2d(self.feat_dim, 600, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(600, 600, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(600, 400, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(400, 1, kernel_size=1, stride=1, padding=0),
                )
            else:
                raise ValueError(f'Unknown head {head}')

        self.connected_indices_cache = None

    def connected_indices(self, h: int, w: int, batch_size: Optional[int] = None) -> np.ndarray:
        """ Returns all connected flattened pixel indices/coordinates for a given image size.
        Two pixels are connected if they are within self.neighbor_radius to each other.
        If batch_size is not None, the indices are tiled accordingly.

        Parameters
        ----------
        h : int
            height
        w : int
            width
        batch_size: int
            batch size

        Returns
        -------
        connected_indices : np.ndarray
            Connected indices [B, I, 2] or [I, 2] (if batch_size is None) for a given image size.
            If batch_size is not None, the indices are tiled accordingly.
            I is the number of connected pixels.
            Last dimension 2 are the flattened indices (indices are between 0 and H * W) of
            two pixels.
        """
        if self.connected_indices_cache is not None and \
                self.connected_indices_cache[0] == h and self.connected_indices_cache[1] == w and \
                self.connected_indices_cache[2] == batch_size:  # check if cache is valid
            return self.connected_indices_cache[3]

        coordinates = np.indices((h, w)).reshape(2, -1).T  # (H * W, 2)
        graph = radius_neighbors_graph(coordinates, radius=self.neighbor_radius,
                                       mode='connectivity',
                                       include_self=False)  # (H * W, H * W)
        connected_indices = np.argwhere(graph == 1)  # (I, 2)
        if batch_size is not None:
            connected_indices = np.tile(connected_indices, (batch_size, 1, 1))  # (B, I, 2)

        self.connected_indices_cache = (h, w, batch_size, connected_indices)
        return connected_indices

    def forward(self, img: torch.Tensor, connected_indices: np.array = None,
                segment_mask=None) -> torch.Tensor:
        x = self.forward_encoder(img)  # (B, feat_dim, H, W)

        if self.mode == 'affinity':
            batch_size = x.shape[0]
            h = x.shape[2]
            w = x.shape[3]

            x = x.view((batch_size, self.feat_dim, h * w))  # (B, feat_dim, H*W)
            x = x.permute((0, 2, 1)).contiguous()  # (B, H*W, feat_dim)
            if segment_mask is not None:  # (B, H, W)
                segment_mask = segment_mask.view(-1)  # (B*H*W)
                x = x.view(-1, self.feat_dim)  # (B*H*W, feat_dim)
                x = x[segment_mask == 1, :]  # (B*K, feat_dim)
                x = x.view(batch_size, -1, self.feat_dim)  # (B, K, feat_dim)
            if connected_indices is None:
                connected_indices = self.connected_indices(h, w, batch_size)  # (B, I, 2)
            x = x.view(-1, self.feat_dim)  # (B*K, feat_dim)
            connected_indices = connected_indices.reshape(-1, 2)  # (B*I, 2)
            x1 = x[connected_indices[:, 0], :]  # (B*I, feat_dim)
            x2 = x[connected_indices[:, 1], :]  # (B*I, feat_dim)
            x = torch.cat((x1, x2), dim=1)  # (B*I, 2*feat_dim)
            x = x.view(batch_size, -1, 2 * self.feat_dim)  # (B, I, 2*feat_dim)
            x = self.head(x)  # (B, I, 1)
            x = torch.sigmoid(x)  # (B, I, 1)
        elif self.mode == 'direct':
            if isinstance(self.head, KNeighborsClassifier):
                if self.training:
                    return x
                feat_shape = x.shape
                x = x.permute(0, 2, 3, 1).reshape(-1, feat_shape[1])
                x = x.detach().cpu().numpy()
                x = self.head.predict_proba(x)  # (B * H * W, 2)
                x = np.expand_dims(x[:, 1], axis=1)  # (B * H * W, 1)
                x = torch.from_numpy(x).to(self.device)
                x = x.reshape(feat_shape[0], feat_shape[2], feat_shape[3], -1) \
                    .permute(0, 3, 1, 2)  # (B, 1, H, W)
            else:
                x = self.head(x)  # (B, 1, H, W)
                x = torch.sigmoid(x)  # (B, 1, H, W)
        return x

    def training_step(self, train_batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        rgb = train_batch['rgb']
        ins = train_batch['instance'].long()

        device = rgb.device
        batch_size = rgb.shape[0]
        rgb_h, rgb_w = rgb.shape[2:]
        patches_h, patches_w = rgb_h // self.patch_size, rgb_w // self.patch_size
        upsample_factor = 1.0 if self.upsample_factor is None else self.upsample_factor
        network_output_size = (int(patches_h * upsample_factor), int(patches_w * upsample_factor))

        ins = TF.resize(ins, size=[network_output_size[0], network_output_size[1]],
                        interpolation=TF.InterpolationMode.NEAREST)  # (B, H, W)
        ins_flattened = ins.view(ins.shape[0], -1)  # (B, H * W)
        connected_indices = self.connected_indices(network_output_size[0],
                                                   network_output_size[1])  # (I, 2)

        if self.mode == 'affinity':
            ins_boundary = (ins_flattened[:, connected_indices[:, 0]] ==
                            ins_flattened[:, connected_indices[:, 1]]).to(torch.float)  # (B, I)

            pred = self(rgb)  # (B, I, 1)
            pred = pred.squeeze(2)  # (B, I)
        elif self.mode == 'direct':
            # Compute the GT boundary map
            ins_boundary = (ins_flattened[:, connected_indices[:, 0]] !=
                            ins_flattened[:, connected_indices[:, 1]]).cpu().numpy().astype(
                int)  # (B, I)
            connected_indices = np.tile(connected_indices, (batch_size, 1, 1))  # (B, I, 2)
            indices = connected_indices[:, :, 0]  # (B, I)
            ins_boundary = np.add.reduceat(ins_boundary,
                                           np.unique(indices, return_index=True, axis=1)[1],
                                           axis=1)  # (B, H*W)
            ins_boundary = np.logical_not(ins_boundary >= self.num_boundary_neighbors)  # (B, H*W)
            # Note: ins_boundary is 0 for boundary pixels and 1 for non-boundary pixels
            ins_boundary = torch.Tensor(ins_boundary.reshape(batch_size, network_output_size[0],
                                                             network_output_size[1])).to(
                device)  # (B, H, W)

            if isinstance(self.head, KNeighborsClassifier):
                x = self(rgb)  # (B, feat_dim, H, W)
                x = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])  # (B * H * W, feat_dim)
                x = x.detach().cpu().numpy()
                self.knn_X.append(x)

                ins_boundary = ins_boundary.reshape(-1)
                ins_boundary = ins_boundary.detach().cpu().numpy()
                self.knn_y.append(ins_boundary)
            else:
                pred = self(rgb)  # (B, 1, H, W)
                pred = pred.squeeze(1)  # (B, H, W)

        if isinstance(self.head, KNeighborsClassifier):
            loss = torch.tensor([0.0], requires_grad=True).to(self.device)
        else:
            loss = F.binary_cross_entropy(pred, ins_boundary)

        self.log('train_loss', loss)
        return loss

    def predict(self, rgb: torch.Tensor) -> torch.Tensor:
        batch_size = rgb.shape[0]
        rgb_h, rgb_w = rgb.shape[2:]
        patches_h, patches_w = rgb_h // self.patch_size, rgb_w // self.patch_size
        upsample_factor = 1.0 if self.upsample_factor is None else self.upsample_factor
        network_output_size = (int(patches_h * upsample_factor), int(patches_w * upsample_factor))

        if self.mode == 'affinity':
            if self.test_multi_scales is None:
                pred = self(rgb)  # (B, I, 1)
                pred = pred.squeeze(2)  # (B, I)
                pred = pred.detach().cpu().numpy()  # (B, I)

                if self.threshold_boundary is not None:
                    pred = (pred > self.threshold_boundary).astype(float)  # (B, I)

                connected_indices = self.connected_indices(network_output_size[0],
                                                           network_output_size[1],
                                                           batch_size)  # (B, I, 2)
                indices = connected_indices[:, :, 0]  # (B, I)
                pred = np.add.reduceat(pred, np.unique(indices, return_index=True, axis=1)[1],
                                       axis=1)  # (B, I)
                pred = (pred >= self.num_boundary_neighbors).astype(float)  # (B, I)
                pred = pred.reshape(batch_size, network_output_size[0],
                                    network_output_size[1])  # (B, H, W)
                pred = torch.Tensor(pred).to(rgb.device)  # (B, H, W)

            else:
                raise NotImplementedError

        elif self.mode == 'direct':
            if self.test_multi_scales is None:
                pred = self(rgb)  # (B, 1, H, W)
                pred = pred.squeeze(1)  # (B, H, W)
            else:
                pred = self.multi_scale_test_augmentation(rgb,
                                                          self.test_output_size)  # (B, 1, H, W)
                pred = pred.squeeze(1)  # (B, H, W)

            pred = (pred > self.threshold_boundary).to(torch.float)  # (B, H, W)

        pred = nn.functional.interpolate(pred.unsqueeze(1), size=self.test_output_size,
                                         mode='nearest').squeeze(1)  # (B, H, W)

        return pred

    def multi_scale_test_augmentation(self, rgb: torch.Tensor,
                                      output_size: Tuple[int, int]) -> torch.Tensor:
        # Splitting and upscaling at multiple scales
        all_preds = []
        batch_size = rgb.shape[0]
        rgb_h, rgb_w = rgb.shape[2:]

        for scale in self.test_multi_scales:
            image_h_split, image_w_split = rgb_h // scale, rgb_w // scale
            output_h_split, output_w_split = output_size[0] // scale, output_size[1] // scale

            rgb_split = torch.split(rgb, image_h_split, dim=2)
            rgb_split = [torch.split(split, image_w_split, dim=3) for split in rgb_split]

            pred_scale = torch.zeros(
                (batch_size, 1, output_size[0], output_size[1]))
            for row, row_splits in enumerate(rgb_split):
                for col, rgb_split_i in enumerate(row_splits):
                    rgb_split_i_upscaled = T.functional.resize(
                        rgb_split_i, [rgb_h, rgb_w], interpolation=InterpolationMode.BILINEAR)
                    pred = self(rgb_split_i_upscaled)  # (B, num_classes, H, W)

                    # Resize to output size. If rgb is not divisible by output size, need to
                    # resize accordingly
                    row_end = (row + 1) * output_h_split
                    i_output_h_split = output_h_split
                    if row_end > output_size[0]:
                        i_output_h_split = output_size[0] - row * output_h_split
                        row_end = output_size[0]
                    col_end = (col + 1) * output_w_split
                    i_output_w_split = output_w_split
                    if col_end > output_size[1]:
                        i_output_w_split = output_size[1] - col * output_w_split
                        col_end = output_size[1]

                    pred = nn.functional.interpolate(pred,
                                                     size=[i_output_h_split, i_output_w_split],
                                                     mode='area')  # (B, num_classes, H, W)

                    pred_scale[:, :, row * output_h_split:row_end, col * output_w_split:col_end] \
                        = pred  # (B, num_classes, H, W)

            all_preds.append(pred_scale)

        # Concatenate and fuse scales
        pred = torch.stack(all_preds, dim=1)  # (B, S, num_classes, H, W)
        pred = pred.mean(dim=1)  # (B, num_classes, H, W)
        return pred

    @staticmethod
    def plot(rgb: np.array, pred: np.array):
        plt.figure(figsize=(20, 6))
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(10, 10)

        rgb = rgb.transpose((1, 2, 0))  # (H, W, 3)

        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.grid(False)
        plt.imshow(rgb)

        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.grid(False)
        plt.imshow(rgb)
        plt.imshow(pred, cmap='jet', alpha=0.5, interpolation='nearest')
        plt.show()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        rgb = batch['rgb']  # (B, 3, H, W)

        pred = self.predict(rgb)  # (B, H, W)

        pred = pred.cpu().numpy()  # (B, H, W)
        rgb_original = batch['rgb_original']  # (B, 3, H, W)
        rgb_original = rgb_original.cpu().numpy()  # (B, 3, H, W)

        if self.test_plot:
            for rgb_i, pred_i in zip(rgb_original, pred):
                self.plot(rgb_i, pred_i)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        if isinstance(self.head, KNeighborsClassifier):
            checkpoint['knn_X'] = self.knn_X
            checkpoint['knn_y'] = self.knn_y

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        if isinstance(self.head, KNeighborsClassifier):
            self.knn_X = np.concatenate(checkpoint['knn_X'], axis=0)
            self.knn_y = np.concatenate(checkpoint['knn_y'], axis=0)
            self.head = self.head.fit(self.knn_X, self.knn_y)


class BoundaryFineTunerCLI(LightningCLI):

    def __init__(self):
        super().__init__(
            model_class=BoundaryFineTuner,
            seed_everything_default=0,
            parser_kwargs={'parser_mode': 'omegaconf'},
            save_config_callback=None,
        )

    def add_arguments_to_parser(self, parser):
        # Dataset
        parser.add_argument('--data_params', type=Dict)


if __name__ == '__main__':
    cli = BoundaryFineTunerCLI()
