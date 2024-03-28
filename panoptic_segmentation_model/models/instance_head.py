from typing import Optional, Tuple

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor, nn
from torch.nn import functional as F


class _InstanceDecoder(nn.Module):

    def __init__(self, num_ch_enc: ArrayLike, use_skips: bool, is_dino=False):
        super().__init__()

        self.num_ch_enc = num_ch_enc
        self.use_skips = use_skips
        self.num_ch_dec_project = np.array([64, 32, 16])
        self.num_ch_dec = np.array([256, 128, 128])
        self.is_dino = is_dino
        if self.is_dino:
            self.num_ch_dec = np.array([320, 128, 128])

        self.net = nn.ModuleDict()

        for i, _ in enumerate(self.num_ch_dec):
            # 1x1 convolutions in skip connections (project)
            if self.use_skips:
                num_ch_in = int(self.num_ch_enc[-(i + 2)])
                num_ch_out = int(self.num_ch_dec_project[i])
                self.net[f'project_{i}'] = nn.Sequential(
                    nn.Conv2d(num_ch_in, num_ch_out, kernel_size=1, bias=False),
                    nn.BatchNorm2d(num_ch_out),
                    nn.ReLU(inplace=True),
                )

            # "normal" upsampling pass
            num_ch_in = int(self.num_ch_dec[0]) if i == 0 else int(self.num_ch_dec[i - 1])
            if self.use_skips:
                num_ch_in += int(self.num_ch_dec_project[i])
            num_ch_out = int(self.num_ch_dec[i])
            # Depthwise separable convolution with BatchNorm and ReLU
            # https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/model/decoder/conv_module.py
            self.net[f'fuse_{i}'] = nn.Sequential(
                nn.Conv2d(num_ch_in,
                          num_ch_in,
                          kernel_size=5,
                          stride=1,
                          padding=2,
                          groups=num_ch_in,
                          bias=False),
                nn.BatchNorm2d(num_ch_in),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_ch_in, num_ch_out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_ch_out),
                nn.ReLU(inplace=True),
            )
        # Initial 1x1 connection shown in paper
        self.net['conv'] = nn.Sequential(
            nn.Conv2d(int(self.num_ch_enc[-1]),
                      int(self.num_ch_dec[0]),
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False))

    def forward(self, in_feats: Tensor) -> Tensor:
        feat = in_feats[-1]
        feat = self.net['conv'](feat)
        for i, _ in enumerate(self.num_ch_dec):
            if self.use_skips:
                skip_feat = self.net[f'project_{i}'](in_feats[-(i + 2)])
            if not self.is_dino or self.use_skips or (self.is_dino and i > 0):
                feat = F.interpolate(feat, scale_factor=2, mode='bilinear')
            if self.use_skips:
                feat = torch.cat([feat, skip_feat], 1)
            feat = self.net[f'fuse_{i}'](feat)

        return feat

    def __repr__(self):
        return self.net.__repr__()


class _InstanceHead(nn.Module):

    def __init__(self, num_ch_dec: ArrayLike, num_classes: int, feed_img_size):
        super().__init__()

        self.num_ch_dec = num_ch_dec  # [..., 128]
        self.num_ch_head = np.array([32, num_classes])
        self.feed_img_size = feed_img_size

        self.net = nn.ModuleDict()

        # Depthwise separable convolution with BatchNorm and ReLU
        # https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/model/decoder/conv_module.py
        self.net['depth_conv'] = nn.Sequential(
            nn.Conv2d(int(self.num_ch_dec[-1]),
                      int(self.num_ch_dec[-1]),
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      groups=int(self.num_ch_dec[-1]),
                      bias=False),
            nn.BatchNorm2d(int(self.num_ch_dec[-1])),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(self.num_ch_dec[-1]),
                      int(self.num_ch_head[0]),
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(int(self.num_ch_head[0])),
            nn.ReLU(inplace=True),
        )
        # Vanilla convolution
        self.net['conv'] = nn.Conv2d(int(self.num_ch_head[0]),
                                     int(self.num_ch_head[1]),
                                     kernel_size=1)

    def forward(self, in_feats: Tensor) -> Tensor:
        feat = F.interpolate(in_feats, scale_factor=2, mode='bilinear', align_corners=True)
        feat = self.net['depth_conv'](feat)
        feat = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=True)
        pred = self.net['conv'](feat)
        pred = F.interpolate(pred, self.feed_img_size, mode='bilinear', align_corners=False)
        return pred

    def __repr__(self):
        return self.net.__repr__()


class InstanceHead(nn.Module):
    """ Instance head following PanopticDeepLab architecture
    https://github.com/bowenc0221/panoptic-deeplab/blob/master/segmentation/model/decoder/panoptic_deeplab.py

    The network consists of a shared decoder and two task-specific heads, namely a CenterHead to
    predict the center of instances and an OffsetHead to predict the offset of each pixel to the
    nearest instance center.
    """

    def __init__(self, num_ch_enc: ArrayLike, use_skips, feed_img_size, is_dino):
        super().__init__()

        self.decoder = _InstanceDecoder(num_ch_enc, use_skips, is_dino=is_dino)
        self.center_head = _InstanceHead(self.decoder.num_ch_dec, num_classes=1,
                                         feed_img_size=feed_img_size)
        self.offset_head = _InstanceHead(self.decoder.num_ch_dec, num_classes=2,
                                         feed_img_size=feed_img_size)

    def forward(self, in_feats: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        decoder_feats = self.decoder(in_feats)
        center = self.center_head(decoder_feats)
        offset = self.offset_head(decoder_feats)

        return center, offset
