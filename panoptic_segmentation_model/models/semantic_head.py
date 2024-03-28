import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike
from torch import Tensor, nn


class SemanticHead(nn.Module):
    def __init__(self, num_ch_enc: ArrayLike, num_classes: int, feed_img_size, use_skips: bool,
                 use_guda_fusion: bool = False, is_dino=False):
        super().__init__()

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.feed_img_size = feed_img_size
        self.use_skips = use_skips
        self.use_guda_fusion = use_guda_fusion
        self.is_dino = is_dino

        # Set up the up convolutions
        self.upconvs_0, self.upconvs_1 = nn.ModuleDict(), nn.ModuleDict()

        for i in range(4, -1, -1):
            # Upconv 0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.upconvs_0[str(i)] = nn.Sequential(
                nn.Conv2d(int(num_ch_in), int(num_ch_out), 3, stride=1, padding=1),
                nn.ELU(inplace=True))

            # Upconv 1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.upconvs_1[str(i)] = nn.Sequential(
                nn.Conv2d(int(num_ch_in), int(num_ch_out), 3, stride=1, padding=1),
                nn.ELU(inplace=True))

        if self.use_guda_fusion:
            num_ch_concat_guda = self.num_ch_dec.sum() - self.num_ch_dec[-1]
            self.semconv_guda = nn.Conv2d(num_ch_concat_guda, num_classes, 3, padding=1)
        else:
            # Final layer to get the number of channels equal to 'num_classes'
            self.semconv = nn.Sequential(nn.ReflectionPad2d(1),
                                         nn.Conv2d(int(self.num_ch_dec[0]), num_classes, 3))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats: Tensor) -> Tensor:
        feat = in_feats[-1]
        feat_guda_mem = []

        for i in range(4, -1, -1):
            feat = self.upconvs_0[str(i)](feat)
            if not self.is_dino or (self.is_dino and i < 4):
                feat = [F.interpolate(feat, scale_factor=2, mode="nearest")]
            else:
                feat = [feat]
            if self.use_skips and i > 0:
                feat += [in_feats[i - 1]]
            feat = torch.cat(feat, 1)
            feat = self.upconvs_1[str(i)](feat)

            if i < 4 and self.use_guda_fusion:
                feat_guda_mem.append(feat)

        if self.use_guda_fusion:
            res_out = feat_guda_mem[-1].shape[-2:]
            for i, feat_i in enumerate(feat_guda_mem):
                feat_guda_mem[i] = F.interpolate(feat_i, res_out, mode="bilinear",
                                                 align_corners=False)
            feat_cat = torch.cat(feat_guda_mem, dim=1)
            sem_feat = self.semconv_guda(feat_cat)
        else:
            sem_feat = self.semconv(feat)

        sem_logits = self.softmax(sem_feat)
        sem_logits = F.interpolate(sem_logits, self.feed_img_size, mode="bilinear",
                                   align_corners=False)

        return sem_logits
