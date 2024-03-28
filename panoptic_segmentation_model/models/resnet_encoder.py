# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import dataclasses

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils import model_zoo
from torchvision import models

NEW_TORCHVISION_API = "0.13" in torchvision.__version__

@dataclasses.dataclass
class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, block, layers, num_input_images=1, num_channels_input=3):
        super().__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * num_channels_input, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1, num_channels_input=3):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_channels_input >= 3, "Require at least RGB image (3 channels)"
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images, num_channels_input)

    if pretrained:
        if NEW_TORCHVISION_API:
            model_url = eval(f"models.resnet.ResNet{num_layers}_Weights.IMAGENET1K_V1.value.url")   # pylint: disable=eval-used
        else:
            model_url = models.resnet.model_urls[f"resnet{num_layers}"]
        loaded = model_zoo.load_url(model_url)

        if num_channels_input > 3:
            # Use mean of R, G, and B channels for additional channels
            additional_channels = num_channels_input - 3

            weights = [loaded["conv1.weight"]] + additional_channels * [
                loaded["conv1.weight"].mean(1, keepdim=True)]
            weights = [torch.cat(weights, dim=1)]
        else:
            weights = [loaded["conv1.weight"]]

        loaded["conv1.weight"] = torch.cat(weights * num_input_images, 1) / num_input_images

        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers, pretrained, num_input_images=1, num_channels_input=3):
        super().__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError(f"{num_layers} is not a valid number of resnet layers")

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images,
                                                   num_channels_input)
        else:
            if pretrained:
                if NEW_TORCHVISION_API:
                    self.encoder = resnets[num_layers](weights="IMAGENET1K_V1")
                else:
                    self.encoder = resnets[num_layers](pretrained=True)
            else:
                self.encoder = resnets[num_layers]()

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        features = []
        x = input_image
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        features.append(self.encoder.relu(x))
        features.append(self.encoder.layer1(self.encoder.maxpool(features[-1])))
        features.append(self.encoder.layer2(features[-1]))
        features.append(self.encoder.layer3(features[-1]))
        features.append(self.encoder.layer4(features[-1]))

        return features
