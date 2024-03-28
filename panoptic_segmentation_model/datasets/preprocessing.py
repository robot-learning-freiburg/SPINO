import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
from skimage.exposure import match_histograms
from torchvision.transforms import Compose, Lambda, Normalize, ToTensor
from torchvision.transforms import functional as F
from yacs.config import CfgNode as CN


def prepare_for_network(output: Dict[str, Any], cfg: CN):
    # Convert PIL image to torch.Tensor.
    output['rgb'] = ToTensor()(output['rgb'])
    if cfg.active:
        output['rgb'] = Normalize(mean=cfg.rgb_mean, std=cfg.rgb_std)(output['rgb'])


def augment_data(output: Dict[str, Any], cfg: CN):
    if not cfg.active:
        return

    brightness = None
    contrast = None
    saturation = None
    hue = None
    if cfg.brightness_jitter is not None:
        brightness = (1 - cfg.brightness_jitter, 1 + cfg.brightness_jitter)
    if cfg.contrast_jitter is not None:
        contrast = (1 - cfg.contrast_jitter, 1 + cfg.contrast_jitter)
    if cfg.saturation_jitter is not None:
        saturation = (1 - cfg.saturation_jitter, 1 + cfg.saturation_jitter)
    if cfg.hue_jitter is not None:
        hue = (-cfg.hue_jitter, cfg.hue_jitter)
    color_augmentation = _get_random_color_jitter(brightness, contrast, saturation, hue)

    do_flip = cfg.horizontal_flipping and random.random() > .5

    for key, value in output.items():
        if key == 'rgb':
            if do_flip:
                output[key] = output[key].transpose(Image.FLIP_LEFT_RIGHT)
            output[key] = color_augmentation(output[key])
        elif do_flip:
            output[key] = np.flip(value, axis=-1).copy()
            if key == 'offset':
                # We have to make sure that x_coords are still correct
                output[key][1] *= -1


def transfer_histogram_style(img: Image, reference_img: Image, mode: str) -> Image:
    if mode == 'rgb':
        new_img = match_histograms(np.array(img), np.array(reference_img), channel_axis=-1)
        new_img = Image.fromarray(new_img, mode='RGB')
    elif mode in ['hsv', 'saturation']:
        img_hsv = np.array(img.convert('HSV'))
        reference_img_hsv = np.array(reference_img.convert('HSV'))
        new_img = match_histograms(img_hsv, reference_img_hsv, channel_axis=-1)
        if mode == 'saturation':
            new_img[:, :, 0] = img_hsv[:, :, 0]
            new_img[:, :, 2] = img_hsv[:, :, 2]
        new_img = Image.fromarray(new_img, mode='HSV').convert('RGB')
    elif mode is None:
        new_img = img
    else:
        raise ValueError(f'Unknown histogram transfer mode: {mode}')
    return new_img


# -------------------------------------------------------- #
# Adapted from:
# https://github.com/pytorch/vision/pull/3001#issuecomment-814919958
def _get_random_color_jitter(
        brightness: Optional[Tuple[float, float]] = None,
        contrast: Optional[Tuple[float, float]] = None,
        saturation: Optional[Tuple[float, float]] = None,
        hue: Optional[Tuple[float, float]] = None,
) -> Compose:
    transforms_ = []

    if brightness is not None:
        brightness_factor = random.uniform(brightness[0], brightness[1])
        transforms_.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))
    if contrast is not None:
        contrast_factor = random.uniform(contrast[0], contrast[1])
        transforms_.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))
    if saturation is not None:
        saturation_factor = random.uniform(saturation[0], saturation[1])
        transforms_.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))
    if hue is not None:
        hue_factor = random.uniform(hue[0], hue[1])
        transforms_.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

    random.shuffle(transforms_)
    transforms_ = Compose(transforms_)
    return transforms_
