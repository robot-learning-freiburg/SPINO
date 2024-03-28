# Copied from: https://github.com/facebookresearch/dinov2/blob/main/hubconf.py

import torch

_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


def _make_dinov2_model_name(arch_name: str, patch_size: int) -> str:
    compact_arch_name = arch_name.replace("_", "")[:4]
    return f"dinov2_{compact_arch_name}{patch_size}"


def _make_dinov2_model(
        *,
        arch_name: str = "vit_large",
        img_size: int = 518,
        patch_size: int = 14,
        init_values: float = 1.0,
        ffn_layer: str = "mlp",
        block_chunks: int = 0,
        pretrained: bool = True,
        **kwargs,
):
    from .vit import vision_transformer as vits

    model_name = _make_dinov2_model_name(arch_name, patch_size)
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
    )
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)

    if pretrained:
        url = _DINOV2_BASE_URL + f"/{model_name}/{model_name}_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    return model


def dinov2_vits14(*, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-S/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_small", pretrained=pretrained, **kwargs)


def dinov2_vitb14(*, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-B/14 model pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_base", pretrained=pretrained, **kwargs)


def dinov2_vitl14(*, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-L/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_large", pretrained=pretrained, **kwargs)


def dinov2_vitg14(*, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-g/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_giant2", ffn_layer="swiglufused",
                              pretrained=pretrained, **kwargs)
