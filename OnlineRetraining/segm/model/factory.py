from pathlib import Path
import yaml
import torch
import math
import os
import torch.nn as nn

from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.vision_transformer import default_cfgs
from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer

from segm.model.vit import VisionTransformer, DINOV2VisionTransformer, EVA02VisionTransformer
from segm.model.utils import checkpoint_filter_fn
from segm.model.decoder import DecoderLinear
from segm.model.decoder import MaskTransformer, MultiMaskTransformer
from segm.model.segmenter import Segmenter, MultiSegmenter
import segm.utils.torch as ptu

from apex.normalization import FusedLayerNorm

# 添加多个键值对

default_cfgs.update({
    "dino_small_patch16_224": {"url": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"},
    "dinov2_small_patch16_224": {"url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"},
    "eva02_small_patch16_224": {"url": "eva02_S_pt_in21k_p14.pt"},
    "eva02_tiny_patch16_224": {"url": "eva02_T_pt_in21k_p14.pt"},  
})

@register_model
def vit_base_patch8_384(pretrained=False, **kwargs):
    """ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch8_384",
        pretrained=pretrained,
        default_cfg=dict(
            url="",
            input_size=(3, 384, 384),
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            num_classes=1000,
        ),
        **model_kwargs,
    )
    return model


def create_vit(model_cfg):
    model_cfg = model_cfg.copy()
    backbone = model_cfg.pop("backbone")

    normalization = model_cfg.pop("normalization")
    model_cfg["n_cls"] = 1000
    mlp_expansion_ratio = 4
    model_cfg["d_ff"] = mlp_expansion_ratio * model_cfg["d_model"]

    if backbone in default_cfgs:
        default_cfg = default_cfgs[backbone]
    else:
        default_cfg = dict(
            pretrained=False,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.0,
            drop_block_rate=None,
        )

    default_cfg["input_size"] = (
        3,
        model_cfg["image_size"][0],
        model_cfg["image_size"][1],
    )
    if "dinov2" in backbone:
        model = DINOV2VisionTransformer(**model_cfg)
    elif "eva02" in backbone:
        mlp_expansion_ratio = 4*2/3
        model_cfg["d_ff"] = int(mlp_expansion_ratio * model_cfg["d_model"])
        model_cfg["norm_layer"] = FusedLayerNorm
        model = EVA02VisionTransformer(**model_cfg)
    else:
        model = VisionTransformer(**model_cfg)
    
    from torch.hub import get_dir
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, 'checkpoints')
    if backbone == "vit_base_patch8_384":
        path = os.path.join(model_dir, "vit_base_patch8_384.pth")
        state_dict = torch.load(path, map_location="cpu")
        filtered_dict = checkpoint_filter_fn(state_dict, model)
        model.load_state_dict(filtered_dict, strict=True)
    elif "deit" in backbone:
        load_pretrained(model, default_cfg, filter_fn=checkpoint_filter_fn)
    elif "dino" in backbone:
        # without head
        load_pretrained(model, default_cfg, filter_fn=checkpoint_filter_fn, strict=False)
    elif "eva02" in backbone:
        path = os.path.join(model_dir, default_cfg["url"])
        state_dict = torch.load(path, map_location="cpu")
        filtered_dict = checkpoint_filter_fn(state_dict, model)
        model.load_state_dict(filtered_dict, strict=False)
    else:
        load_custom_pretrained(model, default_cfg)

    return model


def create_decoder(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    decoder_cfg["d_encoder"] = encoder.d_model
    decoder_cfg["patch_size"] = encoder.patch_size

    if "linear" in name:
        decoder = DecoderLinear(**decoder_cfg)
    elif name == "mask_transformer":
        dim = encoder.d_model
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder = MaskTransformer(**decoder_cfg)
    elif name == "multi_mask_transformer":
        dim = encoder.d_model
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder = MultiMaskTransformer(**decoder_cfg)
    else:
        raise ValueError(f"Unknown decoder: {name}")
    return decoder


def create_segmenter(model_cfg):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop("decoder")
    decoder_cfg["n_cls"] = model_cfg["n_cls"]

    encoder = create_vit(model_cfg)
    decoder = create_decoder(encoder, decoder_cfg)
    model = Segmenter(encoder, decoder, n_cls=model_cfg["n_cls"])

    return model


def create_multi_segmenter(model_cfg):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop("decoder")
    decoder_cfg["n_cls"] = model_cfg["n_cls"]

    encoder = create_vit(model_cfg)
    decoder = create_decoder(encoder, decoder_cfg)
    model = MultiSegmenter(encoder, decoder, n_cls=model_cfg["n_cls"])
    return model


def load_model(model_path, backbone=None):
    variant_path = Path(model_path).parent / "variant.yml"
    with open(variant_path, "r") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    net_kwargs = variant["net_kwargs"]
    if backbone is None:
        backbone = net_kwargs["backbone"]
    if "multi" in backbone:
        net_kwargs["decoder"]["name"] = "multi_mask_transformer"
        model = create_multi_segmenter(net_kwargs)
    else:
        model = create_segmenter(net_kwargs)
    data = torch.load(model_path, map_location="cpu")
    checkpoint = data["model"]

    model.load_state_dict(checkpoint, strict=False)

    return model, variant
