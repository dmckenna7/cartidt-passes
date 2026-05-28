"""Frozen DINOv2 ViT-L/16 backbone loader (Ref: Sec. III.C, ref. [13]).

Returns a `timm`-compatible vision transformer with all parameters frozen.
The actual weights live at `facebook/dinov2-large` on the HuggingFace hub or
inside the local `timm` cache. Network access is not required when a cached
checkpoint is on disk.
"""

from __future__ import annotations

import logging

import torch.nn as nn

_LOG = logging.getLogger(__name__)


def load_frozen_dinov2_large(weights: str = "vit_large_patch14_dinov2", pretrained: bool = True) -> nn.Module:
    try:
        import timm
    except ImportError as exc:
        raise ImportError("timm is required to load DINOv2 weights") from exc
    _LOG.info("loading DINOv2 backbone: %s (pretrained=%s)", weights, pretrained)
    model = timm.create_model(weights, pretrained=pretrained, num_classes=0)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    return model
