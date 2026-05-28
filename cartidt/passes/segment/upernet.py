"""Full UPerNet 3-D decoder for cartilage / meniscus segmentation (Ref: Sec. III.C-3).

Inputs are four pyramid feature maps from the LoRA ViT backbone. The output is
a dense `(B, C+1, D, H, W)` logit volume at the original MRI resolution.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cartidt.passes.segment.pooling import PyramidPooling
from cartidt.passes.segment.pyramid import FeaturePyramid


class UPerNetDecoder(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        decoder_dim: int = 256,
        num_classes: int = 7,
        ppm_scales: Sequence[int] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        if len(in_channels) < 2:
            raise ValueError("UPerNet expects at least two pyramid levels")
        self.ppm = PyramidPooling(in_channels[-1], decoder_dim, scales=ppm_scales)
        intermediate = [*in_channels[:-1], decoder_dim]
        self.fpn = FeaturePyramid(intermediate, decoder_dim)
        self.fuse = nn.Sequential(
            nn.Conv3d(decoder_dim * len(intermediate), decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=decoder_dim),
            nn.GELU(),
        )
        self.head = nn.Conv3d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, feats: Sequence[torch.Tensor], target_shape: tuple[int, int, int]) -> torch.Tensor:
        if len(feats) < 2:
            raise ValueError("UPerNet expects at least two pyramid levels")
        deepest = self.ppm(feats[-1])
        merged = [*feats[:-1], deepest]
        fpn_feats = self.fpn(merged)
        base = fpn_feats[0]
        same_size = [
            F.interpolate(f, size=base.shape[-3:], mode="trilinear", align_corners=False) for f in fpn_feats
        ]
        fused = self.fuse(torch.cat(same_size, dim=1))
        logits = self.head(fused)
        return F.interpolate(logits, size=target_shape, mode="trilinear", align_corners=False)
