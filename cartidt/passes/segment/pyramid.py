"""Feature Pyramid top-down path of UPerNet (Ref: Sec. III.C-3)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeaturePyramid(nn.Module):
    def __init__(self, in_channels: Sequence[int], out_channels: int) -> None:
        super().__init__()
        self.lateral = nn.ModuleList(
            [nn.Conv3d(c, out_channels, kernel_size=1, bias=False) for c in in_channels]
        )
        self.smooth = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(num_groups=8, num_channels=out_channels),
                    nn.GELU(),
                )
                for _ in in_channels
            ]
        )

    def forward(self, feats: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(feats) != len(self.lateral):
            raise ValueError(f"expected {len(self.lateral)} feature maps, got {len(feats)}")
        laterals = [conv(feat) for conv, feat in zip(self.lateral, feats, strict=True)]
        for idx in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(
                laterals[idx + 1], size=laterals[idx].shape[-3:], mode="trilinear", align_corners=False
            )
            laterals[idx] = laterals[idx] + up
        return [smooth(lat) for smooth, lat in zip(self.smooth, laterals, strict=True)]
