"""Pyramid Pooling Module of UPerNet (Ref: Sec. III.C-3, ref. [14])."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scales: Sequence[int] = (1, 2, 3, 6)) -> None:
        super().__init__()
        self.scales = tuple(scales)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool3d(output_size=s),
                    nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(num_groups=8, num_channels=out_channels),
                    nn.GELU(),
                )
                for s in scales
            ]
        )
        self.project = nn.Sequential(
            nn.Conv3d(
                in_channels + out_channels * len(scales), out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = [x]
        for branch in self.branches:
            y = branch(x)
            pooled.append(F.interpolate(y, size=x.shape[-3:], mode="trilinear", align_corners=False))
        return self.project(torch.cat(pooled, dim=1))
