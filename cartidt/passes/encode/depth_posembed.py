"""Depth-axis positional interpolation (Ref: Sec. III.C-2, Eq. (2)).

The 2-D DINOv2 positional embedding `P_2D` lives on an (H/u)² grid with patch
size u = 16. To process a 3-D volume with D depth slices we (a) bicubic-resample
`P_2D` over the same (H/u)² grid and (b) learn a small depth-axis embedding
`P_depth ∈ R^{D/s_d × d}` with stride `s_d = 4`. The 3-D positional embedding is

    P_3D[z, i, j] = Interp(P_2D)[i, j] + P_depth[floor(z / s_d)].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthInterpolatedPosEmbed(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        patch_size: int = 16,
        depth_stride: int = 4,
        target_hw: tuple[int, int] = (384, 384),
        target_depth: int = 160,
        pos_2d_init: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        h, w = target_hw
        if h % patch_size or w % patch_size:
            raise ValueError("H and W must be divisible by patch_size")
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depth_stride = depth_stride
        self.target_hw = target_hw
        self.target_depth = target_depth
        self.grid_hw = (h // patch_size, w // patch_size)
        self.depth_slots = (target_depth + depth_stride - 1) // depth_stride
        if pos_2d_init is None:
            pos_2d_init = torch.zeros(1, embed_dim, *self.grid_hw)
            nn.init.trunc_normal_(pos_2d_init, std=0.02)
        if pos_2d_init.ndim != 4:
            raise ValueError("pos_2d_init must be (1, C, gH, gW)")
        self.register_buffer("pos_2d", pos_2d_init, persistent=True)
        self.pos_depth = nn.Parameter(torch.zeros(self.depth_slots, embed_dim))
        nn.init.trunc_normal_(self.pos_depth, std=0.02)

    def forward(self, depth: int) -> torch.Tensor:
        gh, gw = self.grid_hw
        plane = F.interpolate(self.pos_2d, size=(gh, gw), mode="bicubic", align_corners=False)
        plane = plane.squeeze(0).permute(1, 2, 0).reshape(gh * gw, self.embed_dim)
        depth_index = torch.div(
            torch.arange(depth, device=plane.device), self.depth_stride, rounding_mode="floor"
        )
        depth_index = depth_index.clamp(max=self.depth_slots - 1)
        depth_terms = self.pos_depth.index_select(0, depth_index)
        return plane.unsqueeze(0) + depth_terms.unsqueeze(1)
