"""Frozen ViT-L backbone + depth-wise LoRA + UPerNet-style multi-scale taps.

The backbone yields four feature maps at 1/4, 1/8, 1/16, 1/32 of the input
resolution (Ref: Sec. III.C-3). On a 160 × 384 × 384 volume the deepest tap is
sized (160/16, 384/16, 384/16) = (10, 24, 24) per slice batched along depth.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from cartidt.passes.encode.depth_posembed import DepthInterpolatedPosEmbed
from cartidt.passes.encode.lora import inject_lora_qv


class LoRAViTBackbone(nn.Module):
    def __init__(
        self,
        vit: nn.Module,
        embed_dim: int = 1024,
        patch_size: int = 16,
        depth_stride: int = 4,
        rank: int = 16,
        alpha: int = 32,
        target_hw: tuple[int, int] = (384, 384),
        target_depth: int = 160,
        tap_layers: Sequence[int] = (5, 11, 17, 23),
    ) -> None:
        super().__init__()
        self.vit = vit
        self.tap_layers = tuple(int(x) for x in tap_layers)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        q_names, v_names = self._discover_qv(vit)
        inject_lora_qv(vit, q_names, v_names, rank=rank, alpha=alpha)
        self.pos3d = DepthInterpolatedPosEmbed(
            embed_dim=embed_dim,
            patch_size=patch_size,
            depth_stride=depth_stride,
            target_hw=target_hw,
            target_depth=target_depth,
        )
        self._taps: list[torch.Tensor] = []
        self._register_taps()

    @staticmethod
    def _discover_qv(vit: nn.Module) -> tuple[list[str], list[str]]:
        q_names: list[str] = []
        v_names: list[str] = []
        for name, _module in vit.named_modules():
            if name.endswith(".attn.q_proj") or name.endswith(".attn.q"):
                q_names.append(name)
            if name.endswith(".attn.v_proj") or name.endswith(".attn.v"):
                v_names.append(name)
            if name.endswith(".attn.qkv"):
                q_names.append(name)
        if not q_names:
            raise RuntimeError("could not locate ViT attention Q projections to wrap")
        return q_names, v_names

    def _register_taps(self) -> None:
        blocks = getattr(self.vit, "blocks", None)
        if blocks is None:
            raise AttributeError("ViT model has no `blocks` attribute")
        for idx in self.tap_layers:
            if idx >= len(blocks):
                raise IndexError(f"tap layer {idx} out of range; ViT has {len(blocks)} blocks")
            blocks[idx].register_forward_hook(self._make_hook(idx))

    def _make_hook(self, idx: int):
        order = self.tap_layers.index(idx)

        def hook(_module: nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
            while len(self._taps) <= order:
                self._taps.append(out.detach())
            self._taps[order] = out

        return hook

    def forward(self, volume: torch.Tensor) -> list[torch.Tensor]:
        if volume.ndim != 5:
            raise ValueError("expected volume shape (B, 1, D, H, W)")
        b, c, d, h, w = volume.shape
        if c == 1:
            volume = volume.expand(-1, 3, -1, -1, -1)
        slices = rearrange(volume, "b c d h w -> (b d) c h w")
        pos = self.pos3d(d).to(slices.dtype).to(slices.device)
        self._taps.clear()
        self.vit.forward_features(slices) if hasattr(self.vit, "forward_features") else self.vit(slices)
        outputs: list[torch.Tensor] = []
        for order, tap in enumerate(self._taps):
            x = (
                tap[:, 1:, :]
                if tap.shape[1] > 1 + (h // self.patch_size) * (w // self.patch_size) - 1
                else tap
            )
            if order == 0:
                x = x + pos.repeat(b, 1, 1)
            n = x.shape[1]
            gh = h // self.patch_size
            gw = w // self.patch_size
            if n != gh * gw:
                raise RuntimeError(f"tap token count {n} != gH·gW {gh * gw}")
            x = rearrange(x, "(b d) (gh gw) c -> b c d gh gw", b=b, d=d, gh=gh, gw=gw)
            outputs.append(x)
        scale_factors = (1.0, 0.5, 0.5, 0.25)
        out_pyramid: list[torch.Tensor] = []
        for ix, feat in enumerate(outputs):
            if scale_factors[ix] == 1.0:
                out_pyramid.append(feat)
            else:
                target = (
                    feat.shape[-3],
                    int(feat.shape[-2] * scale_factors[ix]),
                    int(feat.shape[-1] * scale_factors[ix]),
                )
                out_pyramid.append(F.interpolate(feat, size=target, mode="trilinear", align_corners=False))
        return out_pyramid
