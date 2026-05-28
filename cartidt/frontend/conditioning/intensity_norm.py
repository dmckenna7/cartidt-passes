"""Per-volume z-score intensity normalisation (Ref: Sec. V.C ¶1)."""

from __future__ import annotations

import torch


def zscore_per_volume(volume: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if volume.ndim < 3:
        raise ValueError("volume must be at least 3D (D, H, W)")
    flat_axes = tuple(range(volume.ndim - 3, volume.ndim))
    mean = volume.mean(dim=flat_axes, keepdim=True)
    std = volume.std(dim=flat_axes, keepdim=True, unbiased=False)
    return (volume - mean) / (std + eps)
