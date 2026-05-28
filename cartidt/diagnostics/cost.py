"""Parameter / FLOP / per-volume timing counters (Ref: Sec. V.G, Table X)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True, slots=True)
class ProfileResult:
    params_millions: float
    flops_billions: float
    seconds_per_volume: float
    peak_memory_gb: float


def count_params(model: nn.Module, trainable_only: bool = False) -> float:
    params = model.parameters()
    total = sum(p.numel() for p in params if (p.requires_grad or not trainable_only))
    return float(total) / 1_000_000.0


def profile_per_volume(
    model: nn.Module,
    sample_volume: torch.Tensor,
    iterations: int = 5,
) -> ProfileResult:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    device = sample_volume.device
    model.eval()
    with torch.no_grad():
        model(sample_volume)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(iterations):
            model(sample_volume)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / iterations
    peak_gb = float(torch.cuda.max_memory_allocated()) / (1024.0**3) if device.type == "cuda" else 0.0
    return ProfileResult(
        params_millions=count_params(model),
        flops_billions=float("nan"),
        seconds_per_volume=elapsed,
        peak_memory_gb=peak_gb,
    )
