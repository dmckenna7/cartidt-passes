"""Paired bootstrap CI + Holm-Bonferroni correction (Ref: Sec. VI ¶1).

`paired_bootstrap_ci` follows the seeded paired-sample resampling scheme used in
the manuscript (`n = 10 000`). `holm_bonferroni` returns adjusted p-values for a
family of comparisons against a common baseline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    point_estimate: float
    lower: float
    upper: float


def paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapCI:
    if a.shape != b.shape:
        raise ValueError("a and b must have identical shape")
    rng = np.random.default_rng(seed)
    point = float(statistic(a, b))
    n = a.shape[0]
    samples = np.empty(n_resamples, dtype=np.float64)
    for k in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples[k] = statistic(a[idx], b[idx])
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1.0 - alpha / 2))
    return BootstrapCI(point_estimate=point, lower=lo, upper=hi)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
    m = len(p_values)
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (original_index, p) in enumerate(indexed):
        scale = m - rank
        scaled = min(1.0, scale * p)
        running_max = max(running_max, scaled)
        adjusted[original_index] = running_max
    return adjusted
