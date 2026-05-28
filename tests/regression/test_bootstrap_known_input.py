from __future__ import annotations

import numpy as np

from cartidt.diagnostics.significance import holm_bonferroni, paired_bootstrap_ci


def test_paired_bootstrap_ci_brackets_known_mean_diff() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.5, scale=0.1, size=(256,))
    b = rng.normal(loc=0.0, scale=0.1, size=(256,))

    def mean_diff(x: np.ndarray, y: np.ndarray) -> float:
        return float(x.mean() - y.mean())

    ci = paired_bootstrap_ci(a, b, statistic=mean_diff, n_resamples=2_000, seed=42)
    assert ci.lower < 0.5 < ci.upper


def test_holm_bonferroni_orders_correctly() -> None:
    p_values = [0.01, 0.04, 0.03, 0.5]
    adjusted = holm_bonferroni(p_values)
    assert all(0.0 <= p <= 1.0 for p in adjusted)
    assert adjusted[0] <= adjusted[3]
