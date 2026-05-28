from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from cartidt.diagnostics.ordinal import auroc_any_damage


def test_auroc_matches_sklearn_baseline() -> None:
    rng = np.random.default_rng(0)
    n = 64
    target = torch.from_numpy(rng.integers(low=0, high=4, size=(n,)))
    alpha = torch.from_numpy(rng.uniform(1.0, 5.0, size=(n, 4)).astype(np.float32))
    ours = auroc_any_damage(alpha, target)
    y_true = (target >= 1).numpy().astype(int)
    y_score = 1.0 - (alpha[:, 0] / alpha.sum(dim=-1)).numpy()
    expected = roc_auc_score(y_true, y_score)
    assert ours == pytest.approx(float(expected), rel=1e-5)
