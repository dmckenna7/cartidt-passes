"""Grade-detection metrics (Ref: Sec. V.D, Table IV).

`auroc_any_damage` is the binary AUROC for "any damage" (grade ≥ 1).
`quadratic_weighted_kappa` is Cohen's κ_w on the four-grade ordinal labels.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, roc_auc_score


def auroc_any_damage(alpha: torch.Tensor, target: torch.Tensor) -> float:
    if alpha.ndim != target.ndim + 1:
        raise ValueError("alpha must have one more dim than target")
    alpha_zero = alpha.sum(dim=-1)
    prob_any = 1.0 - alpha[..., 0] / alpha_zero
    valid = target >= 0
    if valid.sum() < 2:
        return float("nan")
    y_true = (target[valid] >= 1).long().cpu().numpy()
    y_score = prob_any[valid].detach().cpu().numpy()
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def quadratic_weighted_kappa(alpha: torch.Tensor, target: torch.Tensor) -> float:
    valid = target >= 0
    if valid.sum() < 2:
        return float("nan")
    predicted = alpha.argmax(dim=-1)
    y_true = target[valid].cpu().numpy()
    y_pred = predicted[valid].cpu().numpy()
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
