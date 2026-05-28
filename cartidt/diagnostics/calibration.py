"""Calibration metrics (Ref: Sec. V.D, Table V).

`expected_calibration_error` follows the equally-spaced-bin definition (Naeini
et al.). `area_under_risk_coverage` integrates the risk-coverage curve over the
sorted-by-confidence list. `fpr_at_threshold` returns the false-positive rate at
the abstention threshold τ that retains `1 − τ` fraction of the test set.
"""

from __future__ import annotations

import numpy as np
import torch


def expected_calibration_error(
    confidence: torch.Tensor, correctness: torch.Tensor, num_bins: int = 15
) -> float:
    if confidence.shape != correctness.shape:
        raise ValueError("confidence and correctness must have identical shape")
    conf = confidence.detach().cpu().numpy().astype(np.float64).ravel()
    correct = correctness.detach().cpu().numpy().astype(np.float64).ravel()
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_id = np.clip(np.digitize(conf, bins) - 1, 0, num_bins - 1)
    n = float(len(conf))
    ece = 0.0
    for b in range(num_bins):
        mask = bin_id == b
        if not np.any(mask):
            continue
        bin_conf = conf[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def area_under_risk_coverage(uncertainty: torch.Tensor, correctness: torch.Tensor) -> float:
    if uncertainty.shape != correctness.shape:
        raise ValueError("shape mismatch")
    unc = uncertainty.detach().cpu().numpy().astype(np.float64).ravel()
    correct = correctness.detach().cpu().numpy().astype(np.float64).ravel()
    order = np.argsort(unc)
    correct_sorted = correct[order]
    n = len(correct_sorted)
    risks: list[float] = []
    coverages: list[float] = []
    cum_correct = 0.0
    for i in range(n):
        cum_correct += correct_sorted[i]
        coverage = (i + 1) / n
        risk = 1.0 - cum_correct / (i + 1)
        risks.append(risk)
        coverages.append(coverage)
    return float(np.trapz(risks, coverages))


def fpr_at_threshold(
    uncertainty: torch.Tensor,
    target_any_damage: torch.Tensor,
    prediction_any_damage: torch.Tensor,
    abstention_rate: float = 0.15,
) -> float:
    if uncertainty.shape != target_any_damage.shape or uncertainty.shape != prediction_any_damage.shape:
        raise ValueError("shape mismatch")
    if not (0.0 <= abstention_rate < 1.0):
        raise ValueError("abstention_rate must be in [0, 1)")
    unc = uncertainty.detach().cpu().numpy().astype(np.float64).ravel()
    target = target_any_damage.detach().cpu().numpy().astype(np.int64).ravel()
    pred = prediction_any_damage.detach().cpu().numpy().astype(np.int64).ravel()
    threshold = np.quantile(unc, 1.0 - abstention_rate)
    keep = unc <= threshold
    if not np.any(keep):
        return float("nan")
    neg = keep & (target == 0)
    if not np.any(neg):
        return float("nan")
    return float(np.mean(pred[neg] == 1))
