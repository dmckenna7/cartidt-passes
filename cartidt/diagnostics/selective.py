"""Selective-prediction sweep over abstention thresholds (Ref: Sec. V.C, Fig. 4 lower-left)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class RiskCoveragePoint:
    coverage: float
    fpr: float
    accuracy: float


def sweep_risk_coverage(
    uncertainty: torch.Tensor,
    target_any_damage: torch.Tensor,
    prediction_any_damage: torch.Tensor,
    fractions: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50),
) -> list[RiskCoveragePoint]:
    unc = uncertainty.detach().cpu().numpy().astype(np.float64).ravel()
    target = target_any_damage.detach().cpu().numpy().astype(np.int64).ravel()
    pred = prediction_any_damage.detach().cpu().numpy().astype(np.int64).ravel()
    points: list[RiskCoveragePoint] = []
    for frac in fractions:
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"fraction {frac} outside (0, 1]")
        threshold = np.quantile(unc, 1.0 - frac)
        keep = unc <= threshold
        if not np.any(keep):
            points.append(RiskCoveragePoint(coverage=float(frac), fpr=float("nan"), accuracy=float("nan")))
            continue
        kept_pred = pred[keep]
        kept_target = target[keep]
        neg = kept_target == 0
        fpr = float(np.mean(kept_pred[neg] == 1)) if np.any(neg) else float("nan")
        acc = float(np.mean(kept_pred == kept_target))
        points.append(RiskCoveragePoint(coverage=float(np.mean(keep)), fpr=fpr, accuracy=acc))
    return points
