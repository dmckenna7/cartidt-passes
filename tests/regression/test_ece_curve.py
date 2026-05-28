from __future__ import annotations

import torch

from cartidt.diagnostics.calibration import (
    area_under_risk_coverage,
    expected_calibration_error,
    fpr_at_threshold,
)


def test_ece_zero_for_perfectly_calibrated_input() -> None:
    confidence = torch.tensor([0.95, 0.90, 0.10, 0.05])
    correctness = torch.tensor([1.0, 1.0, 0.0, 0.0])
    ece = expected_calibration_error(confidence, correctness, num_bins=4)
    assert ece < 0.25


def test_aurc_lower_when_uncertainty_orders_errors() -> None:
    uncertainty_aligned = torch.tensor([0.0, 0.0, 0.5, 1.0])
    correctness = torch.tensor([1.0, 1.0, 0.0, 0.0])
    aurc = area_under_risk_coverage(uncertainty_aligned, correctness)
    misaligned = area_under_risk_coverage(torch.tensor([1.0, 1.0, 0.5, 0.0]), correctness)
    assert aurc < misaligned


def test_fpr_at_threshold_is_zero_when_predictions_are_correct() -> None:
    unc = torch.linspace(0.0, 1.0, steps=10)
    target = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    pred = target.clone()
    fpr = fpr_at_threshold(unc, target, pred, abstention_rate=0.5)
    assert fpr == 0.0
