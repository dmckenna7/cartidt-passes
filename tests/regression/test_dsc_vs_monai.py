from __future__ import annotations

import pytest
import torch

from cartidt.diagnostics.overlap import dice_similarity_coefficient

monai = pytest.importorskip("monai")
from monai.metrics import DiceMetric  # noqa: E402


def test_dsc_matches_monai_for_random_volume() -> None:
    torch.manual_seed(1)
    pred = torch.randint(low=0, high=3, size=(1, 4, 8, 8))
    target = torch.randint(low=0, high=3, size=(1, 4, 8, 8))
    for label in (1, 2):
        ours = dice_similarity_coefficient(pred[0], target[0], label=label)
        metric = DiceMetric(include_background=True, reduction="none")
        pred_onehot = (pred == label).float().unsqueeze(1)
        target_onehot = (target == label).float().unsqueeze(1)
        metric(pred_onehot, target_onehot)
        reference = float(metric.aggregate().item())
        assert ours == pytest.approx(reference, abs=1e-4)
