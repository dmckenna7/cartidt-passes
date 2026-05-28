from __future__ import annotations

import pytest
import torch

from cartidt.backend.segmentation_loss import SoftDiceCELoss

monai = pytest.importorskip("monai")
from monai.losses import DiceCELoss  # noqa: E402


def test_soft_dice_ce_within_tolerance_of_monai() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 4, 8, 8)
    target = torch.randint(low=0, high=4, size=(1, 4, 8, 8))
    ours = SoftDiceCELoss(num_classes=4, dice_weight=1.0, ce_weight=1.0)(logits, target)
    ref = DiceCELoss(softmax=True, to_onehot_y=True, include_background=True)(logits, target.unsqueeze(1))
    assert float(ours.item()) == pytest.approx(float(ref.item()), rel=0.25)
