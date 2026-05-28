from __future__ import annotations

import torch

from cartidt.backend.reproducibility import set_seed
from cartidt.backend.segmentation_loss import SoftDiceCELoss


def test_set_seed_yields_identical_losses() -> None:
    set_seed(42)
    logits_a = torch.randn(1, 4, 2, 4, 4)
    target_a = torch.randint(low=0, high=4, size=(1, 2, 4, 4))
    loss_a = SoftDiceCELoss(num_classes=4)(logits_a, target_a)

    set_seed(42)
    logits_b = torch.randn(1, 4, 2, 4, 4)
    target_b = torch.randint(low=0, high=4, size=(1, 2, 4, 4))
    loss_b = SoftDiceCELoss(num_classes=4)(logits_b, target_b)

    assert torch.allclose(logits_a, logits_b)
    assert torch.equal(target_a, target_b)
    assert float(loss_a.item()) == float(loss_b.item())
