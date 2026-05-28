from __future__ import annotations

import torch

from cartidt.passes.evidence.edl import EDLLoss


def test_edl_loss_drops_with_correct_evidence() -> None:
    loss_fn = EDLLoss(num_grades=4, anneal_epochs=10)
    target = torch.tensor([[1]])
    confused = torch.full((1, 1, 4), 2.0)
    confident = torch.tensor([[[1.05, 50.0, 1.05, 1.05]]])
    l_confused = loss_fn(confused, target, epoch=5.0)
    l_confident = loss_fn(confident, target, epoch=5.0)
    assert float(l_confident.item()) < float(l_confused.item())


def test_edl_loss_anneal_factor_grows() -> None:
    loss_fn = EDLLoss(num_grades=4, anneal_epochs=10)
    alpha = torch.full((1, 1, 4), 2.0)
    target = torch.tensor([[0]])
    early = loss_fn(alpha, target, epoch=0.0)
    late = loss_fn(alpha, target, epoch=10.0)
    assert float(late.item()) >= float(early.item())


def test_edl_loss_ignores_negative_targets() -> None:
    loss_fn = EDLLoss(num_grades=4, anneal_epochs=10)
    alpha = torch.full((1, 1, 4), 2.0)
    target = torch.tensor([[-1]])
    out = loss_fn(alpha, target, epoch=3.0)
    assert float(out.item()) == 0.0
