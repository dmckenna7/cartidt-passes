from __future__ import annotations

import torch

from cartidt.passes.evidence.decompose import aleatoric, epistemic


def test_epistemic_equals_k_over_alpha_zero() -> None:
    alpha = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    u_epi = epistemic(alpha)
    assert torch.allclose(u_epi, torch.tensor([1.0]))


def test_aleatoric_zero_when_alpha_one_hot_strong() -> None:
    alpha = torch.tensor([[1e6, 1.0, 1.0, 1.0]])
    u_ale = aleatoric(alpha)
    assert float(u_ale.item()) < 1e-4


def test_epistemic_decreases_with_evidence() -> None:
    weak = epistemic(torch.tensor([[1.0, 1.0, 1.0, 1.0]]))
    strong = epistemic(torch.tensor([[10.0, 10.0, 10.0, 10.0]]))
    assert float(weak.item()) > float(strong.item())
