from __future__ import annotations

import torch

from cartidt.passes.twin.contact_weights import estimate_edge_weights


def test_edge_weights_are_zero_when_seg_is_empty() -> None:
    seg = torch.zeros(8, 16, 16, dtype=torch.long)
    weights = estimate_edge_weights(seg)
    assert torch.all(weights == 0.0)


def test_edge_weights_are_deterministic() -> None:
    torch.manual_seed(0)
    seg = torch.zeros(8, 16, 16, dtype=torch.long)
    seg[:, :8, :8] = 0
    seg[:, 8:, :8] = 3
    seg[:, :8, 8:] = 4
    seg[:, 8:, 8:] = 1
    w1 = estimate_edge_weights(seg)
    w2 = estimate_edge_weights(seg)
    assert torch.allclose(w1, w2)
