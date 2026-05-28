from __future__ import annotations

import torch

from cartidt.passes.evidence.dirichlet_head import EvidenceHead


def test_evidence_head_emits_k_concentrations() -> None:
    head = EvidenceHead(in_dim=64, num_grades=4, hidden_dim=32)
    inputs = torch.randn(2, 14, 64)
    alpha = head(inputs)
    assert alpha.shape == (2, 14, 4)
    assert torch.all(alpha >= 1.0)
