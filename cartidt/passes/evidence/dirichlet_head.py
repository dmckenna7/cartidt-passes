"""Evidential head — node embeddings → Dirichlet α (Ref: Sec. III.E-1, Eq. (6)).

A two-layer MLP maps the L-th node embedding `h_k^{(L)}` to the concentration
parameters `α_k = Softplus(MLP(h_k^{(L)})) + 1`. The additive `+ 1` keeps every
`α_k ≥ 1` and pins the maximum-uncertainty scale at `α_0 = K` (Sec. III.E-1).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EvidenceHead(nn.Module):
    def __init__(self, in_dim: int, num_grades: int = 4, hidden_dim: int = 128) -> None:
        super().__init__()
        if num_grades < 2:
            raise ValueError("num_grades must be ≥ 2")
        self.num_grades = num_grades
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_grades),
        )
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        logits = self.mlp(node_embeddings)
        alpha = self.softplus(logits) + 1.0
        return alpha
