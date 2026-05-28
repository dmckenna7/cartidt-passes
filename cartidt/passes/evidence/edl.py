"""Evidential deep-learning loss (Ref: Sec. III.E-2, Eq. (9), ref. [10]).

    L_EDL = Σ_k [ ψ(α_{0,k}) − ψ(α_{k, y_k*}) + λ_t · KL( Dir(α̃_k) ‖ Dir(1) ) ]

with `α̃_{k, j}` defined as the truncated evidence vector that strips α at the
true class index (Eq. (9)). The KL term uses the closed form for Dirichlet
distributions. `λ_t = min(1, t / T_anneal)` ramps the KL contribution over
`T_anneal = 10` epochs (Sec. III.E-2 final ¶).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    k = alpha.shape[-1]
    alpha_zero = alpha.sum(dim=-1)
    term1 = torch.lgamma(alpha_zero) - torch.lgamma(torch.tensor(float(k), device=alpha.device))
    term2 = -torch.lgamma(alpha).sum(dim=-1)
    term3 = ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(alpha_zero.unsqueeze(-1)))).sum(dim=-1)
    return term1 + term2 + term3


class EDLLoss(nn.Module):
    def __init__(self, num_grades: int = 4, anneal_epochs: int = 10) -> None:
        super().__init__()
        if anneal_epochs <= 0:
            raise ValueError("anneal_epochs must be positive")
        self.num_grades = num_grades
        self.anneal_epochs = anneal_epochs

    def forward(self, alpha: torch.Tensor, target: torch.Tensor, epoch: float) -> torch.Tensor:
        if alpha.ndim < 2 or alpha.shape[-1] != self.num_grades:
            raise ValueError(f"alpha must have grade axis of size {self.num_grades}")
        if target.shape != alpha.shape[:-1]:
            raise ValueError("target shape must match alpha shape without the grade axis")
        valid_mask = target >= 0
        if valid_mask.sum() == 0:
            return alpha.sum() * 0.0
        alpha_zero = alpha.sum(dim=-1)
        target_clamped = target.clamp(min=0)
        alpha_target = alpha.gather(-1, target_clamped.unsqueeze(-1)).squeeze(-1)
        evidence_term = torch.digamma(alpha_zero) - torch.digamma(alpha_target)
        eye = torch.eye(self.num_grades, device=alpha.device)
        truth_onehot = eye[target_clamped]
        alpha_tilde = alpha * (1.0 - truth_onehot) + truth_onehot
        lam = min(1.0, float(epoch) / float(self.anneal_epochs))
        kl_term = lam * _dirichlet_kl_to_uniform(alpha_tilde)
        per_element = evidence_term + kl_term
        return per_element[valid_mask].mean()
