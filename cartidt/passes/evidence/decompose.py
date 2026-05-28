"""Closed-form decomposition of Dirichlet uncertainty (Ref: Sec. III.E-1, Eqs. (7)-(8)).

For α_0 = Σ_k α_k and π̂_{k,j} = α_{k,j} / α_{0,k},

    u_ale[k] = (K / (α_{0,k} (α_{0,k} + 1))) · Σ_j π̂_{k,j} (1 − π̂_{k,j})
    u_epi[k] = K / α_{0,k}
"""

from __future__ import annotations

import torch


def _check_alpha(alpha: torch.Tensor) -> None:
    if alpha.ndim < 1:
        raise ValueError("alpha must have at least one dim (the grade axis)")


def aleatoric(alpha: torch.Tensor) -> torch.Tensor:
    _check_alpha(alpha)
    k = alpha.shape[-1]
    alpha_zero = alpha.sum(dim=-1)
    expected = alpha / alpha_zero.unsqueeze(-1)
    var = (expected * (1.0 - expected)).sum(dim=-1)
    return (k / (alpha_zero * (alpha_zero + 1.0))) * var


def epistemic(alpha: torch.Tensor) -> torch.Tensor:
    _check_alpha(alpha)
    k = alpha.shape[-1]
    return k / alpha.sum(dim=-1)
