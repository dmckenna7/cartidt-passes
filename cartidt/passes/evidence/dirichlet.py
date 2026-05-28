"""Dirichlet PDF in log-space (Ref: Sec. III.E-1, Eq. (5)).

p(π|α) = Γ(α_0) / Π_k Γ(α_k) · Π_k π_k^{α_k − 1}

The log form is used elsewhere for stable EDL loss computation.
"""

from __future__ import annotations

import torch


def dirichlet_log_pdf(pi: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    if pi.shape[-1] != alpha.shape[-1]:
        raise ValueError("pi and alpha must share their last dimension")
    alpha_zero = alpha.sum(dim=-1, keepdim=True)
    log_norm = torch.lgamma(alpha_zero.squeeze(-1)) - torch.lgamma(alpha).sum(dim=-1)
    log_kernel = ((alpha - 1.0) * torch.log(pi.clamp(min=1e-12))).sum(dim=-1)
    return log_norm + log_kernel
