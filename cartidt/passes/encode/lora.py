"""Depth-wise LoRA injection on Q and V projections (Ref: Sec. III.C-1, Eq. (1)).

For every transformer block in a frozen 24-layer ViT-L, the adapted layer weight is

    W_adapted = W_frozen + (alpha / r) * B @ A

with `r = 16`, `alpha = 32`, and only the Q and V projections wrapped.
For 2 × 24 projections × 2 × 16 × 1024 = 1,572,864 trainable parameters
(matches the 1.57 M figure in Sec. III.C-1).
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn


class LoRAProjection(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, base: nn.Linear, rank: int = 16, alpha: int = 32
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scaling = float(alpha) / float(rank)
        self.a = nn.Parameter(torch.zeros(rank, in_features))
        self.b = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=5**0.5)
        nn.init.zeros_(self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = torch.nn.functional.linear(torch.nn.functional.linear(x, self.a), self.b) * self.scaling
        return self.base(x) + delta


def inject_lora_qv(
    model: nn.Module, q_names: Iterable[str], v_names: Iterable[str], rank: int = 16, alpha: int = 32
) -> int:
    targets = list(q_names) + list(v_names)
    if len(targets) == 0:
        raise ValueError("no Q/V target modules supplied")
    n_replaced = 0
    for target in targets:
        parent, attr = _resolve(model, target)
        base = getattr(parent, attr)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"module {target} is not nn.Linear")
        wrapped = LoRAProjection(base.in_features, base.out_features, base, rank=rank, alpha=alpha)
        setattr(parent, attr, wrapped)
        n_replaced += 1
    for name, param in model.named_parameters():
        if ".a" not in name and ".b" not in name:
            param.requires_grad_(False)
    return n_replaced


def _resolve(root: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parts = dotted.split(".")
    parent: nn.Module = root
    for token in parts[:-1]:
        parent = getattr(parent, token)
    return parent, parts[-1]
