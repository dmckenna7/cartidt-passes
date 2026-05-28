"""AdamW optimiser wiring (Ref: Sec. V.C — β1=0.9, β2=0.999, wd=1e-2)."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    lr: float = 2.0e-4,
    weight_decay: float = 1.0e-2,
    betas: tuple[float, float] = (0.9, 0.999),
) -> torch.optim.AdamW:
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no parameters require gradients — refusing to build optimiser")
    return torch.optim.AdamW(
        _decay_groups(trainable, weight_decay),
        lr=lr,
        betas=betas,
    )


def _decay_groups(params: Iterable[nn.Parameter], weight_decay: float) -> list[dict[str, object]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for param in params:
        if param.ndim <= 1:
            no_decay.append(param)
        else:
            decay.append(param)
    groups: list[dict[str, object]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups
