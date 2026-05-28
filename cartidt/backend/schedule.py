"""Linear-warmup → cosine-decay LR schedule (Ref: Sec. V.C).

Linearly warm `lr` from 0 to `base_lr` over the first `warmup_epochs` epochs,
then cosine-decay to `min_lr` over the remaining `total_epochs - warmup_epochs`.
The schedule is parameterised in *epochs* so it composes cleanly with the
gradient-accumulated outer loop.
"""

from __future__ import annotations

import math

import torch


def warmup_cosine_schedule(
    optimiser: torch.optim.Optimizer,
    base_lr: float = 2.0e-4,
    min_lr: float = 1.0e-6,
    warmup_epochs: int = 5,
    total_epochs: int = 120,
    steps_per_epoch: int | None = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_epochs <= warmup_epochs:
        raise ValueError("total_epochs must exceed warmup_epochs")
    unit = float(steps_per_epoch) if steps_per_epoch else 1.0

    def fn(step: int) -> float:
        epoch = step / unit
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1e-6)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1e-6)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        ratio = min_lr / base_lr
        return ratio + (1.0 - ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=fn)
