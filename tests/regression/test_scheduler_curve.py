from __future__ import annotations

import math

import torch.nn as nn

from cartidt.backend.optimizer import build_optimizer
from cartidt.backend.schedule import warmup_cosine_schedule


def _walk_lr(epochs: int, warmup: int, steps_per_epoch: int) -> list[float]:
    layer = nn.Linear(4, 4)
    opt = build_optimizer(layer, lr=1.0e-3, weight_decay=0.0)
    sched = warmup_cosine_schedule(
        opt,
        base_lr=1.0e-3,
        min_lr=1.0e-6,
        warmup_epochs=warmup,
        total_epochs=epochs,
        steps_per_epoch=steps_per_epoch,
    )
    history: list[float] = []
    for _ in range(epochs * steps_per_epoch):
        opt.step()
        sched.step()
        history.append(opt.param_groups[0]["lr"])
    return history


def test_scheduler_warmup_hits_base_lr() -> None:
    history = _walk_lr(epochs=10, warmup=5, steps_per_epoch=4)
    assert math.isclose(history[5 * 4 - 1], 1.0e-3, rel_tol=0.1)


def test_scheduler_floor_approached_at_end() -> None:
    history = _walk_lr(epochs=10, warmup=2, steps_per_epoch=4)
    assert history[-1] < 1.0e-4
