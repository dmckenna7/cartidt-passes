from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from cartidt.backend.objective import CartiDTObjective
from cartidt.backend.optimizer import build_optimizer
from cartidt.backend.schedule import warmup_cosine_schedule
from cartidt.link import build_model


@pytest.mark.slow
def test_four_step_integration_lowers_loss_and_grows_alpha_zero(tiny_config) -> None:
    cfg = OmegaConf.create(OmegaConf.to_container(tiny_config, resolve=True))
    torch.manual_seed(0)
    model = build_model(cfg)
    objective = CartiDTObjective(
        num_classes=int(cfg.model.num_seg_classes),
        num_grades=int(cfg.model.num_grades),
        mu=float(cfg.train.mu),
        anneal_epochs=int(cfg.train.anneal_epochs),
    )
    opt = build_optimizer(model, lr=float(cfg.train.base_lr), weight_decay=float(cfg.train.weight_decay))
    sched = warmup_cosine_schedule(
        opt,
        base_lr=float(cfg.train.base_lr),
        min_lr=float(cfg.train.min_lr),
        warmup_epochs=int(cfg.train.warmup_epochs),
        total_epochs=int(cfg.train.epochs),
        steps_per_epoch=1,
    )
    d, h, w = (int(x) for x in cfg.data.target_shape)
    volume = torch.randn(1, 1, d, h, w)
    seg_target = torch.zeros(1, d, h, w, dtype=torch.long)
    seg_target[..., : w // 2] = 1
    grade = torch.tensor([[1] + [-1] * (NUM_NODES_TEST - 1)])
    losses: list[float] = []
    alpha_zero_history: list[float] = []
    for step in range(4):
        seg_logits, alpha = model(volume)
        outputs = objective(seg_logits, seg_target, alpha, grade, epoch=float(step))
        opt.zero_grad(set_to_none=True)
        outputs.total.backward()
        opt.step()
        sched.step()
        losses.append(float(outputs.total.detach().item()))
        with torch.no_grad():
            alpha_zero_history.append(float(alpha.sum(dim=-1).mean().item()))
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    assert alpha_zero_history[-1] >= alpha_zero_history[0] - 1e-3


NUM_NODES_TEST = 14
