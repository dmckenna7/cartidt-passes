"""CLI entry — train (Ref: Sec. V.C)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from cartidt.backend.checkpointing import atomic_save
from cartidt.backend.distributed import init_distributed, is_main_process
from cartidt.backend.objective import CartiDTObjective
from cartidt.backend.optimizer import build_optimizer
from cartidt.backend.reproducibility import set_seed
from cartidt.backend.schedule import warmup_cosine_schedule
from cartidt.backend.trainer import TrainLoop
from cartidt.frontend.catalog import build_loaders
from cartidt.link import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
_LOG = logging.getLogger("cartidt.train")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CartiDT")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("./runs"))
    parser.add_argument("--steps", type=int, default=0, help="if > 0, exit after this many optimiser steps")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.overrides)))
    dist_info = init_distributed()
    set_seed(int(cfg.train.seed))
    device = torch.device(f"cuda:{dist_info.local_rank}" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    objective = CartiDTObjective(
        num_classes=int(cfg.model.num_seg_classes),
        num_grades=int(cfg.model.num_grades),
        mu=float(cfg.train.mu),
        anneal_epochs=int(cfg.train.anneal_epochs),
    ).to(device)
    loaders = build_loaders(cfg)
    optimiser = build_optimizer(
        model,
        lr=float(cfg.train.base_lr),
        weight_decay=float(cfg.train.weight_decay),
        betas=tuple(float(x) for x in cfg.train.betas),
    )
    scheduler = warmup_cosine_schedule(
        optimiser,
        base_lr=float(cfg.train.base_lr),
        min_lr=float(cfg.train.min_lr),
        warmup_epochs=int(cfg.train.warmup_epochs),
        total_epochs=int(cfg.train.epochs),
        steps_per_epoch=max(len(loaders.train), 1),
    )
    amp_dtype = None
    if str(cfg.train.precision).lower() == "bf16":
        amp_dtype = torch.bfloat16
    elif str(cfg.train.precision).lower() == "fp16":
        amp_dtype = torch.float16
    loop = TrainLoop(
        model=model,
        objective=objective,
        optimiser=optimiser,
        scheduler=scheduler,
        grad_accum=int(cfg.train.grad_accum),
        max_grad_norm=float(cfg.train.max_grad_norm),
        amp_dtype=amp_dtype,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for epoch in range(int(cfg.train.epochs)):
        records = loop.fit_epoch(loaders.train, epoch=float(epoch))
        if is_main_process():
            last = records[-1] if records else None
            _LOG.info(
                "epoch=%d step=%d L=%.4f L_seg=%.4f L_edl=%.4f lr=%.2e",
                epoch,
                loop.global_step,
                last.total_loss if last else float("nan"),
                last.seg_loss if last else float("nan"),
                last.edl_loss if last else float("nan"),
                last.lr if last else float("nan"),
            )
            atomic_save(
                {
                    "epoch": epoch,
                    "global_step": loop.global_step,
                    "model": model.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "seed": int(cfg.train.seed),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                args.out / "last.ckpt",
            )
        if 0 < args.steps <= loop.global_step:
            _LOG.info("stopping at user-requested step budget=%d", args.steps)
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
