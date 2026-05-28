"""Epoch loop with gradient accumulation (Ref: Sec. V.C — batch 4 × accum 2 × 4 GPUs).

The loop is intentionally framework-light: it owns optimiser stepping,
gradient accumulation, scheduler stepping, and per-step logging, but defers
checkpointing to `cartidt.backend.checkpointing`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cartidt.backend.objective import CartiDTObjective, JointLossOutputs
from cartidt.frontend.schema import Batch

_LOG = logging.getLogger(__name__)


@dataclass
class StepRecord:
    epoch: float
    step: int
    total_loss: float
    seg_loss: float
    edl_loss: float
    lr: float


class TrainLoop:
    def __init__(
        self,
        model: nn.Module,
        objective: CartiDTObjective,
        optimiser: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_accum: int = 2,
        max_grad_norm: float | None = 1.0,
        amp_dtype: torch.dtype | None = None,
    ) -> None:
        if grad_accum <= 0:
            raise ValueError("grad_accum must be positive")
        self.model = model
        self.objective = objective
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.grad_accum = grad_accum
        self.max_grad_norm = max_grad_norm
        self.amp_dtype = amp_dtype
        self.global_step = 0

    def fit_epoch(self, loader: DataLoader[Batch], epoch: float) -> list[StepRecord]:
        self.model.train()
        records: list[StepRecord] = []
        self.optimiser.zero_grad(set_to_none=True)
        for micro_idx, batch in enumerate(loader):
            outputs = self._micro_step(batch, epoch)
            (outputs.total / self.grad_accum).backward()
            if (micro_idx + 1) % self.grad_accum == 0:
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        max_norm=self.max_grad_norm,
                    )
                self.optimiser.step()
                self.scheduler.step()
                self.optimiser.zero_grad(set_to_none=True)
                self.global_step += 1
            lr = self.optimiser.param_groups[0]["lr"]
            records.append(
                StepRecord(
                    epoch=epoch,
                    step=self.global_step,
                    total_loss=float(outputs.total.detach()),
                    seg_loss=float(outputs.seg.detach()),
                    edl_loss=float(outputs.edl.detach()),
                    lr=lr,
                )
            )
        return records

    def _micro_step(self, batch: Batch, epoch: float) -> JointLossOutputs:
        device = next(self.model.parameters()).device
        batch = batch.to(device)
        if self.amp_dtype is not None and torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                seg_logits, alpha = self.model(batch.volume, batch.segmentation)
                outputs = self.objective(seg_logits, batch.segmentation, alpha, batch.grade, epoch=epoch)
        else:
            seg_logits, alpha = self.model(batch.volume, batch.segmentation)
            outputs = self.objective(seg_logits, batch.segmentation, alpha, batch.grade, epoch=epoch)
        return outputs
