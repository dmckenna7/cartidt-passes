"""Soft-Dice + cross-entropy segmentation loss (Ref: Sec. III.F)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceCELoss(nn.Module):
    def __init__(
        self, num_classes: int, dice_weight: float = 1.0, ce_weight: float = 1.0, eps: float = 1e-6
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be ≥ 2")
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] != self.num_classes:
            raise ValueError(f"expected {self.num_classes} channels, got {logits.shape[1]}")
        ce = F.cross_entropy(logits, target, reduction="mean")
        probs = logits.softmax(dim=1)
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 4, 1, 2, 3).float()
        intersect = (probs * target_onehot).sum(dim=(0, 2, 3, 4))
        denom = probs.sum(dim=(0, 2, 3, 4)) + target_onehot.sum(dim=(0, 2, 3, 4))
        dice = (2.0 * intersect + self.eps) / (denom + self.eps)
        return self.ce_weight * ce + self.dice_weight * (1.0 - dice.mean())
