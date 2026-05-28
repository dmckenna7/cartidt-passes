"""Joint CartiDT objective L = L_seg + μ · L_EDL (Ref: Sec. III.F, Eq. (10)).

The class glues together the soft-Dice + CE segmentation loss and the EDL
grading loss, with the manuscript's μ = 0.5 mixing constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from cartidt.backend.segmentation_loss import SoftDiceCELoss
from cartidt.passes.evidence.edl import EDLLoss


@dataclass
class JointLossOutputs:
    total: torch.Tensor
    seg: torch.Tensor
    edl: torch.Tensor


class CartiDTObjective(nn.Module):
    def __init__(
        self,
        num_classes: int = 7,
        num_grades: int = 4,
        mu: float = 0.5,
        anneal_epochs: int = 10,
    ) -> None:
        super().__init__()
        if mu < 0.0:
            raise ValueError("mu must be non-negative")
        self.mu = mu
        self.seg_loss = SoftDiceCELoss(num_classes=num_classes)
        self.edl_loss = EDLLoss(num_grades=num_grades, anneal_epochs=anneal_epochs)

    def forward(
        self,
        seg_logits: torch.Tensor,
        seg_target: torch.Tensor,
        alpha: torch.Tensor,
        grade_target: torch.Tensor,
        epoch: float,
    ) -> JointLossOutputs:
        l_seg = self.seg_loss(seg_logits, seg_target)
        l_edl = self.edl_loss(alpha, grade_target, epoch=epoch)
        return JointLossOutputs(total=l_seg + self.mu * l_edl, seg=l_seg, edl=l_edl)
