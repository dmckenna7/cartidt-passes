"""Typed batch and prediction contracts (Ref: Sec. III.A — problem formulation).

`Batch` carries one volumetric MRI sample with paired segmentation and grading labels.
`Prediction` carries the four outputs of a single forward pass: dense logits,
predicted grade, aleatoric uncertainty, and epistemic uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class SubjectMeta:
    subject_id: str
    timepoint: int
    cohort: str
    voxel_spacing_mm: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Batch:
    volume: torch.Tensor
    segmentation: torch.Tensor
    grade: torch.Tensor
    meta: tuple[SubjectMeta, ...]

    def to(self, device: torch.device | str) -> Batch:
        return Batch(
            volume=self.volume.to(device, non_blocking=True),
            segmentation=self.segmentation.to(device, non_blocking=True),
            grade=self.grade.to(device, non_blocking=True),
            meta=self.meta,
        )

    @property
    def batch_size(self) -> int:
        return int(self.volume.shape[0])

    @property
    def num_classes(self) -> int:
        return int(self.segmentation.max().item()) + 1


@dataclass(frozen=True, slots=True)
class Prediction:
    seg_logits: torch.Tensor
    alpha: torch.Tensor
    grade: torch.Tensor
    u_aleatoric: torch.Tensor
    u_epistemic: torch.Tensor

    @property
    def alpha_zero(self) -> torch.Tensor:
        return self.alpha.sum(dim=-1)
