from __future__ import annotations

import torch

from cartidt.frontend.schema import Batch, Prediction, SubjectMeta


def test_batch_basic_shape() -> None:
    meta = SubjectMeta(subject_id="9000099", timepoint=0, cohort="OAI", voxel_spacing_mm=(0.365, 0.365, 0.7))
    b = Batch(
        volume=torch.zeros(2, 1, 16, 32, 32),
        segmentation=torch.zeros(2, 16, 32, 32, dtype=torch.long),
        grade=torch.zeros(2, 6, dtype=torch.long),
        meta=(meta, meta),
    )
    assert b.batch_size == 2
    assert b.num_classes == 1


def test_prediction_alpha_zero() -> None:
    pred = Prediction(
        seg_logits=torch.zeros(1, 7, 4, 4, 4),
        alpha=torch.ones(1, 6, 4),
        grade=torch.zeros(1, 6, dtype=torch.long),
        u_aleatoric=torch.zeros(1, 6),
        u_epistemic=torch.zeros(1, 6),
    )
    assert torch.allclose(pred.alpha_zero, torch.full((1, 6), 4.0))
