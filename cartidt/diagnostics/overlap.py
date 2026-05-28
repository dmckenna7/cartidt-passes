"""Per-compartment segmentation metrics (Ref: Sec. V.D, Table III).

`dice_similarity_coefficient` returns the standard DSC; `average_symmetric_surface_distance`
returns ASSD in millimetres given the volume's voxel spacing.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


def _to_numpy_bool(volume: torch.Tensor) -> np.ndarray:
    return volume.detach().cpu().numpy().astype(bool)


def dice_similarity_coefficient(prediction: torch.Tensor, target: torch.Tensor, label: int) -> float:
    pred_mask = prediction == label
    target_mask = target == label
    if not (pred_mask.any() or target_mask.any()):
        return 1.0
    intersection = float((pred_mask & target_mask).sum().item())
    denom = float(pred_mask.sum().item() + target_mask.sum().item())
    return (2.0 * intersection) / max(denom, 1.0)


def average_symmetric_surface_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    label: int,
    voxel_spacing_mm: tuple[float, float, float],
) -> float:
    pred = _to_numpy_bool(prediction == label)
    targ = _to_numpy_bool(target == label)
    if not pred.any() or not targ.any():
        return float("nan")
    pred_surface = pred & ~binary_erosion(pred)
    targ_surface = targ & ~binary_erosion(targ)
    dt_targ = distance_transform_edt(~targ_surface, sampling=voxel_spacing_mm)
    dt_pred = distance_transform_edt(~pred_surface, sampling=voxel_spacing_mm)
    p2t = dt_targ[pred_surface].mean() if pred_surface.any() else float("nan")
    t2p = dt_pred[targ_surface].mean() if targ_surface.any() else float("nan")
    return float(np.nanmean([p2t, t2p]))
