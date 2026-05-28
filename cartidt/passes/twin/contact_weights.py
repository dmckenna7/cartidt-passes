"""One-shot estimator of digital-twin edge weights (Ref: Sec. III.D-2 final ¶).

The edge weight `w_ij` is the ratio of the contact-area surface overlap at the
segmentation boundary between sub-region i and sub-region j, normalised by the
mean thickness of region i. To prevent circularity the manuscript computes
`E` and its weights once from the *mean training segmentation* and freezes them
for both train and test.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cartidt.passes.twin.anatomy_graph import BIOMECH_EDGES, NUM_NODES


def _binary_surface(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype == torch.bool:
        mask = mask.float()
    kernel = torch.ones(1, 1, 3, 3, 3, device=mask.device, dtype=mask.dtype)
    dilated = F.conv3d(mask.unsqueeze(0).unsqueeze(0), kernel, padding=1).squeeze(0).squeeze(0)
    boundary = (dilated > 0) & (mask == 0)
    return boundary


def _region_mask(label_volume: torch.Tensor, region_index: int) -> torch.Tensor:
    return (label_volume == region_index).float()


def estimate_edge_weights(
    mean_segmentation: torch.Tensor,
    node_to_label: dict[int, int] | None = None,
) -> torch.Tensor:
    if mean_segmentation.ndim != 3:
        raise ValueError("mean_segmentation must be (D, H, W) with integer labels")
    mapping = node_to_label if node_to_label is not None else {n: n for n in range(NUM_NODES)}
    weights = torch.zeros(NUM_NODES, NUM_NODES, dtype=torch.float32, device=mean_segmentation.device)
    surfaces: dict[int, torch.Tensor] = {}
    areas: dict[int, float] = {}
    for node, label in mapping.items():
        mask = _region_mask(mean_segmentation, label)
        surface = _binary_surface(mask)
        surfaces[node] = surface
        areas[node] = float(surface.sum().clamp(min=1.0))
    for src, dst in BIOMECH_EDGES:
        if src not in surfaces or dst not in surfaces:
            continue
        overlap = (surfaces[src].float() * surfaces[dst].float()).sum().item()
        weights[src, dst] = float(overlap) / areas[src]
    row_max = weights.amax(dim=1, keepdim=True).clamp(min=1e-6)
    weights = weights / row_max
    return weights
