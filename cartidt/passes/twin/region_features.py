"""Compartment-masked global average pooling (Ref: Sec. III.D-3, Eq. (3)).

For each subject we use the decoder's predicted segmentation `Ŷ` to select the
voxels belonging to sub-region `k`, then average the corresponding backbone
feature column vectors over those voxels.
"""

from __future__ import annotations

import torch


def compartment_masked_gap(
    features: torch.Tensor,
    segmentation: torch.Tensor,
    node_to_label: dict[int, int],
    num_nodes: int,
) -> torch.Tensor:
    if features.ndim != 5:
        raise ValueError("features must be (B, C, D, H, W)")
    if segmentation.ndim != 4:
        raise ValueError("segmentation must be (B, D, H, W)")
    b, c, _, _, _ = features.shape
    node_feats = features.new_zeros(b, num_nodes, c)
    for node, label in node_to_label.items():
        mask = (segmentation == label).unsqueeze(1).float()
        denom = mask.sum(dim=(2, 3, 4)).clamp(min=1.0)
        num = (features * mask).sum(dim=(2, 3, 4))
        node_feats[:, node, :] = num / denom
    return node_feats
