from __future__ import annotations

import torch

from cartidt.passes.twin.region_features import compartment_masked_gap


def test_node_features_output_dimensions() -> None:
    features = torch.randn(2, 32, 8, 16, 16)
    segmentation = torch.zeros(2, 8, 16, 16, dtype=torch.long)
    segmentation[:, :, :4, :4] = 1
    segmentation[:, :, 4:, 4:] = 2
    feats = compartment_masked_gap(features, segmentation, {0: 1, 1: 2, 2: 0}, num_nodes=14)
    assert feats.shape == (2, 14, 32)


def test_node_features_zero_for_absent_label() -> None:
    features = torch.randn(1, 8, 4, 4, 4)
    segmentation = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    feats = compartment_masked_gap(features, segmentation, {3: 7}, num_nodes=14)
    assert torch.allclose(feats[0, 3], torch.zeros(8))
