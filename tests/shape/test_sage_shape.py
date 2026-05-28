from __future__ import annotations

import torch

from cartidt.passes.twin.graph_sage import BiomechSAGE


def test_sage_preserves_node_count() -> None:
    sage = BiomechSAGE(in_dim=32, hidden_dim=64, num_layers=3)
    node_feats = torch.randn(2, 14, 32)
    weighted_adj = torch.rand(14, 14)
    out = sage(node_feats, weighted_adj)
    assert out.shape == (2, 14, 64)
