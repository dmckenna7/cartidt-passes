"""GraphSAGE-style aggregation over the 14-node biomechanical graph (Ref: Sec. III.D-4, Eq. (4)).

The layer update is

    h_k^{(l+1)} = σ( W_self^{(l)} h_k^{(l)} + W_neigh^{(l)} · MEAN( {w_kj h_j^{(l)}}_{j ∈ N(k)} ) )

with GELU, L = 3 layers, hidden dimension d_g = 256. The neighbour mean is
scaled by the frozen edge weight `w_kj` (Sec. III.D-2 final ¶).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _SAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.w_self = nn.Linear(in_dim, out_dim, bias=True)
        self.w_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.act = nn.GELU()

    def forward(self, node_feats: torch.Tensor, weighted_adj: torch.Tensor) -> torch.Tensor:
        degree = weighted_adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        neigh_mean = torch.einsum("ij,bjc->bic", weighted_adj, node_feats) / degree
        return self.act(self.w_self(node_feats) + self.w_neigh(neigh_mean))


class BiomechSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, num_layers: int = 3) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList([_SAGELayer(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.out_dim = hidden_dim

    def forward(self, node_feats: torch.Tensor, weighted_adj: torch.Tensor) -> torch.Tensor:
        x = node_feats
        for layer in self.layers:
            x = layer(x, weighted_adj)
        return x
