"""Biomechanically-informed digital-twin graph topology (Ref: Sec. III.D-2).

The graph has 14 nodes: six cartilage / meniscus compartments plus the eight
sub-regions required for finer biomechanical coupling (medial / lateral
weight-bearing femoral and tibial regions, trochlear, medial / lateral facets,
medial meniscus body, lateral meniscus body). Edges encode the contact-area
couplings the manuscript enumerates in Sec. III.D-2.
"""

from __future__ import annotations

import torch

NODE_NAMES: tuple[str, ...] = (
    "fem_med_wb",
    "fem_lat_wb",
    "fem_trochlea",
    "tib_med_wb",
    "tib_lat_wb",
    "pat_med_facet",
    "pat_lat_facet",
    "men_med_body",
    "men_lat_body",
    "men_med_ant_horn",
    "men_med_post_horn",
    "men_lat_ant_horn",
    "men_lat_post_horn",
    "pat_apex",
)

NUM_NODES: int = len(NODE_NAMES)
NUM_COMPARTMENT_NODES: int = 6

BIOMECH_EDGES: tuple[tuple[int, int], ...] = (
    (0, 3),
    (3, 0),
    (1, 4),
    (4, 1),
    (2, 5),
    (5, 2),
    (2, 6),
    (6, 2),
    (0, 7),
    (7, 0),
    (1, 8),
    (8, 1),
    (3, 7),
    (7, 3),
    (4, 8),
    (8, 4),
    (7, 9),
    (7, 10),
    (8, 11),
    (8, 12),
    (5, 13),
    (6, 13),
    (13, 5),
    (13, 6),
    (0, 1),
    (3, 4),
)


def build_adjacency(
    num_nodes: int = NUM_NODES, edges: tuple[tuple[int, int], ...] = BIOMECH_EDGES
) -> torch.Tensor:
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
    for src, dst in edges:
        if not (0 <= src < num_nodes) or not (0 <= dst < num_nodes):
            raise IndexError(f"edge ({src}->{dst}) out of range for {num_nodes} nodes")
        adj[src, dst] = True
    return adj
