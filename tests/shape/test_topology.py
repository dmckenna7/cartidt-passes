from __future__ import annotations

from cartidt.passes.twin.anatomy_graph import BIOMECH_EDGES, NODE_NAMES, NUM_NODES, build_adjacency


def test_topology_node_count_matches_paper() -> None:
    assert NUM_NODES == 14
    assert len(NODE_NAMES) == 14


def test_topology_edge_count_matches_paper() -> None:
    assert len(BIOMECH_EDGES) == 26


def test_adjacency_is_square_and_boolean() -> None:
    adj = build_adjacency()
    assert adj.shape == (NUM_NODES, NUM_NODES)
    assert adj.dtype.is_floating_point is False
