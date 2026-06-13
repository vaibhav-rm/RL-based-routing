"""
Tests for the classical routing baselines (Dijkstra, ECMP, random).

They verify that each baseline returns a valid neighbour as the next hop,
respects failed links, and reports unreachability honestly rather than
raising or returning a bogus node.
"""

import networkx as nx
import pytest

from network_rl.env.network_env import NetworkRoutingEnv
from network_rl.baselines.dijkstra import (
    dijkstra_next_hop,
    dijkstra_full_path,
    build_weighted_graph,
)
from network_rl.baselines.ecmp import (
    ecmp_next_hop,
    ecmp_full_path,
    ecmp_load_balanced_path,
)
from network_rl.baselines.random_routing import random_next_hop, random_full_path


@pytest.fixture
def env():
    e = NetworkRoutingEnv(failure_prob=0.0)
    e.reset(seed=0)
    return e


# ── Dijkstra ─────────────────────────────────────────────────────────────────

def test_dijkstra_next_hop_is_a_neighbour(env):
    g = build_weighted_graph(env)
    hop = dijkstra_next_hop(g, 0, 9)
    assert hop in g.neighbors(0)


def test_dijkstra_full_path_is_connected_and_ends_at_dest(env):
    g = build_weighted_graph(env)
    path = dijkstra_full_path(g, 0, 9)
    assert path[0] == 0 and path[-1] == 9
    for u, v in zip(path[:-1], path[1:]):
        assert g.has_edge(u, v)


def test_dijkstra_same_node_returns_self(env):
    g = build_weighted_graph(env)
    assert dijkstra_next_hop(g, 4, 4) == 4


def test_dijkstra_unreachable_returns_none():
    g = nx.Graph()
    g.add_nodes_from([0, 1])  # no edge → disconnected
    assert dijkstra_next_hop(g, 0, 1) is None
    assert dijkstra_full_path(g, 0, 1) == []


def test_build_weighted_graph_drops_failed_links(env):
    u, v = env.edge_list[0]
    env.G.edges[u, v]["active"] = False
    g = build_weighted_graph(env)
    assert not g.has_edge(u, v)


# ── ECMP ─────────────────────────────────────────────────────────────────────

def test_ecmp_next_hop_is_a_neighbour(env):
    g = build_weighted_graph(env)
    hop = ecmp_next_hop(g, 0, 9, flow_id=3)
    assert hop in g.neighbors(0)


def test_ecmp_is_deterministic_per_flow(env):
    g = build_weighted_graph(env)
    a = ecmp_full_path(g, 0, 9, flow_id=7)
    b = ecmp_full_path(g, 0, 9, flow_id=7)
    assert a == b  # same flow → same path (TCP in-order requirement)


def test_ecmp_load_balanced_picks_a_valid_shortest_path(env):
    g = build_weighted_graph(env)
    path = ecmp_load_balanced_path(g, 0, 9)
    assert path and path[0] == 0 and path[-1] == 9


def test_ecmp_unreachable_returns_none_and_empty():
    g = nx.Graph()
    g.add_nodes_from([0, 1])
    assert ecmp_next_hop(g, 0, 1) is None
    assert ecmp_full_path(g, 0, 1) == []


# ── Random ───────────────────────────────────────────────────────────────────

def test_random_next_hop_is_a_neighbour(env):
    hop = random_next_hop(env.G, 0, 9, visited={0})
    assert hop in env.G.neighbors(0)


def test_random_prefers_unvisited(env):
    # With every neighbour but one marked visited, the unvisited one must win.
    nbrs = list(env.G.neighbors(0))
    target = nbrs[0]
    visited = set(nbrs[1:]) | {0}
    for _ in range(20):
        assert random_next_hop(env.G, 0, 9, visited=visited) == target


def test_random_full_path_starts_at_source(env):
    path = random_full_path(env.G, 0, 9)
    assert path[0] == 0


def test_random_dead_end_returns_none():
    g = nx.Graph()
    g.add_node(0)
    g.add_node(1)
    assert random_next_hop(g, 0, 1) is None
