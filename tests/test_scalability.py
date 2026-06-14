"""
Tests for the large-topology scalability harness.

These guard the two things that make the zero-shot GNN test honest: (1) the
generated graphs carry the edge attributes the env/agent read, and (2)
ScalableNetworkEnv correctly drops the parent's hard-coded 10-node assumption so
src/dst are sampled from the real node set of an arbitrary graph.
"""

import networkx as nx

from experiments.scalability import ScalableNetworkEnv, make_ba_graph
from network_rl.agents.gnn_agent import GNNAgent


def test_ba_graph_is_connected_with_env_attributes():
    G = make_ba_graph(30, seed=7)
    assert G.number_of_nodes() == 30
    assert nx.is_connected(G)
    for (u, v) in G.edges():
        e = G.edges[u, v]
        for key in ("bandwidth", "base_delay", "congestion", "loss_rate", "active"):
            assert key in e


def test_scalable_env_respects_real_node_count():
    G = make_ba_graph(25, seed=3)
    env = ScalableNetworkEnv(G, failure_prob=0.0)
    for _ in range(50):
        env.reset()
        assert 0 <= env._current_node < 25
        assert 0 <= env._dest_node < 25
        assert env._current_node != env._dest_node


def test_gnn_routes_on_larger_graph_zero_shot():
    # Architecture is size-independent: a GNN built on a 40-node graph must be
    # able to take a step (no shape error), which is the whole transfer premise.
    G = make_ba_graph(40, seed=11)
    env = ScalableNetworkEnv(G, failure_prob=0.0)
    env.reset()
    agent = GNNAgent(graph=env.G, edge_list=env.edge_list)
    agent.epsilon = 0.0
    a_idx, nbrs = agent.select_action(env, env._current_node, env._dest_node)
    assert 0 <= a_idx < len(nbrs)
    _, _, term, trunc, info = env.step(a_idx)
    assert "path" in info
