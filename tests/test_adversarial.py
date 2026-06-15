"""
Tests for the adversarial-failure harness.

These guard the integrity of the worst-case robustness study: the targeted
attack must actually hit the most critical links (not random ones), both
strategies must respect the same failure budget, and the injector must leave the
graph in exactly the requested up/down state.
"""

import networkx as nx

from network_rl.env.network_env import NetworkRoutingEnv
from experiments.adversarial import (
    adversarial_failures,
    random_failures,
    _set_failures,
)
import numpy as np


def test_adversarial_picks_highest_betweenness_links():
    env = NetworkRoutingEnv()
    k = 3
    chosen = adversarial_failures(env, k)
    assert len(chosen) == k
    bc = nx.edge_betweenness_centrality(env.G)
    chosen_scores = sorted(bc[e] for e in chosen)
    other_scores = [bc[e] for e in env.edge_list if e not in chosen]
    # Every targeted link is at least as central as every spared link.
    assert min(chosen_scores) >= max(other_scores)


def test_random_and_adversarial_share_budget():
    env = NetworkRoutingEnv()
    rng = np.random.default_rng(0)
    k = 5
    assert len(random_failures(env, k, rng)) == k
    assert len(adversarial_failures(env, k)) == k


def test_set_failures_disables_exactly_the_chosen_links():
    env = NetworkRoutingEnv()
    fail = adversarial_failures(env, 4)
    _set_failures(env, fail)
    for (u, v) in env.edge_list:
        expected_active = (u, v) not in fail
        assert env.G.edges[u, v]["active"] == expected_active
