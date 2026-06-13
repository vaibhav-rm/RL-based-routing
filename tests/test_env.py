"""
Environment contract tests for NetworkRoutingEnv.

These pin down the observation layout (the 70-dim shape whose mismatch with a
stale 19-dim adapter once crashed the hardware path) and the core M/M/1 delay
and link-dynamics invariants the agents and baselines rely on.
"""

import numpy as np
import pytest

from network_rl.env.network_env import (
    NetworkRoutingEnv,
    mm1_delay,
    OBS_DIM,
    NUM_NODES,
    NUM_EDGES,
)


def test_obs_dim_matches_layout():
    # 4 features per edge + 2 node scalars. This is the contract the MLP agents
    # are built against; the eval crash of 2026-06-10 was a 19-vs-70 mismatch.
    assert OBS_DIM == NUM_EDGES * 4 + 2 == 70


def test_reset_returns_well_formed_obs():
    env = NetworkRoutingEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
    assert isinstance(info, dict)


def test_reset_is_seed_deterministic():
    obs_a, _ = NetworkRoutingEnv().reset(seed=42)
    obs_b, _ = NetworkRoutingEnv().reset(seed=42)
    np.testing.assert_array_equal(obs_a, obs_b)


def test_observation_space_contains_obs():
    env = NetworkRoutingEnv()
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs)


def test_step_signature_and_bounds():
    env = NetworkRoutingEnv()
    env.reset(seed=3)
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "path" in info and "dest_node" in info


def test_episode_truncates_within_max_steps():
    env = NetworkRoutingEnv(failure_prob=0.0)
    env.reset(seed=7)
    truncated = False
    for _ in range(60):  # MAX_STEPS is 50
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break
    assert terminated or truncated


def test_reaching_destination_terminates_with_bonus():
    # Walk the sorted-neighbour graph deterministically until we land on dest.
    env = NetworkRoutingEnv(failure_prob=0.0)
    env.reset(seed=11)
    env._current_node = 6
    env._dest_node = 9
    env._visited = {6}
    # 6's sorted neighbours include 9; pick its index.
    nbrs = env._get_sorted_neighbors(6)
    action = nbrs.index(9)
    _, reward, terminated, _, _ = env.step(action)
    assert terminated
    assert reward > 0  # +20 arrival bonus dominates the small delay penalty


def test_out_of_range_action_is_penalised_not_crashing():
    env = NetworkRoutingEnv()
    env.reset(seed=2)
    huge = env.action_space.n  # one past the last valid index
    obs, reward, terminated, truncated, _ = env.step(huge)
    assert reward < 0
    assert not terminated


# ── M/M/1 delay model ────────────────────────────────────────────────────────

def test_mm1_monotonic_in_utilisation():
    low = mm1_delay(5.0, 100.0, 0.1)
    high = mm1_delay(5.0, 100.0, 0.9)
    assert high > low


def test_mm1_floor_is_base_delay():
    # At ρ→0 the sojourn time collapses toward the propagation delay.
    assert mm1_delay(5.0, 100.0, 0.0) >= 5.0


def test_mm1_caps_utilisation():
    # ρ is clipped at 0.95 so delay stays finite even at "100%" load.
    assert np.isfinite(mm1_delay(5.0, 100.0, 1.0))


# ── link dynamics ────────────────────────────────────────────────────────────

def test_failed_link_marks_congestion_and_loss_max():
    env = NetworkRoutingEnv()
    env.reset(seed=5)
    u, v = env.edge_list[0]
    env.G.edges[u, v]["active"] = False
    env._update_link_conditions()
    assert env.G.edges[u, v]["congestion"] == 1.0
    assert env.G.edges[u, v]["loss_rate"] == 1.0


def test_active_subgraph_excludes_failed_links():
    env = NetworkRoutingEnv()
    env.reset(seed=5)
    u, v = env.edge_list[0]
    env.G.edges[u, v]["active"] = False
    H = env.get_active_subgraph()
    assert not H.has_edge(u, v)


def test_curriculum_setter_updates_difficulty():
    env = NetworkRoutingEnv()
    env.set_curriculum(failure_prob=0.2, mean_load=0.8)
    assert env.failure_prob == 0.2
    assert env.mean_load == 0.8


def test_topology_constants():
    env = NetworkRoutingEnv()
    assert env.G.number_of_nodes() == NUM_NODES
    assert env.G.number_of_edges() == NUM_EDGES
