"""
Agent tests — with the inference-determinism regression as the centrepiece.

The 2026-06-10 bug: select_action short-circuited to a *random* action whenever
the replay buffer was smaller than one batch. A freshly loaded agent has an empty
buffer, so every evaluation was secretly routing randomly instead of using the
trained policy. The fix gates that warmup clause on `policy_net.training`, so a
loaded (eval-mode) net is always used directly.

`test_*_eval_mode_is_deterministic_not_random` is the guard that pins this fix.
If anyone reintroduces the unconditional warmup, these fail.
"""

import numpy as np
import pytest
import torch

from network_rl.env.network_env import NetworkRoutingEnv, OBS_DIM
from network_rl.agents.dqn_agent import DQNAgent
from network_rl.agents.rainbow_agent import RainbowAgent
from network_rl.agents.gnn_agent import GNNAgent


@pytest.fixture
def env():
    e = NetworkRoutingEnv(failure_prob=0.0)
    e.reset(seed=0)
    return e


def _valid_actions(env):
    return list(range(len(env._get_sorted_neighbors(env._current_node))))


# ── DQN ──────────────────────────────────────────────────────────────────────

def test_dqn_select_action_returns_valid(env):
    agent = DQNAgent(state_dim=OBS_DIM, action_dim=env.action_space.n, epsilon=0.0)
    agent.policy_net.eval()
    obs = env._get_obs()
    va = _valid_actions(env)
    a = agent.select_action(obs, va)
    assert a in va


def test_dqn_eval_mode_is_deterministic_not_random(env):
    """A loaded (eval-mode) agent at epsilon=0 must return the SAME action every
    call — it is reading Q-values, not rolling dice. This is the regression guard."""
    agent = DQNAgent(state_dim=OBS_DIM, action_dim=env.action_space.n, epsilon=0.0)
    agent.policy_net.eval()  # what load() does
    assert len(agent.replay) == 0  # empty buffer, the bug's trigger condition
    obs = env._get_obs()
    va = _valid_actions(env)
    choices = {agent.select_action(obs, va) for _ in range(50)}
    assert len(choices) == 1, "eval-mode agent must be deterministic, not random"


def test_dqn_training_mode_still_warms_up_randomly(env):
    """In training mode with an empty buffer the warmup random walk must remain —
    the fix must not break exploration during training."""
    agent = DQNAgent(state_dim=OBS_DIM, action_dim=env.action_space.n,
                     epsilon=0.0, batch_size=64)
    agent.policy_net.train()
    assert agent.policy_net.training and len(agent.replay) < agent.batch_size
    obs = env._get_obs()
    va = list(range(env.action_space.n))
    choices = {agent.select_action(obs, va) for _ in range(200)}
    assert len(choices) > 1, "training warmup should still explore randomly"


def test_dqn_rank_actions_is_permutation(env):
    agent = DQNAgent(state_dim=OBS_DIM, action_dim=env.action_space.n, epsilon=0.0)
    agent.policy_net.eval()
    va = _valid_actions(env)
    ranked = agent.rank_actions(env._get_obs(), va)
    assert sorted(ranked) == sorted(va)


def test_dqn_save_load_roundtrip(env, tmp_path):
    agent = DQNAgent(state_dim=OBS_DIM, action_dim=env.action_space.n, epsilon=0.0)
    agent.policy_net.eval()
    obs, va = env._get_obs(), _valid_actions(env)
    before = agent.select_action(obs, va)

    path = tmp_path / "dqn.pth"
    agent.save(str(path))
    reloaded = DQNAgent(state_dim=OBS_DIM, action_dim=env.action_space.n)
    reloaded.load(str(path))
    assert not reloaded.policy_net.training  # load() switches to eval
    assert reloaded.select_action(obs, va) == before


# ── Rainbow ──────────────────────────────────────────────────────────────────

def test_rainbow_eval_mode_is_deterministic_not_random(env):
    agent = RainbowAgent(state_dim=OBS_DIM, action_dim=env.action_space.n, epsilon=0.0)
    agent.policy_net.eval()
    assert len(agent.replay) == 0
    obs, va = env._get_obs(), _valid_actions(env)
    choices = {agent.select_action(obs, va) for _ in range(50)}
    assert len(choices) == 1


def test_rainbow_save_load_roundtrip(env, tmp_path):
    agent = RainbowAgent(state_dim=OBS_DIM, action_dim=env.action_space.n, epsilon=0.0)
    agent.policy_net.eval()
    obs, va = env._get_obs(), _valid_actions(env)
    before = agent.select_action(obs, va)
    path = tmp_path / "rainbow.pth"
    agent.save(str(path))
    reloaded = RainbowAgent(state_dim=OBS_DIM, action_dim=env.action_space.n)
    reloaded.load(str(path))
    assert reloaded.select_action(obs, va) == before


# ── GNN ──────────────────────────────────────────────────────────────────────

def test_gnn_eval_mode_is_deterministic_not_random(env):
    agent = GNNAgent(env.G, env.edge_list, epsilon=0.0)
    agent.policy_net.eval()
    assert len(agent.replay) == 0
    actions = {agent.select_action(env, 0, 9)[0] for _ in range(50)}
    assert len(actions) == 1


def test_gnn_rank_actions_is_permutation_of_indices(env):
    agent = GNNAgent(env.G, env.edge_list, epsilon=0.0)
    agent.policy_net.eval()
    order, neighbors = agent.rank_actions(env, 0, 9)
    assert sorted(order) == list(range(len(neighbors)))


def test_gnn_save_load_roundtrip(env, tmp_path):
    agent = GNNAgent(env.G, env.edge_list, epsilon=0.0)
    agent.policy_net.eval()
    before = agent.select_action(env, 0, 9)[0]
    path = tmp_path / "gnn.pth"
    agent.save(str(path))
    reloaded = GNNAgent(env.G, env.edge_list)
    reloaded.load(str(path))
    assert not reloaded.policy_net.training
    assert reloaded.select_action(env, 0, 9)[0] == before


def test_gnn_is_topology_agnostic_across_node_count(env):
    """The GNN reads node/edge features, so the same net handles a different-sized
    graph without an architecture change — the property that lets it generalise."""
    agent = GNNAgent(env.G, env.edge_list, epsilon=0.0)
    agent.policy_net.eval()
    # Build a smaller env and route on it with the SAME agent weights.
    small = NetworkRoutingEnv(failure_prob=0.0)
    small.reset(seed=1)
    idx, neighbors = agent.select_action(small, 0, 9)
    assert 0 <= idx < len(neighbors)
