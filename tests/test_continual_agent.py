"""
Tests for the continual-learning (EWC) DQN agent and the base-agent hook it uses.

These guard the mechanics behind the catastrophic-forgetting study: the EWC
penalty is zero until a task is consolidated and zero again at the snapshot
point, it grows as weights move away, the Fisher estimate is normalised, and the
base DQNAgent's _extra_loss hook is a no-op so vanilla training is unchanged.
"""

import numpy as np
import pytest
import torch

from network_rl.agents.dqn_agent import DQNAgent
from network_rl.agents.continual_dqn_agent import ContinualDQNAgent


STATE_DIM, ACTION_DIM = 70, 4


def _fill_buffer(agent, n=200, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        s  = rng.random(STATE_DIM).astype(np.float32)
        ns = rng.random(STATE_DIM).astype(np.float32)
        agent.store(s, int(rng.integers(ACTION_DIM)), float(rng.normal()), ns, 0.0)


def test_base_agent_extra_loss_is_noop():
    # The hook added to DQNAgent must leave vanilla agents untouched.
    assert DQNAgent(STATE_DIM, ACTION_DIM)._extra_loss() == 0.0


def test_no_penalty_before_consolidation():
    agent = ContinualDQNAgent(STATE_DIM, ACTION_DIM, ewc_lambda=100.0)
    assert agent._extra_loss() == 0.0          # no consolidated task yet


def test_fisher_is_normalised_and_per_parameter():
    agent = ContinualDQNAgent(STATE_DIM, ACTION_DIM, ewc_lambda=100.0)
    _fill_buffer(agent)
    fisher = agent.estimate_fisher(n_batches=8)
    names = {n for n, _ in agent.policy_net.named_parameters()}
    assert set(fisher) == names
    assert all(torch.all(f >= 0) for f in fisher.values())   # squared grads ≥ 0
    # Normalised so the mean importance is ~1.
    total = sum(float(f.sum()) for f in fisher.values())
    count = sum(f.numel() for f in fisher.values())
    assert total / count == pytest.approx(1.0, rel=1e-4)


def test_penalty_zero_at_snapshot_then_grows_when_weights_move():
    agent = ContinualDQNAgent(STATE_DIM, ACTION_DIM, ewc_lambda=100.0)
    _fill_buffer(agent)
    agent.consolidate()
    # At the snapshot θ == θ*, so the quadratic penalty is exactly zero.
    assert float(agent._extra_loss().detach()) == pytest.approx(0.0, abs=1e-6)
    # Perturb the weights → penalty becomes strictly positive.
    with torch.no_grad():
        for p in agent.policy_net.parameters():
            p.add_(0.1)
    assert float(agent._extra_loss().detach()) > 0.0
