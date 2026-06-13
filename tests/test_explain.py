"""
Tests for the local Q-attribution explainability module.

Verify that an explanation faithfully reflects the agent's own ranking: the
chosen hop is the top-ranked candidate, every reachable neighbour appears
exactly once with its live edge features, and the rendered audit text mentions
the chosen hop.
"""

import pytest

from network_rl.env.network_env import NetworkRoutingEnv, OBS_DIM
from network_rl.agents.dqn_agent import DQNAgent
from network_rl.agents.gnn_agent import GNNAgent
from network_rl.analysis.explain import explain_decision, format_explanation


@pytest.fixture
def env():
    e = NetworkRoutingEnv(failure_prob=0.0)
    e.reset(seed=0)
    return e


def test_mlp_explanation_chosen_is_top_ranked(env):
    agent = DQNAgent(state_dim=OBS_DIM, action_dim=env.action_space.n, epsilon=0.0)
    agent.policy_net.eval()
    exp = explain_decision(agent, env, current=0, dest=9, kind="mlp")
    assert exp["chosen"] == exp["candidates"][0]["next_hop"]
    assert exp["candidates"][0]["rank"] == 0


def test_mlp_explanation_covers_every_neighbour_once(env):
    agent = DQNAgent(state_dim=OBS_DIM, action_dim=env.action_space.n, epsilon=0.0)
    agent.policy_net.eval()
    exp = explain_decision(agent, env, current=0, dest=9, kind="mlp")
    hops = [c["next_hop"] for c in exp["candidates"]]
    assert sorted(hops) == sorted(env.G.neighbors(0))


def test_explanation_matches_agent_ranking(env):
    agent = DQNAgent(state_dim=OBS_DIM, action_dim=env.action_space.n, epsilon=0.0)
    agent.policy_net.eval()
    exp = explain_decision(agent, env, current=0, dest=9, kind="mlp")
    chosen = exp["chosen"]
    # The agent, asked to pick directly, must agree with the explained choice.
    neighbors = sorted(env.G.neighbors(0))
    direct = agent.select_action(env._get_obs(), list(range(len(neighbors))))
    assert neighbors[direct] == chosen


def test_gnn_explanation_is_well_formed(env):
    agent = GNNAgent(env.G, env.edge_list, epsilon=0.0)
    agent.policy_net.eval()
    exp = explain_decision(agent, env, current=0, dest=9, kind="gnn")
    assert exp["chosen"] in env.G.neighbors(0)
    assert exp["candidates"][0]["rank"] == 0


def test_format_explanation_mentions_chosen_hop(env):
    agent = DQNAgent(state_dim=OBS_DIM, action_dim=env.action_space.n, epsilon=0.0)
    agent.policy_net.eval()
    exp = explain_decision(agent, env, current=0, dest=9, kind="mlp")
    text = format_explanation(exp)
    assert f"forward to {exp['chosen']}" in text


def test_dead_end_is_reported_not_crashed(env):
    agent = DQNAgent(state_dim=OBS_DIM, action_dim=env.action_space.n, epsilon=0.0)
    agent.policy_net.eval()
    # Isolate a node so it has no neighbours.
    env.G.remove_edges_from(list(env.G.edges(0)))
    exp = explain_decision(agent, env, current=0, dest=9, kind="mlp")
    assert exp["chosen"] is None
    assert "dead end" in exp["reason"]
