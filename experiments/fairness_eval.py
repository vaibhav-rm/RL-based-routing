"""
Per-flow fairness evaluation (addresses the QoS-fairness gap).

Aggregate PDR/throughput hide whether a routing policy quietly starves a subset
of source–destination flows while keeping the mean high. This script measures
the *distribution* of service across flows:

  • For every ordered (src, dst) pair we route `trials` packets and record the
    per-flow delivery ratio.
  • Jain's fairness index over that per-flow vector tells us how evenly service
    is spread (1.0 = every flow treated identically; → 1/F = a few flows hogged
    by the policy while others starve).

We report fairness alongside mean PDR so the trade-off is explicit: a policy can
have high mean PDR yet low fairness, or vice-versa. Nothing here is fabricated —
the numbers come from the same trained models used by evaluate.py.

Usage:
    python experiments/fairness_eval.py [--trials 8] [--failure-rate 20]
"""

import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from network_rl.env.network_env import NetworkRoutingEnv, NUM_NODES
from network_rl.analysis.statistics import jain_fairness_index
from network_rl.baselines.dijkstra import build_weighted_graph, dijkstra_full_path
from network_rl.baselines.ecmp import ecmp_load_balanced_path
from network_rl.baselines.random_routing import random_full_path
from evaluate import inject_failures, valid_actions, load_agents

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def _all_flows():
    """Every ordered source≠dest pair — the set of flows we test fairness over."""
    return [(s, d) for s in range(NUM_NODES) for d in range(NUM_NODES) if s != d]


def _route_rl_flow(agent, env, src, dst, is_gnn, max_steps=50):
    """Drive one packet from src to dst with a trained RL agent. Returns delivered?"""
    env.reset()
    env._current_node, env._dest_node = src, dst
    env._visited, env._path = {src}, [src]
    obs = env._get_obs()
    for _ in range(max_steps):
        if is_gnn:
            a_idx, _ = agent.select_action(env, env._current_node, dst)
            action = a_idx
        else:
            action = agent.select_action(obs, valid_actions(env))
        obs, _, term, trunc, info = env.step(action)
        if term:
            return info["path"][-1] == dst
        if trunc:
            break
    return env._current_node == dst


def _per_flow_pdr(route_fn, flows, trials):
    """Delivery ratio for each flow → vector we feed to Jain's index."""
    vec = []
    for (s, d) in flows:
        delivered = sum(route_fn(s, d) for _ in range(trials))
        vec.append(delivered / trials)
    return vec


def evaluate_fairness(failure_rate: float, trials: int):
    env = NetworkRoutingEnv(use_mm1=True)
    env.reset()
    agents = load_agents(env)
    flows = _all_flows()
    rng = np.random.default_rng(42)

    out = {}

    def record(name, vec):
        out[name] = {
            "mean_pdr": float(np.mean(vec)),
            "min_pdr":  float(np.min(vec)),
            "jain_fairness": jain_fairness_index(vec),
            "n_flows": len(vec),
        }
        print(f"  {name:<12} mean_PDR={out[name]['mean_pdr']:.3f}  "
              f"min_PDR={out[name]['min_pdr']:.3f}  "
              f"Jain={out[name]['jain_fairness']:.4f}")

    # RL agents + Q-routing
    for alg, (kind, agent, is_gnn, _is_rb) in agents.items():
        inject_failures(env, failure_rate, rng)
        if kind == "rl":
            agent.epsilon = 0.0
            vec = _per_flow_pdr(
                lambda s, d: _route_rl_flow(agent, env, s, d, is_gnn), flows, trials)
        else:  # Q-routing: fresh tables, warm up on this failure scenario first
            agent.reset_tables()
            for _ in range(100):
                env.reset()
                agent.route_episode(env, env._current_node, env._dest_node, max_hops=30)
            vec = _per_flow_pdr(
                lambda s, d: agent.route_episode(env, s, d, max_hops=30)[2], flows, trials)
        record(alg, vec)

    # Classical baselines (deterministic given the graph snapshot → trials collapse,
    # but we keep the same interface so fairness is computed identically).
    inject_failures(env, failure_rate, rng)
    graph = build_weighted_graph(env)
    record("Dijkstra", _per_flow_pdr(
        lambda s, d: bool(dijkstra_full_path(graph, s, d)) and
                     dijkstra_full_path(graph, s, d)[-1] == d, flows, 1))
    record("ECMP", _per_flow_pdr(
        lambda s, d: bool(ecmp_load_balanced_path(graph, s, d)) and
                     ecmp_load_balanced_path(graph, s, d)[-1] == d, flows, 1))
    record("Random", _per_flow_pdr(
        lambda s, d: bool(random_full_path(graph, s, d, max_hops=30)) and
                     random_full_path(graph, s, d, max_hops=30)[-1] == d, flows, trials))

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8,
                    help="packets routed per flow (stochastic policies)")
    ap.add_argument("--failure-rate", type=int, default=20,
                    help="percent of links failed during the test")
    args = ap.parse_args()

    print(f"\nPer-flow fairness @ {args.failure_rate}% link failure "
          f"({NUM_NODES*(NUM_NODES-1)} flows × {args.trials} trials)\n" + "-" * 60)
    out = evaluate_fairness(args.failure_rate / 100.0, args.trials)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "fairness.json")
    with open(path, "w") as f:
        json.dump({"failure_rate": args.failure_rate, "results": out}, f, indent=2)
    print(f"\nFairness results → {path}")

    best = max(out, key=lambda a: out[a]["jain_fairness"])
    print(f"Fairest policy: {best} (Jain={out[best]['jain_fairness']:.4f})")


if __name__ == "__main__":
    main()
