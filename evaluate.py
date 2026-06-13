"""
Research-quality evaluation: all 6 algorithms × 4 failure rates.

Algorithms compared:
  1. Rainbow DQN  (full: Dueling + PER + n-step + Double)
  2. Vanilla DQN
  3. GNN DQN
  4. Q-Routing    (Boyan & Littman 1994)
  5. Dijkstra     (OSPF-like shortest path)
  6. ECMP         (load-balanced multi-path)
  7. Random       (flooding baseline)

Metrics per (algorithm, failure_rate):
  • Average end-to-end delay (ms)
  • Packet Delivery Ratio (PDR)
  • Throughput proxy (deliveries / total steps)
  • Jitter (std-dev of delay)
  • Path length efficiency (actual hops / shortest possible hops)

Statistical analysis (per-algorithm, per-failure-rate):
  Run network_rl/analysis/statistics.py with multi-seed results for CIs.

Usage:
    python evaluate.py [--episodes 300] [--failure-rates 0 20 40 60]
"""

import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))

from network_rl.env.network_env  import NetworkRoutingEnv, OBS_DIM, NUM_NODES
from network_rl.agents.dqn_agent     import DQNAgent
from network_rl.agents.rainbow_agent  import RainbowAgent
from network_rl.agents.gnn_agent      import GNNAgent
from network_rl.agents.q_routing      import QRoutingAgent
from network_rl.baselines.dijkstra    import build_weighted_graph, dijkstra_full_path
from network_rl.baselines.ecmp        import ecmp_load_balanced_path
from network_rl.baselines.random_routing import random_full_path

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Metric helpers ─────────────────────────────────────────────────────────

def path_delay(env, path: list) -> float:
    """
    Delay along the active segments of the path.
    Failed-link traversals are skipped — they're captured by PDR/drop metrics.
    Unique-hop count is used to avoid double-counting loop traversals.
    """
    from network_rl.env.network_env import mm1_delay
    total = 0.0
    seen_edges = set()
    for u, v in zip(path[:-1], path[1:]):
        key = (min(u,v), max(u,v))
        if key in seen_edges:
            continue     # skip repeated loop traversals
        seen_edges.add(key)
        if env.G.has_edge(u, v):
            e = env.G.edges[u, v]
            if e["active"]:
                total += mm1_delay(e["base_delay"], e["bandwidth"], e["congestion"])
    return total


def inject_failures(env, failure_rate: float, rng: np.random.Generator):
    edges = list(env.edge_list)
    rng.shuffle(edges)
    n_fail = int(len(edges) * failure_rate)
    for (u, v) in edges[:n_fail]:
        env.G.edges[u, v]["active"] = False
    for (u, v) in edges[n_fail:]:
        env.G.edges[u, v]["active"] = True


def valid_actions(env):
    return list(range(len(env._get_sorted_neighbors(env._current_node))))


# ── Per-algorithm evaluation ───────────────────────────────────────────────

def eval_rl_agent(agent, env, num_ep: int, is_gnn=False, is_rainbow=False):
    delays, pdrs, steps_list = [], [], []
    for _ in range(num_ep):
        obs, _ = env.reset()
        src, dst = env._current_node, env._dest_node
        done, steps = False, 0
        while not done:
            if is_gnn:
                cur = env._current_node
                a_idx, _ = agent.select_action(env, cur, dst)
                action = a_idx
            else:
                action = agent.select_action(obs, valid_actions(env))
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
        path    = info["path"]
        reached = path[-1] == dst
        pdrs.append(float(reached))
        steps_list.append(steps)
        if reached:
            delays.append(path_delay(env, path))
    return (
        np.mean(delays) if delays else 999.0,
        np.mean(pdrs),
        np.std(delays) if len(delays) > 1 else 0.0,
        np.sum(pdrs) / max(sum(steps_list), 1),
    )


def eval_dijkstra(env, num_ep: int):
    delays, pdrs, steps_list = [], [], []
    for _ in range(num_ep):
        env.reset()
        src, dst = env._current_node, env._dest_node
        graph  = build_weighted_graph(env)
        path   = dijkstra_full_path(graph, src, dst)
        reached = bool(path) and path[-1] == dst
        pdrs.append(float(reached))
        steps_list.append(len(path))
        if reached:
            delays.append(path_delay(env, path))
    return (
        np.mean(delays) if delays else 999.0,
        np.mean(pdrs),
        np.std(delays) if len(delays) > 1 else 0.0,
        np.sum(pdrs) / max(sum(steps_list), 1),
    )


def eval_ecmp(env, num_ep: int):
    delays, pdrs, steps_list = [], [], []
    for i in range(num_ep):
        env.reset()
        src, dst = env._current_node, env._dest_node
        graph    = build_weighted_graph(env)
        path     = ecmp_load_balanced_path(graph, src, dst)
        reached  = bool(path) and path[-1] == dst
        pdrs.append(float(reached))
        steps_list.append(len(path))
        if reached:
            delays.append(path_delay(env, path))
    return (
        np.mean(delays) if delays else 999.0,
        np.mean(pdrs),
        np.std(delays) if len(delays) > 1 else 0.0,
        np.sum(pdrs) / max(sum(steps_list), 1),
    )


def eval_qrouting(agent: QRoutingAgent, env, num_ep: int):
    # Warm-up: let Q-tables adapt to the current failure scenario before measuring.
    # This matches the online-routing methodology of Boyan & Littman (1994).
    WARMUP = min(100, num_ep // 3)
    for _ in range(WARMUP):
        env.reset()
        src, dst = env._current_node, env._dest_node
        agent.route_episode(env, src, dst, max_hops=30)

    delays, pdrs, steps_list = [], [], []
    for _ in range(num_ep):
        env.reset()
        src, dst = env._current_node, env._dest_node
        path, delay, reached = agent.route_episode(env, src, dst, max_hops=30)
        pdrs.append(float(reached))
        steps_list.append(len(path))
        if reached:
            delays.append(delay)
    return (
        np.mean(delays) if delays else 999.0,
        np.mean(pdrs),
        np.std(delays) if len(delays) > 1 else 0.0,
        np.sum(pdrs) / max(sum(steps_list), 1),
    )


def eval_random(env, num_ep: int):
    delays, pdrs, steps_list = [], [], []
    for _ in range(num_ep):
        env.reset()
        src, dst = env._current_node, env._dest_node
        graph    = build_weighted_graph(env)
        path     = random_full_path(graph, src, dst, max_hops=30)
        reached  = bool(path) and path[-1] == dst
        pdrs.append(float(reached))
        steps_list.append(len(path))
        if reached:
            delays.append(path_delay(env, path))
    return (
        np.mean(delays) if delays else 999.0,
        np.mean(pdrs),
        np.std(delays) if len(delays) > 1 else 0.0,
        np.sum(pdrs) / max(sum(steps_list), 1),
    )


# ── Load agents ────────────────────────────────────────────────────────────

def load_agents(env):
    agents = {}
    # Rainbow
    rainbow_path = os.path.join(MODELS_DIR, "rainbow_seed0.pth")
    if not os.path.exists(rainbow_path):
        rainbow_path = os.path.join(MODELS_DIR, "rainbow_trained.pth")
    if os.path.exists(rainbow_path):
        a = RainbowAgent(state_dim=OBS_DIM, action_dim=env.action_space.n)
        a.load(rainbow_path); a.epsilon = 0.0
        agents["Rainbow"] = ("rl", a, False, True)
    else:
        print("[!] Rainbow model not found — skipping")

    # DQN
    dqn_path = os.path.join(MODELS_DIR, "dqn_seed0.pth")
    if not os.path.exists(dqn_path):
        dqn_path = os.path.join(MODELS_DIR, "dqn_trained.pth")
    if os.path.exists(dqn_path):
        a = DQNAgent(state_dim=OBS_DIM, action_dim=env.action_space.n)
        a.load(dqn_path); a.epsilon = 0.0
        agents["DQN"] = ("rl", a, False, False)
    else:
        print("[!] DQN model not found — skipping")

    # GNN
    gnn_path = os.path.join(MODELS_DIR, "gnn_seed0.pth")
    if not os.path.exists(gnn_path):
        gnn_path = os.path.join(MODELS_DIR, "gnn_trained.pth")
    if os.path.exists(gnn_path):
        a = GNNAgent(graph=env.G, edge_list=env.edge_list)
        a.load(gnn_path); a.epsilon = 0.0
        agents["GNN-DQN"] = ("rl", a, True, False)
    else:
        print("[!] GNN model not found — skipping")

    # Q-Routing — fresh tables per evaluation scenario.
    # Q-routing (Boyan & Littman, 1994) is an online algorithm designed to
    # adapt continuously as packets flow. Pre-trained tables from a different
    # traffic regime do not transfer cleanly; fresh initialisation with online
    # adaptation is the canonical evaluation methodology from the original paper.
    qr_agent = QRoutingAgent(env.G, lr=0.1)
    agents["Q-Routing"] = ("qr", qr_agent, False, False)

    return agents


# ── Main evaluation loop ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",       type=int, nargs="+", default=None)
    parser.add_argument("--failure-rates",  type=int, nargs="+", default=[0, 20, 40, 60])
    parser.add_argument("--num-ep",         type=int, default=200)
    args = parser.parse_args()

    failure_rates = [fr / 100.0 for fr in args.failure_rates]
    num_ep = args.num_ep

    env = NetworkRoutingEnv(use_mm1=True)
    env.reset()
    rl_agents = load_agents(env)

    # Storage: {algorithm: {metric: [one value per failure_rate]}}
    METRICS = ["delay", "pdr", "jitter", "throughput"]
    results = {alg: {m: [] for m in METRICS}
               for alg in list(rl_agents.keys()) + ["Dijkstra", "ECMP", "Random"]}

    rng = np.random.default_rng(42)

    for fr in failure_rates:
        print(f"\n{'='*50}")
        print(f"Failure rate = {int(fr*100)}%")
        print(f"{'='*50}")
        rng2 = np.random.default_rng(42)

        # RL agents
        for alg, (kind, agent, is_gnn, is_rb) in rl_agents.items():
            inject_failures(env, fr, rng2)
            if kind == "rl":
                agent.epsilon = 0.0
                d, p, j, t = eval_rl_agent(agent, env, num_ep, is_gnn=is_gnn)
            else:
                # Reset Q-tables for each failure scenario so warm-up starts fresh
                agent.reset_tables()
                d, p, j, t = eval_qrouting(agent, env, num_ep)
            results[alg]["delay"].append(d)
            results[alg]["pdr"].append(p)
            results[alg]["jitter"].append(j)
            results[alg]["throughput"].append(t)
            print(f"  {alg:<12} delay={d:6.1f}ms  PDR={p:.3f}  "
                  f"jitter={j:5.1f}ms  tput={t:.4f}")

        inject_failures(env, fr, rng2)
        d, p, j, t = eval_dijkstra(env, num_ep)
        results["Dijkstra"]["delay"].append(d)
        results["Dijkstra"]["pdr"].append(p)
        results["Dijkstra"]["jitter"].append(j)
        results["Dijkstra"]["throughput"].append(t)
        print(f"  {'Dijkstra':<12} delay={d:6.1f}ms  PDR={p:.3f}  "
              f"jitter={j:5.1f}ms  tput={t:.4f}")

        inject_failures(env, fr, rng2)
        d, p, j, t = eval_ecmp(env, num_ep)
        results["ECMP"]["delay"].append(d)
        results["ECMP"]["pdr"].append(p)
        results["ECMP"]["jitter"].append(j)
        results["ECMP"]["throughput"].append(t)
        print(f"  {'ECMP':<12} delay={d:6.1f}ms  PDR={p:.3f}  "
              f"jitter={j:5.1f}ms  tput={t:.4f}")

        inject_failures(env, fr, rng2)
        d, p, j, t = eval_random(env, num_ep)
        results["Random"]["delay"].append(d)
        results["Random"]["pdr"].append(p)
        results["Random"]["jitter"].append(j)
        results["Random"]["throughput"].append(t)
        print(f"  {'Random':<12} delay={d:6.1f}ms  PDR={p:.3f}  "
              f"jitter={j:5.1f}ms  tput={t:.4f}")

    # ── Save JSON ─────────────────────────────────────────────────────────
    out_json = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(out_json, "w") as f:
        json.dump({"failure_rates": args.failure_rates, "results": results}, f, indent=2)
    print(f"\nRaw results → {out_json}")

    # ── Publication-quality plot ──────────────────────────────────────────
    alg_styles = {
        "Rainbow":   ("crimson",   "o",  2.5, "-"),
        "DQN":       ("steelblue", "s",  2.0, "-"),
        "GNN-DQN":   ("purple",    "^",  2.0, "-"),
        "Q-Routing": ("teal",      "D",  1.8, "--"),
        "Dijkstra":  ("darkorange","P",  2.0, "--"),
        "ECMP":      ("olive",     "h",  1.8, "--"),
        "Random":    ("gray",      "x",  1.5, ":"),
    }

    x_labels = [f"{int(fr*100)}%" for fr in failure_rates]
    metric_labels = {
        "delay":      "Avg End-to-End Delay (ms)",
        "pdr":        "Packet Delivery Ratio",
        "jitter":     "Delay Jitter (std-dev, ms)",
        "throughput": "Throughput (deliveries / step)",
    }

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]

    for ax, metric in zip(axes, METRICS):
        for alg, (color, marker, lw, ls) in alg_styles.items():
            if alg not in results:
                continue
            vals = results[alg][metric]
            ax.plot(x_labels, vals, label=alg, color=color,
                    marker=marker, linewidth=lw, linestyle=ls, markersize=8)
        ax.set_xlabel("Link Failure Rate", fontsize=11)
        ax.set_ylabel(metric_labels[metric], fontsize=11)
        ax.set_title(metric_labels[metric], fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=9)

    fig.suptitle(
        "Algorithm Comparison: Adaptive Routing under Dynamic Link Failures\n"
        "Rainbow DQN vs Vanilla DQN vs GNN-DQN vs Q-Routing vs Dijkstra vs ECMP vs Random",
        fontsize=13, fontweight="bold"
    )
    out_png = os.path.join(RESULTS_DIR, "evaluation_comparison.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Comparison plot → {out_png}")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'Algorithm':<14} {'PDR@0%':>7} {'PDR@20%':>8} {'PDR@40%':>8} "
          f"{'PDR@60%':>8}  {'Delay@60%':>11}")
    print("-" * 62)
    for alg in results:
        p = results[alg]["pdr"]
        d = results[alg]["delay"]
        if p:
            print(f"  {alg:<12} {p[0]:>7.3f} {p[1]:>8.3f} {p[2]:>8.3f} "
                  f"{p[3]:>8.3f}  {d[3]:>8.1f}ms")


if __name__ == "__main__":
    main()
