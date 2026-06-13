"""
Publication-quality visualisations:
  1. Network topology with link properties
  2. Congestion heatmap over time
  3. Multi-seed reward curves with confidence bands (loaded from CSV logs)
  4. PDR vs failure-rate comparison (from evaluation JSON)
  5. Q-table visualisation for Q-routing agent
  6. Path comparison: Dijkstra vs Rainbow routing decisions

Usage:
    python visualize.py [--mode all|topology|heatmap|curves|comparison|qtable]
"""

import os, sys, csv, argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))
from network_rl.env.network_env import NetworkRoutingEnv, NUM_NODES

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
LOGS_DIR    = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Fixed layout for the 10-node topology
POS = {
    0: (0.0, 1.0), 1: (1.0, 1.5), 2: (1.0, 0.5),
    3: (2.0, 1.5), 4: (2.0, 2.2), 5: (2.0, 0.8),
    6: (3.0, 1.8), 7: (3.0, 1.0), 8: (3.0, 0.2),
    9: (4.0, 1.0),
}


# ── 1. Topology ────────────────────────────────────────────────────────────

def plot_topology(env, ax, title="Network Topology"):
    edge_labels = {}
    edge_colors = []
    edge_widths = []
    for (u, v) in env.edge_list:
        e = env.G.edges[u, v]
        bw    = e["bandwidth"]
        delay = e["base_delay"]
        edge_labels[(u, v)] = f"{delay}ms\n{bw}Mb"
        edge_colors.append("#2e7d32" if e["active"] else "#c62828")
        edge_widths.append(1.5 + bw / 50.0)

    nx.draw_networkx(
        env.G, pos=POS, ax=ax,
        node_color="#1565c0", node_size=600, font_color="white",
        font_size=10, font_weight="bold",
        edge_color=edge_colors, width=edge_widths,
    )
    nx.draw_networkx_edge_labels(env.G, pos=POS, edge_labels=edge_labels,
                                  font_size=6.5, ax=ax, rotate=False)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")


def save_topology(env):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: base topology
    plot_topology(env, axes[0], "Network Topology\n(green=active link, red=failed)")

    # Panel 2: congestion overlay
    env.reset()
    for _ in range(10):
        env._update_link_conditions()

    cong = [env.G.edges[u, v]["congestion"] for (u, v) in env.edge_list]
    cmap = plt.cm.RdYlGn_r
    edge_colors_cong = [cmap(c) for c in cong]
    nx.draw_networkx(
        env.G, pos=POS, ax=axes[1],
        node_color="#1565c0", node_size=600, font_color="white",
        font_size=10, edge_color=edge_colors_cong, width=3,
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=axes[1], label="Congestion (0=idle, 1=saturated)", shrink=0.8)
    axes[1].set_title("Link Congestion State\n(after 10 simulation steps)",
                       fontsize=12, fontweight="bold")
    axes[1].axis("off")

    fig.suptitle("10-Node Router Topology — AI-Assisted Adaptive Routing",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "topology.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Topology → {out}")


# ── 2. Congestion heatmap ───────────────────────────────────────────────────

def save_congestion_heatmap(env):
    STEPS = 30
    env.reset()
    matrix = []
    for _ in range(STEPS):
        row = [env.G.edges[u, v]["congestion"] for (u, v) in env.edge_list]
        matrix.append(row)
        env._update_link_conditions()
    matrix = np.array(matrix)

    edge_labels = [f"{u}–{v}" for (u, v) in env.edge_list]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

    # Heatmap
    im = ax1.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax1.set_xticks(range(len(env.edge_list)))
    ax1.set_xticklabels(edge_labels, rotation=45, ha="right", fontsize=8)
    ax1.set_yticks(range(STEPS))
    ax1.set_yticklabels([f"t={i}" for i in range(STEPS)], fontsize=7)
    ax1.set_xlabel("Link (u–v)", fontsize=10)
    ax1.set_ylabel("Timestep", fontsize=10)
    ax1.set_title("Link Congestion Heatmap Over Time", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax1, label="Congestion", shrink=0.8)

    # Average congestion time series
    ax2.plot(matrix.mean(1), color="steelblue", linewidth=2, label="Mean")
    ax2.fill_between(range(STEPS),
                     matrix.mean(1) - matrix.std(1),
                     matrix.mean(1) + matrix.std(1),
                     alpha=0.25, color="steelblue", label="±1 std")
    ax2.axhline(0.5, color="red", linestyle="--", alpha=0.6, label="saturation threshold")
    ax2.set_xlabel("Timestep"); ax2.set_ylabel("Avg Congestion")
    ax2.set_title("Network-wide Average Congestion (AR(1) + Pareto bursts)")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "congestion_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Heatmap → {out}")


# ── 3. Multi-seed reward curves from CSV logs ──────────────────────────────

def load_csv_rewards(agent_name: str) -> list:
    """Load per-episode rewards for all seeds of an agent from CSV logs."""
    runs = []
    for seed in range(10):
        path = os.path.join(LOGS_DIR, f"{agent_name}_seed{seed}.csv")
        if not os.path.exists(path):
            break
        rewards = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rewards.append(float(row["reward"]))
        if rewards:
            runs.append(rewards)
    return runs


def save_training_curves():
    agents = {
        "dqn":      ("Vanilla DQN",  "steelblue"),
        "rainbow":  ("Rainbow DQN",  "crimson"),
        "gnn":      ("GNN-DQN",      "darkorange"),
        "qrouting": ("Q-Routing",    "green"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    W    = 30

    for idx, (key, (label, color)) in enumerate(agents.items()):
        runs = load_csv_rewards(key)
        ax   = axes[idx]
        if not runs:
            ax.text(0.5, 0.5, f"No data for {label}\n(run train.py first)",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue

        # Align runs to same length by truncating to the shortest
        min_len = min(len(r) for r in runs)
        arr   = np.array([r[:min_len] for r in runs])
        mean  = arr.mean(0)
        std   = arr.std(0)
        n_ep  = len(mean)
        sm    = np.convolve(mean, np.ones(W)/W, mode="valid")
        ss    = np.convolve(std,  np.ones(W)/W, mode="valid")
        x     = np.arange(len(sm))

        ax.plot(x, sm, color=color, linewidth=2.5, label=f"{label} (mean)")
        ax.fill_between(x, sm - ss, sm + ss, alpha=0.2, color=color,
                        label=f"±1 std ({len(runs)} seeds)")
        ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
        ax.set_title(f"{label} — Training Curve", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # Mark curriculum stages
        for thresh, _, _ in [(200, None, None), (500, None, None)]:
            if thresh < n_ep:
                ax.axvline(thresh, color="gray", linestyle=":", alpha=0.7)
                ax.text(thresh + 2, ax.get_ylim()[0] * 0.9,
                        f"Stage\n{thresh}", fontsize=7, color="gray")

    fig.suptitle("Training Curves — Multi-Seed with Std-Dev Bands",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "training_reward.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Training curves → {out}")


# ── 4. Evaluation comparison (from JSON) ──────────────────────────────────

def save_eval_comparison():
    json_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    if not os.path.exists(json_path):
        print("[!] evaluation_results.json not found — run evaluate.py first")
        return

    with open(json_path) as f:
        data = json.load(f)

    failure_rates = data["failure_rates"]
    results       = data["results"]
    x_labels      = [f"{fr}%" for fr in failure_rates]

    alg_styles = {
        "Rainbow":   ("crimson",   "o",  2.5),
        "DQN":       ("steelblue", "s",  2.0),
        "GNN-DQN":   ("purple",    "^",  2.0),
        "Q-Routing": ("teal",      "D",  1.8),
        "Dijkstra":  ("darkorange","P",  2.0),
        "ECMP":      ("olive",     "h",  1.8),
        "Random":    ("lightgray", "x",  1.5),
    }

    metrics_to_plot = [
        ("pdr",       "Packet Delivery Ratio",      True),
        ("delay",     "Avg Delay (ms)",              False),
        ("jitter",    "Delay Jitter (ms)",           False),
        ("throughput","Throughput (delivers/step)",  True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (metric, ylabel, higher_better) in zip(axes, metrics_to_plot):
        for alg, (color, marker, lw) in alg_styles.items():
            if alg not in results:
                continue
            vals = results[alg].get(metric, [])
            if vals:
                ax.plot(x_labels, vals, label=alg, color=color,
                        marker=marker, linewidth=lw, markersize=8)
        arrow = "↑ better" if higher_better else "↓ better"
        ax.set_xlabel("Link Failure Rate", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{ylabel}  ({arrow})", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)

    fig.suptitle("Algorithm Comparison — Adaptive Routing Under Link Failures\n"
                 "(M/M/1 queuing model, RED-like loss, Pareto background traffic)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "evaluation_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Evaluation comparison → {out}")


# ── 5. Q-table visualisation ───────────────────────────────────────────────

def save_qtable_heatmap(env):
    from network_rl.agents.q_routing import QRoutingAgent
    agent = QRoutingAgent(env.G, lr=0.1)
    env.reset()
    # Quick training run
    for _ in range(200):
        env.reset()
        src, dst = env._current_node, env._dest_node
        agent.route_episode(env, src, dst)

    # Show Q-table for destination=9 (most common target)
    dest = 9
    matrix = np.full((NUM_NODES, NUM_NODES), np.nan)
    for node in range(NUM_NODES):
        if dest in agent.Q.get(node, {}):
            for nb, val in agent.Q[node][dest].items():
                matrix[node, nb] = val

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="YlOrRd_r", aspect="auto",
                   vmin=0, vmax=20)
    ax.set_xticks(range(NUM_NODES)); ax.set_xticklabels([f"nb{i}" for i in range(NUM_NODES)])
    ax.set_yticks(range(NUM_NODES)); ax.set_yticklabels([f"node{i}" for i in range(NUM_NODES)])
    ax.set_xlabel("Next-Hop Neighbour"); ax.set_ylabel("Current Node")
    ax.set_title(f"Q-Routing Table for Destination=Node {dest}\n"
                 f"(lower = better estimated delivery time)",
                 fontsize=11, fontweight="bold")
    for i in range(NUM_NODES):
        for j in range(NUM_NODES):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center",
                        fontsize=7, color="white" if matrix[i,j] > 10 else "black")
    plt.colorbar(im, ax=ax, label="Q-value (estimated delivery time)", shrink=0.8)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "qtable_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Q-table heatmap → {out}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                        choices=["all", "topology", "heatmap", "curves",
                                 "comparison", "qtable"])
    args = parser.parse_args()

    env = NetworkRoutingEnv()
    env.reset()

    modes = {
        "topology":   lambda: save_topology(env),
        "heatmap":    lambda: save_congestion_heatmap(env),
        "curves":     save_training_curves,
        "comparison": save_eval_comparison,
        "qtable":     lambda: save_qtable_heatmap(env),
    }

    if args.mode == "all":
        for name, fn in modes.items():
            print(f"Generating: {name}")
            try:
                fn()
            except Exception as e:
                print(f"  [!] {name} failed: {e}")
    else:
        modes[args.mode]()


if __name__ == "__main__":
    main()
