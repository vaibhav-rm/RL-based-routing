"""
Generalisation test: evaluate a trained GNN vs MLP agents on an UNSEEN topology.

This is a key research contribution of the GNN approach:
  The GNN routes from GRAPH STRUCTURE (node/edge features + message passing),
  not a memorised, fixed-size topology. It can therefore be applied to a graph
  it has never seen — the message-passing operation is defined for any graph.

  The MLP agents (DQN / Rainbow) consume a FIXED-LENGTH observation vector
  (OBS_DIM = NUM_EDGES * 4 + 2 = 70 for the 17-edge training graph). Change the
  edge set and the observation length changes, so the first linear layer can no
  longer accept the input. MLP agents are thus topology-specific *by construction*
  and cannot even be evaluated on the modified graph — we report that explicitly
  rather than fabricating a number.

We construct an unseen topology by adding 4 cross-links to the training graph
(17 → 21 edges). This changes the observation length from 70 → 86, which is
exactly the regime that separates a structure-aware GNN from a fixed-input MLP.

Usage:
    python experiments/generalisation_test.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from network_rl.env.network_env import NetworkRoutingEnv, OBS_DIM, TOPOLOGY_EDGES
from network_rl.agents.dqn_agent     import DQNAgent
from network_rl.agents.rainbow_agent import RainbowAgent
from network_rl.agents.gnn_agent     import GNNAgent

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Unseen topology ─────────────────────────────────────────────────────────
# Four extra cross-links over the 17-edge training graph (none already present).
EXTRA_EDGES = [
    (0, 9,  70, 6),   # direct source→sink shortcut
    (2, 7,  60, 9),   # cross-link
    (5, 9,  55, 11),  # alternate path to sink
    (4, 8,  45, 13),  # new path
]


class ExtendedNetworkEnv(NetworkRoutingEnv):
    """Training env plus 4 extra edges — an unseen topology (17 → 21 edges)."""
    def __init__(self):
        super().__init__(use_mm1=True)
        for (u, v, bw, delay) in EXTRA_EDGES:
            self.G.add_edge(u, v, bandwidth=bw, base_delay=delay,
                            congestion=0.2, loss_rate=0.01, active=True)
        self.edge_list  = list(self.G.edges())
        self.max_degree = max(dict(self.G.degree()).values())


def eval_gnn_pdr(agent, env, n_ep=150):
    """Packet-delivery ratio for a (structure-aware) GNN agent."""
    delivered = 0
    for _ in range(n_ep):
        env.reset()
        dst, info = env._dest_node, {}
        done, steps = False, 0
        while not done and steps < 50:
            action, _ = agent.select_action(env, env._current_node, dst)
            _, _, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
        if info.get("path", [None])[-1] == dst:
            delivered += 1
    return delivered / n_ep


def main():
    n_ep = 150
    env_train = NetworkRoutingEnv()
    env_test  = ExtendedNetworkEnv()

    train_obs_dim = env_train.observation_space.shape[0]
    test_obs_dim  = len(env_test._get_obs())
    print(f"Training topology : {len(env_train.edge_list)} edges, obs_dim={train_obs_dim}")
    print(f"Unseen   topology : {len(env_test.edge_list)} edges, obs_dim={test_obs_dim}")
    print()

    # ── MLP agents: report architectural incompatibility (no fabricated PDR) ──
    incompatible = []
    for label, fname, Cls in [("DQN", "dqn_seed0.pth", DQNAgent),
                              ("Rainbow", "rainbow_seed0.pth", RainbowAgent)]:
        if not os.path.exists(os.path.join(MODELS_DIR, fname)):
            continue
        incompatible.append(label)
        print(f"{label:<10}  topology-specific: fixed {OBS_DIM}-dim input cannot "
              f"accept the {test_obs_dim}-dim observation of the unseen graph "
              f"→ not evaluable.")

    # ── GNN: genuine transfer measurement ───────────────────────────────────
    gnn_train = gnn_test = None
    gnn_path = os.path.join(MODELS_DIR, "gnn_seed0.pth")
    if os.path.exists(gnn_path):
        agent = GNNAgent(graph=env_train.G, edge_list=env_train.edge_list)
        agent.load(gnn_path); agent.epsilon = 0.0
        gnn_train = eval_gnn_pdr(agent, env_train, n_ep)
        gnn_test  = eval_gnn_pdr(agent, env_test,  n_ep)
        transfer  = gnn_test / max(gnn_train, 1e-6)
        print(f"\nGNN        train_PDR={gnn_train:.3f}  "
              f"unseen_PDR={gnn_test:.3f}  transfer={transfer:.3f}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    w = 0.6

    # GNN bars (train vs unseen)
    if gnn_train is not None:
        ax.bar(x, [gnn_train, gnn_test], w,
               color=["steelblue", "crimson"], alpha=0.85)
        for i, val in enumerate([gnn_train, gnn_test]):
            ax.text(i, val + 0.02, f"{val:.2f}", ha="center", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(["GNN-DQN\n(training topology)",
                        "GNN-DQN\n(unseen topology, +4 edges)"], fontsize=10)
    ax.set_ylabel("Packet Delivery Ratio (PDR)", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    note = ("MLP agents (DQN, Rainbow) are topology-specific:\n"
            f"their fixed {OBS_DIM}-dim input cannot process the\n"
            f"{test_obs_dim}-dim observation of the unseen graph.")
    if incompatible:
        ax.text(0.98, 0.04, note, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, bbox=dict(boxstyle="round", fc="wheat", alpha=0.7))

    ax.set_title("Generalisation to an Unseen Topology\n"
                 "GNN transfers via message passing; fixed-input MLPs cannot run",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "generalisation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nGeneralisation plot → {out}")


if __name__ == "__main__":
    main()
