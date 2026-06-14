"""
Large-topology scalability: the 10-node-trained GNN applied zero-shot to graphs
of 20-50 nodes it has never seen.

generalisation_test.py shows the GNN transfers to a *slightly* perturbed graph
(+4 edges). The open question for learned routing is whether a structure-aware
policy keeps working as the graph grows by an order of magnitude. Because the
GraphSAGE message-passing and Q-head operate on fixed-width node/edge *feature*
vectors (not on node counts), the trained weights apply unchanged to any graph —
so we can measure this directly instead of speculating.

Why PDR alone is the wrong metric here: on a well-connected graph with no
failures, *any* router (even a random walk given enough hops) eventually
delivers, so PDR ≈ 1.0 for everything and proves nothing. The discriminating
metric is DELAY STRETCH — the mean end-to-end delay of a policy's delivered
paths divided by Dijkstra's optimal delay on the same flows. Stretch = 1.0 means
optimal routing; a wandering policy has stretch ≫ 1. A zero-shot GNN that keeps
stretch near 1.0 on a 50-node graph is genuinely transferring route quality, not
merely "reaching the destination eventually".

Method:
  • Generate Barabási-Albert scale-free graphs (Barabási & Albert, 1999 — the
    canonical model for the Internet's degree distribution) at N = 20, 30, 50,
    with the edge attributes the env uses. BA(m=2) is connected with realistic
    sparse average degree (~4), unlike a dense random graph.
  • Drive the trained GNN (models/gnn_seed0.pth, trained on the 10-node graph)
    zero-shot, and measure PDR + mean delivered-path delay.
  • Compare against Dijkstra (optimal delay → stretch denominator) and a random
    walk (the stretch floor) on the same graphs.

Nothing is fabricated: the GNN weights are the trained ones, and every number is
measured on freshly generated graphs.

Usage:
    python experiments/scalability.py [--sizes 20 30 50] [--num-ep 200] [--seed 7]
"""

import os, sys, json, argparse
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from network_rl.env.network_env import NetworkRoutingEnv
from network_rl.agents.gnn_agent import GNNAgent
from network_rl.baselines.dijkstra import build_weighted_graph, dijkstra_full_path
from network_rl.baselines.random_routing import random_full_path
from evaluate import path_delay

MODELS_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


# ── A NetworkRoutingEnv over an arbitrary generated graph ───────────────────

class ScalableNetworkEnv(NetworkRoutingEnv):
    """
    NetworkRoutingEnv driven by a caller-supplied graph of any size. Reuses the
    parent's M/M/1 delay, congestion dynamics and step logic unchanged; only the
    graph and the node-count-dependent reset are overridden. The MLP observation
    layout is left unused (the GNN reads env.G directly), so a varying obs length
    is fine here.
    """

    def __init__(self, graph: nx.Graph, **kwargs):
        super().__init__(**kwargs)
        self.G = graph
        self.edge_list  = list(self.G.edges())
        self.max_degree = max(dict(self.G.degree()).values())
        self.action_space.n = self.max_degree
        self._n_nodes = self.G.number_of_nodes()

    def reset(self, *, seed=None, options=None):
        # Bypass the parent's NUM_NODES-bound src/dst sampling.
        if seed is not None:
            np.random.seed(seed)
        super(NetworkRoutingEnv, self).reset(seed=seed)
        self._current_node = int(self.np_random.integers(0, self._n_nodes))
        self._dest_node    = int(self.np_random.integers(0, self._n_nodes))
        while self._dest_node == self._current_node:
            self._dest_node = int(self.np_random.integers(0, self._n_nodes))
        self._visited = {self._current_node}
        self._steps   = 0
        self._path    = [self._current_node]
        self.episode_delays = []
        self.episode_drops  = 0
        self._update_link_conditions()
        return None, {}


def make_ba_graph(n: int, seed: int) -> nx.Graph:
    """Sparse, connected Barabási-Albert scale-free graph with env attributes."""
    rng = np.random.default_rng(seed)
    G = nx.barabasi_albert_graph(n, m=2, seed=seed)  # connected by construction
    H = nx.Graph()
    H.add_nodes_from(range(n))
    for (u, v) in G.edges():
        H.add_edge(u, v,
                   bandwidth=float(rng.integers(30, 100)),
                   base_delay=float(rng.integers(3, 15)),
                   congestion=0.2, loss_rate=0.01, active=True)
    return H


# ── Per-policy measurement (PDR + mean delivered-path delay) ────────────────

def _gnn_eval(agent, env, num_ep):
    delivered, delays = 0, []
    for _ in range(num_ep):
        env.reset()
        dst = env._dest_node
        info, done, steps = {}, False, 0
        while not done and steps < 60:
            a_idx, _ = agent.select_action(env, env._current_node, dst)
            _, _, term, trunc, info = env.step(a_idx)
            done = term or trunc
            steps += 1
        path = info.get("path", [None])
        if path[-1] == dst:
            delivered += 1
            delays.append(path_delay(env, path))
    return delivered / num_ep, (float(np.mean(delays)) if delays else float("nan"))


def _baseline_eval(env, num_ep, kind):
    delivered, delays = 0, []
    for _ in range(num_ep):
        env.reset()
        src, dst = env._current_node, env._dest_node
        graph = build_weighted_graph(env)
        path = (dijkstra_full_path(graph, src, dst) if kind == "dijkstra"
                else random_full_path(graph, src, dst, max_hops=80))
        if path and path[-1] == dst:
            delivered += 1
            delays.append(path_delay(env, path))
    return delivered / num_ep, (float(np.mean(delays)) if delays else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[20, 30, 50])
    ap.add_argument("--num-ep", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    gnn_path = os.path.join(MODELS_DIR, "gnn_seed0.pth")
    if not os.path.exists(gnn_path):
        print(f"[!] {gnn_path} not found — train the GNN first."); return

    sizes = [10] + list(args.sizes)   # include the training size as a reference
    results = {}
    print(f"GNN trained on 10 nodes → zero-shot transfer to {args.sizes}")
    print("Stretch = policy mean delay / Dijkstra optimal delay (1.0 = optimal)\n" + "-" * 72)
    print(f"  {'N':>4} {'edges':>6} {'GNN PDR':>8} {'GNN stretch':>12} "
          f"{'Rand PDR':>9} {'Rand stretch':>13}")
    for n in sizes:
        if n == 10:
            env = NetworkRoutingEnv(use_mm1=True, failure_prob=0.0)
        else:
            env = ScalableNetworkEnv(make_ba_graph(n, args.seed),
                                     use_mm1=True, failure_prob=0.0)
        env.reset()
        agent = GNNAgent(graph=env.G, edge_list=env.edge_list)
        agent.load(gnn_path); agent.epsilon = 0.0

        gnn_pdr, gnn_delay = _gnn_eval(agent, env, args.num_ep)
        dij_pdr, dij_delay = _baseline_eval(env, args.num_ep, "dijkstra")
        rnd_pdr, rnd_delay = _baseline_eval(env, args.num_ep, "random")

        gnn_stretch = gnn_delay / dij_delay if dij_delay else float("nan")
        rnd_stretch = rnd_delay / dij_delay if dij_delay else float("nan")
        results[str(n)] = {
            "n_nodes": n, "n_edges": len(env.edge_list),
            "gnn_pdr": gnn_pdr, "dijkstra_pdr": dij_pdr, "random_pdr": rnd_pdr,
            "gnn_delay": gnn_delay, "dijkstra_delay": dij_delay, "random_delay": rnd_delay,
            "gnn_stretch": gnn_stretch, "random_stretch": rnd_stretch,
        }
        print(f"  {n:>4} {len(env.edge_list):>6} {gnn_pdr:>8.3f} {gnn_stretch:>12.3f} "
              f"{rnd_pdr:>9.3f} {rnd_stretch:>13.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "scalability.json")
    with open(out_path, "w") as f:
        json.dump({"trained_nodes": 10, "results": results}, f, indent=2)
    print(f"\nScalability results → {out_path}")
    _plot(results)


def _plot(results):
    ns = sorted(int(k) for k in results)
    gnn_s = [results[str(n)]["gnn_stretch"] for n in ns]
    rnd_s = [results[str(n)]["random_stretch"] for n in ns]
    gnn_p = [results[str(n)]["gnn_pdr"] for n in ns]
    dij_p = [results[str(n)]["dijkstra_pdr"] for n in ns]
    rnd_p = [results[str(n)]["random_pdr"] for n in ns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.axhline(1.0, color="darkorange", ls="--", lw=1.5, label="Dijkstra optimal (stretch=1)")
    ax1.plot(ns, gnn_s, "-", color="purple", marker="^", lw=2.5,
             label="GNN-DQN (zero-shot, trained on N=10)")
    ax1.plot(ns, rnd_s, ":", color="gray", marker="x", lw=1.5, label="Random walk")
    ax1.axvline(10, color="black", alpha=0.3, lw=1)
    ax1.set_xlabel("Topology size (nodes)", fontsize=11)
    ax1.set_ylabel("Delay stretch  (policy delay / optimal)", fontsize=11)
    ax1.set_title("Route quality vs. graph size", fontsize=11, fontweight="bold")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=9, loc="upper left")

    ax2.plot(ns, dij_p, "--", color="darkorange", marker="P", lw=2, label="Dijkstra")
    ax2.plot(ns, gnn_p, "-", color="purple", marker="^", lw=2.5, label="GNN-DQN (zero-shot)")
    ax2.plot(ns, rnd_p, ":", color="gray", marker="x", lw=1.5, label="Random walk")
    ax2.axvline(10, color="black", alpha=0.3, lw=1)
    ax2.set_xlabel("Topology size (nodes)", fontsize=11)
    ax2.set_ylabel("Packet Delivery Ratio", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Delivery vs. graph size", fontsize=11, fontweight="bold")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=9, loc="lower left")

    fig.suptitle("GNN scalability: zero-shot transfer to larger unseen topologies\n"
                 "Weights trained on a 10-node graph, evaluated on Barabási-Albert graphs up to 50 nodes",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(RESULTS_DIR, "scalability.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Scalability plot → {out}")


if __name__ == "__main__":
    main()
