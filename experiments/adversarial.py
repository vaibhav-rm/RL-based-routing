"""
Adversarial-failure robustness (worst-case vs. average-case link failures).

The headline evaluation (evaluate.py) fails a *random* subset of links. Random
failures are average-case: most of them hit redundant links the topology can
route around. A real adversary — or a correlated outage (a shared conduit, a
power feeder, a targeted attack) — does not pick links at random. It removes the
links that matter most.

This script measures how each policy degrades under a *targeted* attack and
contrasts it with random failures at the SAME budget, so the gap is purely an
effect of failure *placement*:

  • Random       — k links chosen uniformly at random (the average case).
  • Adversarial  — the k highest edge-betweenness-centrality links, i.e. the
    links carrying the most shortest paths. Cutting these maximally fragments
    the reachable topology (Girvan & Newman, 2002, use the same centrality to
    find a network's most critical edges).

To isolate the effect of *which* links fail, we disable the env's stochastic
failure/recovery dynamics (failure_prob = 0) so the injected outage persists for
the whole measurement. Congestion dynamics are left untouched. Nothing is
fabricated — PDR comes from the same trained models used by evaluate.py.

Failure semantics (project-wide convention): a failed link is a *high-penalty
but still-traversable* link, not a hard cut — so for the learned policies PDR
measures successful navigation to the destination within the hop budget over a
degraded cost landscape. Dijkstra, by contrast, routes only over the ACTIVE
subgraph (a hard cut), so it is included as a topology *reachability reference*:
under a heavy targeted attack it can deliver fewer flows than a soft-routing
policy that pushes through, which is a property of the two failure models — NOT
evidence that a learned policy "beats shortest-path optimal".

Usage:
    python experiments/adversarial.py [--failure-rates 20 40] [--num-ep 200]
"""

import os, sys, json, argparse
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from network_rl.env.network_env import NetworkRoutingEnv
from network_rl.baselines.dijkstra import build_weighted_graph, dijkstra_full_path
from evaluate import eval_rl_agent, load_agents, eval_qrouting

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


# ── Failure-placement strategies ───────────────────────────────────────────

def _set_failures(env, fail_edges):
    """Activate every link, then knock down exactly the chosen ones."""
    for (u, v) in env.edge_list:
        env.G.edges[u, v]["active"] = True
    for (u, v) in fail_edges:
        env.G.edges[u, v]["active"] = False


def random_failures(env, k, rng):
    edges = list(env.edge_list)
    rng.shuffle(edges)
    return edges[:k]


def adversarial_failures(env, k):
    """The k most critical links by edge-betweenness centrality."""
    bc = nx.edge_betweenness_centrality(env.G)
    ranked = sorted(env.edge_list, key=lambda e: bc[e], reverse=True)
    return ranked[:k]


# ── Evaluation under a fixed outage ─────────────────────────────────────────

def _eval_dijkstra_fixed(env, num_ep):
    """Dijkstra PDR with the current (already-injected) outage held fixed."""
    pdrs = []
    for _ in range(num_ep):
        env.reset()
        src, dst = env._current_node, env._dest_node
        path = dijkstra_full_path(build_weighted_graph(env), src, dst)
        pdrs.append(float(bool(path) and path[-1] == dst))
    return float(np.mean(pdrs))


def pdr_under_outage(env, agents, fail_edges, num_ep):
    """Mean PDR for every policy with `fail_edges` held down for the run."""
    out = {}
    for alg, (kind, agent, is_gnn, _is_rb) in agents.items():
        _set_failures(env, fail_edges)
        if kind == "rl":
            agent.epsilon = 0.0
            _, pdr, _, _ = eval_rl_agent(env=env, agent=agent, num_ep=num_ep, is_gnn=is_gnn)
        else:  # Q-routing — warm up on this exact outage first
            agent.reset_tables()
            _, pdr, _, _ = eval_qrouting(agent, env, num_ep)
        out[alg] = float(pdr)
    _set_failures(env, fail_edges)
    out["Dijkstra"] = _eval_dijkstra_fixed(env, num_ep)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--failure-rates", type=int, nargs="+", default=[20, 40])
    ap.add_argument("--num-ep", type=int, default=200)
    args = ap.parse_args()

    # failure_prob=0 → injected outage persists; only placement differs.
    env = NetworkRoutingEnv(use_mm1=True, failure_prob=0.0)
    env.reset()
    agents = load_agents(env)
    n_edges = len(env.edge_list)
    rng = np.random.default_rng(42)

    results = {}
    for fr in args.failure_rates:
        k = int(round(n_edges * fr / 100.0))
        print(f"\n{'='*58}\nFailure budget = {fr}%  ({k}/{n_edges} links down)\n{'='*58}")

        rnd = pdr_under_outage(env, agents, random_failures(env, k, rng), args.num_ep)
        adv = pdr_under_outage(env, agents, adversarial_failures(env, k), args.num_ep)

        print(f"  {'Policy':<12} {'PDR random':>11} {'PDR adversarial':>16} {'Δ (drop)':>10}")
        print("  " + "-" * 51)
        per_alg = {}
        for alg in rnd:
            drop = rnd[alg] - adv[alg]
            per_alg[alg] = {"pdr_random": rnd[alg], "pdr_adversarial": adv[alg],
                            "robustness_drop": float(drop)}
            print(f"  {alg:<12} {rnd[alg]:>11.3f} {adv[alg]:>16.3f} {drop:>10.3f}")
        results[str(fr)] = {"k_failed": k, "n_edges": n_edges, "per_algorithm": per_alg}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "adversarial.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAdversarial results → {out_path}")

    _plot(results, args.failure_rates)


def _plot(results, failure_rates):
    algs = list(next(iter(results.values()))["per_algorithm"].keys())
    fig, axes = plt.subplots(1, len(failure_rates), figsize=(7 * len(failure_rates), 5),
                             squeeze=False)
    x = np.arange(len(algs))
    w = 0.38
    for ax, fr in zip(axes[0], failure_rates):
        per = results[str(fr)]["per_algorithm"]
        rnd = [per[a]["pdr_random"] for a in algs]
        adv = [per[a]["pdr_adversarial"] for a in algs]
        ax.bar(x - w / 2, rnd, w, label="Random failures", color="steelblue", alpha=0.85)
        ax.bar(x + w / 2, adv, w, label="Adversarial (critical-link) failures",
               color="crimson", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(algs, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Packet Delivery Ratio")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{fr}% of links down "
                     f"({results[str(fr)]['k_failed']}/{results[str(fr)]['n_edges']})",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Robustness to worst-case vs. average-case link failures\n"
                 "Targeted attack removes the highest edge-betweenness links",
                 fontsize=12, fontweight="bold")
    fig.text(0.5, 0.005,
             "Learned policies route a degraded (penalised) cost graph; Dijkstra "
             "routes only the active subgraph (hard-cut reachability reference).",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    out = os.path.join(RESULTS_DIR, "adversarial.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Adversarial plot → {out}")


if __name__ == "__main__":
    main()
