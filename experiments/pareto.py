"""
Multi-objective Pareto fronts: routing is a trade-off, not a single number.

The reward is a scalarised sum  r = … − w_delay·delay − w_drop·1[packet dropped].
Any single choice of weights commits the agent to ONE point on a
latency-vs-reliability trade-off, and the headline comparison (evaluate.py)
reports each algorithm at one operating point. This script makes the
multi-objective structure explicit in two complementary ways.

(A) Cross-algorithm front (from the already-validated evaluate.py results).
    Every algorithm is a point in (delay, PDR) space at a fixed failure rate.
    statistics.pareto_front identifies the non-dominated set — the algorithms
    that are not strictly beaten on both objectives. This needs no training; it
    re-reads results/evaluation_results.json.

(B) Learned-policy front (real training sweep).
    We train a vanilla DQN at a sweep of w_drop values (holding w_delay fixed)
    and measure the achieved operating point of each:
      • objective 1 — mean end-to-end delay of delivered packets (ms)
      • objective 2 — mean packet drops per episode (reliability cost)
    Each point is averaged over several seeds to separate the weight's effect
    from training noise. The non-dominated points trace the Pareto front that a
    single hard-coded scalarisation would hide.

Nothing is fabricated: (A) reuses validated numbers; (B) wires the weights
straight into network_env.step and trains/evaluates each point separately.

Usage:
    python experiments/pareto.py [--w-drop 0 5 15 30 60] [--episodes 350]
                                 [--seeds 3] [--eval-ep 300] [--failure-rate 20]
"""

import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from network_rl.env.network_env import NetworkRoutingEnv, OBS_DIM
from network_rl.agents.dqn_agent import DQNAgent
from network_rl.analysis.statistics import pareto_front
from evaluate import path_delay

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

# Elevated background load so some links are genuinely fast-but-lossy and the
# latency/reliability tension actually exists to be traded off.
MEAN_LOAD = 0.5
W_DELAY   = 10.0


def _valid_actions(env):
    return list(range(len(env._get_sorted_neighbors(env._current_node))))


# ── (B) Learned-policy front — real training ────────────────────────────────

def train_dqn_operating_point(w_drop: float, episodes: int, seed: int) -> DQNAgent:
    """Train a fresh DQN whose reward uses (W_DELAY, w_drop)."""
    np.random.seed(seed)
    env = NetworkRoutingEnv(use_mm1=True, failure_prob=0.002, mean_load=MEAN_LOAD,
                            w_delay=W_DELAY, w_drop=w_drop)
    agent = DQNAgent(state_dim=OBS_DIM, action_dim=env.action_space.n)
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            a = agent.select_action(obs, _valid_actions(env))
            nobs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            agent.store(obs, a, r, nobs, float(done))
            agent.learn()
            obs = nobs
        agent.decay_epsilon()
        if ep % agent.target_update == 0:
            agent.sync_target()
    return agent


def eval_operating_point(agent: DQNAgent, eval_ep: int, seed: int):
    """
    Mean delivered-path delay and mean drops/episode on a fixed flow stream.
    The eval env's reward weights are irrelevant here (reward isn't measured —
    only realised delay/drops/PDR), so we leave them at their defaults.
    """
    agent.epsilon = 0.0
    env = NetworkRoutingEnv(use_mm1=True, failure_prob=0.002, mean_load=MEAN_LOAD)
    delays, drops, pdrs = [], [], []
    for i in range(eval_ep):
        obs, _ = env.reset(seed=seed + i)   # identical flow stream for every point
        done = False
        while not done:
            a = agent.select_action(obs, _valid_actions(env))
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
        reached = info["path"][-1] == env._dest_node
        pdrs.append(float(reached))
        drops.append(info["drops"])
        if reached:
            delays.append(path_delay(env, info["path"]))
    return (float(np.mean(delays)) if delays else float("nan"),
            float(np.mean(drops)), float(np.mean(pdrs)))


def learned_policy_front(w_drops, episodes, seeds, eval_ep):
    print(f"(B) Learned-policy front: w_drop sweep {w_drops}, "
          f"{episodes} ep × {seeds} seeds\n" + "-" * 64)
    print(f"  {'w_drop':>7} {'delay(ms)':>10} {'drops/ep':>9} {'PDR':>6}  (mean over seeds)")
    points = []
    for wd in w_drops:
        ds, dr, pd = [], [], []
        for s in range(seeds):
            agent = train_dqn_operating_point(wd, episodes, seed=s)
            delay, drop, pdr = eval_operating_point(agent, eval_ep, seed=10_000 + s)
            ds.append(delay); dr.append(drop); pd.append(pdr)
        pt = {"w_drop": wd,
              "delay": float(np.nanmean(ds)),     "delay_std": float(np.nanstd(ds)),
              "drops": float(np.mean(dr)),         "drops_std": float(np.std(dr)),
              "pdr":   float(np.mean(pd))}
        points.append(pt)
        print(f"  {wd:>7.0f} {pt['delay']:>10.2f} {pt['drops']:>9.3f} {pt['pdr']:>6.3f}")

    front_idx = set(pareto_front([[p["delay"], p["drops"]] for p in points]))
    for i, p in enumerate(points):
        p["pareto_optimal"] = i in front_idx
    print("Non-dominated operating points: "
          + ", ".join(f"w_drop={points[i]['w_drop']:.0f}" for i in sorted(front_idx)))
    return points


# ── (A) Cross-algorithm front — from validated eval results ─────────────────

def cross_algorithm_front(failure_rate: int):
    path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    if not os.path.exists(path):
        print(f"[!] {path} not found — run evaluate.py first; skipping panel (A).")
        return None
    data = json.load(open(path))
    rates = data["failure_rates"]
    if failure_rate not in rates:
        print(f"[!] failure rate {failure_rate}% not in {rates}; skipping panel (A).")
        return None
    idx = rates.index(failure_rate)
    algs = {a: {"delay": m["delay"][idx], "pdr": m["pdr"][idx]}
            for a, m in data["results"].items()}
    names = list(algs)
    # Both objectives lower-is-better: minimise delay and minimise (1 − PDR).
    objs = [[algs[a]["delay"], 1.0 - algs[a]["pdr"]] for a in names]
    front_idx = set(pareto_front(objs))
    for i, a in enumerate(names):
        algs[a]["pareto_optimal"] = i in front_idx
    print(f"\n(A) Cross-algorithm front @ {failure_rate}% failure "
          f"(delay vs PDR)\n" + "-" * 64)
    for i, a in enumerate(names):
        tag = "  ← Pareto-optimal" if i in front_idx else ""
        print(f"  {a:<12} delay={algs[a]['delay']:6.2f}ms  PDR={algs[a]['pdr']:.3f}{tag}")
    return {"failure_rate": failure_rate, "algorithms": algs}


# ── Plot ────────────────────────────────────────────────────────────────────

def _plot(cross, learned, failure_rate):
    n_panels = 1 + int(cross is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6), squeeze=False)
    axes = axes[0]
    ax_i = 0

    if cross is not None:
        ax = axes[ax_i]; ax_i += 1
        algs = cross["algorithms"]
        front = sorted((a for a in algs if algs[a]["pareto_optimal"]),
                       key=lambda a: algs[a]["delay"])
        if len(front) > 1:
            ax.plot([algs[a]["delay"] for a in front], [algs[a]["pdr"] for a in front],
                    "-", color="purple", lw=1.5, alpha=0.6, zorder=1, label="Pareto frontier")
        for a, m in algs.items():
            opt = m["pareto_optimal"]
            ax.scatter(m["delay"], m["pdr"], s=130,
                       color="crimson" if opt else "lightgray",
                       edgecolor="black", zorder=3)
            ax.annotate(a, (m["delay"], m["pdr"]),
                        textcoords="offset points", xytext=(7, 5), fontsize=9)
        ax.scatter([], [], s=130, color="crimson", edgecolor="black", label="Pareto-optimal")
        ax.scatter([], [], s=130, color="lightgray", edgecolor="black", label="dominated")
        ax.set_xlabel("Mean end-to-end delay (ms)  — lower better", fontsize=11)
        ax.set_ylabel("Packet Delivery Ratio  — higher better", fontsize=11)
        ax.set_title(f"(A) Cross-algorithm trade-off @ {failure_rate}% link failure",
                     fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="best")

    ax = axes[ax_i]
    front = sorted((p for p in learned if p["pareto_optimal"]), key=lambda p: p["delay"])
    if len(front) > 1:
        ax.plot([p["delay"] for p in front], [p["drops"] for p in front],
                "-", color="purple", lw=1.5, alpha=0.6, zorder=1, label="Pareto frontier")
    for p in learned:
        opt = p["pareto_optimal"]
        ax.errorbar(p["delay"], p["drops"], xerr=p["delay_std"], yerr=p["drops_std"],
                    fmt="o", ms=11, color="crimson" if opt else "lightgray",
                    ecolor="gray", elinewidth=1, capsize=3,
                    markeredgecolor="black", zorder=3)
        ax.annotate(f"w_drop={p['w_drop']:.0f}", (p["delay"], p["drops"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.scatter([], [], s=110, color="crimson", edgecolor="black", label="Pareto-optimal")
    ax.scatter([], [], s=110, color="lightgray", edgecolor="black", label="dominated")
    ax.set_xlabel("Mean end-to-end delay (ms)  — lower better", fontsize=11)
    ax.set_ylabel("Mean packet drops per episode  — lower better", fontsize=11)
    ax.set_title("(B) Learned-policy front (DQN, drop-penalty sweep)",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="best")

    fig.suptitle("Multi-objective routing: latency vs. reliability is a trade-off, "
                 "not a single number", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(RESULTS_DIR, "pareto.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPareto plot → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w-drop", type=float, nargs="+", default=[0, 5, 15, 30, 60])
    ap.add_argument("--episodes", type=int, default=350)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eval-ep", type=int, default=300)
    ap.add_argument("--failure-rate", type=int, default=20)
    args = ap.parse_args()

    cross   = cross_algorithm_front(args.failure_rate)
    learned = learned_policy_front(args.w_drop, args.episodes, args.seeds, args.eval_ep)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "pareto.json")
    with open(out_path, "w") as f:
        json.dump({"cross_algorithm": cross,
                   "learned_policy": {"w_delay": W_DELAY, "mean_load": MEAN_LOAD,
                                      "train_episodes": args.episodes,
                                      "seeds": args.seeds, "points": learned}},
                  f, indent=2)
    print(f"Pareto results → {out_path}")
    _plot(cross, learned, args.failure_rate)


if __name__ == "__main__":
    main()
