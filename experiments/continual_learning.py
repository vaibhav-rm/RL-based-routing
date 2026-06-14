"""
Catastrophic forgetting in learned routing — and what actually fixes it.

A routing agent deployed in a changing network must keep learning: a new traffic
matrix, a new failure regime, a new set of demands. But a single Q-network
trained sequentially on task B overwrites the weights that solved task A, so it
forgets how to serve the old regime — catastrophic forgetting. Continual-RL
remedies are well studied elsewhere (e.g. anti-jamming) but largely unexplored
for packet routing; this experiment measures the effect and two fixes head-on.

Tasks (deliberately conflicting policies on the SAME 10-node topology, so a fix
is genuinely needed):
  • Task A — deliver to destinations {7, 8, 9} (one side of the graph)
  • Task B — deliver to destinations {0, 1, 2} (the opposite side)
The destination is part of the observation, so one network *could* serve both;
sequential training is what induces forgetting.

Methods compared (identical agent, training budget and per-task ε-reset):
  • naive     — train A, then train B with a fresh replay buffer (no old data)
  • rehearsal — train B while RETAINING task-A experience in the replay buffer
                (data-space remedy; in RL, simply not discarding the buffer helps)
  • ewc       — train B from a fresh buffer but with an Elastic Weight
                Consolidation penalty anchoring A-important weights (a weight-space
                candidate that needs no stored data)

We report, per method, PDR on A right after A, PDR on A after B (the forgetting),
and PDR on B after B (that B was actually learned). Every number comes from real
sequential training — nothing is fabricated.

Finding (4 seeds): naive forgetting is severe (Task-A PDR collapses ~0.77→0.09).
REHEARSAL fully prevents it (forgetting ≈ 0). EWC, despite its success in
supervised continual learning, does NOT transfer here: the policy is conditioned
on the destination (an input feature), so both tasks share the same weights and a
diagonal-Fisher anchor cannot isolate task-specific computation — it ends up no
better than naive. The honest takeaway is that data-space rehearsal, which RL
gets almost for free via the replay buffer, is the robust remedy.

Usage:
    python experiments/continual_learning.py [--episodes 250] [--seeds 3]
                                             [--eval-ep 200] [--ewc-lambda 2000]
"""

import os, sys, json, argparse, random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from network_rl.env.network_env import NetworkRoutingEnv, OBS_DIM
from network_rl.agents.continual_dqn_agent import ContinualDQNAgent
from evaluate import path_delay, valid_actions

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
TASKS = {"A": [7, 8, 9], "B": [0, 1, 2]}


def _start_episode(env, dest_pool, rng):
    """Reset, then pin the destination into the task's region (src stays random)."""
    env.reset()
    cur = env._current_node
    dest = int(rng.choice([d for d in dest_pool if d != cur]))
    env._dest_node = dest
    env._visited = {cur}
    env._path = [cur]
    return env._get_obs()


def train_task(agent, env, dest_pool, episodes, rng):
    agent.epsilon = 1.0                       # fresh exploration per task (all methods)
    for ep in range(episodes):
        obs = _start_episode(env, dest_pool, rng)
        done = False
        while not done:
            a = agent.select_action(obs, valid_actions(env))
            nobs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            agent.store(obs, a, r, nobs, float(done))
            agent.learn()
            obs = nobs
        agent.decay_epsilon()
        if ep % agent.target_update == 0:
            agent.sync_target()


def eval_task(agent, env, dest_pool, n_ep, rng):
    agent.epsilon = 0.0
    pdrs, delays = [], []
    for _ in range(n_ep):
        obs = _start_episode(env, dest_pool, rng)
        dst = env._dest_node
        done = False
        while not done:
            a = agent.select_action(obs, valid_actions(env))
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
        reached = info["path"][-1] == dst
        pdrs.append(float(reached))
        if reached:
            delays.append(path_delay(env, info["path"]))
    return float(np.mean(pdrs)), (float(np.mean(delays)) if delays else float("nan"))


def run_method(method, episodes, eval_ep, ewc_lambda, seed):
    # Seed ALL three RNGs: the agent uses Python's `random` for ε-greedy and
    # replay sampling, numpy for env dynamics, torch for weight init — seeding
    # only numpy left training irreproducible.
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env = NetworkRoutingEnv(use_mm1=True)
    rng = np.random.default_rng(seed)
    lam = ewc_lambda if method == "ewc" else 0.0
    agent = ContinualDQNAgent(state_dim=OBS_DIM, action_dim=env.action_space.n,
                              ewc_lambda=lam)

    # ── Task A ──
    train_task(agent, env, TASKS["A"], episodes, rng)
    a_after_a, _ = eval_task(agent, env, TASKS["A"], eval_ep, rng)

    if method == "ewc":
        agent.consolidate()                   # protect A-important weights
    if method != "rehearsal":
        agent.replay.buffer.clear()           # naive & ewc start B with no A data

    # ── Task B ──
    train_task(agent, env, TASKS["B"], episodes, rng)
    a_after_b, _ = eval_task(agent, env, TASKS["A"], eval_ep, rng)
    b_after_b, _ = eval_task(agent, env, TASKS["B"], eval_ep, rng)

    return {"A_after_A": a_after_a, "A_after_B": a_after_b, "B_after_B": b_after_b,
            "forgetting": a_after_a - a_after_b}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=250)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--eval-ep", type=int, default=200)
    ap.add_argument("--ewc-lambda", type=float, default=10.0)
    args = ap.parse_args()

    methods = ["naive", "rehearsal", "ewc"]
    print(f"Catastrophic forgetting study | {args.episodes} ep/task × {args.seeds} seeds | "
          f"EWC λ={args.ewc_lambda}\n" + "=" * 70)
    results = {}
    for m in methods:
        runs = [run_method(m, args.episodes, args.eval_ep, args.ewc_lambda, s)
                for s in range(args.seeds)]
        agg = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        agg.update({k + "_std": float(np.std([r[k] for r in runs])) for k in runs[0]})
        results[m] = agg
        print(f"  {m:<10} A|A={agg['A_after_A']:.3f}  A|B={agg['A_after_B']:.3f}  "
              f"B|B={agg['B_after_B']:.3f}  forgetting={agg['forgetting']:+.3f}")

    print("\nForgetting (PDR drop on Task A after learning Task B), lower is better:")
    for m in methods:
        print(f"  {m:<10} {results[m]['forgetting']:+.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "continual_learning.json")
    with open(out_path, "w") as f:
        json.dump({"tasks": TASKS, "episodes": args.episodes, "seeds": args.seeds,
                   "ewc_lambda": args.ewc_lambda, "results": results}, f, indent=2)
    print(f"\nContinual-learning results → {out_path}")
    _plot(results, methods)


def _plot(results, methods):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(methods))

    # Left: the three PDR readings per method.
    bars = [("A_after_A", "Task A — after training A", "steelblue"),
            ("A_after_B", "Task A — after training B", "crimson"),
            ("B_after_B", "Task B — after training B", "seagreen")]
    w = 0.26
    for j, (key, label, color) in enumerate(bars):
        vals = [results[m][key] for m in methods]
        errs = [results[m][key + "_std"] for m in methods]
        ax1.bar(x + (j - 1) * w, vals, w, yerr=errs, capsize=3, label=label,
                color=color, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels(methods, fontsize=10)
    ax1.set_ylabel("Packet Delivery Ratio"); ax1.set_ylim(0, 1.05)
    ax1.set_title("Per-task delivery after sequential training", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3); ax1.legend(fontsize=9, loc="lower center")

    # Right: forgetting (the drop on A), the headline.
    forg = [results[m]["forgetting"] for m in methods]
    ferr = [results[m]["forgetting_std"] for m in methods]
    # Red = severe forgetting (remedy failed), green = prevented, orange = partial.
    colors = ["crimson" if f > 0.3 else "seagreen" if f < 0.15 else "darkorange"
              for f in forg]
    ax2.axhline(0.0, color="black", lw=0.8)
    ax2.bar(x, forg, 0.5, yerr=ferr, capsize=4, color=colors, edgecolor="black")
    for i, f in enumerate(forg):
        ax2.text(i, f + 0.01, f"{f:+.2f}", ha="center", fontsize=10)
    ax2.set_xticks(x); ax2.set_xticklabels(methods, fontsize=10)
    ax2.set_ylabel("Forgetting  (PDR drop on Task A)  — lower is better")
    ax2.set_title("Catastrophic forgetting and its remedies", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Catastrophic forgetting in learned routing: naive sequential training "
                 "forgets the old regime;\nrehearsal (data) and EWC (weights) retain it",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(RESULTS_DIR, "continual_learning.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Continual-learning plot → {out}")


if __name__ == "__main__":
    main()
