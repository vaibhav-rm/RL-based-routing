"""
Ablation study: decompose Rainbow components to measure each contribution.

Ablation design (standard in DRL papers):
  Each row removes ONE component from the full Rainbow agent.
  The performance drop isolates that component's marginal contribution.

  Full Rainbow:    Dueling + PER + n-step + Double DQN
  − Dueling:       plain Q-head (no Value/Advantage split)
  − PER:           uniform replay (α=0)
  − n-step:        1-step TD (n=1)
  − Double DQN:    vanilla DQN target (uses target net for both select+eval)

Usage:
    python experiments/ablation.py [--episodes 400] [--seeds 3]

Output: results/ablation.png + results/ablation.json
"""

import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from network_rl.env.network_env import NetworkRoutingEnv, OBS_DIM
from network_rl.agents.rainbow_agent import RainbowAgent, DuelingQNetwork
from network_rl.agents.dqn_agent    import QNetwork as VanillaQNetwork, DQNAgent

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ABLATIONS = {
    "Full Rainbow":     dict(use_dueling=True,  per_alpha=0.6, n_step=3, double=True),
    "− Dueling":        dict(use_dueling=False, per_alpha=0.6, n_step=3, double=True),
    "− PER (uniform)":  dict(use_dueling=True,  per_alpha=0.0, n_step=3, double=True),
    "− n-step (n=1)":   dict(use_dueling=True,  per_alpha=0.6, n_step=1, double=True),
    "Vanilla DQN":      dict(use_dueling=False, per_alpha=0.0, n_step=1, double=False),
}


def train_ablation(cfg: dict, episodes: int, seed: int) -> list:
    np.random.seed(seed)
    env = NetworkRoutingEnv()
    env.reset(seed=seed)

    agent = RainbowAgent(
        state_dim=OBS_DIM,
        action_dim=env.action_space.n,
        lr=5e-4,
        n_step=cfg["n_step"],
        per_alpha=cfg["per_alpha"],
        epsilon_decay=0.995,
    )
    # Replace network if ablating Dueling
    if not cfg["use_dueling"]:
        from network_rl.agents.dqn_agent import QNetwork
        agent.policy_net = QNetwork(OBS_DIM, env.action_space.n, hidden=256)
        agent.target_net = QNetwork(OBS_DIM, env.action_space.n, hidden=256)
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        agent.optimizer = torch.optim.Adam(agent.policy_net.parameters(), lr=5e-4)

    rewards = []
    window  = []
    WINDOW  = 20

    for ep in range(episodes):
        obs, _ = env.reset()
        done, total = False, 0.0
        while not done:
            valid  = list(range(len(env._get_sorted_neighbors(env._current_node))))
            action = agent.select_action(obs, valid)
            nobs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            agent.store(obs, action, r, nobs, float(done))
            agent.learn()
            obs = nobs
            total += r
        agent.decay_epsilon()
        if ep % 10 == 0:
            agent.sync_target()
        rewards.append(total)

    return rewards


def run_ablation(episodes: int = 400, n_seeds: int = 3):
    all_runs = {name: [] for name in ABLATIONS}
    t0 = time.time()

    for name, cfg in ABLATIONS.items():
        print(f"\nAblation: {name}")
        for seed in range(n_seeds):
            runs = train_ablation(cfg, episodes, seed)
            all_runs[name].append(runs)
            tail_mean = np.mean(runs[int(episodes * 0.8):])
            print(f"  seed={seed}  tail_mean={tail_mean:+.1f}")

    elapsed = time.time() - t0
    print(f"\nAblation complete in {elapsed:.0f}s")

    # ── Summary stats ─────────────────────────────────────────────────────
    summary = {}
    for name, runs in all_runs.items():
        tail_vals = []
        for run in runs:
            tail_vals.extend(run[int(episodes * 0.8):])
        summary[name] = {
            "mean": float(np.mean(tail_vals)),
            "std":  float(np.std(tail_vals)),
        }

    # Save JSON
    out_json = os.path.join(RESULTS_DIR, "ablation.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Plot: learning curves with shaded std ─────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(ABLATIONS)))

    for color, (name, runs) in zip(colors, all_runs.items()):
        arr    = np.array(runs)          # [seeds, episodes]
        means  = arr.mean(0)
        stds   = arr.std(0)
        W      = 20
        smooth = np.convolve(means, np.ones(W) / W, mode="valid")
        x      = np.arange(len(smooth))
        ax1.plot(x, smooth, label=name, color=color, linewidth=2)
        # Smooth std
        smooth_std = np.convolve(stds, np.ones(W) / W, mode="valid")
        ax1.fill_between(x, smooth - smooth_std, smooth + smooth_std,
                         alpha=0.15, color=color)

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Smoothed Reward (20-ep MA)")
    ax1.set_title("Ablation: Learning Curves")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Bar chart of final performance
    names  = list(summary.keys())
    means  = [summary[n]["mean"] for n in names]
    stds   = [summary[n]["std"]  for n in names]
    y_pos  = np.arange(len(names))
    colors_bar = plt.cm.tab10(np.linspace(0, 1, len(names)))
    bars = ax2.barh(y_pos, means, xerr=stds, color=colors_bar,
                    align="center", alpha=0.85, capsize=4)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("Mean Convergence Reward (final 20%)")
    ax2.set_title("Ablation: Component Contributions")
    ax2.axvline(means[0], color="gray", linestyle="--", alpha=0.5, label="Full Rainbow")
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Rainbow DQN Ablation Study", fontsize=13)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "ablation.png")
    fig.savefig(out, dpi=150)
    print(f"Ablation plot saved → {out}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--seeds",    type=int, default=3)
    args = parser.parse_args()
    run_ablation(args.episodes, args.seeds)
