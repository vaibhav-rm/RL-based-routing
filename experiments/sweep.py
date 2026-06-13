"""
Hyperparameter sweep over Rainbow DQN parameters.

Sweeps are essential in research to:
  1. Rule out that a baseline was simply mis-tuned (Henderson et al. 2018).
  2. Find the Pareto frontier between convergence speed and final performance.
  3. Quantify sensitivity to hyperparameters (robustness analysis).

We sweep the 3 most impactful Rainbow parameters:
  lr, per_alpha, n_step

Usage:
    python experiments/sweep.py [--episodes 300] [--seeds 3]

Output: results/sweep_results.json + results/sweep_heatmap.png
"""

import os, sys, json, time, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from network_rl.env.network_env import NetworkRoutingEnv, OBS_DIM
from network_rl.agents.rainbow_agent import RainbowAgent


SWEEP_GRID = {
    "lr":        [1e-3, 5e-4, 1e-4],
    "per_alpha": [0.4, 0.6, 0.8],
    "n_step":    [1, 3, 5],
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_single(lr, per_alpha, n_step, episodes, seed):
    np.random.seed(seed)
    env = NetworkRoutingEnv()
    obs, _ = env.reset(seed=seed)
    state_dim  = OBS_DIM
    action_dim = env.action_space.n

    agent = RainbowAgent(
        state_dim=state_dim, action_dim=action_dim,
        lr=lr, per_alpha=per_alpha, n_step=n_step,
        epsilon_decay=0.99, batch_size=64,
    )

    rewards = []
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

    # Return mean of last 20% (convergence metric)
    tail = rewards[int(episodes * 0.8):]
    return float(np.mean(tail))


def run_sweep(episodes: int = 300, n_seeds: int = 3):
    keys   = list(SWEEP_GRID.keys())
    values = list(SWEEP_GRID.values())
    combos = list(itertools.product(*values))

    results = []
    t0 = time.time()
    print(f"Sweeping {len(combos)} configs × {n_seeds} seeds = {len(combos)*n_seeds} runs")

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        run_scores = []
        for seed in range(n_seeds):
            score = train_single(
                lr=params["lr"],
                per_alpha=params["per_alpha"],
                n_step=params["n_step"],
                episodes=episodes,
                seed=seed,
            )
            run_scores.append(score)
        mean_score = float(np.mean(run_scores))
        std_score  = float(np.std(run_scores))
        results.append({**params, "mean": mean_score, "std": std_score})
        elapsed = time.time() - t0
        print(f"  [{i+1:2d}/{len(combos)}] lr={params['lr']:.0e}  "
              f"alpha={params['per_alpha']}  n={params['n_step']}  "
              f"→ {mean_score:+.1f} ± {std_score:.1f}   ({elapsed:.0f}s)")

    # Sort by mean performance
    results.sort(key=lambda x: x["mean"], reverse=True)

    # Save JSON
    out_json = os.path.join(RESULTS_DIR, "sweep_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTop 5 configs:")
    for r in results[:5]:
        print(f"  lr={r['lr']:.0e}  alpha={r['per_alpha']}  n_step={r['n_step']}"
              f"  mean={r['mean']:+.1f} ± {r['std']:.1f}")

    # ── Heatmap: lr × per_alpha for best n_step ───────────────────────────
    best_n  = results[0]["n_step"]
    lrs     = sorted(set(r["lr"]        for r in results))
    alphas  = sorted(set(r["per_alpha"] for r in results))
    matrix  = np.zeros((len(alphas), len(lrs)))

    for r in results:
        if r["n_step"] != best_n:
            continue
        i = alphas.index(r["per_alpha"])
        j = lrs.index(r["lr"])
        matrix[i, j] = r["mean"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(lrs)));    ax.set_xticklabels([f"{l:.0e}" for l in lrs])
    ax.set_yticks(range(len(alphas))); ax.set_yticklabels(alphas)
    ax.set_xlabel("Learning Rate"); ax.set_ylabel("PER α")
    ax.set_title(f"Mean Convergence Reward (n_step={best_n})")
    for i in range(len(alphas)):
        for j in range(len(lrs)):
            ax.text(j, i, f"{matrix[i,j]:.0f}", ha="center", va="center",
                    fontsize=9, color="black")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "sweep_heatmap.png"), dpi=150)
    print(f"Heatmap saved → {os.path.join(RESULTS_DIR, 'sweep_heatmap.png')}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seeds",    type=int, default=3)
    args = parser.parse_args()
    run_sweep(args.episodes, args.seeds)
