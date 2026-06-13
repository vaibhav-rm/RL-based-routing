"""
Convergence and generalisation analysis.

Research value:
  1. Convergence speed — how many episodes to reach 90% of final performance?
     Faster convergence = better sample efficiency.
  2. Stability — does reward variance decrease after convergence?
     High variance after convergence suggests policy instability.
  3. Generalisation — does the trained policy transfer to unseen topologies?
     Only the GNN agent is expected to generalise; DQN/Rainbow are topology-specific.

Usage:
    python -c "from network_rl.analysis.convergence import *; run_all()"
"""

import os, sys, csv
import numpy as np
from typing import List, Tuple, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def convergence_point(
    rewards:    List[float],
    threshold:  float = 0.9,
    window:     int   = 20,
) -> Optional[int]:
    """
    Return the episode index at which the smoothed reward first reaches
    `threshold` fraction of the final (last-window) performance.
    Returns None if never reached.
    """
    if not rewards:
        return None
    smoothed  = np.convolve(rewards, np.ones(window) / window, mode="valid")
    final_val = smoothed[-max(1, len(smoothed)//10):].mean()
    target    = threshold * final_val
    for i, v in enumerate(smoothed):
        if v >= target:
            return i + window
    return None


def sample_efficiency_score(rewards: List[float]) -> float:
    """
    Area Under the Curve (AUC) normalised by final performance.
    Higher = learns the same performance in fewer episodes.
    """
    if not rewards or max(rewards) == min(rewards):
        return 0.0
    r = np.array(rewards)
    r_norm = (r - r.min()) / (r.max() - r.min() + 1e-9)
    return float(r_norm.mean())


def stability_score(rewards: List[float], tail_frac: float = 0.2) -> float:
    """
    Coefficient of variation in the tail of training.
    Lower = more stable convergence.
    """
    if not rewards:
        return float("inf")
    tail = rewards[int(len(rewards) * (1 - tail_frac)):]
    mean = np.mean(tail)
    std  = np.std(tail)
    return float(std / max(abs(mean), 1e-6))


def analyse_csv_logs(logs_dir: str) -> Dict[str, Dict]:
    """
    Load all CSV logs from logs_dir and compute convergence statistics.
    Returns {agent_name: {metric: value}} table.
    """
    agents = ["dqn", "rainbow", "gnn", "qrouting"]
    results = {}

    for agent in agents:
        all_conv   = []
        all_se     = []
        all_stable = []
        seed = 0
        while True:
            path = os.path.join(logs_dir, f"{agent}_seed{seed}.csv")
            if not os.path.exists(path):
                break
            rewards = []
            with open(path) as f:
                for row in csv.DictReader(f):
                    rewards.append(float(row["reward"]))
            if rewards:
                cp = convergence_point(rewards)
                all_conv.append(cp if cp is not None else len(rewards))
                all_se.append(sample_efficiency_score(rewards))
                all_stable.append(stability_score(rewards))
            seed += 1

        if all_conv:
            results[agent] = {
                "convergence_ep_mean":  float(np.mean(all_conv)),
                "convergence_ep_std":   float(np.std(all_conv)),
                "sample_efficiency":    float(np.mean(all_se)),
                "stability_cv":         float(np.mean(all_stable)),
                "n_seeds":              len(all_conv),
            }

    return results


def print_convergence_table(results: Dict[str, Dict]):
    print(f"\n{'Agent':<12} {'Conv.Ep (mean±std)':>22} {'Sample Eff.':>14} {'Stability CV':>14}")
    print("-" * 65)
    for agent, stats in results.items():
        conv_str = f"{stats['convergence_ep_mean']:.0f} ± {stats['convergence_ep_std']:.0f}"
        print(f"  {agent.upper():<10} {conv_str:>22} "
              f"{stats['sample_efficiency']:>14.4f} "
              f"{stats['stability_cv']:>14.4f}")
    print()
