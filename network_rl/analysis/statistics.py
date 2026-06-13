"""
Statistical analysis utilities for rigorous evaluation.

Research necessity:
  Single-run comparisons are unreliable due to stochastic environments
  and random weight initialisation. We follow best practices from:
    Henderson et al., "Deep Reinforcement Learning that Matters", AAAI 2018.
    Agarwal et al., "Deep RL at the Edge of the Statistical Precipice", NeurIPS 2021.

Methods:
  • Bootstrapped confidence intervals (non-parametric, no Gaussian assumption)
  • Welch's t-test (unequal variance — appropriate for comparing RL algorithms)
  • Cohen's d (effect size, not just significance)
  • Inter-quartile Mean (IQM) — robust alternative to mean for RL (Agarwal 2021)
"""

import numpy as np
from typing import List, Tuple, Dict
from scipy import stats


def bootstrap_ci(
    data:    List[float],
    n_boot:  int   = 10_000,
    ci:      float = 0.95,
    stat_fn  = np.mean,
) -> Tuple[float, float, float]:
    """
    Non-parametric bootstrap confidence interval.
    Returns (point_estimate, lower_bound, upper_bound).
    """
    data   = np.array(data)
    point  = stat_fn(data)
    boot   = np.array([stat_fn(np.random.choice(data, len(data), replace=True))
                       for _ in range(n_boot)])
    alpha  = 1.0 - ci
    lo, hi = np.percentile(boot, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return float(point), float(lo), float(hi)


def welch_ttest(a: List[float], b: List[float]) -> Dict:
    """
    Welch's t-test (does not assume equal variance).
    Returns dict with t-statistic, p-value, and interpretation.
    """
    a, b = np.array(a), np.array(b)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {
        "t_stat":     float(t),
        "p_value":    float(p),
        "significant": bool(p < 0.05),
        "n_a": len(a),
        "n_b": len(b),
    }


def cohens_d(a: List[float], b: List[float]) -> float:
    """
    Effect size (Cohen's d).
    Conventions: small=0.2, medium=0.5, large=0.8
    """
    a, b   = np.array(a), np.array(b)
    pooled = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2.0)
    if pooled < 1e-9:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def iqm(data: List[float]) -> float:
    """
    Inter-quartile Mean — robust estimator recommended by Agarwal et al. 2021.
    Trims bottom and top 25% before computing mean.
    """
    data = np.array(data)
    q25, q75 = np.percentile(data, [25, 75])
    trimmed  = data[(data >= q25) & (data <= q75)]
    return float(np.mean(trimmed)) if len(trimmed) > 0 else float(np.mean(data))


def summarise_runs(
    metric_runs: Dict[str, List[List[float]]],
    metric_name: str = "reward",
) -> Dict[str, Dict]:
    """
    Given {algorithm_name: [run1_episodes, run2_episodes, ...]} where
    each run is a list of per-episode values, compute:
      mean, std, IQM, bootstrap CI (across final 20% of training).

    Returns nested dict suitable for printing/plotting.
    """
    out = {}
    for alg, runs in metric_runs.items():
        # Use final 20% of each run (convergence region)
        final_vals = []
        for run in runs:
            if run:
                tail = run[max(0, int(len(run) * 0.8)):]
                final_vals.extend(tail)

        if not final_vals:
            continue

        est, lo, hi = bootstrap_ci(final_vals)
        out[alg] = {
            "mean":  float(np.mean(final_vals)),
            "std":   float(np.std(final_vals)),
            "iqm":   iqm(final_vals),
            "ci_lo": lo,
            "ci_hi": hi,
            "n_samples": len(final_vals),
        }

    # Pairwise significance vs first algorithm
    algs = list(out.keys())
    if len(algs) >= 2:
        base_alg = algs[0]
        base_runs = [metric_runs[base_alg][i] for i in range(len(metric_runs[base_alg]))]
        base_vals = []
        for run in base_runs:
            if run:
                tail = run[max(0, int(len(run) * 0.8)):]
                base_vals.extend(tail)
        for alg in algs[1:]:
            cmp_runs = metric_runs[alg]
            cmp_vals = []
            for run in cmp_runs:
                if run:
                    tail = run[max(0, int(len(run) * 0.8)):]
                    cmp_vals.extend(tail)
            if base_vals and cmp_vals:
                out[alg]["vs_base_ttest"] = welch_ttest(base_vals, cmp_vals)
                out[alg]["vs_base_cohens_d"] = cohens_d(base_vals, cmp_vals)

    return out


def print_comparison_table(summary: Dict[str, Dict], metric_name: str = "reward"):
    """Pretty-print a research-style comparison table."""
    print(f"\n{'Algorithm':<15} {'Mean':>10} {'Std':>8} {'IQM':>8} "
          f"{'95% CI':>20}  {'p-value':>10}  {'Cohen d':>8}")
    print("-" * 85)
    for alg, stats_d in summary.items():
        ci = f"[{stats_d['ci_lo']:+.2f}, {stats_d['ci_hi']:+.2f}]"
        p  = stats_d.get("vs_base_ttest", {}).get("p_value", float("nan"))
        d  = stats_d.get("vs_base_cohens_d", float("nan"))
        p_str = f"{p:.4f}" if not np.isnan(p) else "   —"
        d_str = f"{d:+.2f}" if not np.isnan(d) else "   —"
        print(f"{alg:<15} {stats_d['mean']:>10.3f} {stats_d['std']:>8.3f} "
              f"{stats_d['iqm']:>8.3f} {ci:>20}  {p_str:>10}  {d_str:>8}")
    print()
