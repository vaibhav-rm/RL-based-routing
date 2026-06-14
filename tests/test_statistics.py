"""
Tests for the statistical-analysis utilities.

Beyond basic correctness these guard a quality concern the user raised: that
effect-size / significance stats must produce *real* output, not a degenerate
"Cohen's d = +0.00 for every comparison" that signals a broken computation.
"""

import numpy as np
import pytest

from network_rl.analysis.statistics import (
    bootstrap_ci,
    welch_ttest,
    cohens_d,
    iqm,
    summarise_runs,
    jain_fairness_index,
    pareto_front,
)


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    data = list(rng.normal(5.0, 1.0, 200))
    point, lo, hi = bootstrap_ci(data, n_boot=1000)
    assert lo <= point <= hi
    assert abs(point - 5.0) < 0.5


def test_welch_detects_clear_difference():
    a = list(np.full(30, 10.0) + np.random.default_rng(1).normal(0, 0.1, 30))
    b = list(np.full(30, 1.0) + np.random.default_rng(2).normal(0, 0.1, 30))
    res = welch_ttest(a, b)
    assert res["significant"]
    assert res["p_value"] < 0.05


def test_welch_no_difference_is_not_significant():
    rng = np.random.default_rng(3)
    a = list(rng.normal(0, 1, 100))
    b = list(rng.normal(0, 1, 100))
    res = welch_ttest(a, b)
    assert not res["significant"]


def test_cohens_d_is_nonzero_and_signed_for_real_difference():
    # The degenerate-output bug produced d=+0.00 everywhere; a genuine gap must
    # yield a large, correctly-signed effect size.
    a = [10.0, 10.1, 9.9, 10.2, 9.8]
    b = [1.0, 1.1, 0.9, 1.2, 0.8]
    d = cohens_d(a, b)
    assert d > 2.0            # large effect
    assert cohens_d(b, a) == pytest.approx(-d)  # sign flips with order


def test_cohens_d_zero_when_identical():
    a = [1.0, 2.0, 3.0]
    assert cohens_d(a, a) == 0.0


def test_iqm_trims_outliers():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 1000]  # one wild outlier
    assert iqm(data) < np.mean(data)


def test_jain_perfectly_fair_is_one():
    assert jain_fairness_index([0.7, 0.7, 0.7, 0.7]) == pytest.approx(1.0)


def test_jain_maximally_unfair_approaches_one_over_n():
    # One flow served, n-1 starved → J = 1/n.
    vec = [1.0] + [0.0] * 9
    assert jain_fairness_index(vec) == pytest.approx(1.0 / 10)


def test_jain_is_between_inv_n_and_one():
    rng = np.random.default_rng(0)
    vec = list(rng.random(20))
    j = jain_fairness_index(vec)
    assert 1.0 / 20 <= j <= 1.0


def test_jain_uneven_is_less_fair_than_even():
    even = jain_fairness_index([0.5, 0.5, 0.5, 0.5])
    uneven = jain_fairness_index([0.9, 0.9, 0.1, 0.1])
    assert uneven < even == pytest.approx(1.0)


def test_jain_all_zero_is_defined():
    # No flow served anywhere: denominator guard returns a finite value, not NaN.
    assert jain_fairness_index([0.0, 0.0, 0.0]) == 1.0


def test_pareto_front_drops_dominated_points():
    # Lower is better on both axes. (2,2) dominates (3,3); (1,5) and (5,1) are
    # each non-dominated trade-offs.
    pts = [[1, 5], [5, 1], [2, 2], [3, 3]]
    front = set(pareto_front(pts))
    assert front == {0, 1, 2}           # index 3 (3,3) is dominated by (2,2)


def test_pareto_front_all_nondominated_when_strict_tradeoff():
    # A clean decreasing trade-off: every point is on the frontier.
    pts = [[1, 4], [2, 3], [3, 2], [4, 1]]
    assert set(pareto_front(pts)) == {0, 1, 2, 3}


def test_pareto_front_single_winner_dominates_all():
    pts = [[1, 1], [2, 2], [3, 3]]
    assert set(pareto_front(pts)) == {0}


def test_summarise_runs_emits_real_pairwise_stats():
    # Two algorithms with clearly different reward levels: the summary must carry
    # a significant t-test and a non-zero Cohen's d for the second vs the first.
    rng = np.random.default_rng(7)
    runs = {
        "weak":   [list(rng.normal(1.0, 0.2, 50)) for _ in range(3)],
        "strong": [list(rng.normal(8.0, 0.2, 50)) for _ in range(3)],
    }
    summary = summarise_runs(runs)
    assert set(summary) == {"weak", "strong"}
    assert "vs_base_cohens_d" in summary["strong"]
    assert abs(summary["strong"]["vs_base_cohens_d"]) > 0.8
    assert summary["strong"]["vs_base_ttest"]["significant"]
    assert summary["strong"]["mean"] > summary["weak"]["mean"]
