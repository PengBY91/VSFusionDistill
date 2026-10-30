"""Tests for §6.4 branch-correlation metrics (setup doc §4.5, row 5).

Constructed data with known answers:
  * independent misses -> phi≈0, ΔP_miss≈0
  * perfectly correlated misses -> phi=1, ΔP_miss=1−P(m_s=1)
  * hand-computed naive-Bayes fusion and ECE
"""

import numpy as np
import pytest

from aodet.evaluation.branch_correlation import (
    bootstrap_ci,
    conditional_miss_lift,
    expected_calibration_error,
    fuse_existence_naive_bayes,
    phi_coefficient,
    spearman_conf,
    summarize_matched_pairs,
)

RNG = np.random.default_rng(42)


def _independent_misses(n, p_c, p_s):
    m_c = (RNG.random(n) < p_c).astype(float)
    m_s = (RNG.random(n) < p_s).astype(float)
    return m_c, m_s


def test_phi_independent_close_to_zero():
    m_c, m_s = _independent_misses(200_000, 0.3, 0.4)
    assert abs(phi_coefficient(m_c, m_s)) < 0.01
    assert abs(conditional_miss_lift(m_c, m_s)) < 0.01


def test_phi_perfect_correlation():
    m = (RNG.random(10_000) < 0.25).astype(float)
    assert phi_coefficient(m, m) == pytest.approx(1.0)
    # ΔP_miss = P(m_s=1|m_c=1) − P(m_s=1) = 1 − P(m_s=1) (empirical rate)
    assert conditional_miss_lift(m, m) == pytest.approx(1.0 - m.mean())


def test_phi_perfect_anticorrelation():
    m_c = np.array([1, 1, 0, 0] * 100, dtype=float)
    m_s = 1.0 - m_c
    assert phi_coefficient(m_c, m_s) == pytest.approx(-1.0)
    # camera misses => sonar never misses
    assert conditional_miss_lift(m_c, m_s) == pytest.approx(-0.5)


def test_lift_matches_manual_computation():
    m_c = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    m_s = np.array([1, 0, 1, 0, 0, 1], dtype=float)
    # P(m_s=1) = 3/6; P(m_s=1|m_c=1) = 2/3 -> lift = 2/3 - 1/2 = 1/6
    assert conditional_miss_lift(m_c, m_s) == pytest.approx(1 / 6)


def test_spearman_perfect_and_zero():
    x = RNG.random(4000)
    assert spearman_conf(x, x) == pytest.approx(1.0)
    y = RNG.random(4000)
    # under independence |rho| ~ 1/sqrt(n); n=4000 gives ~0.016 std
    assert abs(spearman_conf(x, y)) < 0.08


def test_bootstrap_ci_brackets_estimate():
    m_c, m_s = _independent_misses(4000, 0.3, 0.4)
    lo, hi = bootstrap_ci(phi_coefficient, m_c, m_s, n_boot=500, seed=1)
    est = phi_coefficient(m_c, m_s)
    assert lo <= est <= hi
    assert hi - lo < 0.15  # tight for n=4000


def test_naive_bayes_fusion_hand_case():
    # pd=0.8 both, pfa=0.1 both, prior 0.5
    # P(e|1,1) = .64 / (.64 + .01) = 64/65
    f = fuse_existence_naive_bayes([1, 1], [1, 1], 0.8, 0.8, 0.1, 0.1)
    assert f[0] == pytest.approx(64 / 65)
    # P(e|0,0) = .04 / (.04 + .81) = 4/85
    f = fuse_existence_naive_bayes([0], [0], 0.8, 0.8, 0.1, 0.1)
    assert f[0] == pytest.approx(4 / 85)
    # P(e|1,0) = .16 / (.16 + .09) = 16/25
    f = fuse_existence_naive_bayes([1], [0], 0.8, 0.8, 0.1, 0.1)
    assert f[0] == pytest.approx(16 / 25)


def test_naive_bayes_fusion_vectorized_consistent():
    a = fuse_existence_naive_bayes([1, 0, 1, 0], [1, 0, 0, 1], 0.9, 0.7,
                                   0.05, 0.2)
    b = [
        fuse_existence_naive_bayes([dc], [ds], 0.9, 0.7, 0.05, 0.2)[0]
        for dc, ds in [(1, 1), (0, 0), (1, 0), (0, 1)]
    ]
    np.testing.assert_allclose(a, b)


def test_ece_hand_computed():
    # Two equal-width bins in [0,1): bin0 conf mean .3 acc 0; bin1 conf .8
    # acc 1. ECE with n_bins=2 = .5*|.3-0| + .5*|.8-1| = .15+.1 = .25
    prob = [0.2, 0.4, 0.7, 0.9]
    exists = [0, 0, 1, 1]
    assert expected_calibration_error(prob, exists, n_bins=2) == pytest.approx(0.25)
    # perfectly calibrated -> 0
    prob = [0.25] * 4 + [0.75] * 4
    exists = [0, 1, 0, 0, 1, 1, 1, 0]  # bin accs .25/.75 with 8 samples
    assert expected_calibration_error(prob, exists, n_bins=4) == pytest.approx(
        abs(0.25 - 0.25) * 0.5 + abs(0.75 - 0.75) * 0.5, abs=1e-9
    )


def test_ece_rejects_bad_prob():
    with pytest.raises(ValueError):
        expected_calibration_error([1.2], [1])
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [2])


def test_summarize_report_fields():
    m_c = np.array([1] * 30 + [0] * 70, dtype=float)
    m_s = m_c.copy()  # perfect correlation
    rep = summarize_matched_pairs(
        m_c, m_s, conf_c=m_c * 0.5 + 0.2, conf_s=m_s * 0.5 + 0.2,
        n_boot=200, seed=3,
    )
    row = rep.as_row()
    assert row["n_pairs"] == 100
    assert row["phi"] == pytest.approx(1.0)
    assert row["p_miss_camera"] == pytest.approx(0.3)
    assert row["delta_p_miss"] == pytest.approx(0.7)
    assert row["phi_ci95"][0] <= 1.0 <= row["phi_ci95"][1] + 1e-9
    assert row["spearman_conf"] == pytest.approx(1.0)


def test_mismatched_sizes_rejected():
    with pytest.raises(ValueError):
        phi_coefficient([1, 0], [1])
    with pytest.raises(ValueError):
        summarize_matched_pairs([1, 0], [1, 0], [0.5], [0.5, 0.6])
