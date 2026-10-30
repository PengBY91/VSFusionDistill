"""Branch error-correlation metrics (F2 draft §6.4).

The F2 draft defines four quantities, reported per degradation level on the
cross-modal matched pairs (same object has GT boxes in both modalities):

1. conditional miss lift  ΔP_miss = P(m_s=1 | m_c=1) − P(m_s=1)
2. phi coefficient of the two binary miss variables (+ bootstrap 95% CI)
3. Spearman correlation of the two branches' confidence scores
4. a downstream proxy: naive-Bayes (conditional-independence) fused
   existence probability, scored with ECE

Conventions: ``m_c``/``m_s`` are *miss* indicators (1 = the branch missed the
object at IoU 0.5 and the validation-chosen operating point); ``d_c``/``d_s``
are *detection* indicators (d = 1 − m on matched pairs, but the fusion proxy
also consumes false alarms, where "exists" is 0).

The naive-Bayes fusion follows the ProbEn-style product rule: with branch
likelihoods pD_m = P(d_m=1 | exists) and pFA_m = P(d_m=1 | absent), the fused
existence odds are the prior odds times the per-branch likelihood ratios.
Positive branch correlation makes P(d_c=1, d_s=1 | exists) larger than the
product pD_c·pD_s, so the CI-fused probability is over-confident — which is
exactly what the ECE is meant to expose (ProbEn only reports ranking metrics,
which are insensitive to monotone miscalibration).

Everything here is pure numpy/scipy so it can be unit-tested without the
detection stack, and re-run on saved prediction files without re-inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats

__all__ = [
    "phi_coefficient",
    "conditional_miss_lift",
    "spearman_conf",
    "bootstrap_ci",
    "fuse_existence_naive_bayes",
    "expected_calibration_error",
    "BranchCorrelationReport",
    "summarize_matched_pairs",
]


def _as_binary(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if not np.all((a == 0) | (a == 1)):
        raise ValueError(f"{name} must be binary 0/1")
    return a


def phi_coefficient(miss_c, miss_s) -> float:
    """Pearson/phi correlation of two binary miss variables."""
    a = _as_binary(miss_c, "miss_c")
    b = _as_binary(miss_s, "miss_s")
    if a.size != b.size:
        raise ValueError("size mismatch")
    if a.size < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def conditional_miss_lift(miss_c, miss_s) -> float:
    """ΔP_miss = P(m_s=1 | m_c=1) − P(m_s=1); 0 under independence."""
    a = _as_binary(miss_c, "miss_c")
    b = _as_binary(miss_s, "miss_s")
    p_s = b.mean()
    n_c1 = a.sum()
    if n_c1 == 0:
        return float("nan")
    p_s_given_c = b[a == 1].mean()
    return float(p_s_given_c - p_s)


def spearman_conf(conf_c, conf_s) -> float:
    """Spearman rank correlation of matched-pair confidence scores."""
    x = np.asarray(conf_c, dtype=np.float64).ravel()
    y = np.asarray(conf_s, dtype=np.float64).ravel()
    if x.size != y.size or x.size < 2:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def bootstrap_ci(
    fn,
    *arrays: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI of ``fn(*resampled_arrays)``.

    Arrays are resampled jointly (paired bootstrap over matched objects).
    """
    n = arrays[0].shape[0]
    rng = np.random.default_rng(seed)
    stats_ = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats_.append(fn(*(a[idx] for a in arrays)))
    lo, hi = np.percentile(stats_, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def fuse_existence_naive_bayes(
    d_c,
    d_s,
    pd_c: float,
    pd_s: float,
    pfa_c: float,
    pfa_s: float,
    prior: float = 0.5,
) -> np.ndarray:
    """P(exists | d_c, d_s) under the conditional-independence assumption.

    Binary decisions only; computed in log-odds space for stability.
    """
    d_c = _as_binary(d_c, "d_c")
    d_s = _as_binary(d_s, "d_s")
    eps = 1e-12
    pdc, pds = max(pd_c, eps), max(pd_s, eps)
    pfc, pfs = min(max(pfa_c, eps), 1 - eps), min(max(pfa_s, eps), 1 - eps)
    prior = min(max(prior, eps), 1 - eps)

    def log_lr(d: np.ndarray, pd: float, pfa: float) -> np.ndarray:
        # log P(d|exists)/P(d|absent) for the *observed* d (both outcomes
        # carry evidence — missing the d=0 term silently collapses the
        # posterior to the prior for non-detections)
        return np.where(d == 1, np.log(pd / pfa),
                        np.log((1 - pd) / (1 - pfa)))

    log_odds = (np.log(prior / (1 - prior))
                + log_lr(d_c, pdc, pfc) + log_lr(d_s, pds, pfs))
    return 1.0 / (1.0 + np.exp(-log_odds))


def expected_calibration_error(
    prob, exists, n_bins: int = 10
) -> float:
    """Standard equal-width-bin ECE: Σ_b (n_b/N) |acc_b − conf_b|."""
    p = np.asarray(prob, dtype=np.float64).ravel()
    e = _as_binary(exists, "exists")
    if p.size != e.size or p.size == 0:
        raise ValueError("prob/exists size mismatch or empty")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("prob outside [0,1]")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        ece += n_b / p.size * abs(e[mask].mean() - p[mask].mean())
    return float(ece)


@dataclass
class BranchCorrelationReport:
    """The §6.4 quantities for one (method, degradation level) cell."""

    n_pairs: int
    p_miss_camera: float
    p_miss_sonar: float
    delta_p_miss: float
    phi: float
    phi_ci: Tuple[float, float]
    delta_p_miss_ci: Tuple[float, float]
    spearman_conf: float

    def as_row(self) -> Dict[str, object]:
        return {
            "n_pairs": self.n_pairs,
            "p_miss_camera": self.p_miss_camera,
            "p_miss_sonar": self.p_miss_sonar,
            "delta_p_miss": self.delta_p_miss,
            "delta_p_miss_ci95": self.delta_p_miss_ci,
            "phi": self.phi,
            "phi_ci95": self.phi_ci,
            "spearman_conf": self.spearman_conf,
        }


def summarize_matched_pairs(
    miss_c,
    miss_s,
    conf_c,
    conf_s,
    n_boot: int = 2000,
    seed: int = 0,
) -> BranchCorrelationReport:
    """Quantities 1–3 of §6.4 on the matched pairs of one cell."""
    m_c = _as_binary(miss_c, "miss_c")
    m_s = _as_binary(miss_s, "miss_s")
    if m_c.size != m_s.size:
        raise ValueError("miss_c/miss_s size mismatch")
    for name, conf in (("conf_c", conf_c), ("conf_s", conf_s)):
        if np.asarray(conf).size != m_c.size:
            raise ValueError(f"{name} size mismatch with miss vectors")
    phi_ci = bootstrap_ci(
        lambda a, b: phi_coefficient(a, b), m_c, m_s,
        n_boot=n_boot, seed=seed,
    )
    lift_ci = bootstrap_ci(
        lambda a, b: conditional_miss_lift(a, b), m_c, m_s,
        n_boot=n_boot, seed=seed + 1,
    )
    return BranchCorrelationReport(
        n_pairs=int(m_c.size),
        p_miss_camera=float(m_c.mean()),
        p_miss_sonar=float(m_s.mean()),
        delta_p_miss=conditional_miss_lift(m_c, m_s),
        phi=phi_coefficient(m_c, m_s),
        phi_ci=phi_ci,
        delta_p_miss_ci=lift_ci,
        spearman_conf=spearman_conf(conf_c, conf_s),
    )
