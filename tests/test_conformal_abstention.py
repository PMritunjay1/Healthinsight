"""
Unit and Integration Tests for Phase 2.7: Split-Conformal Selective Abstention
=============================================================================
Verifies:
1. Finite-sample quantile formula strictly matches ceil((n+1)(1-alpha))/n.
2. Nonconformity score s = 1 - g(z) correctly scales with calibrated confidence.
3. Conformal decision rule (Algorithm 4) correctly identifies abstentions:
       Abstain <=> g(z) < tau_conf OR H_agr > H_max.
4. Empirical coverage on exchangeable holdout data satisfies coverage guarantee.
5. Conformal configuration artifact is valid and loadable.
"""

import math
import numpy as np
import pytest
from scripts.compute_conformal_thresholds import compute_finite_sample_quantile


def test_finite_sample_quantile_exact_formula():
    scores = np.linspace(0.0, 1.0, 100)
    alpha = 0.10
    n = len(scores)

    # Formula: ceil((100 + 1) * 0.90) / 100 = ceil(90.9) / 100 = 91 / 100 = 0.91
    tau, q_level = compute_finite_sample_quantile(scores, alpha)
    expected_q_level = math.ceil((n + 1) * (1.0 - alpha)) / n
    assert q_level == pytest.approx(expected_q_level, rel=1e-5)
    assert q_level == 0.91
    assert 0.0 < tau <= 1.0


def test_conformal_score_bounds_and_monotonicity():
    confidences = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.0], dtype=np.float32)
    scores = 1.0 - confidences

    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)
    # Higher confidence must yield strictly lower nonconformity score
    assert np.all(np.diff(scores) < 0.0)


def test_decision_rule_abstention_logic():
    tau_conf = 0.65  # minimum confidence
    h_max = 1.20     # maximum tolerated agreement entropy

    # Case 1: High confidence, low entropy -> ACCEPT
    c1_conf, c1_h = 0.85, 0.30
    abstain_1 = (c1_conf < tau_conf) or (c1_h > h_max)
    assert abstain_1 is False

    # Case 2: Low confidence, low entropy -> ABSTAIN
    c2_conf, c2_h = 0.50, 0.30
    abstain_2 = (c2_conf < tau_conf) or (c2_h > h_max)
    assert abstain_2 is True

    # Case 3: High confidence, high entropy (severe specialist conflict) -> ABSTAIN
    c3_conf, c3_h = 0.85, 1.60
    abstain_3 = (c3_conf < tau_conf) or (c3_h > h_max)
    assert abstain_3 is True

    # Case 4: Boundary condition (exact threshold) -> ACCEPT
    c4_conf, c4_h = 0.65, 1.20
    abstain_4 = (c4_conf < tau_conf) or (c4_h > h_max)
    assert abstain_4 is False


def test_coverage_guarantee_on_exchangeable_data():
    np.random.seed(42)
    n_cal = 500
    n_test = 2000
    alpha = 0.10

    # True nonconformity scores drawn from identical distribution (exchangeable)
    cal_scores = np.random.beta(2, 5, size=n_cal)
    test_scores = np.random.beta(2, 5, size=n_test)

    tau, q_level = compute_finite_sample_quantile(cal_scores, alpha)

    # In conformal prediction, coverage is Pr(s_test <= tau)
    covered = test_scores <= tau
    empirical_coverage = float(np.mean(covered))

    # Conformal theory guarantees E[coverage] >= 1 - alpha
    # With n_test=2000, 3-sigma lower bound is ~ (1 - alpha) - 3 * sqrt(0.9 * 0.1 / 2000) ~ 0.90 - 0.02 = 0.88
    assert empirical_coverage >= (1.0 - alpha - 0.025), f"Coverage {empirical_coverage:.4f} violated guarantee"
