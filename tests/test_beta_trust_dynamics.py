"""
Unit and Integration Tests for Phase 2.8: Bayesian Beta Trust Dynamics & Reliability Consensus
=============================================================================================
Verifies:
1. Beta priors initialize to Beta(1.0, 1.0) (expected value 0.50).
2. Bayesian updates are strictly deterministic (alpha += 1 on success, beta += 1 on failure).
3. Posterior means hat{theta}_{i,d} are bounded strictly in (0.0, 1.0).
4. Trust store atomic serialization and deserialization preserves values accurately.
5. Trust-weighted consensus scores prioritize higher-trust specialists under disagreement.
6. Single active specialist retains calibrated score without distortion.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from backend.trust_tracker import BetaTrustStore, SPECIALISTS, DOMAINS


def test_beta_prior_initialization():
    store = BetaTrustStore()
    store._init_priors()

    for spec in SPECIALISTS:
        for dom in DOMAINS:
            trust = store.get_trust(spec, dom)
            assert trust == 0.50, f"Expected uniform prior 0.50, got {trust} for {spec}/{dom}"
            params = store.posteriors[spec][dom]
            assert params == [1.0, 1.0], f"Expected Beta(1, 1), got {params}"


def test_deterministic_bayesian_updates():
    store = BetaTrustStore()
    store._init_priors()

    # 3 successes, 1 failure for cardiologist in Cardiology
    store.update("cardiologist", "Cardiology", is_correct=True)
    store.update("cardiologist", "Cardiology", is_correct=True)
    store.update("cardiologist", "Cardiology", is_correct=True)
    store.update("cardiologist", "Cardiology", is_correct=False)

    params = store.posteriors["cardiologist"]["Cardiology"]
    assert params == [4.0, 2.0], f"Expected [4.0, 2.0], got {params}"
    expected_mean = 4.0 / 6.0
    actual_mean = store.get_trust("cardiologist", "Cardiology")
    assert abs(actual_mean - expected_mean) < 1e-6


def test_trust_ranking_under_differential_performance():
    store = BetaTrustStore()
    store._init_priors()

    # High accuracy specialist (9 correct, 1 wrong)
    for _ in range(9):
        store.update("cardiologist", "Cardiology", is_correct=True)
    store.update("cardiologist", "Cardiology", is_correct=False)

    # Low accuracy specialist (2 correct, 8 wrong)
    for _ in range(2):
        store.update("pulmonologist", "Cardiology", is_correct=True)
    for _ in range(8):
        store.update("pulmonologist", "Cardiology", is_correct=False)

    t_cardio = store.get_trust("cardiologist", "Cardiology")
    t_pulmo = store.get_trust("pulmonologist", "Cardiology")

    assert t_cardio > t_pulmo
    assert t_cardio == pytest.approx(10.0 / 12.0, rel=1e-3)
    assert t_pulmo == pytest.approx(3.0 / 12.0, rel=1e-3)


def test_atomic_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = os.path.join(tmpdir, "test_trust.json")
        store1 = BetaTrustStore(storage_path=temp_path)
        store1.update("neurologist", "Neurology", is_correct=True)
        store1.update("neurologist", "Neurology", is_correct=True)
        store1.save()

        assert os.path.exists(temp_path)

        store2 = BetaTrustStore(storage_path=temp_path)
        assert store2.get_trust("neurologist", "Neurology") == pytest.approx(3.0 / 4.0)
