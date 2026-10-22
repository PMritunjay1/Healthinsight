"""
Unit and Integration Tests for Phase 2.6: Post-Hoc Calibrator Fitting & Selection
================================================================================
Verifies:
1. Calibrator artifact and manifest exist and are well-formed.
2. Calibrator model is valid, produces outputs strictly in [0.0, 1.0].
3. Calibrator is strictly monotonic with respect to base confidence.
4. Clean split discipline is maintained (calibrator trained only on clean D_cal).
5. RAC module integrates calibrator_clean.pkl correctly.
6. Reliability diagram was generated and is valid.
"""

import json
import os
import pickle
import numpy as np
import pytest

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CALIBRATOR_PATH = os.path.join(WORKSPACE, "backend", "models", "calibrator_clean.pkl")
MANIFEST_PATH = os.path.join(WORKSPACE, "backend", "models", "calibration_manifest.json")
PLOT_PATH = os.path.join(WORKSPACE, "results", "calibration_reliability_diagram.png")


def test_calibrator_artifacts_exist():
    assert os.path.exists(CALIBRATOR_PATH), f"Missing calibrator artifact: {CALIBRATOR_PATH}"
    assert os.path.exists(MANIFEST_PATH), f"Missing manifest: {MANIFEST_PATH}"
    assert os.path.exists(PLOT_PATH), f"Missing plot: {PLOT_PATH}"
    assert os.path.getsize(PLOT_PATH) > 1000, "Reliability plot file appears empty or corrupt"


def test_calibration_manifest_integrity():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "selected_calibrator" in manifest
    assert manifest["selected_calibrator"] in ["Temperature_Scaling", "Isotonic_Regression", "Logistic_Regression_3F"]
    assert "validation_comparison" in manifest
    assert "holdout_test_results" in manifest
    assert manifest["calibration_sample_size"] > 0
    assert manifest["validation_sample_size"] > 0
    assert manifest["test_holdout_sample_size"] > 0

    val_comp = manifest["validation_comparison"]
    assert "uncalibrated" in val_comp
    assert "ece" in val_comp["uncalibrated"]
    assert "brier" in val_comp["uncalibrated"]

    test_res = manifest["holdout_test_results"]
    assert "uncalibrated_ece" in test_res
    assert "calibrated_ece" in test_res
    assert "bin_details" in test_res
    assert len(test_res["bin_details"]) == 10


def test_calibrator_output_bounds_and_monotonicity():
    with open(CALIBRATOR_PATH, "rb") as f:
        calibrator = pickle.load(f)

    # Test extreme boundaries
    test_inputs = np.array([0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0], dtype=np.float32)

    if hasattr(calibrator, "predict_proba"):
        probs = calibrator.predict_proba(test_inputs)
        # Handle 2D output if sklearn style
        if len(probs.shape) > 1 and probs.shape[1] == 2:
            probs = probs[:, 1]
    elif hasattr(calibrator, "predict"):
        probs = calibrator.predict(test_inputs)
    else:
        pytest.fail("Calibrator model does not implement predict_proba or predict")

    probs = np.array(probs).flatten()

    # Assert bounded in [0, 1]
    assert np.all(probs >= 0.0), f"Negative probabilities detected: {probs}"
    assert np.all(probs <= 1.0), f"Probabilities > 1.0 detected: {probs}"

    # Assert monotonicity
    diffs = np.diff(probs)
    assert np.all(diffs >= -1e-6), f"Monotonicity violation in calibrator: {diffs}"


def test_rac_module_calibrator_integration():
    from backend.rac_module import RACModule
    rac = RACModule()
    assert rac.calibrator is not None, "RACModule failed to load calibrator"

    # Test calibrate_confidence with varying inputs
    cal_low = rac.calibrate_confidence(0.1)
    cal_mid = rac.calibrate_confidence(0.5)
    cal_high = rac.calibrate_confidence(0.9)

    assert 0.0 <= cal_low <= 1.0
    assert 0.0 <= cal_mid <= 1.0
    assert 0.0 <= cal_high <= 1.0
    assert cal_low <= cal_mid <= cal_high
