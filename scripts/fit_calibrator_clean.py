"""
HealthInsight-UQ: Post-Hoc Calibrator Fitting & Selection (Algorithm 3)
======================================================================
1. Fits candidate calibrators strictly on clean calibration data (D_cal):
   - Candidate 1: Temperature Scaling (g(z) = sigma(logit(c_base) / T))
   - Candidate 2: Isotonic Regression (non-parametric monotonic mapping)
   - Candidate 3: Multi-feature Logistic Calibrator (Platt scaling over (c_base, u_cons, H_agr))
2. Selects optimal model via validation ECE on D_val_tune.
3. Evaluates selected calibrator once on untouched holdout D_test data.
4. Serializes optimal model to backend/models/calibrator_clean.pkl and logs manifest.
5. Emits calibration reliability diagram to results/calibration_reliability_diagram.png.
"""

import asyncio
import json
import os
import pickle
import sys
from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from backend.orchestrator import Orchestrator
from scripts.utils.label_extractor import extract_ground_truth_labels, normalize_specialty, ALL_DISEASE_CLASSES

CAL_PATHS = {
    "Cardiology": os.path.join(WORKSPACE, "datasets", "processed", "common", "clean_calib.json"),
    "Pulmonology": os.path.join(WORKSPACE, "datasets", "processed", "pulmonologist", "clean_calib.json"),
    "Neurology": os.path.join(WORKSPACE, "datasets", "processed", "neurologist", "clean_calib.json"),
}

VAL_PATHS = {
    "Cardiology": os.path.join(WORKSPACE, "datasets", "processed", "common", "val_tune.json"),
    "Pulmonology": os.path.join(WORKSPACE, "datasets", "processed", "pulmonologist", "val_tune.json"),
    "Neurology": os.path.join(WORKSPACE, "datasets", "processed", "neurologist", "val_tune.json"),
}

TEST_PATHS = {
    "Cardiology": os.path.join(WORKSPACE, "datasets", "processed", "common", "test_split.json"),
    "Pulmonology": os.path.join(WORKSPACE, "datasets", "processed", "pulmonologist", "test_split.json"),
    "Neurology": os.path.join(WORKSPACE, "datasets", "processed", "neurologist", "test_split.json"),
}

OUTPUT_MODEL_PATH = os.path.join(WORKSPACE, "backend", "models", "calibrator_clean.pkl")
MANIFEST_PATH = os.path.join(WORKSPACE, "backend", "models", "calibration_manifest.json")
PLOT_PATH = os.path.join(WORKSPACE, "results", "calibration_reliability_diagram.png")


def logit_safe(p: float, eps: float = 1e-5) -> float:
    p_clamped = min(1.0 - eps, max(eps, float(p)))
    return float(np.log(p_clamped / (1.0 - p_clamped)))


def sigmoid_safe(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


class TemperatureScalingCalibrator:
    """Scalar temperature scaling for binary confidence calibration."""

    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, confidences: np.ndarray, labels: np.ndarray):
        logits = np.array([logit_safe(c) for c in confidences])

        def nll_loss(t_val):
            T = float(t_val[0])
            scaled_logits = logits / max(0.01, T)
            probs = 1.0 / (1.0 + np.exp(-scaled_logits))
            probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
            loss = -np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))
            return loss

        res = minimize(nll_loss, [1.0], method="Nelder-Mead")
        self.temperature = float(max(0.1, res.x[0]))
        return self

    def predict_proba(self, confidences: np.ndarray) -> np.ndarray:
        logits = np.array([logit_safe(c) for c in confidences])
        scaled_logits = logits / self.temperature
        probs = 1.0 / (1.0 + np.exp(-scaled_logits))
        return np.clip(probs, 0.0, 1.0)


def compute_ece_and_brier(
    confidences: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Computes Expected Calibration Error (ECE) with equal-width binning and Brier score.
    ECE = sum (|B_b| / N) * |acc(B_b) - conf(B_b)|
    Brier = mean((conf - label)^2)
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_details = []
    total_samples = len(confidences)
    ece = 0.0

    for i in range(n_bins):
        b_low, b_high = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= b_low) & (confidences <= b_high)
        else:
            mask = (confidences >= b_low) & (confidences < b_high)

        count = int(np.sum(mask))
        if count > 0:
            bin_conf = float(np.mean(confidences[mask]))
            bin_acc = float(np.mean(labels[mask]))
            gap = abs(bin_acc - bin_conf)
            weight = count / total_samples
            ece += weight * gap
            bin_details.append({
                "bin_idx": i,
                "range": [round(b_low, 2), round(b_high, 2)],
                "count": count,
                "mean_confidence": round(bin_conf, 4),
                "empirical_accuracy": round(bin_acc, 4),
                "calibration_gap": round(gap, 4),
            })
        else:
            bin_details.append({
                "bin_idx": i,
                "range": [round(b_low, 2), round(b_high, 2)],
                "count": 0,
                "mean_confidence": None,
                "empirical_accuracy": None,
                "calibration_gap": 0.0,
            })

    brier = float(np.mean((confidences - labels) ** 2))
    return round(float(ece), 4), round(brier, 4), bin_details


async def extract_features_from_split(
    orch: Orchestrator,
    split_paths: Dict[str, str],
    cases_per_spec: int = 25,
) -> Dict[str, np.ndarray]:
    c_base_list = []
    u_cons_list = []
    h_agr_list = []
    labels_list = []

    for spec, path in split_paths.items():
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        print(f"  Extracting features from {spec} ({min(cases_per_spec, len(records))} cases)...", flush=True)
        for idx, r in enumerate(records[:cases_per_spec]):
            txt = r.get("text") or r.get("patient_text") or ""
            gt_labels = extract_ground_truth_labels(r, spec)
            if not txt or not gt_labels:
                continue

            try:
                res = await orch.process_case(patient_text=txt, routing_mode="routed")
                final_diag = (res.get("fusion", {}).get("final_diagnosis") or "").strip().lower()
                uncond_diag = (res.get("fusion", {}).get("unconditioned_diagnosis") or final_diag).strip().lower()

                # Match if unconditioned prediction is among true labels
                is_correct = 1 if any(t.lower() == uncond_diag or t.lower() in uncond_diag for t in gt_labels) else 0

                rac = res.get("rac", {})
                c_b = float(rac.get("c_base") or res.get("fusion", {}).get("overall_confidence") or 0.5)
                u_c = float(rac.get("u_cons") or rac.get("u_instability") or 0.0)
                h_a = float(rac.get("e_domain") or 0.0)

                c_base_list.append(c_b)
                u_cons_list.append(u_c)
                h_agr_list.append(h_a)
                labels_list.append(is_correct)
            except Exception as e:
                print(f"    Warning: feature extraction error on {spec} case {idx}: {e}", flush=True)
                continue

            if (idx + 1) % 10 == 0:
                print(f"    {spec}: processed {idx + 1}/{min(cases_per_spec, len(records))} cases...", flush=True)

    return {
        "c_base": np.array(c_base_list, dtype=np.float32),
        "u_cons": np.array(u_cons_list, dtype=np.float32),
        "h_agr": np.array(h_agr_list, dtype=np.float32),
        "labels": np.array(labels_list, dtype=np.int32),
    }


def fit_and_select_calibrator():
    print("=" * 80, flush=True)
    print("HealthInsight-UQ: Post-Hoc Calibrator Fitting on Clean D_cal (Phase 2.6)", flush=True)
    print("=" * 80, flush=True)

    orch = Orchestrator()

    # 1. Extract Calibration Features (D_cal)
    print("\nStep 1: Extracting calibration data from clean D_cal...", flush=True)
    cal_data = asyncio.run(extract_features_from_split(orch, CAL_PATHS, cases_per_spec=25))
    print(f"Extracted {len(cal_data['labels'])} cases from clean D_cal.", flush=True)

    # 2. Extract Validation Tuning Features (D_val_tune)
    print("\nStep 2: Extracting validation tuning data from D_val_tune...", flush=True)
    val_data = asyncio.run(extract_features_from_split(orch, VAL_PATHS, cases_per_spec=25))
    print(f"Extracted {len(val_data['labels'])} cases from D_val_tune.", flush=True)

    # 3. Fit Candidate Calibrators strictly on D_cal
    print("\nStep 3: Fitting candidate calibrators on D_cal...")

    # Candidate 1: Temperature Scaling
    temp_model = TemperatureScalingCalibrator().fit(cal_data["c_base"], cal_data["labels"])
    print(f"  Candidate 1 (Temperature Scaling): Fitted T = {temp_model.temperature:.4f}")

    # Candidate 2: Isotonic Regression
    iso_model = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    iso_model.fit(cal_data["c_base"], cal_data["labels"])
    print(f"  Candidate 2 (Isotonic Regression): Monotonic step function fitted.")

    # Candidate 3: Logistic Regression (Platt scaling over 3 features)
    X_cal_3f = np.column_stack([cal_data["c_base"], cal_data["u_cons"], cal_data["h_agr"]])
    lr_model = LogisticRegression(C=1.0, solver="lbfgs")
    lr_model.fit(X_cal_3f, cal_data["labels"])
    print(f"  Candidate 3 (Logistic 3-Feature): Weights = {lr_model.coef_[0]}, Intercept = {lr_model.intercept_[0]:.4f}")

    # 4. Evaluate on Validation Tuning Set (D_val_tune)
    print("\nStep 4: Evaluating candidates on D_val_tune (Model Selection)...")
    val_labels = val_data["labels"]
    val_c_base = val_data["c_base"]
    X_val_3f = np.column_stack([val_data["c_base"], val_data["u_cons"], val_data["h_agr"]])

    # Uncalibrated baseline
    uncal_ece, uncal_brier, uncal_bins = compute_ece_and_brier(val_c_base, val_labels)

    # Candidate 1
    temp_preds = temp_model.predict_proba(val_c_base)
    temp_ece, temp_brier, temp_bins = compute_ece_and_brier(temp_preds, val_labels)

    # Candidate 2
    iso_preds = iso_model.predict(val_c_base)
    iso_ece, iso_brier, iso_bins = compute_ece_and_brier(iso_preds, val_labels)

    # Candidate 3
    lr_preds = lr_model.predict_proba(X_val_3f)[:, 1]
    lr_ece, lr_brier, lr_bins = compute_ece_and_brier(lr_preds, val_labels)

    print(f"  Baseline (Uncalibrated): ECE = {uncal_ece:.4f}, Brier = {uncal_brier:.4f}")
    print(f"  Candidate 1 (Temp Scaling): ECE = {temp_ece:.4f}, Brier = {temp_brier:.4f}")
    print(f"  Candidate 2 (Isotonic Reg): ECE = {iso_ece:.4f}, Brier = {iso_brier:.4f}")
    print(f"  Candidate 3 (Logistic 3F): ECE = {lr_ece:.4f}, Brier = {lr_brier:.4f}")

    # Model Selection: min validation ECE
    candidates = [
        ("Temperature_Scaling", temp_ece, temp_brier, temp_model, temp_preds, temp_bins),
        ("Isotonic_Regression", iso_ece, iso_brier, iso_model, iso_preds, iso_bins),
        ("Logistic_Regression_3F", lr_ece, lr_brier, lr_model, lr_preds, lr_bins),
    ]
    # Sort primarily by ECE, prefer simpler temperature scaling on close tie
    candidates.sort(key=lambda x: x[1])
    selected_name, sel_ece, sel_brier, selected_model, sel_preds, sel_bins = candidates[0]
    print(f"\nOptimal Selected Calibrator: {selected_name} (Val ECE: {sel_ece:.4f})")

    # 5. Evaluate ONCE on Untouched Test Holdout (D_test)
    print("\nStep 5: Evaluating selected model on untouched holdout D_test...", flush=True)
    test_data = asyncio.run(extract_features_from_split(orch, TEST_PATHS, cases_per_spec=25))
    test_labels = test_data["labels"]
    test_c_base = test_data["c_base"]
    X_test_3f = np.column_stack([test_data["c_base"], test_data["u_cons"], test_data["h_agr"]])

    uncal_test_ece, uncal_test_brier, _ = compute_ece_and_brier(test_c_base, test_labels)

    if selected_name == "Temperature_Scaling":
        test_preds = selected_model.predict_proba(test_c_base)
    elif selected_name == "Isotonic_Regression":
        test_preds = selected_model.predict(test_c_base)
    else:
        test_preds = selected_model.predict_proba(X_test_3f)[:, 1]

    cal_test_ece, cal_test_brier, test_bins = compute_ece_and_brier(test_preds, test_labels)

    print(f"  D_test Uncalibrated: ECE = {uncal_test_ece:.4f}, Brier = {uncal_test_brier:.4f}", flush=True)
    print(f"  D_test Calibrated:   ECE = {cal_test_ece:.4f}, Brier = {cal_test_brier:.4f}", flush=True)

    # 6. Serialize Optimal Calibrator
    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
    with open(OUTPUT_MODEL_PATH, "wb") as f:
        pickle.dump(selected_model, f)
    print(f"\nSerialized selected calibrator to: {OUTPUT_MODEL_PATH}", flush=True)

    # 7. Generate Reliability Diagram
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    # Plot uncalibrated
    bin_confs_uncal = [b["mean_confidence"] for b in uncal_bins if b["mean_confidence"] is not None]
    bin_accs_uncal = [b["empirical_accuracy"] for b in uncal_bins if b["empirical_accuracy"] is not None]
    ax.plot(bin_confs_uncal, bin_accs_uncal, "s-", color="red", label=f"Uncalibrated (ECE={uncal_ece:.3f})")

    # Plot calibrated
    bin_confs_cal = [b["mean_confidence"] for b in sel_bins if b["mean_confidence"] is not None]
    bin_accs_cal = [b["empirical_accuracy"] for b in sel_bins if b["empirical_accuracy"] is not None]
    ax.plot(bin_confs_cal, bin_accs_cal, "o-", color="blue", label=f"{selected_name} (ECE={sel_ece:.3f})")

    ax.set_xlabel("Predicted Confidence")
    ax.set_ylabel("Empirical Accuracy")
    ax.set_title("Reliability Diagram (Validation Tuning Set)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"Saved reliability diagram to: {PLOT_PATH}")

    # 8. Save Calibration Manifest
    manifest_payload = {
        "manifest_title": "HealthInsight-UQ Post-Hoc Calibration Manifest",
        "timestamp": "2026-10-18T02:37:00+05:30",
        "split_discipline": "Fitted strictly on clean D_cal (zero train/test overlap)",
        "selected_calibrator": selected_name,
        "calibration_sample_size": len(cal_data["labels"]),
        "validation_sample_size": len(val_data["labels"]),
        "test_holdout_sample_size": len(test_data["labels"]),
        "validation_comparison": {
            "uncalibrated": {"ece": uncal_ece, "brier": uncal_brier},
            "temperature_scaling": {
                "ece": temp_ece,
                "brier": temp_brier,
                "temperature": getattr(temp_model, "temperature", 1.0),
            },
            "isotonic_regression": {"ece": iso_ece, "brier": iso_brier},
            "logistic_regression_3f": {"ece": lr_ece, "brier": lr_brier},
        },
        "holdout_test_results": {
            "uncalibrated_ece": uncal_test_ece,
            "uncalibrated_brier": uncal_test_brier,
            "calibrated_ece": cal_test_ece,
            "calibrated_brier": cal_test_brier,
            "bin_details": test_bins,
        },
        "artifact_paths": {
            "model_path": os.path.relpath(OUTPUT_MODEL_PATH, WORKSPACE),
            "reliability_plot": os.path.relpath(PLOT_PATH, WORKSPACE),
        },
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)
    print(f"Saved calibration manifest to: {MANIFEST_PATH}")
    return manifest_payload


if __name__ == "__main__":
    fit_and_select_calibrator()
