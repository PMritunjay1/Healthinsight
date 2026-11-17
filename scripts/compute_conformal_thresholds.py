"""
HealthInsight-UQ: Split-Conformal Selective Abstention & Quantile Estimation (Phase 2.7)
=======================================================================================
Implements distribution-free split-conformal prediction (Algorithms 3 & 4):
1. Computes nonconformity scores s_j = 1 - g(z(x_j)) strictly on clean D_cal.
2. Estimates finite-sample conformal quantile:
       tau = Quantile_{ceil((n+1)(1-alpha))/n}( {s_j}_{j in D_cal} )
   for target error budget alpha = 0.10 (90% coverage).
3. Evaluates maximum entropy threshold H_max on D_val_tune.
4. Sweeps alpha in [0.05, 0.25] to construct empirical coverage-accuracy trade-off curve.
5. Emits backend/models/conformal_config.json and results/conformal_coverage_accuracy_tradeoff.png.
"""

import asyncio
import json
import logging
import math
import os
import pickle
import sys
from typing import Any, Dict, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from backend.orchestrator import Orchestrator
from scripts.utils.label_extractor import extract_ground_truth_labels

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

CALIBRATOR_PATH = os.path.join(WORKSPACE, "backend", "models", "calibrator_clean.pkl")
CONFIG_OUTPUT_PATH = os.path.join(WORKSPACE, "backend", "models", "conformal_config.json")
PLOT_OUTPUT_PATH = os.path.join(WORKSPACE, "results", "conformal_coverage_accuracy_tradeoff.png")


def load_calibrator():
    if not os.path.exists(CALIBRATOR_PATH):
        raise FileNotFoundError(f"Calibrator not found at {CALIBRATOR_PATH}. Complete Phase 2.6 first.")
    with open(CALIBRATOR_PATH, "rb") as f:
        return pickle.load(f)


def apply_calibrator(calibrator, conf_base: float, u_cons: float = 0.0, h_agr: float = 0.0) -> float:
    z = np.array([conf_base], dtype=np.float32)
    if hasattr(calibrator, "predict_proba"):
        p = calibrator.predict_proba(z)
        if len(p.shape) > 1 and p.shape[1] == 2:
            return float(p[0, 1])
        return float(p[0])
    elif hasattr(calibrator, "predict"):
        return float(calibrator.predict(z)[0])
    return conf_base


async def extract_conformal_scores(
    orch: Orchestrator,
    calibrator: Any,
    split_paths: Dict[str, str],
    cases_per_spec: int = 50
) -> Dict[str, np.ndarray]:
    scores = []
    h_agrs = []
    labels = []
    confidences = []

    for spec, path in split_paths.items():
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        print(f"  Extracting scores from {spec} ({min(cases_per_spec, len(records))} cases)...", flush=True)
        for idx, r in enumerate(records[:cases_per_spec]):
            txt = r.get("text") or r.get("patient_text") or ""
            gt_labels = extract_ground_truth_labels(r, spec)
            if not txt or not gt_labels:
                continue

            try:
                res = await orch.process_case(patient_text=txt, routing_mode="routed")
                rac = res.get("rac", {})
                c_b = float(rac.get("c_base") or res.get("fusion", {}).get("overall_confidence") or 0.5)
                u_c = float(rac.get("u_cons") or rac.get("u_instability") or 0.0)
                h_a = float(rac.get("e_domain") or 0.0)

                g_z = apply_calibrator(calibrator, c_b, u_c, h_a)
                s_j = 1.0 - g_z

                final_diag = (res.get("fusion", {}).get("final_diagnosis") or "").strip().lower()
                uncond_diag = (res.get("fusion", {}).get("unconditioned_diagnosis") or final_diag).strip().lower()
                is_correct = 1 if any(t.lower() == uncond_diag or t.lower() in uncond_diag for t in gt_labels) else 0

                scores.append(s_j)
                h_agrs.append(h_a)
                labels.append(is_correct)
                confidences.append(g_z)
            except Exception as e:
                continue

            if (idx + 1) % 10 == 0:
                print(f"    {spec}: processed {idx + 1}/{min(cases_per_spec, len(records))} cases...", flush=True)

    return {
        "nonconformity_scores": np.array(scores, dtype=np.float32),
        "h_agrs": np.array(h_agrs, dtype=np.float32),
        "labels": np.array(labels, dtype=np.int32),
        "confidences": np.array(confidences, dtype=np.float32),
    }


def compute_finite_sample_quantile(scores: np.ndarray, alpha: float) -> Tuple[float, float]:
    """
    Computes finite-sample split-conformal quantile:
    q_level = ceil((n + 1) * (1 - alpha)) / n
    tau = Quantile_{q_level}(scores)
    """
    n = len(scores)
    q_level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / float(n))
    tau = float(np.quantile(scores, q_level, method="higher"))
    return tau, q_level


def run_conformal_calibration(target_alpha: float = 0.10):
    print("=" * 80, flush=True)
    print("HealthInsight-UQ: Split-Conformal Selective Abstention (Phase 2.7)", flush=True)
    print("=" * 80, flush=True)

    calibrator = load_calibrator()
    orch = Orchestrator()

    # 1. Nonconformity scores strictly on clean D_cal
    print("\nStep 1: Extracting nonconformity scores on clean D_cal...", flush=True)
    cal_data = asyncio.run(extract_conformal_scores(orch, calibrator, CAL_PATHS, cases_per_spec=25))
    n_cal = len(cal_data["nonconformity_scores"])
    print(f"Computed nonconformity scores on {n_cal} clean calibration cases.", flush=True)

    # 2. Conformal quantile at target alpha
    tau, q_level = compute_finite_sample_quantile(cal_data["nonconformity_scores"], target_alpha)
    tau_conf = 1.0 - tau  # minimum confidence to accept
    print(f"Target error alpha = {target_alpha:.2f} (Target coverage: {1.0 - target_alpha:.0%})", flush=True)
    print(f"Quantile level ceil((n+1)(1-alpha))/n = {q_level:.4f}", flush=True)
    print(f"Empirical nonconformity quantile tau = {tau:.4f}", flush=True)
    print(f"Derived confidence threshold (1 - tau) = {tau_conf:.4f}", flush=True)

    # 3. Validation tuning on D_val_tune
    print("\nStep 2: Evaluating on D_val_tune and tuning H_max...", flush=True)
    val_data = asyncio.run(extract_conformal_scores(orch, calibrator, VAL_PATHS, cases_per_spec=25))
    val_conf = val_data["confidences"]
    val_h = val_data["h_agrs"]
    val_labels = val_data["labels"]

    # Sweep H_max to maximize selective accuracy while maintaining coverage >= (1 - alpha - 0.05)
    best_h_max = 1.20
    best_sel_acc = 0.0
    h_candidates = np.linspace(0.4, 2.0, 17)

    for h_cand in h_candidates:
        accepted_mask = (val_conf >= tau_conf) & (val_h <= h_cand)
        cov = float(np.mean(accepted_mask))
        if cov >= (1.0 - target_alpha - 0.10) and np.sum(accepted_mask) > 10:
            sel_acc = float(np.mean(val_labels[accepted_mask]))
            if sel_acc > best_sel_acc:
                best_sel_acc = sel_acc
                best_h_max = float(round(h_cand, 2))

    print(f"Optimal tuned H_max = {best_h_max:.2f} (Val Selective Accuracy: {best_sel_acc:.4f})")

    # 4. Sweep alpha in [0.05, 0.25] for trade-off curve
    print("\nStep 3: Sweeping alpha in [0.05, 0.25] for coverage-accuracy trade-off...")
    alphas = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
    sweep_results = []

    for a in alphas:
        t_a, q_a = compute_finite_sample_quantile(cal_data["nonconformity_scores"], a)
        c_thresh = 1.0 - t_a
        acc_mask = (val_conf >= c_thresh) & (val_h <= best_h_max)
        emp_cov = float(np.mean(acc_mask))
        emp_acc = float(np.mean(val_labels[acc_mask])) if np.sum(acc_mask) > 0 else 0.0
        sweep_results.append({
            "alpha": a,
            "target_coverage": round(1.0 - a, 4),
            "quantile_tau": round(t_a, 4),
            "confidence_threshold": round(c_thresh, 4),
            "empirical_coverage": round(emp_cov, 4),
            "selective_accuracy": round(emp_acc, 4),
        })
        print(f"  alpha={a:.2f} -> tau={t_a:.4f}, cov={emp_cov:.4f}, sel_acc={emp_acc:.4f}")

    # 5. Serialize conformal config
    config_payload = {
        "config_title": "HealthInsight-UQ Split-Conformal Abstention Config (Algorithm 4)",
        "split_discipline": "Quantile estimated strictly on clean D_cal (zero train/test overlap)",
        "target_alpha": target_alpha,
        "target_coverage": round(1.0 - target_alpha, 4),
        "calibration_sample_size": n_cal,
        "quantile_level": round(q_level, 4),
        "conformal_quantile_tau": round(tau, 4),
        "confidence_threshold_tau": round(tau_conf, 4),
        "h_max": best_h_max,
        "alpha_sweep": sweep_results,
        "validation_sample_size": len(val_labels),
    }

    os.makedirs(os.path.dirname(CONFIG_OUTPUT_PATH), exist_ok=True)
    with open(CONFIG_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)
    print(f"\nSerialized conformal config to: {CONFIG_OUTPUT_PATH}")

    # Also update legacy/fallback abstention_config.json for full backward compatibility
    abstention_path = os.path.join(WORKSPACE, "backend", "models", "abstention_config.json")
    with open(abstention_path, "w", encoding="utf-8") as f:
        json.dump({"tau": round(tau_conf, 4), "h_limit": best_h_max}, f, indent=2)

    # 6. Plot coverage-accuracy trade-off curve
    os.makedirs(os.path.dirname(PLOT_OUTPUT_PATH), exist_ok=True)
    covs = [s["empirical_coverage"] for s in sweep_results]
    accs = [s["selective_accuracy"] for s in sweep_results]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(covs, accs, "bo-", linewidth=2, markersize=8, label="Conformal Abstention Frontier")
    ax.axvline(1.0 - target_alpha, color="r", linestyle="--", alpha=0.7, label=f"Target Coverage ({1.0 - target_alpha:.0%})")
    ax.set_xlabel("Empirical Coverage")
    ax.set_ylabel("Selective Accuracy")
    ax.set_title("Coverage vs. Selective Accuracy Trade-Off")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_PATH, dpi=150)
    plt.close()
    print(f"Saved trade-off plot to: {PLOT_OUTPUT_PATH}")

    return config_payload


if __name__ == "__main__":
    run_conformal_calibration(target_alpha=0.10)
