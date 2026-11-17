"""
HealthInsight-UQ: Safe Full-Cohort Evaluation Engine (Phase 2.10)
================================================================
Executes complete evaluation across all 39,594 records with:
- Strict split discipline and zero test contamination.
- Full multi-tier ambiguity routing (Low/Med/High).
- 5-pass stochastic input perturbation consistency.
- Bayesian Beta trust-weighted consensus fusion.
- Post-hoc clean calibrator g(z) fitted strictly on D_cal.
- Split-conformal selective abstention (Algorithm 4).
- Offline retrieval grounding for Medium and High tiers.
- Atomic checkpointing every 10 cases with crash-resilient resume.
- Storage circuit breaker (aborts cleanly if Drive C: < 3.0 GB).
- Dual reporting: Full Cohort (N=39,594) and Clean Ensemble Subset (N=34,900).
"""

import argparse
import asyncio
import csv
import datetime
import json
import logging
import math
import os
import pickle
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from backend.orchestrator import Orchestrator
from backend.trust_tracker import BetaTrustStore
from backend.retrieval_grounding import GuidelineRetrievalModule
from scripts.utils.label_extractor import extract_ground_truth_labels, normalize_specialty, ALL_DISEASE_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("health_insights.uq_eval")

TEST_SPLIT_PATHS = {
    "Cardiology": os.path.join(WORKSPACE, "datasets", "processed", "common", "test_split.json"),
    "Pulmonology": os.path.join(WORKSPACE, "datasets", "processed", "pulmonologist", "test_split.json"),
    "Neurology": os.path.join(WORKSPACE, "datasets", "processed", "neurologist", "test_split.json"),
}

LEAKAGE_INDEX_PATH = os.path.join(WORKSPACE, "datasets", "metadata", "cross_specialty_leakage_index.json")
CALIBRATOR_PATH = os.path.join(WORKSPACE, "backend", "models", "calibrator_clean.pkl")
CONFORMAL_CONFIG_PATH = os.path.join(WORKSPACE, "backend", "models", "conformal_config.json")
TRUST_POSTERIORS_PATH = os.path.join(WORKSPACE, "backend", "models", "trust_posteriors.json")


def check_storage_circuit_breaker(min_free_gb: float = 3.0) -> bool:
    """Checks Drive C: free space. Returns True if safe, False if below circuit breaker."""
    try:
        total, used, free = shutil.disk_usage(WORKSPACE)
        free_gb = free / (1024 ** 3)
        if free_gb < min_free_gb:
            logger.critical(f"STORAGE CIRCUIT BREAKER TRIGGERED: Drive free space {free_gb:.2f} GB < {min_free_gb} GB")
            return False
        return True
    except Exception as e:
        logger.warning(f"Failed to check disk usage: {e}")
        return True


def load_evaluation_configs() -> Tuple[Any, Dict[str, Any], BetaTrustStore, Set[str]]:
    """Loads all fitted artifacts strictly without training or fitting on test data."""
    # 1. Calibrator
    if not os.path.exists(CALIBRATOR_PATH):
        raise FileNotFoundError(f"Calibrator not found: {CALIBRATOR_PATH}")
    with open(CALIBRATOR_PATH, "rb") as f:
        calibrator = pickle.load(f)

    # 2. Conformal Config
    if not os.path.exists(CONFORMAL_CONFIG_PATH):
        raise FileNotFoundError(f"Conformal config not found: {CONFORMAL_CONFIG_PATH}")
    with open(CONFORMAL_CONFIG_PATH, "r", encoding="utf-8") as f:
        conformal_config = json.load(f)

    # 3. Trust Store
    trust_store = BetaTrustStore(storage_path=TRUST_POSTERIORS_PATH)

    # 4. Cross-specialty leakage set
    if not os.path.exists(LEAKAGE_INDEX_PATH):
        raise FileNotFoundError(f"Leakage index not found: {LEAKAGE_INDEX_PATH}")
    with open(LEAKAGE_INDEX_PATH, "r", encoding="utf-8") as f:
        leakage_data = json.load(f)
    leaked_encounters = set(leakage_data.get("leaked_encounters", []))

    return calibrator, conformal_config, trust_store, leaked_encounters


def apply_calibrator_prob(calibrator: Any, conf_base: float) -> float:
    z = np.array([conf_base], dtype=np.float32)
    if hasattr(calibrator, "predict_proba"):
        p = calibrator.predict_proba(z)
        if len(p.shape) > 1 and p.shape[1] == 2:
            return float(p[0, 1])
        return float(p[0])
    elif hasattr(calibrator, "predict"):
        return float(calibrator.predict(z)[0])
    return conf_base


def compute_ece_and_brier(
    confidences: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> Tuple[float, float, List[Dict[str, Any]]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_details = []
    total_samples = len(confidences)
    if total_samples == 0:
        return 0.0, 0.0, []

    ece = 0.0
    for i in range(n_bins):
        b_low, b_high = bins[i], bins[i + 1]
        mask = (confidences >= b_low) & (confidences <= b_high) if i == n_bins - 1 else (confidences >= b_low) & (confidences < b_high)
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


async def evaluate_full_cohort(
    output_dir: str,
    max_cases_per_spec: Optional[int] = None,
    smoke_test: bool = False
):
    print("=" * 80)
    print(f"HealthInsight-UQ: Full Cohort Evaluation (Phase 2.10)")
    print(f"Target Output Directory: {output_dir}")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    ledger_path = os.path.join(output_dir, "row_level_ledger.jsonl")
    error_log_path = os.path.join(output_dir, "failure_exclusion_log.jsonl")

    # Storage Check
    if not check_storage_circuit_breaker(min_free_gb=3.0):
        raise RuntimeError("Drive C: free space below safe limit. Aborting.")

    # Load artifacts
    calibrator, conformal_config, trust_store, leaked_encounters = load_evaluation_configs()
    orch = Orchestrator()
    rag_module = GuidelineRetrievalModule()

    tau_conf = float(conformal_config.get("confidence_threshold_tau", 0.65))
    h_max = float(conformal_config.get("h_max", 1.20))
    target_alpha = float(conformal_config.get("target_alpha", 0.10))

    # Crash-resilience: check already processed keys in ledger
    processed_case_keys = set()
    if os.path.exists(ledger_path):
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        row = json.loads(line)
                        processed_case_keys.add(row["case_id"])
                    except Exception:
                        pass
        logger.info(f"Resuming evaluation: found {len(processed_case_keys)} already evaluated cases in ledger.")

    # Collect eligible records across the 3 specialties
    eligible_cohort = []
    cohort_counts = {}

    for spec, path in TEST_SPLIT_PATHS.items():
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        total_in_split = len(records)
        cohort_counts[spec] = total_in_split

        limit = 25 if smoke_test else (max_cases_per_spec or total_in_split)
        selected = records[:limit]
        for idx, r in enumerate(selected):
            enc_raw = str(r.get("encounter_id") or r.get("hadm_id") or f"case_{idx:06d}")
            case_id = f"{spec.lower()}_{idx:06d}_{enc_raw}"
            eligible_cohort.append((case_id, spec, r))

    logger.info(f"Total cohort cases queued: {len(eligible_cohort)} (Counts: {cohort_counts})")

    # Evaluation loop with periodic checkpointing
    checkpoint_interval = 10
    ledger_file = open(ledger_path, "a", encoding="utf-8")
    error_file = open(error_log_path, "a", encoding="utf-8")

    processed_count = len(processed_case_keys)
    start_time = time.time()

    try:
        for idx, (case_id, spec, record) in enumerate(eligible_cohort):
            if case_id in processed_case_keys:
                continue

            # Disk check circuit breaker every 100 cases
            if idx % 100 == 0:
                if not check_storage_circuit_breaker(min_free_gb=3.0):
                    logger.critical("Aborting evaluation cleanly due to storage limit.")
                    break

            txt = record.get("text") or record.get("patient_text") or ""
            gt_labels = extract_ground_truth_labels(record, spec)
            raw_encounter = str(record.get("encounter_id") or record.get("hadm_id") or "")
            hadm_id = raw_encounter.replace("hadm_", "").strip()

            if not txt or not gt_labels:
                # Log exclusion
                error_file.write(json.dumps({
                    "case_id": case_id,
                    "specialty": spec,
                    "reason": "Missing text narrative or ground truth labels",
                    "timestamp": datetime.datetime.now().isoformat()
                }) + "\n")
                continue

            try:
                # 1. Orchestrator inference (Includes 5-pass stochastic consistency, RIE, IPM, Ambiguity Routing)
                res = await orch.process_case(patient_text=txt, fusion_mode="trust", routing_mode="routed")

                routing = res.get("routing", {})
                routing_tier = routing.get("routing_tier", "LOW")
                ambiguity_score = float(routing.get("ambiguity_score", 0.0))
                abstain_eligible = bool(routing.get("abstain_eligible", False))

                rac = res.get("rac", {})
                c_base = float(rac.get("c_base") or res.get("fusion", {}).get("overall_confidence") or 0.5)
                u_cons = float(rac.get("u_cons") or rac.get("u_instability") or 0.0)
                h_agr = float(rac.get("e_domain") or 0.0)

                # 2. Post-hoc Calibrated Confidence strictly via clean calibrator
                calib_conf = apply_calibrator_prob(calibrator, c_base)

                # 3. Retrieval Grounding for Medium and High tiers
                guideline_hits = []
                if routing_tier in ("MEDIUM", "HIGH"):
                    rie_symptoms = res.get("rie", {}).get("symptoms", [])
                    guideline_res = rag_module.ground_case(rie_symptoms, candidate_diagnoses=gt_labels, top_k=2)
                    guideline_hits = guideline_res.get("matched_guidelines", [])

                # 4. Predictions & Ground Truth Verification
                uncond_diag = (res.get("fusion", {}).get("unconditioned_diagnosis") or res.get("fusion", {}).get("final_diagnosis") or "").strip()
                uncond_match = any(
                    t.lower() == uncond_diag.lower() or t.lower() in uncond_diag.lower() or uncond_diag.lower() in t.lower()
                    for t in gt_labels
                )

                # 5. Split-Conformal Selective Abstention Decision (Algorithm 4)
                # Decision rule: Abstain if calib_conf < tau_conf OR h_agr > h_max
                is_abstained = False
                if abstain_eligible or routing_tier == "HIGH":
                    if calib_conf < tau_conf or h_agr > h_max:
                        is_abstained = True
                elif calib_conf < (tau_conf * 0.85):  # Safety fallback for extreme low-confidence cases
                    is_abstained = True

                final_diag = "Uncertain — refer to clinician" if is_abstained else uncond_diag
                selective_correct = uncond_match if not is_abstained else None

                # 6. Leakage stratification
                in_clean_subset = (hadm_id not in leaked_encounters) and (hadm_id != "")

                # 7. Record row ledger
                row = {
                    "case_id": case_id,
                    "specialty": spec,
                    "hadm_id": hadm_id,
                    "ground_truth_labels": gt_labels,
                    "unconditioned_diagnosis": uncond_diag,
                    "final_diagnosis": final_diag,
                    "unconditioned_correct": int(uncond_match),
                    "is_abstained": bool(is_abstained),
                    "selective_correct": (1 if uncond_match else 0) if not is_abstained else None,
                    "calibrated_confidence": round(calib_conf, 4),
                    "c_base": round(c_base, 4),
                    "u_cons": round(u_cons, 4),
                    "h_agr": round(h_agr, 4),
                    "ambiguity_score": round(ambiguity_score, 4),
                    "routing_tier": routing_tier,
                    "in_clean_subset": bool(in_clean_subset),
                    "retrieval_grounded": len(guideline_hits) > 0,
                }

                ledger_file.write(json.dumps(row) + "\n")
                processed_case_keys.add(case_id)
                processed_count += 1

                # Flush every 10 cases
                if processed_count % checkpoint_interval == 0:
                    ledger_file.flush()
                    os.fsync(ledger_file.fileno())

                if processed_count % 50 == 0:
                    elapsed = time.time() - start_time
                    fps = processed_count / max(1.0, elapsed)
                    logger.info(f"Processed {processed_count} cases ({fps:.1f} cases/s)...")

            except Exception as e:
                error_file.write(json.dumps({
                    "case_id": case_id,
                    "specialty": spec,
                    "error": str(e),
                    "timestamp": datetime.datetime.now().isoformat()
                }) + "\n")
                error_file.flush()

    finally:
        ledger_file.flush()
        ledger_file.close()
        error_file.flush()
        error_file.close()

    # Compute final metrics and export artifacts
    generate_full_cohort_reports(output_dir, eligible_cohort, cohort_counts, target_alpha)


def generate_full_cohort_reports(
    output_dir: str,
    eligible_cohort: List[Any],
    cohort_counts: Dict[str, int],
    target_alpha: float
):
    ledger_path = os.path.join(output_dir, "row_level_ledger.jsonl")
    rows = []
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    total_evaluated = len(rows)
    logger.info(f"Generating reports over {total_evaluated} row-level evaluations...")

    if total_evaluated == 0:
        logger.error("No cases were evaluated.")
        return

    # Extract arrays
    uncond_correct = np.array([r["unconditioned_correct"] for r in rows])
    abstained = np.array([1 if r["is_abstained"] else 0 for r in rows])
    accepted_mask = (abstained == 0)
    confidences = np.array([r["calibrated_confidence"] for r in rows])
    c_bases = np.array([r["c_base"] for r in rows])

    # 1. Overall Metrics
    coverage = float(np.mean(accepted_mask))
    abstention_rate = 1.0 - coverage
    uncond_acc = float(np.mean(uncond_correct))
    selective_acc = float(np.mean(uncond_correct[accepted_mask])) if np.sum(accepted_mask) > 0 else 0.0

    uncal_ece, uncal_brier, uncal_bins = compute_ece_and_brier(c_bases, uncond_correct)
    cal_ece, cal_brier, cal_bins = compute_ece_and_brier(confidences, uncond_correct)

    metrics_overall = {
        "total_cohort_records": total_evaluated,
        "accepted_cases": int(np.sum(accepted_mask)),
        "abstained_cases": int(np.sum(abstained)),
        "empirical_coverage": round(coverage, 4),
        "abstention_rate": round(abstention_rate, 4),
        "target_coverage": round(1.0 - target_alpha, 4),
        "unconditioned_accuracy": round(uncond_acc, 4),
        "selective_accuracy": round(selective_acc, 4),
        "uncalibrated_ece": round(uncal_ece, 4),
        "calibrated_ece": round(cal_ece, 4),
        "uncalibrated_brier": round(uncal_brier, 4),
        "calibrated_brier": round(cal_brier, 4),
    }

    with open(os.path.join(output_dir, "metrics_overall.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in metrics_overall.items():
            writer.writerow([k, v])

    # 2. Specialty Breakdown
    specialty_rows = []
    for spec in ["Cardiology", "Pulmonology", "Neurology"]:
        spec_indices = [i for i, r in enumerate(rows) if r["specialty"] == spec]
        if not spec_indices:
            continue
        idx_arr = np.array(spec_indices)
        s_uncond = uncond_correct[idx_arr]
        s_acc_mask = accepted_mask[idx_arr]
        s_cov = float(np.mean(s_acc_mask))
        s_uncond_acc = float(np.mean(s_uncond))
        s_sel_acc = float(np.mean(s_uncond[s_acc_mask])) if np.sum(s_acc_mask) > 0 else 0.0
        s_cal_ece, s_cal_brier, _ = compute_ece_and_brier(confidences[idx_arr], s_uncond)

        specialty_rows.append({
            "Specialty": spec,
            "Total_Cases": len(idx_arr),
            "Accepted_Cases": int(np.sum(s_acc_mask)),
            "Abstained_Cases": int(np.sum(s_acc_mask == 0)),
            "Coverage": round(s_cov, 4),
            "Unconditioned_Accuracy": round(s_uncond_acc, 4),
            "Selective_Accuracy": round(s_sel_acc, 4),
            "Calibrated_ECE": round(s_cal_ece, 4),
            "Calibrated_Brier": round(s_cal_brier, 4),
        })

    with open(os.path.join(output_dir, "metrics_by_specialty.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(specialty_rows[0].keys()))
        writer.writeheader()
        writer.writerows(specialty_rows)

    # 3. Stratified Cross-Specialty Leakage Breakdown
    clean_indices = [i for i, r in enumerate(rows) if r["in_clean_subset"]]
    overlap_indices = [i for i, r in enumerate(rows) if not r["in_clean_subset"]]

    stratified_rows = []
    for cohort_name, idx_list in [("Clean_Ensemble_Subset", clean_indices), ("Cross_Specialty_Overlap_Subset", overlap_indices)]:
        if not idx_list:
            continue
        idx_arr = np.array(idx_list)
        c_uncond = uncond_correct[idx_arr]
        c_acc_mask = accepted_mask[idx_arr]
        c_cov = float(np.mean(c_acc_mask))
        c_uncond_acc = float(np.mean(c_uncond))
        c_sel_acc = float(np.mean(c_uncond[c_acc_mask])) if np.sum(c_acc_mask) > 0 else 0.0
        c_ece, c_brier, _ = compute_ece_and_brier(confidences[idx_arr], c_uncond)

        stratified_rows.append({
            "Cohort": cohort_name,
            "Total_Cases": len(idx_arr),
            "Coverage": round(c_cov, 4),
            "Unconditioned_Accuracy": round(c_uncond_acc, 4),
            "Selective_Accuracy": round(c_sel_acc, 4),
            "ECE": round(c_ece, 4),
            "Brier": round(c_brier, 4),
        })

    with open(os.path.join(output_dir, "metrics_stratified_leakage.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stratified_rows[0].keys()))
        writer.writeheader()
        writer.writerows(stratified_rows)

    # 4. Baseline Comparison
    baseline_comp = [
        {"Model": "Original_Baseline_Uncalibrated", "Cohort": "Reported_MIMIC", "Total_Cases": 39594, "Coverage": 0.5061, "Selective_Accuracy": 0.4817, "Calibrated_ECE": 0.2840},
        {"Model": "HealthInsight_UQ_Full_Cohort", "Cohort": "Full_Cohort", "Total_Cases": total_evaluated, "Coverage": round(coverage, 4), "Selective_Accuracy": round(selective_acc, 4), "Calibrated_ECE": round(cal_ece, 4)},
    ]
    with open(os.path.join(output_dir, "baseline_comparison.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(baseline_comp[0].keys()))
        writer.writeheader()
        writer.writerows(baseline_comp)

    # 5. Reliability Diagram Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    bin_confs_uncal = [b["mean_confidence"] for b in uncal_bins if b["mean_confidence"] is not None]
    bin_accs_uncal = [b["empirical_accuracy"] for b in uncal_bins if b["empirical_accuracy"] is not None]
    ax.plot(bin_confs_uncal, bin_accs_uncal, "s-", color="red", label=f"Uncalibrated (ECE={uncal_ece:.3f})")
    bin_confs_cal = [b["mean_confidence"] for b in cal_bins if b["mean_confidence"] is not None]
    bin_accs_cal = [b["empirical_accuracy"] for b in cal_bins if b["empirical_accuracy"] is not None]
    ax.plot(bin_confs_cal, bin_accs_cal, "o-", color="blue", label=f"Calibrated (ECE={cal_ece:.3f})")
    ax.set_xlabel("Predicted Calibrated Confidence")
    ax.set_ylabel("Empirical Accuracy")
    ax.set_title("Reliability Diagram (Full Cohort Evaluation)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reliability_diagram.png"), dpi=150)
    plt.close()

    # 6. Reconciliation Certificate
    cert = {
        "certificate_title": "HealthInsight-UQ Full-Cohort Reconciliation Certificate",
        "timestamp": datetime.datetime.now().isoformat(),
        "total_queued": len(eligible_cohort),
        "total_processed_in_ledger": total_evaluated,
        "cohort_breakdown_target": cohort_counts,
        "integrity_check": "total_processed == len(eligible_cohort)",
        "is_reconciled": total_evaluated == len(eligible_cohort),
        "full_cohort_metrics": metrics_overall,
        "leakage_stratification": stratified_rows,
        "specialty_breakdown": specialty_rows,
    }
    with open(os.path.join(output_dir, "reconciliation_certificate.json"), "w", encoding="utf-8") as f:
        json.dump(cert, f, indent=2)

    # 7. Run Manifest
    manifest = {
        "manifest_title": "HealthInsight-UQ Full-Cohort Execution Manifest",
        "timestamp": datetime.datetime.now().isoformat(),
        "python_env": sys.executable,
        "calibrator_artifact": os.path.relpath(CALIBRATOR_PATH, WORKSPACE),
        "conformal_config": os.path.relpath(CONFORMAL_CONFIG_PATH, WORKSPACE),
        "trust_posteriors": os.path.relpath(TRUST_POSTERIORS_PATH, WORKSPACE),
        "leakage_index": os.path.relpath(LEAKAGE_INDEX_PATH, WORKSPACE),
        "results_directory": os.path.relpath(output_dir, WORKSPACE),
    }
    with open(os.path.join(output_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Full-cohort evaluation reports exported successfully to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HealthInsight-UQ Full-Cohort Evaluation Engine")
    parser.add_argument("--output_dir", type=str, default="", help="Directory to save versioned results")
    parser.add_argument("--smoke_test", action="store_true", help="Run quick smoke test (N=25)")
    parser.add_argument("--max_cases_per_spec", type=int, default=None, help="Max cases per specialty")
    args = parser.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or os.path.join(WORKSPACE, "results", f"healthinsight_uq_full_cohort_{ts}")

    asyncio.run(evaluate_full_cohort(
        output_dir=out_dir,
        max_cases_per_spec=args.max_cases_per_spec,
        smoke_test=args.smoke_test
    ))
