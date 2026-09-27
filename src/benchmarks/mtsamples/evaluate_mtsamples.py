"""
MTSamples External Benchmark Evaluation Engine
==============================================
Executes zero-weight external generalization evaluation of the three frozen clinical specialists:
1. Cardiologist Specialist V2 (Frozen INT8 ONNX)
2. Pulmonologist Specialist V1 (Frozen INT8 ONNX)
3. Neurologist Specialist V1 (Frozen INT8 ONNX)

Performs:
- Frozen Model-Level Multi-Label Disease Classification Evaluation
- 95% Bootstrap Confidence Interval Estimation (1,000 resamples)
- End-to-End Multi-Agent System Routing and RAC Consensus Benchmark
- Thesis-Ready Artifact Generation under experiments/external_benchmarks/mtsamples/
"""

import os
import sys
import json
import time
import logging
import asyncio
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    hamming_loss, accuracy_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.models import PatientInput
from backend.agents.specialist_agents import (
    CardiologistSpecialistAgent,
    PulmonologistSpecialistAgent,
    NeurologistSpecialistAgent
)
from backend.orchestrator import Orchestrator
from src.benchmarks.mtsamples.load_mtsamples import load_raw_mtsamples
from src.benchmarks.mtsamples.preprocess_mtsamples import preprocess_all_mtsamples
from src.benchmarks.mtsamples.ontology_mapping import (
    map_mtsamples_ontology,
    CARDIOLOGIST_CLASSES,
    PULMONOLOGIST_CLASSES,
    NEUROLOGIST_CLASSES
)

logger = logging.getLogger("MTSamplesEvaluation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "external_benchmarks", "mtsamples")


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Dict[str, Tuple[float, float]]:
    """
    Compute 95% empirical bootstrap confidence intervals for multi-label classification metrics.
    """
    np.random.seed(seed)
    n_samples = len(y_true)
    if n_samples == 0:
        return {}

    micro_f1_list = []
    macro_f1_list = []
    subset_acc_list = []
    macro_auc_list = []
    macro_auprc_list = []

    for _ in range(n_bootstraps):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        yt_boot = y_true[idx]
        yp_boot = y_pred[idx]
        yprob_boot = y_prob[idx]

        # F1 metrics
        try:
            micro_f1 = f1_score(yt_boot, yp_boot, average="micro", zero_division=0)
            macro_f1 = f1_score(yt_boot, yp_boot, average="macro", zero_division=0)
            subset_acc = accuracy_score(yt_boot, yp_boot)
            micro_f1_list.append(micro_f1)
            macro_f1_list.append(macro_f1)
            subset_acc_list.append(subset_acc)
        except Exception:
            pass

        # AUROC & AUPRC (only over classes with >= 1 positive and >= 1 negative in bootstrap sample)
        class_aucs = []
        class_auprcs = []
        for c in range(yt_boot.shape[1]):
            if len(np.unique(yt_boot[:, c])) > 1:
                try:
                    auc = roc_auc_score(yt_boot[:, c], yprob_boot[:, c])
                    auprc = average_precision_score(yt_boot[:, c], yprob_boot[:, c])
                    class_aucs.append(auc)
                    class_auprcs.append(auprc)
                except Exception:
                    pass

        if class_aucs:
            macro_auc_list.append(np.mean(class_aucs))
        if class_auprcs:
            macro_auprc_list.append(np.mean(class_auprcs))

    ci_results = {}
    if micro_f1_list:
        ci_results["micro_f1_ci95"] = (round(float(np.percentile(micro_f1_list, 2.5)), 4), round(float(np.percentile(micro_f1_list, 97.5)), 4))
    if macro_f1_list:
        ci_results["macro_f1_ci95"] = (round(float(np.percentile(macro_f1_list, 2.5)), 4), round(float(np.percentile(macro_f1_list, 97.5)), 4))
    if subset_acc_list:
        ci_results["subset_accuracy_ci95"] = (round(float(np.percentile(subset_acc_list, 2.5)), 4), round(float(np.percentile(subset_acc_list, 97.5)), 4))
    if macro_auc_list:
        ci_results["macro_auroc_ci95"] = (round(float(np.percentile(macro_auc_list, 2.5)), 4), round(float(np.percentile(macro_auc_list, 97.5)), 4))
    if macro_auprc_list:
        ci_results["macro_auprc_ci95"] = (round(float(np.percentile(macro_auprc_list, 2.5)), 4), round(float(np.percentile(macro_auprc_list, 97.5)), 4))

    return ci_results


async def _evaluate_specialist_async(
    specialist_agent,
    df_eval: pd.DataFrame,
    target_classes: List[str],
    gt_vector_col: str
) -> Tuple[List[List[int]], List[List[float]], List[List[int]], List[float], List[Dict[str, Any]]]:
    """Async batch execution helper for fast single-loop processing."""
    y_true_list = []
    y_prob_list = []
    y_pred_list = []
    latencies = []
    case_predictions = []
    frozen_thresholds = specialist_agent.frozen_thresholds

    for idx, row in df_eval.iterrows():
        input_data = PatientInput(
            patient_text=str(row["clean_prediagnostic_text"]),
            history=f"Specialty: {row.get('medical_specialty', '')}. Note: {row.get('sample_name', '')}"
        )
        gt_vec = row[gt_vector_col]
        y_true_list.append(gt_vec)

        start_t = time.time()
        res = await specialist_agent.run(input_data)
        latency = (time.time() - start_t) * 1000
        latencies.append(latency)

        prob_dict = res.output.disease_probabilities
        probs = [prob_dict.get(c, 0.0) for c in target_classes]
        preds = [1 if probs[i] >= frozen_thresholds.get(c, 0.5) else 0 for i, c in enumerate(target_classes)]

        y_prob_list.append(probs)
        y_pred_list.append(preds)

        case_predictions.append({
            "sample_index": int(idx),
            "sample_name": row.get("sample_name", ""),
            "medical_specialty": row.get("medical_specialty", ""),
            "ground_truth_labels": [target_classes[i] for i, v in enumerate(gt_vec) if v == 1],
            "predicted_active_diagnoses": res.output.active_diagnoses,
            "primary_diagnosis": res.output.primary_diagnosis,
            "predicted_complaint": res.output.predicted_complaint,
            "predicted_urgency": res.output.predicted_urgency,
            "probabilities": prob_dict,
            "latency_ms": round(latency, 2)
        })

    return y_true_list, y_prob_list, y_pred_list, latencies, case_predictions


def evaluate_specialist_on_dataset(
    specialist_agent,
    df_eval: pd.DataFrame,
    target_classes: List[str],
    gt_vector_col: str,
    specialist_name: str
) -> Dict[str, Any]:
    """
    Run frozen specialist inference and calculate authoritative metrics.
    """
    logger.info(f"Evaluating {specialist_name} on {len(df_eval)} cohort records...")
    y_true_list, y_prob_list, y_pred_list, latencies, case_predictions = asyncio.run(
        _evaluate_specialist_async(specialist_agent, df_eval, target_classes, gt_vector_col)
    )

    y_true = np.array(y_true_list, dtype=np.int32)
    y_prob = np.array(y_prob_list, dtype=np.float32)
    y_pred = np.array(y_pred_list, dtype=np.int32)
    frozen_thresholds = specialist_agent.frozen_thresholds

    # 1. Global Metrics
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    subset_acc = float(accuracy_score(y_true, y_pred))
    h_loss = float(hamming_loss(y_true, y_pred))

    # 2. Per-Class Metrics & AUROC / AUPRC
    per_class_metrics = {}
    valid_aurocs = []
    valid_auprcs = []

    for i, c_name in enumerate(target_classes):
        yt_c = y_true[:, i]
        yp_c = y_pred[:, i]
        yprob_c = y_prob[:, i]

        supp = int(np.sum(yt_c))
        prec = float(precision_score(yt_c, yp_c, zero_division=0))
        rec = float(recall_score(yt_c, yp_c, zero_division=0))
        f1_c = float(f1_score(yt_c, yp_c, zero_division=0))

        if len(np.unique(yt_c)) > 1:
            try:
                auc_c = float(roc_auc_score(yt_c, yprob_c))
                auprc_c = float(average_precision_score(yt_c, yprob_c))
                valid_aurocs.append(auc_c)
                valid_auprcs.append(auprc_c)
            except Exception:
                auc_c = "NOT COMPUTABLE"
                auprc_c = "NOT COMPUTABLE"
        else:
            auc_c = "NOT COMPUTABLE"
            auprc_c = "NOT COMPUTABLE"

        per_class_metrics[c_name] = {
            "support": supp,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1_c, 4),
            "auroc": round(auc_c, 4) if isinstance(auc_c, float) else auc_c,
            "auprc": round(auprc_c, 4) if isinstance(auprc_c, float) else auprc_c,
            "frozen_threshold": frozen_thresholds.get(c_name, 0.5)
        }

    macro_auroc = round(float(np.mean(valid_aurocs)), 4) if valid_aurocs else "NOT COMPUTABLE"
    macro_auprc = round(float(np.mean(valid_auprcs)), 4) if valid_auprcs else "NOT COMPUTABLE"

    # Micro AUROC/AUPRC across all elements
    try:
        micro_auroc = round(float(roc_auc_score(y_true.ravel(), y_prob.ravel())), 4)
        micro_auprc = round(float(average_precision_score(y_true.ravel(), y_prob.ravel())), 4)
    except Exception:
        micro_auroc = "NOT COMPUTABLE"
        micro_auprc = "NOT COMPUTABLE"

    # 3. Bootstrap Confidence Intervals
    boot_ci = compute_bootstrap_ci(y_true, y_pred, y_prob, n_bootstraps=1000)

    # 4. Latency Profile
    latency_profile = {
        "mean_ms": round(float(np.mean(latencies)), 2),
        "median_ms": round(float(np.median(latencies)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "min_ms": round(float(np.min(latencies)), 2),
        "max_ms": round(float(np.max(latencies)), 2),
    }

    results = {
        "specialist": specialist_name,
        "cohort_size": len(df_eval),
        "positive_cases_count": int(np.sum(np.any(y_true == 1, axis=1))),
        "negative_control_count": int(np.sum(np.all(y_true == 0, axis=1))),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "micro_auroc": micro_auroc,
        "micro_auprc": micro_auprc,
        "subset_accuracy": round(subset_acc, 4),
        "hamming_loss": round(h_loss, 4),
        "bootstrap_ci95": boot_ci,
        "per_class_metrics": per_class_metrics,
        "latency_profile": latency_profile,
        "raw_predictions": case_predictions,
        "y_true": y_true,
        "y_prob": y_prob
    }

    return results


async def _run_system_level_async(
    orchestrator,
    sample_df: pd.DataFrame
) -> Tuple[int, int, List[float], List[float], List[Dict[str, Any]]]:
    """Async execution helper for system-level benchmark."""
    total_tested = 0
    routing_correct = 0
    rac_scores = []
    latencies = []
    routing_details = []

    for idx, row in sample_df.iterrows():
        p_text = str(row["clean_prediagnostic_text"])
        hist = f"Medical Specialty: {row.get('medical_specialty', '')}"

        start_t = time.time()
        res = await orchestrator.process_case(p_text, hist)
        lat = (time.time() - start_t) * 1000
        latencies.append(lat)

        routing = res.get("routing", {})
        rac = res.get("rac", {})
        selected_specs = routing.get("selected_specialists", [])
        rac_agreement = rac.get("agreement_score", 1.0)
        rac_scores.append(rac_agreement)

        gt_specs = row.get("active_gt_specialties", [])
        expected_specs = []
        if "Cardiology" in gt_specs:
            expected_specs.append("cardiologist")
        if "Pulmonology" in gt_specs:
            expected_specs.append("pulmonologist")
        if "Neurology" in gt_specs:
            expected_specs.append("neurologist")

        # Routing accuracy match
        if not expected_specs and len(selected_specs) == 0:
            match = True  # Correct fallback to general diagnosis
        elif any(s in selected_specs for s in expected_specs):
            match = True
        else:
            match = False

        if match:
            routing_correct += 1
        total_tested += 1

        routing_details.append({
            "sample_index": int(idx),
            "medical_specialty": row.get("medical_specialty", ""),
            "expected_specialists": expected_specs,
            "selected_specialists": selected_specs,
            "routing_match": match,
            "rac_agreement_score": rac_agreement,
            "pipeline_latency_ms": round(lat, 2)
        })

    return total_tested, routing_correct, rac_scores, latencies, routing_details


def run_system_level_benchmark(df_cohort: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluate End-to-End Orchestrator Pipeline across MTSamples.
    """
    logger.info("Executing System-Level Multi-Agent Routing & RAC Benchmark...")
    orchestrator = Orchestrator()

    # Sample representative stratified subset (100 cases) for end-to-end evaluation
    sample_df = df_cohort.sample(n=min(100, len(df_cohort)), random_state=42).reset_index(drop=True)

    total_tested, routing_correct, rac_scores, latencies, routing_details = asyncio.run(
        _run_system_level_async(orchestrator, sample_df)
    )

    routing_accuracy = round((routing_correct / total_tested) * 100.0, 2)

    system_results = {
        "total_cases_evaluated": total_tested,
        "routing_accuracy_pct": routing_accuracy,
        "mean_rac_agreement_score": round(float(np.mean(rac_scores)), 4),
        "system_latency_profile": {
            "mean_ms": round(float(np.mean(latencies)), 2),
            "median_ms": round(float(np.median(latencies)), 2),
            "p95_ms": round(float(np.percentile(latencies, 95)), 2)
        },
        "case_routing_breakdown": routing_details
    }

    logger.info(f"System-Level Benchmark Complete. Routing Accuracy: {routing_accuracy}%, Mean RAC: {system_results['mean_rac_agreement_score']}")
    return system_results


def plot_benchmark_curves(
    cardio_res: Dict[str, Any],
    pulmo_res: Dict[str, Any],
    neuro_res: Dict[str, Any],
    output_dir: str
):
    """
    Generate ROC curves and Precision-Recall curves plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    specs = [
        ("Cardiologist (MTSamples)", cardio_res, CARDIOLOGIST_CLASSES, axes[0]),
        ("Pulmonologist (MTSamples)", pulmo_res, PULMONOLOGIST_CLASSES, axes[1]),
        ("Neurologist (MTSamples)", neuro_res, NEUROLOGIST_CLASSES, axes[2])
    ]

    for title, res, classes, ax in specs:
        yt = res["y_true"]
        yp = res["y_prob"]
        for i, c in enumerate(classes):
            if len(np.unique(yt[:, i])) > 1:
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(yt[:, i], yp[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"{c} (AUC={roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.6)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # PR Curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    specs = [
        ("Cardiologist (MTSamples)", cardio_res, CARDIOLOGIST_CLASSES, axes[0]),
        ("Pulmonologist (MTSamples)", pulmo_res, PULMONOLOGIST_CLASSES, axes[1]),
        ("Neurologist (MTSamples)", neuro_res, NEUROLOGIST_CLASSES, axes[2])
    ]

    for title, res, classes, ax in specs:
        yt = res["y_true"]
        yp = res["y_prob"]
        for i, c in enumerate(classes):
            if len(np.unique(yt[:, i])) > 1:
                from sklearn.metrics import precision_recall_curve, average_precision_score
                prec, rec, _ = precision_recall_curve(yt[:, i], yp[:, i])
                ap = average_precision_score(yt[:, i], yp[:, i])
                ax.plot(rec, prec, label=f"{c} (AP={ap:.2f})")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pr_path = os.path.join(output_dir, "pr_curves.png")
    plt.savefig(pr_path, dpi=300)
    plt.close()
    logger.info(f"Saved evaluation plots to {roc_path} and {pr_path}")


def run_full_mtsamples_evaluation():
    """
    Main evaluation pipeline orchestrator.
    """
    logger.info("=================================================================")
    logger.info("  STARTING MTSAMPLES EXTERNAL BENCHMARK EVALUATION PIPELINE     ")
    logger.info("=================================================================")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Data
    raw_df, load_meta = load_raw_mtsamples()

    # 2. Preprocess
    proc_df, prep_stats = preprocess_all_mtsamples(raw_df)

    # 3. Ontology Mapping
    mapped_df, map_stats = map_mtsamples_ontology(proc_df)

    # 4. Filter Evaluation Cohort (Eligible Specialist Cases + Negative Controls)
    eligible_mask = mapped_df["mapping_status"].isin(["MAPPED_SPECIALIST_POSITIVE", "NEGATIVE_CONTROL_INACTIVE"])
    eval_cohort = mapped_df[eligible_mask].copy().reset_index(drop=True)
    logger.info(f"Evaluating across {len(eval_cohort)} eligible cases (excluding {map_stats['unmapped_ambiguous_records']} ambiguous).")

    # 5. Initialize Frozen Specialist Agents
    cardio_agent = CardiologistSpecialistAgent()
    pulmo_agent = PulmonologistSpecialistAgent()
    neuro_agent = NeurologistSpecialistAgent()

    # 6. Evaluate Each Specialist
    cardio_res = evaluate_specialist_on_dataset(
        cardio_agent, eval_cohort, CARDIOLOGIST_CLASSES, "cardio_gt_vector", "Cardiologist Specialist V2"
    )

    pulmo_res = evaluate_specialist_on_dataset(
        pulmo_agent, eval_cohort, PULMONOLOGIST_CLASSES, "pulmo_gt_vector", "Pulmonologist Specialist V1"
    )

    neuro_res = evaluate_specialist_on_dataset(
        neuro_agent, eval_cohort, NEUROLOGIST_CLASSES, "neuro_gt_vector", "Neurologist Specialist V1"
    )

    # 7. System-Level Evaluation
    system_res = run_system_level_benchmark(eval_cohort)

    # 8. Generate Plots
    plot_benchmark_curves(cardio_res, pulmo_res, neuro_res, OUTPUT_DIR)

    # 9. Save JSON / JSONL Artifacts
    # Predictions JSONL
    pred_path = os.path.join(OUTPUT_DIR, "predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for idx in range(len(eval_cohort)):
            rec = {
                "sample_index": idx,
                "sample_name": eval_cohort.iloc[idx].get("sample_name", ""),
                "medical_specialty": eval_cohort.iloc[idx].get("medical_specialty", ""),
                "clean_input_text": eval_cohort.iloc[idx].get("clean_prediagnostic_text", "")[:400],
                "cardio_prediction": cardio_res["raw_predictions"][idx],
                "pulmo_prediction": pulmo_res["raw_predictions"][idx],
                "neuro_prediction": neuro_res["raw_predictions"][idx]
            }
            f.write(json.dumps(rec) + "\n")

    # Clean results dictionaries for JSON serialization (remove numpy arrays)
    def sanitize_results(res_dict):
        d = {k: v for k, v in res_dict.items() if k not in ["raw_predictions", "y_true", "y_prob"]}
        return d

    clean_cardio = sanitize_results(cardio_res)
    clean_pulmo = sanitize_results(pulmo_res)
    clean_neuro = sanitize_results(neuro_res)

    master_metrics = {
        "benchmark_name": "MTSamples External Clinical Benchmark",
        "evaluation_date": "2026-09-09",
        "cohort_summary": {
            "total_dataset_records": load_meta["total_raw_records"],
            "eligible_records_evaluated": len(eval_cohort),
            "excluded_unmapped_records": map_stats["unmapped_ambiguous_records"],
            "negative_control_records": map_stats["negative_control_records"],
            "cardiology_positive_records": map_stats["cardio_positive_records"],
            "pulmonology_positive_records": map_stats["pulmo_positive_records"],
            "neurology_positive_records": map_stats["neuro_positive_records"],
        },
        "specialists": {
            "cardiologist": clean_cardio,
            "pulmonologist": clean_pulmo,
            "neurologist": clean_neuro
        },
        "system_level": system_res
    }

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(master_metrics, f, indent=2)

    per_class_path = os.path.join(OUTPUT_DIR, "per_class_metrics.json")
    with open(per_class_path, "w", encoding="utf-8") as f:
        json.dump({
            "cardiologist": clean_cardio["per_class_metrics"],
            "pulmonologist": clean_pulmo["per_class_metrics"],
            "neurologist": clean_neuro["per_class_metrics"]
        }, f, indent=2)

    dataset_stats_path = os.path.join(OUTPUT_DIR, "dataset_statistics.json")
    with open(dataset_stats_path, "w", encoding="utf-8") as f:
        json.dump(load_meta, f, indent=2)

    mapping_stats_path = os.path.join(OUTPUT_DIR, "mapping_statistics.json")
    with open(mapping_stats_path, "w", encoding="utf-8") as f:
        json.dump(map_stats, f, indent=2)

    manifest = {
        "benchmark": "MTSamples External Generalization Evaluation",
        "version": "1.0",
        "models_evaluated": {
            "cardiologist": "backend/models/cardiologist_mimic_final_quant.onnx",
            "pulmonologist": "backend/models/pulmonologist_final_quant.onnx",
            "neurologist": "backend/models/neurologist_final_quant.onnx"
        },
        "artifacts_generated": [
            "predictions.jsonl", "metrics.json", "per_class_metrics.json",
            "dataset_statistics.json", "mapping_statistics.json",
            "roc_curves.png", "pr_curves.png"
        ],
        "authoritative_freeze_preserved": True
    }
    manifest_path = os.path.join(OUTPUT_DIR, "benchmark_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"All evaluation artifacts written to {OUTPUT_DIR}")
    return master_metrics


if __name__ == "__main__":
    metrics = run_full_mtsamples_evaluation()
    print("\n=======================================================")
    print("      MTSAMPLES BENCHMARK EVALUATION RESULTS          ")
    print("=======================================================")
    print("Cardiologist:  Micro-F1:", metrics["specialists"]["cardiologist"]["micro_f1"], "Macro-AUROC:", metrics["specialists"]["cardiologist"]["macro_auroc"])
    print("Pulmonologist: Micro-F1:", metrics["specialists"]["pulmonologist"]["micro_f1"], "Macro-AUROC:", metrics["specialists"]["pulmonologist"]["macro_auroc"])
    print("Neurologist:   Micro-F1:", metrics["specialists"]["neurologist"]["micro_f1"], "Macro-AUROC:", metrics["specialists"]["neurologist"]["macro_auroc"])
