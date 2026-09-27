"""
IEEE Multi-Agent Medical Evaluation Pipeline
==============================================
Runs full pipeline on DEV / VALIDATION / BENCHMARK splits.
Classifies errors into 4 categories. Computes statistics.
All metrics derived strictly from real pipeline execution.
Zero synthetic data. Zero randomness.
"""
import asyncio
import json
import logging
import os
import numpy as np
from datetime import datetime
from backend.orchestrator import Orchestrator
from backend.datasets import DEV_CASES, VALIDATION_CASES, BENCHMARK_CASES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Evaluator")

# ──────────────────────────────────────────────
# REQUIRED FIELDS — reject cases missing these
# ──────────────────────────────────────────────
REQUIRED_CASE_FIELDS = {"id", "patient_text", "history", "expected_diagnosis"}



# ──────────────────────────────────────────────
# ERROR CLASSIFICATION SYSTEM (IEEE § V)
# ──────────────────────────────────────────────
# Rule-based clinical equivalence map for ACCEPTABLE VARIATION
CLINICAL_EQUIVALENCE = {
    "myocardial infarction": ["heart attack", "mi", "stemi", "nstemi", "acute myocardial infarction", "acute mi"],
    "alzheimer": ["alzheimer's disease", "alzheimers", "alzheimer disease", "alzheimer's"],
    "asthma": ["asthma exacerbation", "bronchial asthma", "asthma attack", "acute asthma"],
    "ibd": ["inflammatory bowel disease", "crohn's disease", "ulcerative colitis", "crohn", "crohns"],
    "dvt": ["deep vein thrombosis", "deep venous thrombosis"],
    "uti": ["urinary tract infection"],
    "copd": ["chronic obstructive pulmonary disease"],
    "migraine": ["migraine with aura", "migraine without aura", "classic migraine"],
    "diabetes": ["diabetes mellitus", "type 2 diabetes", "type 1 diabetes", "dm"],
    "hyperthyroidism": ["graves disease", "graves' disease", "thyrotoxicosis", "overactive thyroid"],
    "hypothyroidism": ["hashimoto", "hashimoto's", "underactive thyroid", "myxedema"],
    "rheumatoid arthritis": ["ra", "rheumatoid", "inflammatory arthritis"],
    "angina": ["angina pectoris", "stable angina", "unstable angina"],
    "heart failure": ["congestive heart failure", "chf", "hf"],
    "pneumonia": ["community acquired pneumonia", "cap", "bacterial pneumonia", "lobar pneumonia"],
    "anemia": ["iron deficiency anemia", "anaemia", "iron deficiency anaemia"],
    "stroke": ["cerebrovascular accident", "cva", "ischemic stroke", "cerebral infarction"],
    "pulmonary embolism": ["pe", "pulmonary thromboembolism"],
    "panic attack": ["panic disorder", "anxiety attack"],
    "gout": ["gouty arthritis"],
    "appendicitis": ["acute appendicitis"],
    "meningitis": ["bacterial meningitis", "viral meningitis"],
    "lung cancer": ["bronchogenic carcinoma", "pulmonary malignancy"],
    "tuberculosis": ["tb", "pulmonary tuberculosis", "pulmonary tb"],
    "pericarditis": ["acute pericarditis"],
    "subarachnoid hemorrhage": ["sah", "subarachnoid haemorrhage"],
    "parkinson disease": ["parkinson's disease", "parkinsons", "parkinson's"],
    "aortic dissection": ["dissecting aortic aneurysm"],
    "meniere disease": ["meniere's disease", "menieres"],
    "cholecystitis": ["acute cholecystitis"],
    "peptic ulcer": ["peptic ulcer disease", "gastric ulcer", "duodenal ulcer"],
    "nephrotic syndrome": ["nephrosis"],
    "hepatitis": ["viral hepatitis", "acute hepatitis"],
    "tonsillitis": ["acute tonsillitis", "pharyngotonsillitis"],
    "diabetic retinopathy": ["proliferative diabetic retinopathy", "diabetic eye disease"],
    "glaucoma": ["open angle glaucoma", "acute angle closure glaucoma"],
    "lymphoma": ["hodgkin lymphoma", "non-hodgkin lymphoma", "hodgkins"],
    "drug allergy": ["drug hypersensitivity", "medication allergy", "allergic drug reaction"],
    "bronchitis": ["acute bronchitis"],
    "delirium": ["acute confusional state", "acute delirium"],
    "functional dyspepsia": ["non-ulcer dyspepsia", "dyspepsia"],
    "gastroenteritis": ["acute gastroenteritis", "stomach flu", "viral gastroenteritis"],
    "ramsay hunt syndrome": ["herpes zoster oticus"],
    "epiglottitis": ["acute epiglottitis", "supraglottitis"],
}


def _normalize(text: str) -> str:
    """Lowercase, strip whitespace and trailing punctuation."""
    return text.strip().strip(".,;:!?").lower()


def _classify_error(expected: str, predicted: str) -> str:
    """
    Classify the prediction error into one of 4 categories:
    - correct:    exact match (after normalization)
    - semantic:   predicted is a known clinical equivalent
    - acceptable: partial keyword overlap indicating related condition
    - severe:     no meaningful overlap
    """
    exp_norm = _normalize(expected)
    pred_norm = _normalize(predicted)

    # 1. EXACT MATCH
    if exp_norm == pred_norm:
        return "correct"

    # 2. SEMANTIC MATCH — check clinical equivalence map
    # Check if predicted is an alias of expected
    for canonical, aliases in CLINICAL_EQUIVALENCE.items():
        all_forms = [canonical] + aliases
        exp_match = any(exp_norm == form or exp_norm in form or form in exp_norm for form in all_forms)
        pred_match = any(pred_norm == form or pred_norm in form or form in pred_norm for form in all_forms)
        if exp_match and pred_match:
            return "semantic"

    # 3. ACCEPTABLE VARIATION — significant keyword overlap
    exp_words = set(exp_norm.split())
    pred_words = set(pred_norm.split())
    # Remove common stopwords for overlap calculation
    stopwords = {"of", "the", "a", "an", "in", "on", "with", "and", "or", "disease", "syndrome", "acute", "chronic"}
    exp_sig = exp_words - stopwords
    pred_sig = pred_words - stopwords
    if exp_sig and pred_sig:
        overlap = exp_sig & pred_sig
        if len(overlap) >= 1 and len(overlap) / max(len(exp_sig), len(pred_sig)) >= 0.4:
            return "acceptable"

    # 4. SEVERE ERROR
    return "severe"


# ──────────────────────────────────────────────
# EVALUATOR
# ──────────────────────────────────────────────

class Evaluator:
    def __init__(self, output_dir: str = "results"):
        self.orchestrator = Orchestrator()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    async def run_experiment(self, experiment_name: str, test_cases: list) -> str:
        """Run full pipeline on all cases. Returns path to saved JSON."""
        logger.info(f"Starting experiment: {experiment_name} with {len(test_cases)} cases.")
        results = []

        for case in test_cases:
            # Data integrity gate
            missing = REQUIRED_CASE_FIELDS - set(case.keys())
            if missing:
                logger.warning(f"Skipping case {case.get('id', '???')}: missing fields {missing}")
                continue

            case_id = case["id"]

            logger.info(f"Processing case ID: {case_id}")
            result = await self.orchestrator.process_case(
                patient_text=case["patient_text"],
                history=case["history"],
                metadata={"case_id": case_id}
            )

            agents = result["agents"]
            fusion = result["fusion"]
            rac = result.get("rac", {})

            # Latency
            latency_data = {
                "intent_ms": agents.get("intent", {}).get("latency_ms", 0),
                "diagnosis_ms": agents.get("diagnosis", {}).get("latency_ms", 0),
                "treatment_ms": agents.get("treatment", {}).get("latency_ms", 0),
                "risk_ms": agents.get("risk", {}).get("latency_ms", 0),
                "total_ms": result["total_latency_ms"]
            }

            # Predictions
            diag_output = agents.get("diagnosis", {}).get("output")
            if diag_output is None:
                logger.warning(f"Case {case['id']}: Diagnosis agent returned no output.")
            predicted = diag_output.get("primary_diagnosis", "Unknown") if diag_output else "Unknown"
            fusion_diagnosis = fusion.get("final_diagnosis", "Unknown")

            expected = case["expected_diagnosis"]

            # Error classification (check both agent and fusion predictions)
            error_type_agent = _classify_error(expected, predicted)
            error_type_fusion = _classify_error(expected, fusion_diagnosis)

            # Take the better classification (fusion may correct an agent miss)
            error_priority = {"correct": 0, "semantic": 1, "acceptable": 2, "severe": 3}
            final_error = error_type_agent if error_priority[error_type_agent] <= error_priority[error_type_fusion] else error_type_fusion

            # Metrics
            confidence_diagnosis = float(diag_output.get("confidence", 0.0)) if diag_output else 0.0
            confidence_fusion = float(fusion.get("overall_confidence", 0.0))
            agreement_score = float(rac.get("agreement_score", 0.0)) if isinstance(rac, dict) else 0.0

            record = {
                "case_id": case["id"],
                "expected": expected,
                "predicted": predicted,
                "fusion_diagnosis": fusion_diagnosis,
                "is_exact_match": _normalize(expected) == _normalize(predicted) or _normalize(expected) == _normalize(fusion_diagnosis),
                "is_semantic_match": final_error in ("correct", "semantic"),
                "is_accurate": final_error in ("correct", "semantic"),
                "error_type": final_error,
                "confidence_diagnosis": confidence_diagnosis,
                "confidence_fusion": confidence_fusion,
                "agreement_score": agreement_score,
                "conflict_penalty": float(rac.get("conflict_penalty", 0.0)) if isinstance(rac, dict) else 0.0,

                "latency": latency_data
            }

            results.append(record)

        filepath = os.path.join(self.output_dir, f"{experiment_name}.json")
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to {filepath}")
        return filepath


# ──────────────────────────────────────────────
# STATISTICS GENERATOR (IEEE § VI)
# ──────────────────────────────────────────────

def compute_statistics(results_dir: str = "results"):
    """Load all split results and compute mean ± std for key metrics."""
    splits = ["dev_results", "val_results", "benchmark_results"]
    all_stats = {}

    for split in splits:
        filepath = os.path.join(results_dir, f"{split}.json")
        if not os.path.exists(filepath):
            logger.warning(f"Split file not found: {filepath}, skipping.")
            continue

        with open(filepath, 'r') as f:
            data = json.load(f)

        if not data:
            continue

        accuracies = [1.0 if d["is_accurate"] else 0.0 for d in data]
        confidences = [d["confidence_fusion"] for d in data]
        latencies = [d["latency"]["total_ms"] for d in data]
        agreements = [d["agreement_score"] for d in data]

        # Error distribution
        error_counts = {"correct": 0, "semantic": 0, "acceptable": 0, "severe": 0}
        for d in data:
            et = d.get("error_type", "severe")
            error_counts[et] = error_counts.get(et, 0) + 1
        total = len(data)
        error_pcts = {k: round(v / total * 100, 2) for k, v in error_counts.items()}

        all_stats[split] = {
            "num_cases": total,
            "mean_accuracy": round(float(np.mean(accuracies)), 4),
            "std_accuracy": round(float(np.std(accuracies)), 4),
            "mean_confidence": round(float(np.mean(confidences)), 4),
            "std_confidence": round(float(np.std(confidences)), 4),
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "std_latency_ms": round(float(np.std(latencies)), 2),
            "mean_agreement": round(float(np.mean(agreements)), 4),
            "std_agreement": round(float(np.std(agreements)), 4),
            "error_distribution_pct": error_pcts,
        }

    stats_path = os.path.join(results_dir, "statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=4)
    logger.info(f"Statistics saved to {stats_path}")
    return all_stats


# ──────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────

async def run_all_experiments():
    evaluator = Evaluator()

    # Run all 3 splits
    await evaluator.run_experiment("dev_results", DEV_CASES)
    evaluator2 = Evaluator()
    await evaluator2.run_experiment("val_results", VALIDATION_CASES)
    evaluator3 = Evaluator()
    await evaluator3.run_experiment("benchmark_results", BENCHMARK_CASES)

    # Compute aggregate statistics
    compute_statistics()


if __name__ == "__main__":
    asyncio.run(run_all_experiments())
