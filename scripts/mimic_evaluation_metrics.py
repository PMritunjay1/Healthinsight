"""
HealthInsight MIMIC Evaluation Metrics Engine
=============================================
Defines standard primary-diagnosis metrics with explicit denominators for multi-label
and single-label clinical ground truths under deterministic selective abstention.
"""

from typing import List, Dict, Any, Optional
from collections import Counter
import numpy as np

# Canonical 18 HealthInsight specialist classes
CANONICAL_18_CLASSES = [
    # Cardiology (6)
    'angina', 'arrhythmia', 'atrial fibrillation', 'coronary artery disease', 'heart failure', 'hypertension',
    # Pulmonology (6)
    'asthma', 'copd', 'pneumonia', 'respiratory_failure', 'pleural_effusion', 'pulmonary_embolism',
    # Neurology (6)
    'stroke_ischemic', 'intracranial_hemorrhage', 'epilepsy_seizures',
    'altered_mental_status_encephalopathy', 'transient_ischemic_attack', 'neuropathy_neurodegenerative'
]

# Normalization mapping for canonical representation
LABEL_NORMALIZATION = {
    c.lower().replace(" ", "_"): c for c in CANONICAL_18_CLASSES
}
for c in CANONICAL_18_CLASSES:
    LABEL_NORMALIZATION[c.lower()] = c

def normalize_label(label: Optional[str]) -> Optional[str]:
    """Normalize a disease label string to its canonical 18-class name or return None if unmapped."""
    if not label or label in ("Uncertain — refer to clinician", "Unknown", "Unknown Condition", "None"):
        return None
    key = label.strip().lower()
    if key in LABEL_NORMALIZATION:
        return LABEL_NORMALIZATION[key]
    key_under = key.replace(" ", "_")
    if key_under in LABEL_NORMALIZATION:
        return LABEL_NORMALIZATION[key_under]
    key_space = key.replace("_", " ")
    if key_space in LABEL_NORMALIZATION:
        return LABEL_NORMALIZATION[key_space]
    return label.strip()

def compute_case_outcome(
    predicted_diagnosis: Optional[str],
    ground_truth_labels: List[str],
    is_abstained: bool = False,
    is_exception: bool = False,
    is_leakage_excluded: bool = False
) -> Dict[str, Any]:
    """
    Computes diagnostic match status for a single encounter.
    """
    if is_leakage_excluded:
        return {
            "eligible": False,
            "status": "EXCLUDED_LEAKAGE",
            "is_abstained": False,
            "is_exception": False,
            "usable": False,
            "any_label_match": False,
            "strict_exact_set_match": False,
            "is_single_label": len(ground_truth_labels) == 1,
            "single_label_match": False
        }

    # Dedup ground truth
    clean_gt = list(dict.fromkeys([normalize_label(l) for l in ground_truth_labels if normalize_label(l) is not None]))
    n_gt = len(clean_gt)
    is_single = (n_gt == 1)

    if is_exception:
        return {
            "eligible": True,
            "status": "EXCEPTION",
            "is_abstained": False,
            "is_exception": True,
            "usable": False,
            "any_label_match": False,
            "strict_exact_set_match": False,
            "is_single_label": is_single,
            "single_label_match": False
        }

    if is_abstained or predicted_diagnosis == "Uncertain — refer to clinician":
        return {
            "eligible": True,
            "status": "ABSTAINED",
            "is_abstained": True,
            "is_exception": False,
            "usable": False,
            "any_label_match": False,
            "strict_exact_set_match": False,
            "is_single_label": is_single,
            "single_label_match": False
        }

    norm_pred = normalize_label(predicted_diagnosis)
    if norm_pred is None:
        return {
            "eligible": True,
            "status": "INVALID_PREDICTION",
            "is_abstained": False,
            "is_exception": False,
            "usable": True,
            "any_label_match": False,
            "strict_exact_set_match": False,
            "is_single_label": is_single,
            "single_label_match": False
        }

    # Check matches
    any_match = norm_pred in clean_gt
    # Strict multilabel exact set match is only true if ground truth has exactly 1 label AND it matches
    # because the system only outputs 1 primary diagnosis
    strict_set_match = (n_gt == 1 and clean_gt[0] == norm_pred)
    single_match = (is_single and any_match)

    return {
        "eligible": True,
        "status": "SUCCESS",
        "is_abstained": False,
        "is_exception": False,
        "usable": True,
        "normalized_prediction": norm_pred,
        "normalized_ground_truth": clean_gt,
        "any_label_match": any_match,
        "strict_exact_set_match": strict_set_match,
        "is_single_label": is_single,
        "single_label_match": single_match
    }

def aggregate_cohort_metrics(case_outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes population-level metrics with explicit denominators.
    """
    total_records = len(case_outcomes)
    excluded = sum(1 for c in case_outcomes if not c.get("eligible", True))
    eligible_cases = [c for c in case_outcomes if c.get("eligible", True)]
    n_eligible = len(eligible_cases)

    exceptions = sum(1 for c in eligible_cases if c["is_exception"])
    abstentions = sum(1 for c in eligible_cases if c["is_abstained"])
    usable = sum(1 for c in eligible_cases if c["usable"])

    # Denominators
    coverage = (usable / n_eligible * 100.0) if n_eligible > 0 else 0.0
    abstention_rate = (abstentions / n_eligible * 100.0) if n_eligible > 0 else 0.0
    exception_rate = (exceptions / n_eligible * 100.0) if n_eligible > 0 else 0.0

    # Any-label matches
    any_matches = sum(1 for c in eligible_cases if c["any_label_match"])
    selective_any_label_acc = (any_matches / usable * 100.0) if usable > 0 else 0.0
    full_cohort_any_label_acc = (any_matches / n_eligible * 100.0) if n_eligible > 0 else 0.0

    # Single-label subset
    single_eligible = [c for c in eligible_cases if c["is_single_label"]]
    n_single_eligible = len(single_eligible)
    single_usable = sum(1 for c in single_eligible if c["usable"])
    single_matches = sum(1 for c in single_eligible if c["single_label_match"])
    single_coverage = (single_usable / n_single_eligible * 100.0) if n_single_eligible > 0 else 0.0
    single_selective_acc = (single_matches / single_usable * 100.0) if single_usable > 0 else 0.0
    single_full_cohort_acc = (single_matches / n_single_eligible * 100.0) if n_single_eligible > 0 else 0.0

    # Multilabel exact set equality count
    strict_set_matches = sum(1 for c in eligible_cases if c["strict_exact_set_match"])
    strict_set_acc = (strict_set_matches / n_eligible * 100.0) if n_eligible > 0 else 0.0

    return {
        "total_source_encounters": total_records,
        "leakage_excluded_encounters": excluded,
        "eligible_cohort": n_eligible,
        "exceptions": exceptions,
        "exception_rate_pct": round(exception_rate, 2),
        "abstentions": abstentions,
        "abstention_rate_pct": round(abstention_rate, 2),
        "usable_encounters": usable,
        "coverage_pct": round(coverage, 2),
        "any_label_matches": any_matches,
        "selective_any_label_accuracy_pct": round(selective_any_label_acc, 2),
        "full_cohort_any_label_accuracy_pct": round(full_cohort_any_label_acc, 2),
        "strict_single_label_eligible": n_single_eligible,
        "strict_single_label_usable": single_usable,
        "strict_single_label_coverage_pct": round(single_coverage, 2),
        "strict_single_label_matches": single_matches,
        "strict_single_label_selective_acc_pct": round(single_selective_acc, 2),
        "strict_single_label_full_cohort_acc_pct": round(single_full_cohort_acc, 2),
        "strict_multilabel_exact_set_matches": strict_set_matches,
        "strict_multilabel_exact_set_acc_pct": round(strict_set_acc, 2)
    }
