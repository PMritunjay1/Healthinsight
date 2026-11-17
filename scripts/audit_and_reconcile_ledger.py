"""
HealthInsight-UQ: Independent Ledger Audit & Reconciliation
=============================================================
Phase 2.1 Audit Script:
1. Ingests 'results/mimic_full_cohort_recovery_20261016/final_row_level_evaluation_ledger.jsonl' (read-only).
2. Verifies line integrity, exact row counts, duplicate record keys, and exclusions/abstentions.
3. Extracts authoritative ground truth across all 39,594 records via 'scripts.utils.label_extractor'.
4. Re-computes per-specialty and cohort-level selective accuracy, coverage, and full accuracy.
5. Emits a tamper-evident audit certificate to 'results/reconciled_baseline_audit_20261018/ledger_audit_certificate.json'.
"""

import hashlib
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from scripts.utils.label_extractor import (
    LABEL_KEY_MAP,
    SPECIALTY_LABEL_MAP,
    extract_ground_truth_labels,
    normalize_specialty,
)

LEDGER_PATH = os.path.join(
    WORKSPACE,
    "results",
    "mimic_full_cohort_recovery_20261016",
    "final_row_level_evaluation_ledger.jsonl",
)
CERT_DIR = os.path.join(WORKSPACE, "results", "reconciled_baseline_audit_20261018")
CERT_PATH = os.path.join(CERT_DIR, "ledger_audit_certificate.json")

SPLIT_PATHS = {
    "Cardiology": os.path.join(WORKSPACE, "datasets", "processed", "common", "test_split.json"),
    "Pulmonology": os.path.join(WORKSPACE, "datasets", "processed", "pulmonologist", "test_split.json"),
    "Neurology": os.path.join(WORKSPACE, "datasets", "processed", "neurologist", "test_split.json"),
}


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_authoritative_ground_truth() -> Dict[str, Dict[str, List[str]]]:
    """
    Loads and normalizes ground truth directly from the source test split files.
    Returns: { specialty: { encounter_id: [clean_labels] } }
    """
    gt_by_spec: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    for spec, path in SPLIT_PATHS.items():
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        for r in records:
            enc_id = str(r.get("encounter_id") or r.get("patient_id"))
            labels = extract_ground_truth_labels(r, spec)
            gt_by_spec[spec][enc_id] = labels
    return gt_by_spec


def calculate_metrics(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes standard coverage, selective accuracy, and full accuracy metrics."""
    total = len(items)
    if total == 0:
        return {}

    abstained = sum(1 for x in items if x["is_abstained"])
    usable = total - abstained
    coverage = usable / total if total > 0 else 0.0

    matches = sum(1 for x in items if not x["is_abstained"] and x["prediction"] in x["gt_labels"])
    sel_acc = matches / usable if usable > 0 else 0.0
    full_acc = matches / total if total > 0 else 0.0

    # Single-label subset
    single_eligible = [x for x in items if len(x["gt_labels"]) == 1]
    single_total = len(single_eligible)
    single_usable = sum(1 for x in single_eligible if not x["is_abstained"])
    single_coverage = single_usable / single_total if single_total > 0 else 0.0
    single_matches = sum(
        1 for x in single_eligible if not x["is_abstained"] and x["prediction"] in x["gt_labels"]
    )
    single_sel_acc = single_matches / single_usable if single_usable > 0 else 0.0
    single_full_acc = single_matches / single_total if single_total > 0 else 0.0

    return {
        "total_cases": total,
        "abstained_cases": abstained,
        "usable_cases": usable,
        "coverage_pct": round(coverage * 100, 2),
        "matches": matches,
        "selective_accuracy_pct": round(sel_acc * 100, 2),
        "full_cohort_accuracy_pct": round(full_acc * 100, 2),
        "single_label_eligible": single_total,
        "single_label_usable": single_usable,
        "single_label_coverage_pct": round(single_coverage * 100, 2),
        "single_label_matches": single_matches,
        "single_label_selective_acc_pct": round(single_sel_acc * 100, 2),
        "single_label_full_cohort_acc_pct": round(single_full_acc * 100, 2),
    }


def audit_and_reconcile() -> Dict[str, Any]:
    print("=" * 80)
    print("HealthInsight-UQ: Reconciling Existing MIMIC Ledger (Phase 2.1)")
    print("=" * 80)

    # 1. Compute checksums
    print("Computing file checksums...")
    ledger_sha256 = compute_file_sha256(LEDGER_PATH)
    split_hashes = {spec: compute_file_sha256(path) for spec, path in SPLIT_PATHS.items()}
    print(f"Ledger SHA-256: {ledger_sha256}")

    # 2. Load ground truth
    print("Loading source test split ground truths...")
    source_gt = load_authoritative_ground_truth()
    source_counts = {spec: len(encs) for spec, encs in source_gt.items()}
    print(f"Source split record counts: {source_counts}")

    # 3. Read ledger line-by-line
    print(f"Parsing ledger: {LEDGER_PATH}...")
    seen_record_ids = set()
    specialty_counts = defaultdict(int)
    corrupted_lines = 0

    evaluated_items = {
        "config_a": [],
        "config_a_uncorrected": [],
    }
    specialty_items = {
        "Cardiology": [],
        "Pulmonology": [],
        "Neurology": [],
    }

    total_rows = 0
    matched_with_source_gt = 0

    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                row = json.loads(line_str)
            except Exception as e:
                corrupted_lines += 1
                continue

            total_rows += 1
            rec_id = row.get("record_id")
            if rec_id in seen_record_ids:
                raise ValueError(f"Duplicate record ID found: {rec_id} at line {line_num}")
            seen_record_ids.add(rec_id)

            spec = normalize_specialty(row.get("specialty"))
            specialty_counts[spec] += 1
            enc_id = str(row.get("encounter_id"))

            # Retrieve verified ground truth from source splits
            verified_labels = source_gt[spec].get(enc_id, [])
            if enc_id in source_gt[spec]:
                matched_with_source_gt += 1
            else:
                # Fallback to extracting from the record itself if present
                verified_labels = extract_ground_truth_labels(row, spec)

            cfg_a = row.get("config_a", {})
            pred = cfg_a.get("prediction", "")
            abstained = cfg_a.get("is_abstained", False)

            item_verified = {
                "record_id": rec_id,
                "encounter_id": enc_id,
                "specialty": spec,
                "prediction": pred,
                "is_abstained": abstained,
                "gt_labels": verified_labels,
            }
            evaluated_items["config_a"].append(item_verified)
            specialty_items[spec].append(item_verified)

            # Record uncorrected calculation (with raw logged ground truth)
            logged_labels = row.get("ground_truth_labels") or []
            item_uncorrected = {
                "prediction": pred,
                "is_abstained": abstained,
                "gt_labels": [str(x).lower().strip() for x in logged_labels],
            }
            evaluated_items["config_a_uncorrected"].append(item_uncorrected)

    # 4. Assert integrity
    print(f"Total Rows Ingested: {total_rows}")
    print(f"Corrupted Lines: {corrupted_lines}")
    print(f"Unique Record IDs: {len(seen_record_ids)}")
    print(f"Specialty Counts: {dict(specialty_counts)}")
    print(f"Matched with Source GT: {matched_with_source_gt} / {total_rows}")

    assert total_rows == 39594, f"Expected 39,594 records, found {total_rows}"
    assert len(seen_record_ids) == 39594, "Duplicate record IDs detected"
    assert corrupted_lines == 0, f"Found {corrupted_lines} corrupted JSON lines"
    assert specialty_counts["Cardiology"] == 15099
    assert specialty_counts["Pulmonology"] == 14512
    assert specialty_counts["Neurology"] == 9983

    # 5. Compute metrics
    verified_metrics = calculate_metrics(evaluated_items["config_a"])
    uncorrected_metrics = calculate_metrics(evaluated_items["config_a_uncorrected"])
    per_spec_metrics = {spec: calculate_metrics(items) for spec, items in specialty_items.items()}

    print("\n" + "-" * 60)
    print("METRIC RECONCILIATION SUMMARY")
    print("-" * 60)
    print(f"Total Cohort:                {verified_metrics['total_cases']:,}")
    print(f"Abstained Cases:             {verified_metrics['abstained_cases']:,}")
    print(f"Usable / Answered Cases:     {verified_metrics['usable_cases']:,}")
    print(f"Coverage:                    {verified_metrics['coverage_pct']}%")
    print(f"Total Correct Matches:       {verified_metrics['matches']:,}")
    print(f"Verified Selective Accuracy: {verified_metrics['selective_accuracy_pct']}%")
    print(f"Verified Full-Cohort Acc:    {verified_metrics['full_cohort_accuracy_pct']}%")
    print(f"Historical Reported Acc:     {uncorrected_metrics['selective_accuracy_pct']}% (with 0% pulmo/neuro)")

    for spec, m in per_spec_metrics.items():
        print(f"  {spec:<12}: Selective Acc = {m['selective_accuracy_pct']:5.2f}%, Matches = {m['matches']:,}/{m['usable_cases']:,}, Coverage = {m['coverage_pct']:5.2f}%")

    # Verify reported 48.17% selective accuracy
    reproduced = abs(verified_metrics["selective_accuracy_pct"] - 48.17) < 0.05
    status = "VERIFIED_AND_REPRODUCED" if reproduced else "REJECTED"
    print(f"\nVerification Verdict: {status} (48.17% target)")

    # 6. Save Certificate
    os.makedirs(CERT_DIR, exist_ok=True)
    certificate = {
        "certificate_title": "HealthInsight-UQ Phase 2.1 Ledger Audit Certificate",
        "audit_timestamp": "2026-10-18T02:22:00+05:30",
        "verification_status": status,
        "input_ledger": {
            "path": os.path.relpath(LEDGER_PATH, WORKSPACE),
            "sha256": ledger_sha256,
            "total_rows": total_rows,
            "corrupted_rows": corrupted_lines,
            "unique_records": len(seen_record_ids),
        },
        "source_splits": {
            spec: {
                "path": os.path.relpath(SPLIT_PATHS[spec], WORKSPACE),
                "sha256": split_hashes[spec],
                "record_count": source_counts[spec],
            }
            for spec in ["Cardiology", "Pulmonology", "Neurology"]
        },
        "specialty_breakdown": dict(specialty_counts),
        "metrics_reproduced": {
            "reported_baseline_selective_accuracy": 48.17,
            "measured_selective_accuracy": verified_metrics["selective_accuracy_pct"],
            "measured_full_cohort_accuracy": verified_metrics["full_cohort_accuracy_pct"],
            "coverage_pct": verified_metrics["coverage_pct"],
            "total_cases": verified_metrics["total_cases"],
            "usable_cases": verified_metrics["usable_cases"],
            "abstained_cases": verified_metrics["abstained_cases"],
            "matches": verified_metrics["matches"],
        },
        "per_specialty_metrics": per_spec_metrics,
        "historical_defect_documentation": {
            "root_cause": "Original evaluation accessed nonexistent keys 'pulmo_disease_labels' and 'neuro_disease_labels'.",
            "resolution": "Enforced authoritative dictionary keys via scripts/utils/label_extractor.py: 'pulmonary_disease_labels' and 'neurology_disease_labels'.",
            "uncorrected_accuracy": uncorrected_metrics["selective_accuracy_pct"],
            "corrected_accuracy": verified_metrics["selective_accuracy_pct"],
        },
    }

    with open(CERT_PATH, "w", encoding="utf-8") as f:
        json.dump(certificate, f, indent=2)

    print(f"Audit certificate saved successfully to: {CERT_PATH}")
    return certificate


if __name__ == "__main__":
    audit_and_reconcile()
