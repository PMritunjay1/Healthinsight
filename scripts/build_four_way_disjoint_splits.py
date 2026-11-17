"""
HealthInsight-UQ: Four-Way Disjoint Split Construction
======================================================
Implements strict Four-Way Split Discipline (Specification Section 9.3):
  1. D_train: Model fitting only (unmodified).
  2. D_val_tune: Threshold tuning and calibrator selection (50% of validation).
  3. D_cal (clean_calib): Calibrator fitting, conformal quantile estimation, Beta trust updates.
     Subdivided from validation by deterministic patient hashing.
     STRICTLY 0% overlap with D_train, D_test, and D_val_tune.
  4. D_test: Single untouched holdout evaluation (39,594 records).

Exports:
  - datasets/processed/{specialist}/val_tune.json, val_tune.csv
  - datasets/processed/{specialist}/clean_calib.json, clean_calib.csv
  - datasets/metadata/four_way_split_manifest.json
"""

import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Set

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from scripts.utils.label_extractor import extract_ground_truth_labels, normalize_specialty

VAL_SPLIT_PATHS = {
    "Cardiology": os.path.join(WORKSPACE, "datasets", "processed", "common", "val_split.json"),
    "Pulmonology": os.path.join(WORKSPACE, "datasets", "processed", "pulmonologist", "val_split.json"),
    "Neurology": os.path.join(WORKSPACE, "datasets", "processed", "neurologist", "val_split.json"),
}

TRAIN_SPLIT_PATHS = {
    "Cardiology": os.path.join(WORKSPACE, "datasets", "processed", "common", "train_split.json"),
    "Pulmonology": os.path.join(WORKSPACE, "datasets", "processed", "pulmonologist", "train_split.json"),
    "Neurology": os.path.join(WORKSPACE, "datasets", "processed", "neurologist", "train_split.json"),
}

TEST_SPLIT_PATHS = {
    "Cardiology": os.path.join(WORKSPACE, "datasets", "processed", "common", "test_split.json"),
    "Pulmonology": os.path.join(WORKSPACE, "datasets", "processed", "pulmonologist", "test_split.json"),
    "Neurology": os.path.join(WORKSPACE, "datasets", "processed", "neurologist", "test_split.json"),
}

OUTPUT_DIRS = {
    "Cardiology": [
        os.path.join(WORKSPACE, "datasets", "processed", "common"),
        os.path.join(WORKSPACE, "datasets", "processed", "cardiologist"),
    ],
    "Pulmonology": [
        os.path.join(WORKSPACE, "datasets", "processed", "pulmonologist"),
    ],
    "Neurology": [
        os.path.join(WORKSPACE, "datasets", "processed", "neurologist"),
    ],
}

MANIFEST_PATH = os.path.join(WORKSPACE, "datasets", "metadata", "four_way_split_manifest.json")


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def assign_patient_partition(patient_id: str) -> str:
    """Deterministic, reproducible 50/50 patient partition assignment via SHA-256."""
    h = int(hashlib.sha256(str(patient_id).encode("utf-8")).hexdigest()[:8], 16)
    return "val_tune" if (h % 2 == 0) else "calibration"


def export_json_and_csv(records: List[Dict[str, Any]], base_path_no_ext: str, specialty: str):
    """Exports both .json and .csv versions of the dataset."""
    json_path = f"{base_path_no_ext}.json"
    csv_path = f"{base_path_no_ext}.csv"

    # Export JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    # Export CSV
    fieldnames = ["encounter_id", "patient_id", "text", "labels", "chief_complaint", "urgency"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            enc_id = r.get("encounter_id") or r.get("patient_id")
            pt_id = r.get("patient_id")
            txt = r.get("text") or ""
            labels = extract_ground_truth_labels(r, specialty)
            labels_str = ";".join(labels)
            cc = r.get("chief_complaint") or ""
            urg = r.get("urgency") or ""
            writer.writerow({
                "encounter_id": enc_id,
                "patient_id": pt_id,
                "text": txt,
                "labels": labels_str,
                "chief_complaint": cc,
                "urgency": urg,
            })


def build_four_way_splits() -> Dict[str, Any]:
    print("=" * 80)
    print("HealthInsight-UQ: Constructing Four-Way Disjoint Splits (Phase 2.2)")
    print("=" * 80)

    split_manifest: Dict[str, Any] = {
        "manifest_title": "HealthInsight-UQ Four-Way Disjoint Split Manifest",
        "timestamp": "2026-10-18T02:24:00+05:30",
        "discipline": "Section 9.3 Four-Way Partition Discipline",
        "specialties": {},
    }

    for spec in ["Cardiology", "Pulmonology", "Neurology"]:
        print(f"\nProcessing {spec}...")
        val_path = VAL_SPLIT_PATHS[spec]
        train_path = TRAIN_SPLIT_PATHS[spec]
        test_path = TEST_SPLIT_PATHS[spec]

        # Load train, val, test
        with open(train_path, "r", encoding="utf-8") as f:
            train_records = json.load(f)
        with open(val_path, "r", encoding="utf-8") as f:
            val_records = json.load(f)
        with open(test_path, "r", encoding="utf-8") as f:
            test_records = json.load(f)

        train_pts = {r.get("patient_id") for r in train_records if r.get("patient_id")}
        test_pts = {r.get("patient_id") for r in test_records if r.get("patient_id")}
        test_encs = {str(r.get("encounter_id") or r.get("patient_id")) for r in test_records}

        # Group validation records by patient
        patient_to_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in val_records:
            pt = r.get("patient_id")
            patient_to_records[pt].append(r)

        val_tune_records: List[Dict[str, Any]] = []
        cal_records: List[Dict[str, Any]] = []

        val_tune_pts: Set[str] = set()
        cal_pts: Set[str] = set()

        for pt, recs in patient_to_records.items():
            part = assign_patient_partition(pt)
            if part == "val_tune":
                val_tune_pts.add(pt)
                val_tune_records.extend(recs)
            else:
                cal_pts.add(pt)
                cal_records.extend(recs)

        cal_encs = {str(r.get("encounter_id") or r.get("patient_id")) for r in cal_records}
        val_tune_encs = {str(r.get("encounter_id") or r.get("patient_id")) for r in val_tune_records}

        print(f"  Total Validation: {len(val_records)} records across {len(patient_to_records)} patients")
        print(f"  -> D_val_tune:    {len(val_tune_records)} records across {len(val_tune_pts)} patients")
        print(f"  -> D_cal (clean): {len(cal_records)} records across {len(cal_pts)} patients")

        # Guardrail assertions
        assert len(train_pts & cal_pts) == 0, f"{spec}: D_train and D_cal overlap!"
        assert len(train_pts & val_tune_pts) == 0, f"{spec}: D_train and D_val_tune overlap!"
        assert len(test_pts & cal_pts) == 0, f"{spec}: D_test and D_cal overlap!"
        assert len(test_pts & val_tune_pts) == 0, f"{spec}: D_test and D_val_tune overlap!"
        assert len(cal_pts & val_tune_pts) == 0, f"{spec}: D_cal and D_val_tune overlap!"
        assert len(cal_encs & test_encs) == 0, f"{spec}: Encounter overlap between D_cal and D_test!"

        print(f"  [PASS] Zero patient or encounter leakage verified for {spec}.")

        # Export to target directories
        exported_files = {}
        for out_dir in OUTPUT_DIRS[spec]:
            os.makedirs(out_dir, exist_ok=True)
            vt_base = os.path.join(out_dir, "val_tune")
            cal_base = os.path.join(out_dir, "clean_calib")

            export_json_and_csv(val_tune_records, vt_base, spec)
            export_json_and_csv(cal_records, cal_base, spec)

            exported_files[out_dir] = {
                "val_tune_json": f"{vt_base}.json",
                "val_tune_csv": f"{vt_base}.csv",
                "clean_calib_json": f"{cal_base}.json",
                "clean_calib_csv": f"{cal_base}.csv",
                "val_tune_sha256": compute_sha256(f"{vt_base}.json"),
                "clean_calib_sha256": compute_sha256(f"{cal_base}.json"),
            }

        split_manifest["specialties"][spec] = {
            "d_train_count": len(train_records),
            "d_train_patients": len(train_pts),
            "d_val_tune_count": len(val_tune_records),
            "d_val_tune_patients": len(val_tune_pts),
            "d_cal_count": len(cal_records),
            "d_cal_patients": len(cal_pts),
            "d_test_count": len(test_records),
            "d_test_patients": len(test_pts),
            "leakage_checks": {
                "train_vs_cal_patient_overlap": 0,
                "test_vs_cal_patient_overlap": 0,
                "val_tune_vs_cal_patient_overlap": 0,
                "test_vs_cal_encounter_overlap": 0,
            },
            "exported_files": exported_files,
        }

    # Save Manifest
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    print(f"\nManifest successfully written to: {MANIFEST_PATH}")
    return split_manifest


if __name__ == "__main__":
    build_four_way_splits()
