"""
HealthInsight-UQ: Real-Time Evaluation Progress Monitor
======================================================
Usage:
    python scripts/check_progress.py
"""

import glob
import json
import os
import sys

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(WORKSPACE, "results", "healthinsight_uq_full_cohort_20261018_173000")
TOTAL_COHORT = 39594


def main():
    if not os.path.exists(OUTPUT_DIR):
        print(f"Target evaluation directory does not exist yet: {OUTPUT_DIR}")
        return

    part_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "part_*.jsonl")))

    total_cases = 0
    correct_cases = 0
    correct_routing = 0
    abstained_cases = 0
    per_part = {}
    per_spec = {}

    for pf in part_files:
        p_name = os.path.basename(pf)
        p_count = 0
        p_correct = 0
        with open(pf, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    p_count += 1
                    total_cases += 1
                    
                    if row.get("unconditioned_correct") == 1:
                        p_correct += 1
                        correct_cases += 1
                    
                    if row.get("domain_match") is True:
                        correct_routing += 1
                    
                    if row.get("is_abstained") is True:
                        abstained_cases += 1

                    spec = row.get("specialty", "Unknown")
                    if spec not in per_spec:
                        per_spec[spec] = {"total": 0, "correct": 0}
                    per_spec[spec]["total"] += 1
                    if row.get("unconditioned_correct") == 1:
                        per_spec[spec]["correct"] += 1

                except Exception:
                    pass

        per_part[p_name] = {"count": p_count, "correct": p_correct}

    pct = (total_cases / TOTAL_COHORT) * 100 if TOTAL_COHORT > 0 else 0.0
    acc = (correct_cases / total_cases * 100) if total_cases > 0 else 0.0
    route_acc = (correct_routing / total_cases * 100) if total_cases > 0 else 0.0
    coverage = ((total_cases - abstained_cases) / total_cases * 100) if total_cases > 0 else 0.0

    print("=" * 70)
    print("HEALTHINSIGHT-UQ: LIVE EVALUATION PROGRESS")
    print("=" * 70)
    print(f"Total Completed:          {total_cases:,} / {TOTAL_COHORT:,} ({pct:.2f}%)")
    print(f"Running Unconditioned Accuracy: {acc:.2f}% ({correct_cases:,}/{total_cases:,})")
    print(f"Running Routing Accuracy:      {route_acc:.2f}% ({correct_routing:,}/{total_cases:,})")
    print(f"Selective Coverage:            {coverage:.2f}% (Abstained: {abstained_cases:,})")
    print("-" * 70)
    print("Per-Worker Partition Status:")
    for p_name, data in per_part.items():
        p_acc = (data["correct"] / data["count"] * 100) if data["count"] > 0 else 0.0
        print(f"  • {p_name}: {data['count']:,} cases | Acc: {p_acc:.2f}%")
    if per_spec:
        print("-" * 70)
        print("Per-Specialty Domain Breakdown:")
        for s_name, data in per_spec.items():
            s_acc = (data["correct"] / data["total"] * 100) if data["total"] > 0 else 0.0
            print(f"  • {s_name}: {data['correct']:,} / {data['total']:,} ({s_acc:.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
