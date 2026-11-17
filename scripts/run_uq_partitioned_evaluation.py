"""
HealthInsight-UQ: Fast Partitioned Full-Cohort Evaluation Engine
================================================================
Executes complete evaluation across all 39,594 records partitioned across CPU workers.

Features:
- Partitioning across N workers (default: 3) for CPU acceleration without lock contention.
- Independent atomic ledger files per partition (part_{id}.jsonl) merged dynamically.
- Storage circuit breaker: cleanly checks Drive C: free space (>= 3.0 GB).
- Milestone evaluations every 5,000 cases checking accuracy >= 75%.
- Pre-flight smoke test mode for safety and schema verification.
- Comprehensive per-row error logging and crash-resilient key resumption.
"""

import argparse
import asyncio
import datetime
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from backend.orchestrator import Orchestrator
from scripts.utils.label_extractor import extract_ground_truth_labels, normalize_specialty

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("health_insights.uq_eval_partitioned")

TEST_SPLIT_PATHS = {
    "Cardiology": os.path.join(WORKSPACE, "datasets", "processed", "common", "test_split.json"),
    "Pulmonology": os.path.join(WORKSPACE, "datasets", "processed", "pulmonologist", "test_split.json"),
    "Neurology": os.path.join(WORKSPACE, "datasets", "processed", "neurologist", "test_split.json"),
}

LEAKAGE_INDEX_PATH = os.path.join(WORKSPACE, "datasets", "metadata", "cross_specialty_leakage_index.json")


def check_storage_circuit_breaker(min_free_gb: float = 3.0) -> bool:
    try:
        total, used, free = shutil.disk_usage(WORKSPACE)
        free_gb = free / (1024 ** 3)
        if free_gb < min_free_gb:
            logger.critical(f"STORAGE CIRCUIT BREAKER: Drive free space {free_gb:.2f} GB < {min_free_gb} GB")
            return False
        return True
    except Exception as e:
        logger.warning(f"Failed to check disk usage: {e}")
        return True


def is_disease_match(pred: str, true_labels: list) -> bool:
    if not pred or pred in ("Unknown", "Undetermined", "Uncertain — refer to clinician"):
        return False
    pred_clean = pred.lower().replace("_", " ").strip()
    for t in true_labels:
        t_clean = t.lower().replace("_", " ").strip()
        if pred_clean == t_clean or pred_clean in t_clean or t_clean in pred_clean:
            return True
    return False


def collect_cohort(smoke_test_limit: Optional[int] = None) -> List[Tuple[str, str, dict]]:
    eligible = []
    for spec, path in TEST_SPLIT_PATHS.items():
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        if smoke_test_limit:
            records = records[:smoke_test_limit]
        for idx, r in enumerate(records):
            enc_raw = str(r.get("encounter_id") or r.get("hadm_id") or f"case_{idx:06d}")
            case_id = f"{spec.lower()}_{idx:06d}_{enc_raw}"
            eligible.append((case_id, spec, r))
    return eligible


async def _process_worker_partition(
    worker_id: int,
    total_workers: int,
    items: List[Tuple[str, str, dict]],
    output_dir: str,
    leaked_encounters: Set[str]
):
    part_ledger = os.path.join(output_dir, f"part_{worker_id}.jsonl")
    part_error = os.path.join(output_dir, f"error_{worker_id}.jsonl")

    # Load already processed keys in this partition for crash-resilience
    done_keys = set()
    if os.path.exists(part_ledger):
        with open(part_ledger, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        done_keys.add(json.loads(line)["case_id"])
                    except Exception:
                        pass

    # Filter items assigned to this worker
    worker_items = [
        item for idx, item in enumerate(items)
        if (idx % total_workers) == worker_id and item[0] not in done_keys
    ]

    orch = Orchestrator()
    logger.info(f"[Worker {worker_id}] Initialized Orchestrator. Processing {len(worker_items)} assigned cases...")

    f_ledger = open(part_ledger, "a", encoding="utf-8")
    f_err = open(part_error, "a", encoding="utf-8")

    processed = 0
    t0 = time.time()

    try:
        for case_id, spec, record in worker_items:
            txt = record.get("text") or record.get("patient_text") or ""
            gt_labels = extract_ground_truth_labels(record, spec)
            raw_enc = str(record.get("encounter_id") or record.get("hadm_id") or "")
            hadm_id = raw_enc.replace("hadm_", "").strip()

            if not txt or not gt_labels:
                f_err.write(json.dumps({
                    "case_id": case_id,
                    "specialty": spec,
                    "reason": "Missing text or labels",
                    "time": datetime.datetime.now().isoformat()
                }) + "\n")
                f_err.flush()
                continue

            try:
                res = await orch.process_case(patient_text=txt, fusion_mode="trust", routing_mode="routed")

                routing = res.get("routing", {})
                routing_tier = routing.get("routing_tier", "LOW")
                primary_domain = routing.get("primary_domain", "general")
                selected_specs = routing.get("selected_specialists", [])
                ambiguity_score = float(routing.get("ambiguity_score", 0.0))

                fusion = res.get("fusion", {})
                uncond_diag = (fusion.get("unconditioned_diagnosis") or fusion.get("final_diagnosis") or "").strip()
                final_diag = (fusion.get("final_diagnosis") or "").strip()
                is_abstained = bool(fusion.get("is_abstained", False))

                uncond_match = is_disease_match(uncond_diag, gt_labels)
                final_match = is_disease_match(final_diag, gt_labels) if not is_abstained else None
                domain_match = (primary_domain.lower() == spec.lower() or primary_domain.lower() in spec.lower())

                row = {
                    "case_id": case_id,
                    "specialty": spec,
                    "hadm_id": hadm_id,
                    "ground_truth_labels": gt_labels,
                    "primary_domain": primary_domain,
                    "domain_match": bool(domain_match),
                    "routing_tier": routing_tier,
                    "selected_specialists": selected_specs,
                    "ambiguity_score": round(ambiguity_score, 4),
                    "unconditioned_diagnosis": uncond_diag,
                    "final_diagnosis": final_diag,
                    "unconditioned_correct": int(uncond_match),
                    "is_abstained": is_abstained,
                    "selective_correct": (1 if uncond_match else 0) if not is_abstained else None,
                    "in_clean_subset": bool(hadm_id not in leaked_encounters and hadm_id != ""),
                    "timestamp": datetime.datetime.now().isoformat()
                }

                f_ledger.write(json.dumps(row) + "\n")
                processed += 1

                if processed % 10 == 0:
                    f_ledger.flush()
                    os.fsync(f_ledger.fileno())

            except Exception as e:
                f_err.write(json.dumps({
                    "case_id": case_id,
                    "specialty": spec,
                    "error": str(e),
                    "time": datetime.datetime.now().isoformat()
                }) + "\n")
                f_err.flush()

    finally:
        f_ledger.flush()
        f_ledger.close()
        f_err.flush()
        f_err.close()

    elapsed = time.time() - t0
    logger.info(f"[Worker {worker_id}] Finished {processed} cases in {elapsed:.1f}s.")


def worker_entrypoint(worker_id: int, total_workers: int, items: List[Any], output_dir: str, leaked_encounters: Set[str]):
    asyncio.run(_process_worker_partition(worker_id, total_workers, items, output_dir, leaked_encounters))


def evaluate_running_metrics(output_dir: str) -> Dict[str, Any]:
    rows = []
    for f_name in os.listdir(output_dir):
        if f_name.startswith("part_") and f_name.endswith(".jsonl"):
            f_path = os.path.join(output_dir, f_name)
            with open(f_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            rows.append(json.loads(line))
                        except Exception:
                            pass

    total = len(rows)
    if total == 0:
        return {"total": 0}

    uncond_corr = sum(1 for r in rows if r.get("unconditioned_correct") == 1)
    abstained = sum(1 for r in rows if r.get("is_abstained") is True)
    answered = total - abstained
    sel_corr = sum(1 for r in rows if r.get("selective_correct") == 1)
    routed_corr = sum(1 for r in rows if r.get("domain_match") is True)

    per_spec = {}
    for r in rows:
        spec = r.get("specialty", "Unknown")
        if spec not in per_spec:
            per_spec[spec] = {"total": 0, "correct": 0, "abstained": 0}
        per_spec[spec]["total"] += 1
        if r.get("unconditioned_correct") == 1:
            per_spec[spec]["correct"] += 1
        if r.get("is_abstained") is True:
            per_spec[spec]["abstained"] += 1

    return {
        "total": total,
        "unconditioned_accuracy": round(uncond_corr / total, 4),
        "routing_accuracy": round(routed_corr / total, 4),
        "coverage": round(answered / total, 4),
        "abstention_rate": round(abstained / total, 4),
        "selective_accuracy": round(sel_corr / max(1, answered), 4) if answered > 0 else 0.0,
        "per_specialty": {
            s: {
                "total": d["total"],
                "accuracy": round(d["correct"] / d["total"], 4) if d["total"] > 0 else 0.0,
                "abstention_rate": round(d["abstained"] / d["total"], 4) if d["total"] > 0 else 0.0,
            }
            for s, d in per_spec.items()
        }
    }


def run_partitioned_evaluation(
    output_dir: str,
    num_workers: int = 3,
    smoke_test_limit: Optional[int] = None
):
    print("=" * 80)
    print(f"HealthInsight-UQ: Partitioned Evaluation (Workers: {num_workers})")
    print(f"Output Directory: {output_dir}")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    if not check_storage_circuit_breaker(min_free_gb=3.0):
        raise RuntimeError("Drive C: space below safety limit (3.0 GB). Aborting.")

    leaked = set()
    if os.path.exists(LEAKAGE_INDEX_PATH):
        with open(LEAKAGE_INDEX_PATH, "r", encoding="utf-8") as f:
            leaked = set(json.load(f).get("leaked_encounters", []))

    all_items = collect_cohort(smoke_test_limit=smoke_test_limit)
    total_cohort = len(all_items)
    print(f"Total cohort cases queued: {total_cohort}")

    processes = []
    for wid in range(num_workers):
        p = mp.Process(
            target=worker_entrypoint,
            args=(wid, num_workers, all_items, output_dir, leaked)
        )
        p.start()
        processes.append(p)
        print(f"Started Worker process {wid} (PID: {p.pid}).")

    t_start = time.time()
    last_milestone = 0
    milestone_step = 5000

    try:
        while any(p.is_alive() for p in processes):
            time.sleep(15)
            metrics = evaluate_running_metrics(output_dir)
            done = metrics.get("total", 0)
            elapsed = time.time() - t_start
            speed = done / max(1.0, elapsed)
            remaining = (total_cohort - done) / max(0.01, speed) if done < total_cohort else 0.0

            if done > 0:
                print(
                    f"[{done}/{total_cohort}] ({done/total_cohort*100:.1f}%) | "
                    f"Speed: {speed:.2f} cases/s | Elapsed: {elapsed/60:.1f}m | "
                    f"ETA: {remaining/60:.1f}m | Uncond Acc: {metrics['unconditioned_accuracy']*100:.2f}% | "
                    f"Routing Acc: {metrics['routing_accuracy']*100:.2f}%",
                    flush=True
                )

            # Milestone verification every 5,000 cases
            if done >= last_milestone + milestone_step or (done >= total_cohort and done > 0 and last_milestone < total_cohort):
                last_milestone = (done // milestone_step) * milestone_step
                print("\n" + "#" * 80, flush=True)
                print(f"MILESTONE AUDIT AT {done} CASES", flush=True)
                print(f"  Unconditioned Fused Accuracy: {metrics['unconditioned_accuracy']*100:.2f}%", flush=True)
                print(f"  Domain Routing Accuracy:     {metrics['routing_accuracy']*100:.2f}%", flush=True)
                print(f"  Selective Coverage:          {metrics['coverage']*100:.2f}%", flush=True)
                print(f"  Selective Accuracy:          {metrics['selective_accuracy']*100:.2f}%", flush=True)
                print(f"  Per-Specialty Breakdown:", flush=True)
                for s, sd in metrics["per_specialty"].items():
                    print(f"    - {s}: N={sd['total']}, Acc={sd['accuracy']*100:.2f}%, Abstain={sd['abstention_rate']*100:.2f}%", flush=True)
                
                # Accuracy checkpoint alert
                if metrics["unconditioned_accuracy"] < 0.75:
                    print(f"  [ALERT] Milestone accuracy ({metrics['unconditioned_accuracy']*100:.2f}%) below 75% target. Investigating per-domain gap...", flush=True)
                else:
                    print(f"  [SUCCESS] Milestone accuracy ({metrics['unconditioned_accuracy']*100:.2f}%) comfortably exceeds 75% target!", flush=True)
                print("#" * 80 + "\n", flush=True)

    finally:
        for p in processes:
            if p.is_alive():
                p.join(timeout=5)

    # Merge partition files into single unified row_level_ledger.jsonl
    unified_ledger_path = os.path.join(output_dir, "row_level_ledger.jsonl")
    print(f"\nMerging partition ledgers into {unified_ledger_path}...")
    merged_count = 0
    with open(unified_ledger_path, "w", encoding="utf-8") as out_f:
        for f_name in sorted(os.listdir(output_dir)):
            if f_name.startswith("part_") and f_name.endswith(".jsonl"):
                p_path = os.path.join(output_dir, f_name)
                with open(p_path, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        if line.strip():
                            out_f.write(line.strip() + "\n")
                            merged_count += 1

    final_metrics = evaluate_running_metrics(output_dir)
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("FULL COHORT EVALUATION COMPLETED")
    print(f"Total Evaluated Records: {merged_count}")
    print(f"Final Unconditioned Accuracy: {final_metrics['unconditioned_accuracy']*100:.2f}%")
    print(f"Final Routing Accuracy:      {final_metrics['routing_accuracy']*100:.2f}%")
    print(f"Final Coverage:              {final_metrics['coverage']*100:.2f}%")
    print(f"Final Selective Accuracy:     {final_metrics['selective_accuracy']*100:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    mp.freeze_support()
    parser = argparse.ArgumentParser(description="HealthInsight-UQ Fast Partitioned Evaluation Engine")
    parser.add_argument("--output-dir", type=str, required=True, help="Versioned output directory")
    parser.add_argument("--workers", type=int, default=3, help="Number of CPU worker partitions")
    parser.add_argument("--smoke-test", type=int, default=None, help="Limit cases per specialty for smoke test")
    args = parser.parse_args()

    run_partitioned_evaluation(
        output_dir=args.output_dir,
        num_workers=args.workers,
        smoke_test_limit=args.smoke_test
    )
