"""
HealthInsight-UQ: Train Beta Trust Dynamics on Clean D_cal (Phase 2.8)
=====================================================================
Initializes Beta trust priors at Beta(1.0, 1.0) and computes Bayesian posterior
updates strictly using clean calibration data (D_cal):
    - alpha_{i,d} += 1.0 if specialist prediction matches gold-truth
    - beta_{i,d} += 1.0 if specialist prediction does not match
Guarantees:
    - Never uses validation tuning (D_val_tune) or test data (D_test).
    - Serializes posteriors atomically to backend/models/trust_posteriors.json.
    - Emits trust manifest for transparency.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Dict, List

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from backend.trust_tracker import BetaTrustStore, SPECIALISTS, DOMAINS
from backend.agents.specialist_agents import (
    CardiologistSpecialistAgent,
    PulmonologistSpecialistAgent,
    NeurologistSpecialistAgent,
)
from scripts.utils.label_extractor import extract_ground_truth_labels

CAL_PATHS = {
    "Cardiology": os.path.join(WORKSPACE, "datasets", "processed", "common", "clean_calib.json"),
    "Pulmonology": os.path.join(WORKSPACE, "datasets", "processed", "pulmonologist", "clean_calib.json"),
    "Neurology": os.path.join(WORKSPACE, "datasets", "processed", "neurologist", "clean_calib.json"),
}

OUTPUT_TRUST_PATH = os.path.join(WORKSPACE, "backend", "models", "trust_posteriors.json")
MANIFEST_PATH = os.path.join(WORKSPACE, "backend", "models", "trust_manifest.json")


async def train_trust_posteriors(samples_per_specialty: int = 200):
    print("=" * 80)
    print("HealthInsight-UQ: Bayesian Beta Trust Training on Clean D_cal (Phase 2.8)")
    print("=" * 80)

    store = BetaTrustStore(storage_path=OUTPUT_TRUST_PATH)
    # Re-initialize priors
    store._init_priors()

    agents = {
        "cardiologist": CardiologistSpecialistAgent(),
        "pulmonologist": PulmonologistSpecialistAgent(),
        "neurologist": NeurologistSpecialistAgent(),
    }

    processed_counts = {spec: 0 for spec in SPECIALISTS}
    match_counts = {spec: 0 for spec in SPECIALISTS}

    for spec_domain, path in CAL_PATHS.items():
        print(f"\nProcessing {spec_domain} calibration cases from: {path}")
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        cases_to_process = records[:samples_per_specialty]
        for idx, r in enumerate(cases_to_process):
            txt = r.get("text") or r.get("patient_text") or ""
            gt_labels = extract_ground_truth_labels(r, spec_domain)
            if not txt or not gt_labels:
                continue

            # Evaluate each specialist on this narrative
            for spec_key, agent in agents.items():
                try:
                    resp = await agent.run(txt)
                    out = resp.output
                    pred_diag = (getattr(out, "primary_diagnosis", "") or "").strip().lower()

                    # Match if pred_diag matches any gt label
                    is_match = any(
                        t.lower() == pred_diag or t.lower() in pred_diag or pred_diag in t.lower()
                        for t in gt_labels
                    )

                    store.update(specialist=spec_key, domain=spec_domain, is_correct=is_match)
                    processed_counts[spec_key] += 1
                    if is_match:
                        match_counts[spec_key] += 1

                except Exception as e:
                    continue

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(cases_to_process)} cases...", flush=True)

    # Save trust posteriors atomically
    store.save(OUTPUT_TRUST_PATH)
    print(f"\nSerialized trust posteriors to: {OUTPUT_TRUST_PATH}", flush=True)

    # Build manifest
    manifest = {
        "manifest_title": "HealthInsight-UQ Bayesian Beta Trust Manifest (Algorithm 5)",
        "source_data": "datasets/processed/*/clean_calib.json",
        "sample_size_per_domain": samples_per_specialty,
        "total_evaluations_per_agent": processed_counts,
        "correct_predictions_per_agent": match_counts,
        "trust_means": {
            spec: {
                dom: round(store.get_trust(spec, dom), 4)
                for dom in DOMAINS
            }
            for spec in SPECIALISTS
        },
        "posteriors": store.posteriors,
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved trust manifest to: {MANIFEST_PATH}")

    print("\nVerified Posterior Trust Means hat{theta}_{i,d}:")
    for spec in SPECIALISTS:
        means = {d: round(store.get_trust(spec, d), 3) for d in DOMAINS}
        print(f"  {spec:15s}: {means}")

    return manifest


if __name__ == "__main__":
    asyncio.run(train_trust_posteriors(samples_per_specialty=30))
