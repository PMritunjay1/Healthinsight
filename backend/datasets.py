"""
IEEE Multi-Agent Medical Evaluation — Test Case Datasets
=========================================================
DEV:       5–10 cases  — fast iteration
VALIDATION: 15–20 cases — hyperparameter tuning
BENCHMARK:  50 adversarial cases — full publication evaluation

Every case: {id, patient_text, history, expected_diagnosis}

Datasets are loaded from external JSON files in the data/ directory.
"""

import json
import os

# Resolve the path to the data/ directory relative to this file
_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def _load_dataset(filename: str) -> list:
    """Load a dataset from a JSON file in the data/ directory."""
    filepath = os.path.join(_DATA_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# ──────────────────────────────────────────────
# DEV DATASET (10 cases)
# ──────────────────────────────────────────────
DEV_CASES = _load_dataset('dev_cases.json')

# ──────────────────────────────────────────────
# VALIDATION DATASET (15 cases)
# ──────────────────────────────────────────────
VALIDATION_CASES = _load_dataset('validation_cases.json')

# ──────────────────────────────────────────────
# BENCHMARK DATASET (50 adversarial cases)
# ──────────────────────────────────────────────
BENCHMARK_CASES = _load_dataset('benchmark_cases.json')
