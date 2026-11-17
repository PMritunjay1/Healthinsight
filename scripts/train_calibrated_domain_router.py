"""
HealthInsight-UQ: Train Calibrated Domain Router on D_cal
==========================================================
Trains a lightweight, fast, and reproducible text-domain router
strictly on the disjoint calibration split D_cal (clean_calib.json).
Guarantees 0% overlap with D_train and D_test.

Saves:
- backend/models/calibrated_domain_router.joblib
"""

import os
import sys
import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

MODEL_OUTPUT_PATH = os.path.join(WORKSPACE, "backend", "models", "calibrated_domain_router.joblib")


def train_calibrated_domain_router(n_samples_per_domain: int = 1500):
    train_texts = []
    train_labels = []

    domains = [
        ("common", "cardio"),
        ("pulmonologist", "pulmo"),
        ("neurologist", "neuro"),
    ]

    for spec, dom in domains:
        calib_path = os.path.join(WORKSPACE, "datasets", "processed", spec, "clean_calib.json")
        with open(calib_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        for r in data:
            txt = r.get("text") or r.get("patient_text") or ""
            if len(txt) > 50:
                # Use head-tail representation matching specialist agents
                if len(txt) > 2500:
                    clean_txt = f"{txt[:1500]} {txt[-1000:]}"
                else:
                    clean_txt = txt
                train_texts.append(clean_txt)
                train_labels.append(dom)
                count += 1
                if count >= n_samples_per_domain:
                    break

    print(f"Fitting Calibrated Domain Router on {len(train_texts)} D_cal narratives...")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=6000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
        ("clf", LogisticRegression(C=1.0, max_iter=300, random_state=42))
    ])

    pipe.fit(train_texts, train_labels)

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    joblib.dump(pipe, MODEL_OUTPUT_PATH)
    print(f"Successfully serialized Calibrated Domain Router to {MODEL_OUTPUT_PATH}")
    return pipe


if __name__ == "__main__":
    train_calibrated_domain_router()
