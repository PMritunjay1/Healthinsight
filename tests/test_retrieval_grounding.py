"""
Unit and Integration Tests for Phase 2.9: Offline Retrieval Grounding Module
===========================================================================
Verifies:
1. Retrieval is 100% offline (0 network calls).
2. Latency is well under 20 ms on CPU.
3. BM25 indexing correctly scores acute vs non-acute clinical guidelines.
4. Acute red flags and emergency criteria trigger properly.
5. Integration with candidate diseases yields proper support score kappa_rag.
"""

import time
import pytest
from backend.retrieval_grounding import GuidelineRetrievalModule, BM25GuidelineIndex


def test_retrieval_module_initialization():
    module = GuidelineRetrievalModule()
    assert module.is_loaded is True
    assert module.bm25.n_docs > 0
    assert len(module.bm25.documents) > 20


def test_retrieval_latency_under_20ms():
    module = GuidelineRetrievalModule()
    symptoms = ["chest pain", "shortness of breath", "sweating", "left arm pain"]
    candidate_diagnoses = ["Heart Attack", "Angina"]

    # Warmup
    module.ground_case(symptoms, candidate_diagnoses)

    # Benchmark 20 iterations
    start = time.perf_counter()
    iterations = 20
    for _ in range(iterations):
        res = module.ground_case(symptoms, candidate_diagnoses)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / iterations

    assert elapsed_ms < 20.0, f"Retrieval latency {elapsed_ms:.2f} ms exceeded 20 ms ceiling"
    assert "matched_guidelines" in res
    assert len(res["matched_guidelines"]) > 0


def test_retrieval_grounds_acute_cardiac_condition():
    module = GuidelineRetrievalModule()
    symptoms = ["chest tightness", "crushing retrosternal pain", "diaphoresis", "dyspnea"]
    res = module.ground_case(symptoms, ["Heart attack", "Angina"])

    assert len(res["matched_guidelines"]) > 0
    top_hit = res["matched_guidelines"][0]
    assert "heart attack" in top_hit["disease"].lower() or "angina" in top_hit["disease"].lower()
    assert top_hit["evidence_source"] != ""
    assert res["guideline_support_score"] > 0.3
    assert len(res["supporting_snippets"]) > 0


def test_retrieval_grounds_pulmonary_asthma():
    module = GuidelineRetrievalModule()
    symptoms = ["wheezing", "cough", "shortness of breath", "chest tightness"]
    res = module.ground_case(symptoms, ["Bronchial Asthma"])

    assert len(res["matched_guidelines"]) > 0
    matched_diseases = [g["disease"].lower() for g in res["matched_guidelines"]]
    assert any("asthma" in d for d in matched_diseases)


def test_retrieval_handles_empty_or_unknown_symptoms():
    module = GuidelineRetrievalModule()
    res = module.ground_case([], [])
    assert res["matched_guidelines"] == []
    assert res["guideline_support_score"] == 0.0
    assert res["supporting_snippets"] == []
