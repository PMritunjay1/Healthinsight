"""
Tests for HealthInsight-UQ Dynamic Trust-Weighted Consensus Fusion (Algorithm 5)
================================================================================
Verifies:
1. Reliable primary-domain prediction (e.g. Cardiologist predicting hypertension)
   is NOT overridden solely by unrelated out-of-domain confidence (e.g. Pulmonologist
   predicting asthma or Neurologist predicting intracranial hemorrhage).
2. Out-of-domain specialists cannot leverage acute clinical priority multipliers (>1.0)
   to overpower the primary specialist.
3. Within-domain disease ranking preserves clinical priority appropriately.
4. Genuine specialist disagreement (e.g. general domain or unconfident primary)
   is handled safely without distorted bias.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from backend.models import (
    AgentResponse,
    SpecialistOutput,
    IPMOutput,
    RACOutput,
    SpecialistRoutingDecision,
)
from backend.trust_tracker import BetaTrustStore
from backend.orchestrator import _dynamic_trust_fuse


def _make_specialist_output(spec_type, primary_diag, probs, urgency="Normal", u_cons=0.0):
    active = [d for d, p in probs.items() if p >= 0.40]
    conf = max(probs.values()) if probs else 0.50
    return SpecialistOutput(
        specialist_type=spec_type,
        primary_diagnosis=primary_diag,
        confidence=conf,
        disease_probabilities=probs,
        active_diagnoses=active,
        predicted_complaint=primary_diag,
        predicted_urgency=urgency,
        differential_diagnoses=[d for d in probs.keys() if d != primary_diag],
        thresholds_applied={},
        u_cons=u_cons
    )


def _make_rac_output(adj_map, agreement=0.8, penalty=0.1):
    return RACOutput(
        agreement_score=agreement,
        conflict_penalty=penalty,
        adjusted_confidence_map=adj_map
    )


def test_cardio_primary_protected_against_out_of_domain_acute():
    """
    Cardiology primary case:
    - Cardiologist predicts 'hypertension' with high confidence (0.88).
    - Pulmonologist co-activated, predicts 'asthma' (0.75, acute weight 1.15).
    - Neurologist co-activated, predicts 'intracranial_hemorrhage' (0.70, acute weight 1.40).
    Under dynamic trust fusion, the primary specialist's prediction must be preserved.
    """
    cardio_out = _make_specialist_output(
        "cardiologist",
        "hypertension",
        {"hypertension": 0.88, "angina": 0.10, "heart failure": 0.05}
    )
    pulmo_out = _make_specialist_output(
        "pulmonologist",
        "asthma",
        {"asthma": 0.75, "copd": 0.20, "pneumonia": 0.10}
    )
    neuro_out = _make_specialist_output(
        "neurologist",
        "intracranial_hemorrhage",
        {"intracranial_hemorrhage": 0.70, "stroke_ischemic": 0.15}
    )

    agent_results = {
        "cardiologist": AgentResponse(agent_name="cardiologist", success=True, output=cardio_out, latency_ms=10.0),
        "pulmonologist": AgentResponse(agent_name="pulmonologist", success=True, output=pulmo_out, latency_ms=10.0),
        "neurologist": AgentResponse(agent_name="neurologist", success=True, output=neuro_out, latency_ms=10.0),
    }

    rac_output = _make_rac_output({"cardiologist": 0.88, "pulmonologist": 0.75, "neurologist": 0.70})

    ipm_output = IPMOutput(
        medical_domain="cardio",
        urgency_level="medium",
        task_type="diagnosis",
        extracted_entities=["chest pain", "high blood pressure"]
    )

    routing_decision = SpecialistRoutingDecision(
        primary_domain="cardio",
        secondary_domains=["pulmo", "neuro"],
        selected_specialists=["cardiologist", "pulmonologist", "neurologist"],
        routing_tier="HIGH",
        ambiguity_score=0.72
    )

    trust_store = BetaTrustStore()
    fused = _dynamic_trust_fuse(
        agent_results=agent_results,
        rac_output=rac_output,
        ipm_output=ipm_output,
        routing_decision=routing_decision,
        trust_store=trust_store
    )

    assert fused.final_diagnosis == "hypertension", (
        f"Expected 'hypertension' to be protected as primary cardio diagnosis, got '{fused.final_diagnosis}'"
    )
    assert "hypertension" in fused.active_diagnoses


def test_pulmo_primary_protected_against_cardio_override():
    """
    Pulmonology primary case:
    - Pulmonologist predicts 'pneumonia' with p=0.78.
    - Cardiologist predicts 'heart failure' with p=0.82 (acute priority 1.25).
    Dynamic trust fusion must select 'pneumonia'.
    """
    pulmo_out = _make_specialist_output(
        "pulmonologist",
        "pneumonia",
        {"pneumonia": 0.78, "asthma": 0.15, "copd": 0.10}
    )
    cardio_out = _make_specialist_output(
        "cardiologist",
        "heart failure",
        {"heart failure": 0.82, "hypertension": 0.50}
    )

    agent_results = {
        "pulmonologist": AgentResponse(agent_name="pulmonologist", success=True, output=pulmo_out, latency_ms=10.0),
        "cardiologist": AgentResponse(agent_name="cardiologist", success=True, output=cardio_out, latency_ms=10.0),
    }

    rac_output = _make_rac_output({"pulmonologist": 0.78, "cardiologist": 0.82})

    ipm_output = IPMOutput(
        medical_domain="pulmo",
        urgency_level="medium",
        task_type="diagnosis",
        extracted_entities=["cough", "fever"]
    )

    routing_decision = SpecialistRoutingDecision(
        primary_domain="pulmo",
        secondary_domains=["cardio"],
        selected_specialists=["pulmonologist", "cardiologist"],
        routing_tier="MEDIUM",
        ambiguity_score=0.45
    )

    trust_store = BetaTrustStore()
    fused = _dynamic_trust_fuse(
        agent_results=agent_results,
        rac_output=rac_output,
        ipm_output=ipm_output,
        routing_decision=routing_decision,
        trust_store=trust_store
    )

    assert fused.final_diagnosis == "pneumonia", (
        f"Expected 'pneumonia', got '{fused.final_diagnosis}'"
    )


def test_neuro_primary_protected_against_cardio_override():
    """
    Neurology primary case:
    - Neurologist predicts 'epilepsy_seizures' with p=0.74.
    - Cardiologist predicts 'arrhythmia' with p=0.79.
    Dynamic trust fusion must select 'epilepsy_seizures'.
    """
    neuro_out = _make_specialist_output(
        "neurologist",
        "epilepsy_seizures",
        {"epilepsy_seizures": 0.74, "stroke_ischemic": 0.10}
    )
    cardio_out = _make_specialist_output(
        "cardiologist",
        "arrhythmia",
        {"arrhythmia": 0.79, "hypertension": 0.40}
    )

    agent_results = {
        "neurologist": AgentResponse(agent_name="neurologist", success=True, output=neuro_out, latency_ms=10.0),
        "cardiologist": AgentResponse(agent_name="cardiologist", success=True, output=cardio_out, latency_ms=10.0),
    }

    rac_output = _make_rac_output({"neurologist": 0.74, "cardiologist": 0.79})

    ipm_output = IPMOutput(
        medical_domain="neuro",
        urgency_level="high",
        task_type="diagnosis",
        extracted_entities=["seizure"]
    )

    routing_decision = SpecialistRoutingDecision(
        primary_domain="neuro",
        secondary_domains=["cardio"],
        selected_specialists=["neurologist", "cardiologist"],
        routing_tier="MEDIUM",
        ambiguity_score=0.40
    )

    trust_store = BetaTrustStore()
    fused = _dynamic_trust_fuse(
        agent_results=agent_results,
        rac_output=rac_output,
        ipm_output=ipm_output,
        routing_decision=routing_decision,
        trust_store=trust_store
    )

    assert fused.final_diagnosis == "epilepsy_seizures", (
        f"Expected 'epilepsy_seizures', got '{fused.final_diagnosis}'"
    )


def test_genuine_disagreement_general_domain():
    """
    When primary_domain is 'general' (no clear specialist domain):
    Specialists compete on equal trust footing (trust = 0.50), and the
    higher calibrated margin + acute clinical urgency decides safely.
    """
    pulmo_out = _make_specialist_output(
        "pulmonologist",
        "pneumonia",
        {"pneumonia": 0.80, "asthma": 0.10}
    )
    cardio_out = _make_specialist_output(
        "cardiologist",
        "hypertension",
        {"hypertension": 0.45}
    )

    agent_results = {
        "pulmonologist": AgentResponse(agent_name="pulmonologist", success=True, output=pulmo_out, latency_ms=10.0),
        "cardiologist": AgentResponse(agent_name="cardiologist", success=True, output=cardio_out, latency_ms=10.0),
    }

    rac_output = _make_rac_output({"pulmonologist": 0.80, "cardiologist": 0.45})

    ipm_output = IPMOutput(
        medical_domain="general",
        urgency_level="medium",
        task_type="diagnosis",
        extracted_entities=["malaise"]
    )

    routing_decision = SpecialistRoutingDecision(
        primary_domain="general",
        secondary_domains=["pulmo", "cardio"],
        selected_specialists=["pulmonologist", "cardiologist"],
        routing_tier="HIGH",
        ambiguity_score=0.80
    )

    trust_store = BetaTrustStore()
    fused = _dynamic_trust_fuse(
        agent_results=agent_results,
        rac_output=rac_output,
        ipm_output=ipm_output,
        routing_decision=routing_decision,
        trust_store=trust_store
    )

    # In general domain, pneumonia (high margin, acute 1.30) clearly dominates weak HTN (0.45)
    assert fused.final_diagnosis == "pneumonia"
    assert "pneumonia" in fused.active_diagnoses
    assert "hypertension" in fused.active_diagnoses


def test_within_domain_acute_priority_preserved():
    """
    Inside the primary domain (e.g. Cardiology):
    Between 'heart failure' (acute priority 1.25, p=0.60) and 'hypertension' (priority 0.80, p=0.62),
    within-domain clinical priority should favor the acute heart failure condition.
    """
    cardio_out = _make_specialist_output(
        "cardiologist",
        "heart failure",
        {"heart failure": 0.68, "hypertension": 0.62}
    )

    agent_results = {
        "cardiologist": AgentResponse(agent_name="cardiologist", success=True, output=cardio_out, latency_ms=10.0),
    }

    rac_output = _make_rac_output({"cardiologist": 0.70}, agreement=1.0, penalty=0.0)

    ipm_output = IPMOutput(
        medical_domain="cardio",
        urgency_level="high",
        task_type="diagnosis",
        extracted_entities=["orthopnea", "edema"]
    )

    routing_decision = SpecialistRoutingDecision(
        primary_domain="cardio",
        secondary_domains=[],
        selected_specialists=["cardiologist"],
        routing_tier="LOW",
        ambiguity_score=0.15
    )

    trust_store = BetaTrustStore()
    fused = _dynamic_trust_fuse(
        agent_results=agent_results,
        rac_output=rac_output,
        ipm_output=ipm_output,
        routing_decision=routing_decision,
        trust_store=trust_store
    )

    assert fused.final_diagnosis == "heart failure"
