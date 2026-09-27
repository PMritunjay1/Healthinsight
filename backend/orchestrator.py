import asyncio
import logging
import time
from typing import Dict, List, Any, Optional

from backend.models import (
    PatientInput, FinalFusionOutput, AgentResponse, IPMOutput, RACOutput,
    SpecialistRoutingDecision, SpecialistOutput, DiagnosisOutput
)
from backend.agents.intent_agent import IntentAgent
from backend.agents.diagnosis_agent import DiagnosisAgent
from backend.agents.treatment_agent import TreatmentAgent
from backend.agents.risk_agent import RiskAgent
from backend.agents.specialist_agents import (
    CardiologistSpecialistAgent,
    PulmonologistSpecialistAgent,
    NeurologistSpecialistAgent
)
from backend.specialist_router import SpecialistRouter
from backend.rie_module import RIEModule
from backend.rac_module import RACModule

logger = logging.getLogger("Orchestrator")

# ──────────────────────────────────────────────
# DOMAIN WEIGHT MAP (Specialist-Enhanced)
# ──────────────────────────────────────────────
DOMAIN_RELATION_MAP = {
    "cardio":  {
        "cardiologist": 1.6, "pulmonologist": 1.1, "neurologist": 0.9,
        "diagnosis": 1.1, "risk": 1.4, "treatment": 1.2, "intent": 1.0
    },
    "pulmo":   {
        "pulmonologist": 1.6, "cardiologist": 1.1, "neurologist": 0.9,
        "diagnosis": 1.1, "risk": 1.4, "treatment": 1.2, "intent": 1.0
    },
    "neuro":   {
        "neurologist": 1.6, "cardiologist": 0.9, "pulmonologist": 0.9,
        "diagnosis": 1.1, "risk": 1.4, "treatment": 1.2, "intent": 1.0
    },
    "gastro":  {
        "diagnosis": 1.5, "cardiologist": 0.8, "pulmonologist": 0.8, "neurologist": 0.8,
        "risk": 1.2, "treatment": 1.2, "intent": 1.0
    },
    "general": {
        "diagnosis": 1.4, "cardiologist": 1.0, "pulmonologist": 1.0, "neurologist": 1.0,
        "risk": 1.0, "treatment": 1.0, "intent": 1.0
    },
}


def _get_domain_weight(ipm_domain: str, agent_key: str) -> float:
    """Look up the weight for a given agent relative to the detected domain."""
    domain_lower = (ipm_domain or "general").lower()
    for domain_key, weights in DOMAIN_RELATION_MAP.items():
        if domain_key in domain_lower:
            return weights.get(agent_key, 1.0)
    return DOMAIN_RELATION_MAP["general"].get(agent_key, 1.0)


# ──────────────────────────────────────────────
# DETERMINISTIC MULTI-SPECIALIST FUSION
# ──────────────────────────────────────────────

def _deterministic_fuse(
    agent_results: Dict[str, AgentResponse],
    rac_output: RACOutput,
    ipm_output: IPMOutput,
    routing_decision: SpecialistRoutingDecision
) -> FinalFusionOutput:
    """
    Deterministic fusion engine with multi-specialist support.
    Computes weighted fusion scores, selects the dominant primary diagnosis,
    and aggregates multi-specialist clinical findings.
    """
    scores: Dict[str, float] = {}
    candidate_diagnoses: Dict[str, str] = {}
    multispecialist_findings: Dict[str, Any] = {}
    active_specialists: List[str] = []

    # 1. Evaluate contributing diagnostic & specialist agents
    for agent_key, resp in agent_results.items():
        if agent_key in ("intent", "treatment", "risk") or resp.output is None:
            continue

        adj_conf = rac_output.adjusted_confidence_map.get(agent_key, 0.70)
        primary_dom = routing_decision.primary_domain if routing_decision and routing_decision.primary_domain != "general" else ipm_output.medical_domain
        domain_w = _get_domain_weight(primary_dom, agent_key)
        
        # Specialist-specific scoring
        if isinstance(resp.output, SpecialistOutput) or (isinstance(resp.output, dict) and "specialist_type" in resp.output):
            spec_out = resp.output if isinstance(resp.output, SpecialistOutput) else SpecialistOutput(**resp.output)
            active_specialists.append(agent_key)
            multispecialist_findings[agent_key] = {
                "specialist_type": spec_out.specialist_type,
                "primary_diagnosis": spec_out.primary_diagnosis,
                "active_diagnoses": spec_out.active_diagnoses,
                "disease_probabilities": spec_out.disease_probabilities,
                "predicted_complaint": spec_out.predicted_complaint,
                "predicted_urgency": spec_out.predicted_urgency,
                "confidence": spec_out.confidence
            }
            # Specialist score boosted by active diagnosis detection
            active_boost = 1.15 if spec_out.active_diagnoses else 0.90
            final_score = adj_conf * domain_w * (spec_out.confidence * 0.5 + 0.5) * active_boost
            candidate_diagnoses[agent_key] = spec_out.primary_diagnosis
            scores[agent_key] = final_score

        elif isinstance(resp.output, DiagnosisOutput) or (isinstance(resp.output, dict) and "primary_diagnosis" in resp.output):
            diag_out = resp.output if isinstance(resp.output, DiagnosisOutput) else DiagnosisOutput(**resp.output)
            final_score = adj_conf * domain_w * (diag_out.confidence * 0.5 + 0.5)
            candidate_diagnoses[agent_key] = diag_out.primary_diagnosis
            scores[agent_key] = final_score

    # 2. Resolve winner diagnosis
    if not scores:
        final_diagnosis = "Unknown Condition"
        overall_conf = 0.60
    else:
        winner_key = max(scores, key=scores.get)
        final_diagnosis = candidate_diagnoses.get(winner_key, "Unknown Condition")
        overall_conf = max(0.60, min(0.98, scores[winner_key]))

    # 3. Extract final risk level
    final_risk = "Medium"
    risk_resp = agent_results.get("risk")
    if risk_resp and risk_resp.output:
        if hasattr(risk_resp.output, "risk_level"):
            final_risk = risk_resp.output.risk_level
        elif isinstance(risk_resp.output, dict):
            final_risk = risk_resp.output.get("risk_level", "Medium")

    # If specialist flagged emergency urgency, escalate risk if currently Low/Medium
    for spec_key in active_specialists:
        spec_data = multispecialist_findings.get(spec_key, {})
        if spec_data.get("predicted_urgency") == "Emergency" and final_risk in ("Low", "Medium"):
            final_risk = "High"

    # 4. Extract final treatment recommendations
    final_recs = []
    treat_resp = agent_results.get("treatment")
    if treat_resp and treat_resp.output:
        if hasattr(treat_resp.output, "recommended_treatment"):
            rec_val = treat_resp.output.recommended_treatment
            final_recs = [rec_val] if isinstance(rec_val, str) else rec_val
        elif isinstance(treat_resp.output, dict):
            rec_val = treat_resp.output.get("recommended_treatment", "")
            final_recs = [rec_val] if isinstance(rec_val, str) else rec_val

    if not final_recs:
        final_recs = ["Consult attending physician for comprehensive specialist evaluation."]

    return FinalFusionOutput(
        final_diagnosis=final_diagnosis,
        final_recommendation=final_recs,
        final_risk_level=final_risk,
        overall_confidence=round(overall_conf, 4),
        active_specialists=active_specialists,
        multispecialist_findings=multispecialist_findings
    )


# ──────────────────────────────────────────────
# ORCHESTRATOR
# ──────────────────────────────────────────────

class Orchestrator:
    """
    Main Health Insights Multi-Agent Orchestrator.
    Coordinates RIE preprocessing, IPM intent classification, Specialist Routing,
    parallel agent execution, RAC consensus, and deterministic fusion.
    """
    def __init__(self):
        logger.info("Initializing Orchestrator and loading local models...")
        self.rie = RIEModule()
        self.rac = RACModule()
        self.router = SpecialistRouter()

        # Agent registry — keyed by agent name
        self._agent_registry = {
            "intent":        IntentAgent(),
            "cardiologist":  CardiologistSpecialistAgent(),
            "pulmonologist": PulmonologistSpecialistAgent(),
            "neurologist":   NeurologistSpecialistAgent(),
            "diagnosis":     DiagnosisAgent(),
            "treatment":     TreatmentAgent(),
            "risk":          RiskAgent(),
        }
        logger.info("Orchestrator initialized with 3 frozen specialist agents and core agents.")

    async def process_case(
        self,
        patient_text: str,
        history: str = "",
        metadata: dict = None,
    ) -> dict:
        total_start = time.time()

        input_data = PatientInput(
            patient_text=patient_text,
            history=history,
            metadata=metadata or {},
        )

        # ── STEP 1: RIE Preprocessing (100% deterministic local extraction) ──
        rie_output = await self.rie.run(patient_text, history)

        # ── STEP 2: IPM (Intent Agent local DistilBERT ONNX model) ──
        ipm_resp = await self._agent_registry["intent"].run(input_data)
        ipm_output: IPMOutput = ipm_resp.output
        if ipm_output is None:
            ipm_output = IPMOutput(
                medical_domain="general",
                urgency_level="medium",
                task_type="diagnosis",
                extracted_entities=[],
            )

        # ── STEP 3: Specialist Router (Determines relevant specialty domains) ──
        routing_decision = self.router.route(input_data, rie_output, ipm_output)

        # ── STEP 4: Determine Agents to Run in Parallel ──
        agents_to_run = set(routing_decision.selected_specialists)

        # If no specialists selected, activate general diagnosis agent
        if not agents_to_run:
            agents_to_run.add("diagnosis")

        # Always run RiskAgent for clinical safety and triage
        agents_to_run.add("risk")

        # Run diagnosis alongside specialist if multi-domain or requested
        if routing_decision.primary_domain == "general":
            agents_to_run.add("diagnosis")

        logger.info(f"Agents selected for execution: {sorted(agents_to_run)}")

        # ── STEP 5: Parallel Agent Execution ──
        async def _run_agent(key: str) -> tuple:
            agent = self._agent_registry[key]
            resp = await agent.run(input_data)
            return key, resp

        agent_tasks = [_run_agent(k) for k in agents_to_run]
        agent_raw = await asyncio.gather(*agent_tasks, return_exceptions=True)

        agent_results: Dict[str, AgentResponse] = {"intent": ipm_resp}
        for item in agent_raw:
            if isinstance(item, Exception):
                logger.error(f"Agent task failed: {item}")
                continue
            key, resp = item
            agent_results[key] = resp

        # ── STEP 6: Determine Diagnosis & Risk for Treatment Engine ──
        # Extract best candidate diagnosis from active specialists or diagnosis agent
        primary_diagnosis = "Unknown"
        for spec_key in ["cardiologist", "pulmonologist", "neurologist"]:
            if spec_key in agent_results and agent_results[spec_key].output:
                spec_out = agent_results[spec_key].output
                if hasattr(spec_out, "primary_diagnosis"):
                    primary_diagnosis = spec_out.primary_diagnosis
                    break

        if primary_diagnosis == "Unknown" and "diagnosis" in agent_results and agent_results["diagnosis"].output:
            diag_out = agent_results["diagnosis"].output
            if hasattr(diag_out, "primary_diagnosis"):
                primary_diagnosis = diag_out.primary_diagnosis

        risk_level = "Medium"
        if "risk" in agent_results and agent_results["risk"].output:
            risk_out = agent_results["risk"].output
            if hasattr(risk_out, "risk_level"):
                risk_level = risk_out.risk_level

        # Populate metadata for TreatmentAgent lookup
        input_data.metadata["predicted_diagnosis"] = primary_diagnosis
        input_data.metadata["predicted_risk_level"] = risk_level

        # Run treatment agent deterministically
        try:
            treatment_resp = await self._agent_registry["treatment"].run(input_data)
            agent_results["treatment"] = treatment_resp
        except Exception as e:
            logger.error(f"Treatment agent failed: {e}")

        # ── STEP 7: RAC Consensus Evaluation (0 external calls) ──
        rac_output = self.rac.evaluate(agent_results)

        # ── STEP 8: Deterministic Fusion (0 external calls) ──
        fusion_result = _deterministic_fuse(
            agent_results, rac_output, ipm_output, routing_decision
        )

        total_latency_ms = (time.time() - total_start) * 1000

        # ── STEP 9: Build Backward-Compatible Response Payload ──
        return {
            "input": input_data.model_dump(),
            "routing": routing_decision.model_dump(),
            "agents": {
                k: v.model_dump() for k, v in agent_results.items()
            },
            "fusion": fusion_result.model_dump(),
            "rac": rac_output.model_dump(),
            "rie": rie_output.model_dump(),
            "ipm": ipm_output.model_dump(),
            "total_latency_ms": total_latency_ms,
        }
