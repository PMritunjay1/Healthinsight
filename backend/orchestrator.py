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
from backend.specialist_router import SpecialistRouter, DOMAIN_NAME_MAP
from backend.rie_module import RIEModule
from backend.rac_module import RACModule
from backend.trust_tracker import BetaTrustStore

logger = logging.getLogger("Orchestrator")

# ──────────────────────────────────────────────
# FROZEN DECISION THRESHOLDS & CLINICAL PRIORITY
# ──────────────────────────────────────────────
FROZEN_SPECIALIST_THRESHOLDS = {
    "cardiologist": {
        'angina': 0.32, 'arrhythmia': 0.44, 'atrial fibrillation': 0.56,
        'coronary artery disease': 0.56, 'heart failure': 0.56, 'hypertension': 0.42
    },
    "pulmonologist": {
        'asthma': 0.38, 'copd': 0.48, 'pneumonia': 0.46,
        'respiratory_failure': 0.52, 'pleural_effusion': 0.56, 'pulmonary_embolism': 0.84
    },
    "neurologist": {
        'stroke_ischemic': 0.48, 'intracranial_hemorrhage': 0.66, 'epilepsy_seizures': 0.60,
        'altered_mental_status_encephalopathy': 0.42, 'transient_ischemic_attack': 0.46,
        'neuropathy_neurodegenerative': 0.42
    }
}

ACUTE_CLINICAL_PRIORITY = {
    "pulmonary_embolism": 1.40,
    "intracranial_hemorrhage": 1.40,
    "stroke_ischemic": 1.35,
    "respiratory_failure": 1.35,
    "epilepsy_seizures": 1.30,
    "pneumonia": 1.30,
    "heart failure": 1.25,
    "altered_mental_status_encephalopathy": 1.25,
    "transient_ischemic_attack": 1.20,
    "angina": 1.20,
    "pleural_effusion": 1.20,
    "arrhythmia": 1.15,
    "atrial fibrillation": 1.15,
    "copd": 1.15,
    "asthma": 1.15,
    "coronary artery disease": 1.10,
    "neuropathy_neurodegenerative": 1.05,
    "hypertension": 0.80  # Demote background chronic comorbidity in inpatient settings
}

CHRONIC_BACKGROUND_DIAGNOSES = {
    'hypertension', 'coronary artery disease', 'neuropathy_neurodegenerative'
}
ACUTE_PRESENTING_DIAGNOSES = {
    'pulmonary_embolism', 'intracranial_hemorrhage', 'stroke_ischemic',
    'respiratory_failure', 'epilepsy_seizures', 'pneumonia', 'heart failure'
}

# ──────────────────────────────────────────────
# BALANCED DOMAIN RELATION MAP (Repaired 1.15x Prior)
# ──────────────────────────────────────────────
DOMAIN_RELATION_MAP = {
    "cardio":  {
        "cardiologist": 1.15, "pulmonologist": 1.0, "neurologist": 1.0,
        "diagnosis": 1.1, "risk": 1.2, "treatment": 1.1, "intent": 1.0
    },
    "pulmo":   {
        "pulmonologist": 1.15, "cardiologist": 1.0, "neurologist": 1.0,
        "diagnosis": 1.1, "risk": 1.2, "treatment": 1.1, "intent": 1.0
    },
    "neuro":   {
        "neurologist": 1.15, "cardiologist": 1.0, "pulmonologist": 1.0,
        "diagnosis": 1.1, "risk": 1.2, "treatment": 1.1, "intent": 1.0
    },
    "gastro":  {
        "diagnosis": 1.2, "cardiologist": 1.0, "pulmonologist": 1.0, "neurologist": 1.0,
        "risk": 1.1, "treatment": 1.1, "intent": 1.0
    },
    "general": {
        "diagnosis": 1.1, "cardiologist": 1.0, "pulmonologist": 1.0, "neurologist": 1.0,
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
    routing_decision: SpecialistRoutingDecision,
    candidate_cutoff: Optional[float] = None
) -> FinalFusionOutput:
    """
    Principled multi-specialist consensus fusion engine.
    Uses threshold-normalized margins, acute presentation prioritization,
    and balanced domain relevance without a fixed, distorting multiplier.
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
            
            thresh_map = FROZEN_SPECIALIST_THRESHOLDS.get(agent_key, {})
            best_disease = spec_out.primary_diagnosis
            best_disease_score = -10.0
            spec_active = []

            for d, p in spec_out.disease_probabilities.items():
                if candidate_cutoff is not None and p < candidate_cutoff:
                    continue
                t = thresh_map.get(d, 0.50)
                if p >= t:
                    norm_margin = (p - t) / (1.0 - t + 1e-5)
                    spec_active.append((d, p, norm_margin))
                else:
                    norm_margin = (p - t) / (t + 1e-5)

                d_score = norm_margin * 0.60 + p * 0.40

                if d_score > best_disease_score:
                    best_disease_score = d_score
                    best_disease = d

            # Incorporate domain relevance and RAC consensus confidence
            act_boost = 1.15 if spec_out.active_diagnoses else 0.85
            final_score = adj_conf * domain_w * (best_disease_score + 1.0) * act_boost
            candidate_diagnoses[agent_key] = best_disease
            scores[agent_key] = final_score
            multispecialist_findings[agent_key]["evaluated_active"] = spec_active

        elif isinstance(resp.output, DiagnosisOutput) or (isinstance(resp.output, dict) and "primary_diagnosis" in resp.output):
            diag_out = resp.output if isinstance(resp.output, DiagnosisOutput) else DiagnosisOutput(**resp.output)
            final_score = adj_conf * domain_w * (diag_out.confidence * 0.5 + 0.5)
            candidate_diagnoses[agent_key] = diag_out.primary_diagnosis
            scores[agent_key] = final_score

    # 2. Resolve winner diagnosis and selective multi-specialist active set
    if not scores:
        final_diagnosis = "Unknown Condition"
        overall_conf = 0.60
        fused_active_diagnoses = []
    else:
        winner_key = max(scores, key=scores.get)
        final_diagnosis = candidate_diagnoses.get(winner_key, "Unknown Condition")
        winner_spec_conf = multispecialist_findings.get(winner_key, {}).get("confidence", 0.75)
        overall_conf = max(0.60, min(0.98, winner_spec_conf))

        fused_active_diagnoses = []
        max_active_diagnoses = 3

        # 1. Primary winner diagnosis is ALWAYS admitted first (guarantees primary preservation & top-1 accuracy)
        if final_diagnosis and final_diagnosis != "Unknown Condition":
            fused_active_diagnoses.append(final_diagnosis)

        # 2. Selectively admit additional primary diagnoses from winning specialist (requires margin >= 0.15)
        w_active = multispecialist_findings.get(winner_key, {}).get("evaluated_active", [])
        if w_active:
            w_active_sorted = sorted(w_active, key=lambda x: -x[2])
            for d, p, m in w_active_sorted:
                if d == final_diagnosis:
                    continue
                if m >= 0.15 and len(fused_active_diagnoses) < max_active_diagnoses:
                    if d not in fused_active_diagnoses:
                        fused_active_diagnoses.append(d)

        # 3. Selectively admit secondary specialist diagnoses with evidence-based validation gating:
        #    - Secondary specialist domain must have presenting indication from router (domain_score >= 0.55)
        #    - Chronic/background conditions (hypertension, CAD, neuropathy) require p >= 0.85, m >= 0.20, domain_score >= 0.65
        #    - Acute conditions require p >= 0.70, m >= 0.15
        #    - General conditions require p >= 0.75, m >= 0.20
        routing_domain_scores = routing_decision.domain_scores if routing_decision else {}
        for agent_key in scores:
            if agent_key == winner_key:
                continue
            dom_name = "cardio" if "cardio" in agent_key else ("pulmo" if "pulmo" in agent_key else "neuro")
            dom_score = routing_domain_scores.get(dom_name, 0.0)
            if dom_score < 0.55:
                continue

            sec_active = multispecialist_findings.get(agent_key, {}).get("evaluated_active", [])
            sec_active_sorted = sorted(sec_active, key=lambda x: -x[2])
            for d, p, m in sec_active_sorted:
                if d in CHRONIC_BACKGROUND_DIAGNOSES:
                    if p < 0.85 or m < 0.20 or dom_score < 0.65:
                        continue
                elif d in ACUTE_PRESENTING_DIAGNOSES:
                    if p < 0.70 or m < 0.15:
                        continue
                else:
                    if p < 0.75 or m < 0.20:
                        continue

                if len(fused_active_diagnoses) < max_active_diagnoses:
                    if d not in fused_active_diagnoses:
                        fused_active_diagnoses.append(d)

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

    if not fused_active_diagnoses and final_diagnosis and final_diagnosis != "Unknown Condition":
        fused_active_diagnoses = [final_diagnosis]

    return FinalFusionOutput(
        final_diagnosis=final_diagnosis,
        final_recommendation=final_recs,
        final_risk_level=final_risk,
        overall_confidence=round(overall_conf, 4),
        active_specialists=active_specialists,
        multispecialist_findings=multispecialist_findings,
        active_diagnoses=fused_active_diagnoses,
        unconditioned_active_diagnoses=list(fused_active_diagnoses)
    )


# ──────────────────────────────────────────────
# ORCHESTRATOR
# ──────────────────────────────────────────────


def _dynamic_trust_fuse(
    agent_results: Dict[str, AgentResponse],
    rac_output: RACOutput,
    ipm_output: IPMOutput,
    routing_decision: SpecialistRoutingDecision,
    candidate_cutoff: Optional[float] = None,
    trust_store: Optional[BetaTrustStore] = None
) -> FinalFusionOutput:
    '''
    HealthInsight-UQ Dynamic Trust-Weighted Consensus (Algorithm 5).
    Integrates verified Bayesian Beta trust posteriors theta_{i,d} from BetaTrustStore,
    modulated by stochastic perturbation stability u_cons.
    Prevents cross-domain acute-priority multipliers from overpowering the primary specialist
    while preserving within-domain ranking.
    '''
    multispecialist_findings = {}
    active_specialists = []
    
    # Track domain base confidences for fusion
    raw_domain_scores = {}
    trust_weights = {}
    candidate_diagnoses = {}

    if trust_store is None:
        trust_store = BetaTrustStore()

    primary_domain = routing_decision.primary_domain if routing_decision and routing_decision.primary_domain else "general"

    for agent_key, resp in agent_results.items():
        if isinstance(resp.output, SpecialistOutput) or (isinstance(resp.output, dict) and "specialist_type" in resp.output):
            spec_out = resp.output if isinstance(resp.output, SpecialistOutput) else SpecialistOutput(**resp.output)
            active_specialists.append(agent_key)
            multispecialist_findings[agent_key] = {
                "specialist_type": spec_out.specialist_type,
                "primary_diagnosis": spec_out.primary_diagnosis,
                "active_diagnoses": spec_out.active_diagnoses,
                "disease_probabilities": spec_out.disease_probabilities,
                "confidence": spec_out.confidence,
                "urgency": getattr(spec_out, "predicted_urgency", "Normal"),
                "differential": getattr(spec_out, "differential_diagnoses", [])
            }
            
            # 1. Base domain weight from routing intent
            base_w = _get_domain_weight(primary_domain, agent_key)

            # 2. Extract stochastic consistency uncertainty
            u_instability = getattr(spec_out, "u_cons", getattr(spec_out, "u_instability", 0.0))
            if u_instability is None:
                u_instability = 0.0

            # 3. Within-domain disease selection (preserving within-domain acute priority)
            thresh_map = FROZEN_SPECIALIST_THRESHOLDS.get(agent_key, {})
            best_disease = spec_out.primary_diagnosis
            best_internal_score = -10.0
            best_base_margin = 0.0
            best_acute_weight = 1.0

            for d, p in spec_out.disease_probabilities.items():
                if candidate_cutoff is not None and p < candidate_cutoff:
                    continue
                t = thresh_map.get(d, 0.50)
                if p >= t:
                    norm_margin = (p - t) / (1.0 - t + 1e-5)
                else:
                    norm_margin = (p - t) / (t + 1e-5)

                base_margin = norm_margin * 0.70 + p * 0.30
                acute_weight = ACUTE_CLINICAL_PRIORITY.get(d, 1.0)
                internal_score = base_margin * acute_weight

                if internal_score > best_internal_score:
                    best_internal_score = internal_score
                    best_disease = d
                    best_base_margin = base_margin
                    best_acute_weight = acute_weight

            # 4. Bayesian Beta expected trust conditioned on primary domain strictly from BetaTrustStore
            expected_trust = trust_store.get_trust(agent_key, primary_domain)
            stability = max(0.2, 1.0 - u_instability * 0.5)
            trust = expected_trust * stability

            # 5. Cross-domain acute priority control:
            # Prevent cross-domain acute-priority multipliers from overpowering primary specialist
            is_primary_specialist = (DOMAIN_NAME_MAP.get(agent_key) == primary_domain)
            if is_primary_specialist:
                # Primary specialist maintains acute priority; chronic conditions (e.g. HTN 0.80)
                # are not demoted against competing out-of-domain specialists
                cross_acute = max(1.0, best_acute_weight)
            else:
                # Out-of-domain specialist cannot leverage acute priority bonus (> 1.0) to overpower primary specialist
                cross_acute = min(1.0, best_acute_weight)

            effective_score = max(0.01, best_base_margin * cross_acute)
            act_boost = 1.15 if spec_out.active_diagnoses else 0.85
            trust_weight = base_w * trust * effective_score * act_boost
            
            trust_weights[agent_key] = trust_weight
            raw_domain_scores[agent_key] = trust_weight
            candidate_diagnoses[agent_key] = best_disease

        elif isinstance(resp.output, DiagnosisOutput) or (isinstance(resp.output, dict) and "primary_diagnosis" in resp.output):
            diag_out = resp.output if isinstance(resp.output, DiagnosisOutput) else DiagnosisOutput(**resp.output)
            base_w = _get_domain_weight(primary_domain, agent_key)
            final_score = base_w * 0.5 * (diag_out.confidence * 0.5 + 0.5)
            candidate_diagnoses[agent_key] = diag_out.primary_diagnosis
            raw_domain_scores[agent_key] = final_score
            trust_weights[agent_key] = final_score

    # Select dominant specialist
    if raw_domain_scores:
        dominant_agent = max(raw_domain_scores.items(), key=lambda x: x[1])[0]
        primary_diag = candidate_diagnoses.get(dominant_agent, "Undetermined")
        dominant_conf = multispecialist_findings.get(dominant_agent, {}).get("confidence", 0.75)
        final_conf = max(0.60, min(0.98, dominant_conf))
    else:
        dominant_agent = "none"
        primary_diag = "Undetermined"
        final_conf = 0.0

    # Aggregate urgency
    final_urgency = "Normal"
    for spec_key in active_specialists:
        spec_data = multispecialist_findings.get(spec_key, {})
        urg = spec_data.get("urgency", "Normal")
        if urg == "Emergency":
            final_urgency = "Emergency"
        elif urg == "Urgent" and final_urgency == "Normal":
            final_urgency = "Urgent"
            
    final_recs = ["Consult attending physician for comprehensive specialist evaluation."]
    
    fused_active_diagnoses = []
    for spec_key, spec_data in multispecialist_findings.items():
        for d in spec_data.get("active_diagnoses", []):
            if d not in fused_active_diagnoses:
                fused_active_diagnoses.append(d)
    if not fused_active_diagnoses and primary_diag and primary_diag != "Unknown Condition":
        fused_active_diagnoses = [primary_diag]

    return FinalFusionOutput(
        final_diagnosis=primary_diag,
        overall_confidence=final_conf,
        final_risk_level=final_urgency,
        active_specialists=active_specialists,
        multispecialist_findings=multispecialist_findings,
        final_recommendation=final_recs,
        active_diagnoses=fused_active_diagnoses
    )


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
        self.trust_store = BetaTrustStore()

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
        fusion_mode: str = "deterministic",
        routing_mode: str = "parallel_all",
        candidate_cutoff: Optional[float] = None
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
        if routing_mode in ("all_specialists", "parallel_all", "parallel"):
            # All-specialist parallel inference path (runs all 3 frozen ONNX specialists)
            agents_to_run = {"cardiologist", "pulmonologist", "neurologist"}
        else:
            # Default routed path (retains exact standard router behavior)
            agents_to_run = set(routing_decision.selected_specialists)

        # If no specialists selected, activate general diagnosis agent
        if not agents_to_run:
            agents_to_run.add("diagnosis")

        # Always run RiskAgent for clinical safety and triage
        agents_to_run.add("risk")

        # Run diagnosis alongside specialist if multi-domain or requested
        if routing_decision.primary_domain == "general":
            agents_to_run.add("diagnosis")

        logger.info(f"Agents selected for execution (routing_mode={routing_mode}): {sorted(agents_to_run)}")


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
        if fusion_mode == "trust":
            fusion_result = _dynamic_trust_fuse(
                agent_results, rac_output, ipm_output, routing_decision, candidate_cutoff=candidate_cutoff, trust_store=self.trust_store
            )
        else:
            fusion_result = _deterministic_fuse(
                agent_results, rac_output, ipm_output, routing_decision, candidate_cutoff=candidate_cutoff
            )

        # ── P1.6 / Phase 2.7 Selective Abstention Guardrail ──
        import os
        import json
        conformal_path = os.path.join(os.path.dirname(__file__), 'models', 'conformal_config.json')
        abstention_path = os.path.join(os.path.dirname(__file__), 'models', 'abstention_config.json')
        tau = 0.5
        h_limit = 1.0
        if os.path.exists(conformal_path):
            try:
                with open(conformal_path, 'r') as f:
                    config = json.load(f)
                    tau = config.get("confidence_threshold_tau", config.get("tau", tau))
                    h_limit = config.get("h_max", config.get("h_limit", h_limit))
            except Exception as e:
                logger.error(f"Failed to load conformal config: {e}")
        elif os.path.exists(abstention_path):
            try:
                with open(abstention_path, 'r') as f:
                    config = json.load(f)
                    tau = config.get("tau", tau)
                    h_limit = config.get("h_limit", h_limit)
            except Exception as e:
                logger.error(f"Failed to load abstention config: {e}")

        calib_conf = rac_output.calibrated_confidence if rac_output.calibrated_confidence is not None else fusion_result.overall_confidence
        e_dom = rac_output.e_domain if rac_output.e_domain is not None else 0.0

        fusion_result.unconditioned_diagnosis = fusion_result.final_diagnosis
        fusion_result.unconditioned_active_diagnoses = list(fusion_result.active_diagnoses or [])

        if calib_conf < tau or e_dom > h_limit:
            fusion_result.final_diagnosis = "Uncertain — refer to clinician"
            fusion_result.is_abstained = True
            fusion_result.final_recommendation = ["System confidence is below safe operating threshold or specialist conflict is too high. Immediate human clinical review required."]
            fusion_result.overall_confidence = calib_conf
            fusion_result.active_diagnoses = []

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
