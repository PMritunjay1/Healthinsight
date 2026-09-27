import re
import logging
from typing import Dict, List, Optional, Tuple, Set

from backend.models import PatientInput, RIEOutput, IPMOutput, SpecialistRoutingDecision

logger = logging.getLogger("SpecialistRouter")

DOMAIN_NAME_MAP = {
    "cardiologist": "cardio",
    "pulmonologist": "pulmo",
    "neurologist": "neuro",
}


class SpecialistRouter:
    """
    Deterministic clinical router and domain orchestrator.
    Routes clinical cases to:
    - Cardiologist Specialist (cardiovascular conditions)
    - Pulmonologist Specialist (respiratory conditions)
    - Neurologist Specialist (neurological conditions)
    - Multi-specialist consensus when presentation spans overlapping domains
    - General Diagnostic Agent when presentation is general / non-specialized
    """

    # Domain clinical keyword lexicons (Enriched with standard linguistic & symptom variations)
    CARDIO_KEYWORDS: Set[str] = {
        "chest pain", "pain in chest", "pain in the chest", "chest discomfort", "discomfort in chest",
        "chest tightness", "tightness in chest", "chest pressure", "pressure in chest", "pressure in the chest",
        "substernal", "substernal pain", "substernal chest pain", "substernal pressure", "crushing chest pain",
        "angina", "angina pectoris", "cardiac pain",
        "palpitations", "irregular heartbeat", "arrhythmia", "atrial fibrillation", "afib",
        "flutter", "tachycardia", "bradycardia", "heart failure", "chf", "congestive heart failure",
        "coronary artery disease", "cad", "myocardial infarction", "heart attack", "stemi", "nstemi",
        "st elevation", "troponin", "hypertension", "high blood pressure", "htn",
        "edema", "pedal edema", "swollen ankles", "leg swelling", "orthopnea",
        "paroxysmal nocturnal dyspnea", "pnd", "cardiac", "cardiomegaly", "pericarditis",
        "aortic", "murmur", "ejection fraction", "cardiovascular",
        "radiating to left arm", "radiating to arm", "radiation to left arm", "left arm pain", "jaw pain"
    }

    PULMO_KEYWORDS: Set[str] = {
        "shortness of breath", "dyspnea", "sob", "breathlessness", "difficulty breathing", "breathing difficulty",
        "heavy breathing", "labored breathing", "wheezing", "stridor",
        "cough", "chronic cough", "productive cough", "dry cough", "coughing",
        "sputum", "purulent sputum", "hemoptysis", "coughing blood",
        "chest congestion", "congestion in chest", "lung congestion",
        "aspiration", "aspiration pneumonia", "respiratory distress", "choking",
        "asthma", "asthma exacerbation", "bronchial asthma", "reactive airway",
        "copd", "chronic obstructive pulmonary disease", "emphysema", "chronic bronchitis",
        "pneumonia", "community acquired pneumonia", "cap", "consolidation", "lung infiltrate",
        "respiratory failure", "acute respiratory failure", "hypoxia", "hypoxemia", "cyanosis",
        "pleural effusion", "pulmonary embolism", "pe", "pleuritic chest pain",
        "low oxygen saturation", "spo2", "crackles", "rales", "rhonchi", "bronchospasm",
        "inhaler", "albuterol", "nebulizer", "pulmonary", "respiratory", "lung", "lungs"
    }

    NEURO_KEYWORDS: Set[str] = {
        "headache", "migraine", "severe headache", "thunderclap headache", "head pain",
        "stroke", "ischemic stroke", "cerebrovascular accident", "cva", "infarction",
        "intracranial hemorrhage", "brain bleed", "subarachnoid hemorrhage", "sah", "subdural",
        "transient ischemic attack", "tia", "mini-stroke",
        "seizure", "epilepsy", "convulsions", "tonic-clonic", "postictal", "status epilepticus",
        "altered mental status", "ams", "confusion", "disorientation", "encephalopathy",
        "delirium", "memory loss", "dementia", "lethargy", "unresponsive", "stupor", "coma",
        "dizziness", "vertigo", "lightheadedness", "loss of balance", "ataxia", "gait instability",
        "numbness", "tingling", "paresthesia", "loss of sensation", "neuropathy", "peripheral neuropathy",
        "facial numbness", "numbness in face",
        "weakness", "hemiparesis", "hemiplegia", "facial droop", "arm weakness", "leg weakness",
        "weakness in arm", "weakness in leg", "weakness in limbs",
        "slurred speech", "dysarthria", "aphasia", "difficulty speaking", "speech arrest", "difficulty talking", "loss of speech",
        "vision loss", "visual disturbance", "diplopia", "double vision", "hemianopia",
        "loss of consciousness", "syncope", "passed out", "fainting", "tremor", "neurological", "neuro"
    }

    # Negation words to avoid false positive matches
    NEGATION_TERMS: Set[str] = {
        "no", "denies", "without", "negative for", "ruled out", "free of", "never had", "denied", "denying"
    }

    def __init__(self, single_domain_threshold: float = 0.35, multi_domain_ratio: float = 0.50):
        self.single_domain_threshold = single_domain_threshold
        self.multi_domain_ratio = multi_domain_ratio

    def _extract_matches(self, text_lower: str, keywords: Set[str]) -> List[str]:
        """Find non-negated keyword matches in text."""
        matches = []
        for kw in keywords:
            pattern = r'\b' + re.escape(kw) + r'\b'
            for m in re.finditer(pattern, text_lower):
                start_pos = m.start()
                preceding = text_lower[max(0, start_pos - 35):start_pos].strip()
                is_neg = any(re.search(r'\b' + re.escape(neg) + r'\b', preceding) for neg in self.NEGATION_TERMS)
                if not is_neg:
                    matches.append(kw)
                    break
        return matches

    def _score_domain(
        self,
        text_lower: str,
        keywords: Set[str],
        ipm_domain_match: bool,
        rie_symptoms: List[str]
    ) -> Tuple[float, List[str]]:
        """
        Calculate weighted score for a given specialty domain.
        A single landmark domain symptom grants a baseline score of 0.45 (exceeding 0.35 threshold).
        Additional symptoms scale the confidence up to 0.85 + optional IPM boost up to 1.0.
        """
        matched_kws = self._extract_matches(text_lower, keywords)
        
        rie_matches = []
        for sym in rie_symptoms:
            sym_lower = sym.lower()
            if any(kw == sym_lower or kw in sym_lower for kw in keywords):
                rie_matches.append(sym)

        all_matches = list(set(matched_kws + rie_matches))
        
        if not all_matches:
            kw_score = 0.0
        else:
            # 1 match -> 0.45 (reliably activates specialist)
            # 2 matches -> 0.60
            # 3+ matches -> 0.75 - 0.85
            kw_score = min(0.85, 0.45 + (len(all_matches) - 1) * 0.15)
            
        ipm_boost = 0.25 if ipm_domain_match else 0.0
        total_score = min(1.0, kw_score + ipm_boost)
        return total_score, all_matches

    def route(
        self,
        input_data: PatientInput,
        rie_output: Optional[RIEOutput] = None,
        ipm_output: Optional[IPMOutput] = None
    ) -> SpecialistRoutingDecision:
        """
        Determines relevant specialist domain(s) and returns a structured routing decision.
        """
        patient_text = (input_data.patient_text or "").lower()
        history = (input_data.history or "").lower()
        combined_text = f"{patient_text} {history}".strip()

        rie_symptoms = rie_output.symptoms if rie_output else []
        ipm_domain = (ipm_output.medical_domain or "general").lower() if ipm_output else "general"

        ipm_is_cardio = "cardio" in ipm_domain
        ipm_is_pulmo = "pulmo" in ipm_domain or "respirat" in ipm_domain
        ipm_is_neuro = "neuro" in ipm_domain

        # Score each specialty domain
        cardio_score, cardio_matches = self._score_domain(combined_text, self.CARDIO_KEYWORDS, ipm_is_cardio, rie_symptoms)
        pulmo_score, pulmo_matches = self._score_domain(combined_text, self.PULMO_KEYWORDS, ipm_is_pulmo, rie_symptoms)
        neuro_score, neuro_matches = self._score_domain(combined_text, self.NEURO_KEYWORDS, ipm_is_neuro, rie_symptoms)

        domain_scores = {
            "cardio": round(cardio_score, 4),
            "pulmo": round(pulmo_score, 4),
            "neuro": round(neuro_score, 4),
            "general": 0.0
        }

        max_specialist_score = max(cardio_score, pulmo_score, neuro_score)
        domain_scores["general"] = round(max(0.0, 1.0 - max_specialist_score), 4)

        specialist_candidates = []
        routing_reasons = []

        if cardio_score >= self.single_domain_threshold:
            specialist_candidates.append(("cardiologist", cardio_score, cardio_matches))
        if pulmo_score >= self.single_domain_threshold:
            specialist_candidates.append(("pulmonologist", pulmo_score, pulmo_matches))
        if neuro_score >= self.single_domain_threshold:
            specialist_candidates.append(("neurologist", neuro_score, neuro_matches))

        specialist_candidates.sort(key=lambda x: x[1], reverse=True)

        selected_specialists = []
        is_multispecialist = False

        if not specialist_candidates:
            primary_domain = "general"
            routing_reasons.append("Clinical presentation does not match specific organ-system specialist domains; routed to General Diagnosis Agent.")
        elif len(specialist_candidates) == 1:
            spec_name, score, matches = specialist_candidates[0]
            selected_specialists.append(spec_name)
            primary_domain = DOMAIN_NAME_MAP.get(spec_name, "general")
            match_str = f" (matched indicators: {', '.join(matches[:5])})" if matches else ""
            routing_reasons.append(f"High clinical confidence in {spec_name} domain (score {score:.2f}){match_str}.")
        else:
            top_spec, top_score, top_matches = specialist_candidates[0]
            selected_specialists.append(top_spec)
            primary_domain = DOMAIN_NAME_MAP.get(top_spec, "general")
            routing_reasons.append(f"Primary domain resolved to {top_spec} (score {top_score:.2f}, indicators: {', '.join(top_matches[:4])}).")

            for sec_spec, sec_score, sec_matches in specialist_candidates[1:]:
                if sec_score >= top_score * self.multi_domain_ratio and sec_score >= 0.35:
                    selected_specialists.append(sec_spec)
                    is_multispecialist = True
                    routing_reasons.append(
                        f"Co-activated {sec_spec} for cross-domain differential (score {sec_score:.2f}, indicators: {', '.join(sec_matches[:4])})."
                    )

        logger.info(
            f"SpecialistRouter decision: specialists={selected_specialists}, primary={primary_domain}, "
            f"multispecialist={is_multispecialist}, scores={domain_scores}"
        )

        return SpecialistRoutingDecision(
            selected_specialists=selected_specialists,
            primary_domain=primary_domain,
            domain_scores=domain_scores,
            routing_reasons=routing_reasons,
            is_multispecialist=is_multispecialist
        )
