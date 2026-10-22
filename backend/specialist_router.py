import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Set

from backend.models import PatientInput, RIEOutput, IPMOutput, SpecialistRoutingDecision
from backend.ambiguity_detector import AmbiguityDetector

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

    def __init__(
        self,
        single_domain_threshold: float = 0.35,
        multi_domain_ratio: float = 0.50,
        ambiguity_detector: Optional[AmbiguityDetector] = None
    ):
        self.single_domain_threshold = single_domain_threshold
        self.multi_domain_ratio = multi_domain_ratio
        self.ambiguity_detector = ambiguity_detector or AmbiguityDetector()

        # Phase 2.8 / Calibrated Domain Router fitted on D_cal (clean_calib.json)
        self.calibrated_pipeline = None
        calib_router_path = os.path.join(
            os.path.dirname(__file__), "models", "calibrated_domain_router.joblib"
        )
        if os.path.exists(calib_router_path):
            try:
                import joblib
                self.calibrated_pipeline = joblib.load(calib_router_path)
                logger.info(f"SpecialistRouter successfully loaded calibrated domain router from {calib_router_path}")
            except Exception as e:
                logger.warning(f"Failed to load calibrated domain router: {e}")

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

    def _extract_pmh(self, text: str) -> str:
        """Extract Past Medical History section if present."""
        pmh_match = re.search(r'(Past Medical History|PMH):\s*(.*?)(?=(Social History|Family History|Physical Exam|Brief Hospital Course|Allergies|Medications|Review of Systems|$))', text, re.I | re.DOTALL)
        return pmh_match.group(2).strip().lower() if pmh_match else ""

    def _extract_sections(self, text: str) -> Tuple[str, str, str]:
        """Separate Chief Complaint, HPI, and general narrative, stripping routine vitals."""
        text_clean = re.sub(r'Triage Vitals:[^.]*\.', ' ', text, flags=re.I)
        text_clean = re.sub(r'\b(BP|HR|RR|Temp|SpO2)\s*=\s*[\d./]+', ' ', text_clean, flags=re.I)
        
        cc_match = re.search(r'Chief Complaint:\s*(.*?)(?=(Major Surgical|History of Present Illness|Past Medical History|$))', text_clean, re.I | re.DOTALL)
        cc_text = cc_match.group(1).strip().lower() if cc_match else ""
        
        hpi_match = re.search(r'History of Present Illness:\s*(.*?)(?=(Past Medical History|Physical Exam|Brief Hospital Course|Allergies|Medications|$))', text_clean, re.I | re.DOTALL)
        hpi_text = hpi_match.group(1).strip().lower() if hpi_match else ""
        
        return cc_text, hpi_text, text_clean.lower()

    def _score_domain(
        self,
        full_text_lower: str,
        cc_text_lower: str,
        hpi_text_lower: str,
        keywords: Set[str],
        ipm_domain_match: bool,
        rie_symptoms: List[str],
        pmh_text_lower: str = ""
    ) -> Tuple[float, List[str]]:
        """
        Calculate weighted score for a given specialty domain.
        Matches in Chief Complaint and HPI carry presenting priority over background history.
        Pure Past Medical History matches represent background comorbidities and are scored lower
        so they do not falsely override presenting complaints.
        """
        if not pmh_text_lower:
            pmh_text_lower = self._extract_pmh(full_text_lower)

        matched_cc = self._extract_matches(cc_text_lower, keywords) if cc_text_lower else []
        matched_hpi = self._extract_matches(hpi_text_lower, keywords) if hpi_text_lower else []
        matched_pmh = self._extract_matches(pmh_text_lower, keywords) if pmh_text_lower else []
        matched_full = self._extract_matches(full_text_lower, keywords)
        
        rie_matches = []
        for sym in rie_symptoms:
            sym_lower = sym.lower()
            if any(kw == sym_lower or kw in sym_lower for kw in keywords):
                rie_matches.append(sym)

        all_matches = list(set(matched_full + rie_matches))
        presenting_matches = list(set(matched_cc + matched_hpi + rie_matches))
        
        if not all_matches:
            kw_score = 0.0
        else:
            if presenting_matches:
                # Presenting symptom matches (CC, HPI, or RIE symptoms) grant high confidence
                base_score = 0.60
                kw_score = min(0.90, base_score + (len(presenting_matches) - 1) * 0.10)
                if matched_cc:
                    kw_score = min(0.95, kw_score + 0.10)
            elif matched_pmh and not presenting_matches:
                # Pure Past Medical History mentions represent chronic background conditions
                # Discounted below single_domain_threshold (0.35) unless extensive
                kw_score = min(0.28, 0.18 + (len(matched_pmh) - 1) * 0.04)
            else:
                # General narrative matches when section headers are absent
                base_score = 0.40
                kw_score = min(0.80, base_score + (len(all_matches) - 1) * 0.08)
            
        ipm_boost = 0.20 if ipm_domain_match else 0.0
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
        raw_text = input_data.patient_text or ""
        history = input_data.history or ""
        combined_raw = f"{raw_text} {history}".strip()

        cc_text, hpi_text, full_text_clean = self._extract_sections(combined_raw)
        pmh_text = self._extract_pmh(full_text_clean)

        rie_symptoms = rie_output.symptoms if rie_output else []
        ipm_domain = (ipm_output.medical_domain or "general").lower() if ipm_output else "general"

        ipm_is_cardio = "cardio" in ipm_domain
        ipm_is_pulmo = "pulmo" in ipm_domain or "respirat" in ipm_domain
        ipm_is_neuro = "neuro" in ipm_domain

        # Score each specialty domain with section awareness
        cardio_kw_score, cardio_matches = self._score_domain(full_text_clean, cc_text, hpi_text, self.CARDIO_KEYWORDS, ipm_is_cardio, rie_symptoms, pmh_text_lower=pmh_text)
        pulmo_kw_score, pulmo_matches = self._score_domain(full_text_clean, cc_text, hpi_text, self.PULMO_KEYWORDS, ipm_is_pulmo, rie_symptoms, pmh_text_lower=pmh_text)
        neuro_kw_score, neuro_matches = self._score_domain(full_text_clean, cc_text, hpi_text, self.NEURO_KEYWORDS, ipm_is_neuro, rie_symptoms, pmh_text_lower=pmh_text)
        cardio_score = cardio_kw_score
        pulmo_score = pulmo_kw_score
        neuro_score = neuro_kw_score

        calib_top = None
        calib_scores = {}
        if self.calibrated_pipeline is not None:
            try:
                eval_txt = f"{combined_raw[:1500]} {combined_raw[-1000:]}" if len(combined_raw) > 2500 else combined_raw
                calib_top = str(self.calibrated_pipeline.predict([eval_txt])[0])
                clf_probs = self.calibrated_pipeline.predict_proba([eval_txt])[0]
                classes = list(self.calibrated_pipeline.classes_)
                calib_scores = {c: float(p) for c, p in zip(classes, clf_probs)}

                # Blend keyword scores with calibrated model probabilities
                cardio_score = round(float(cardio_score * 0.25 + calib_scores.get("cardio", 0.0) * 0.75), 4)
                pulmo_score = round(float(pulmo_score * 0.25 + calib_scores.get("pulmo", 0.0) * 0.75), 4)
                neuro_score = round(float(neuro_score * 0.25 + calib_scores.get("neuro", 0.0) * 0.75), 4)
            except Exception as e:
                logger.warning(f"Calibrated domain inference failed: {e}")

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

        # Re-sort candidates based on calibrated score
        specialist_candidates.sort(key=lambda x: x[1], reverse=True)

        selected_specialists = []
        secondary_domains = []
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
            primary_domain = calib_top if calib_top and calib_scores.get(calib_top, 0.0) >= 0.40 else DOMAIN_NAME_MAP.get(top_spec, "general")
            routing_reasons.append(f"Primary domain resolved to {primary_domain} (score {top_score:.2f}, indicators: {', '.join(top_matches[:4])}).")

            for sec_spec, sec_score, sec_matches in specialist_candidates[1:]:
                if sec_score >= top_score * self.multi_domain_ratio and sec_score >= 0.35:
                    if sec_spec not in selected_specialists:
                        selected_specialists.append(sec_spec)
                    secondary_domains.append(sec_spec)
                    is_multispecialist = True
                    routing_reasons.append(
                        f"Co-activated {sec_spec} for cross-domain differential (score {sec_score:.2f}, indicators: {', '.join(sec_matches[:4])})."
                    )

        # Collect all presenting clinical symptom indicators across RIE and domain matches
        all_symptom_indicators = list(set(rie_symptoms + cardio_matches + pulmo_matches + neuro_matches))

        # Symptom-level domain scores for clinical ambiguity calculation (Algorithm 1)
        symptom_domain_scores = {
            "cardio": round(cardio_kw_score, 4),
            "pulmo": round(pulmo_kw_score, 4),
            "neuro": round(neuro_kw_score, 4),
            "general": 0.0
        }

        # Ambiguity Risk Detector & Activation Tiers (Algorithm 1)
        ambiguity_info = self.ambiguity_detector.evaluate(all_symptom_indicators, symptom_domain_scores)
        routing_tier = ambiguity_info["routing_tier"]
        ambiguity_score = ambiguity_info["ambiguity_score"]
        abstain_eligible = ambiguity_info["abstain_eligible"]
        retrieval_enabled = ambiguity_info["retrieval_enabled"]

        if routing_tier == "HIGH":
            # Tier 3 (High Risk): Full specialist board co-activation + retrieval + abstain-eligible
            selected_specialists = ["cardiologist", "pulmonologist", "neurologist"]
            is_multispecialist = True
            secondary_domains = [s for s in ["cardiologist", "pulmonologist", "neurologist"] if DOMAIN_NAME_MAP.get(s) != primary_domain]
            routing_reasons.append(
                f"High risk ambiguity tier (R={ambiguity_score:.2f} >= {self.ambiguity_detector.r_high}): "
                f"activated full specialist board with selective abstention eligibility."
            )
        elif routing_tier == "MEDIUM":
            routing_reasons.append(
                f"Medium risk ambiguity tier (R={ambiguity_score:.2f}): primary specialist activated with guideline retrieval grounding enabled."
            )
        else:
            routing_reasons.append(
                f"Low risk ambiguity tier (R={ambiguity_score:.2f} < {self.ambiguity_detector.r_low}): routed directly to single primary specialist."
            )

        logger.info(
            f"SpecialistRouter decision: specialists={selected_specialists}, primary={primary_domain}, "
            f"tier={routing_tier}, ambiguity={ambiguity_score:.2f}, multispecialist={is_multispecialist}, scores={domain_scores}"
        )

        return SpecialistRoutingDecision(
            selected_specialists=selected_specialists,
            primary_domain=primary_domain,
            secondary_domains=secondary_domains,
            domain_scores=domain_scores,
            routing_reasons=routing_reasons,
            is_multispecialist=is_multispecialist,
            routing_tier=routing_tier,
            ambiguity_score=ambiguity_score,
            abstain_eligible=abstain_eligible,
            retrieval_enabled=retrieval_enabled
        )
