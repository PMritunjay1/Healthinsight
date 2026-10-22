"""
HealthInsight-UQ: Ambiguity Risk Detector & Activation Router (Algorithm 1)
============================================================================
Implements:
  R(x) = gamma_1 * (H(S(x)) / log |S(x)|) + gamma_2 * (1 - rho_rarity(x)) + gamma_3 * O_diff(x)

Governs 3 Risk-Conditioned Activation Tiers:
  - Low Risk (R < R_low): Single domain specialist {a_domain(x)}.
  - Medium Risk (R_low <= R < R_high): Specialist + Offline Guideline Retrieval.
  - High Risk (R >= R_high): Full Specialist Board + Retrieval + Abstain-Eligible.
"""

import math
from typing import Any, Dict, List, Set, Tuple


class AmbiguityDetector:
    """
    Computes clinical ambiguity risk R(x) and assigns risk-conditioned activation tiers.
    """

    # Landmark common symptoms with high empirical prevalence prior (rho ~ 0.85)
    LANDMARK_COMMON_SYMPTOMS: Set[str] = {
        "chest pain", "shortness of breath", "dyspnea", "cough", "headache",
        "dizziness", "palpitations", "edema", "wheezing", "weakness",
        "nausea", "fever", "hypertension", "fatigue", "chest pressure"
    }

    # Lexicons for symptom domain assignment
    CARDIO_INDICATORS: Set[str] = {
        "chest pain", "chest tightness", "chest pressure", "substernal", "angina",
        "palpitations", "irregular heartbeat", "arrhythmia", "atrial fibrillation",
        "edema", "pedal edema", "orthopnea", "pnd", "murmur", "left arm pain"
    }

    PULMO_INDICATORS: Set[str] = {
        "shortness of breath", "dyspnea", "sob", "cough", "wheezing", "stridor",
        "sputum", "hemoptysis", "hypoxia", "crackles", "pleuritic chest pain",
        "respiratory distress", "bronchospasm"
    }

    NEURO_INDICATORS: Set[str] = {
        "headache", "migraine", "seizure", "convulsions", "altered mental status",
        "confusion", "delirium", "vertigo", "loss of balance", "ataxia",
        "numbness", "paresthesia", "facial droop", "slurred speech", "dysarthria",
        "aphasia", "diplopia", "syncope", "loss of consciousness"
    }

    def __init__(
        self,
        gamma_1: float = 0.40,
        gamma_2: float = 0.30,
        gamma_3: float = 0.30,
        r_low: float = 0.35,
        r_high: float = 0.65,
    ):
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.gamma_3 = gamma_3
        self.r_low = r_low
        self.r_high = r_high

    def compute_symptom_domain_entropy(self, symptoms: List[str]) -> Tuple[float, Dict[str, int]]:
        """
        Calculates normalized Shannon entropy of extracted symptoms across clinical domains.
        Returns (H_norm, domain_counts). If symptoms span 0 or 1 domain, H_norm = 0.0.
        """
        if not symptoms or len(symptoms) <= 1:
            return 0.0, {"cardio": 0, "pulmo": 0, "neuro": 0}

        counts = {"cardio": 0, "pulmo": 0, "neuro": 0}
        for s in symptoms:
            s_clean = str(s).lower().strip()
            # Match domain if symptom contains any indicator keyword
            is_c = any(k == s_clean or k in s_clean for k in self.CARDIO_INDICATORS)
            is_p = any(k == s_clean or k in s_clean for k in self.PULMO_INDICATORS)
            is_n = any(k == s_clean or k in s_clean for k in self.NEURO_INDICATORS)

            if is_c:
                counts["cardio"] += 1
            if is_p:
                counts["pulmo"] += 1
            if is_n:
                counts["neuro"] += 1

        active_counts = [c for c in counts.values() if c > 0]
        K_active = len(active_counts)
        if K_active <= 1:
            return 0.0, counts

        total_active = sum(active_counts)
        probs = [c / total_active for c in active_counts]
        shannon_h = -sum(p * math.log(p) for p in probs if p > 0)
        max_h = math.log(K_active)
        h_norm = shannon_h / max_h if max_h > 0 else 0.0

        return round(min(1.0, max(0.0, h_norm)), 4), counts

    def compute_rarity_score(self, symptoms: List[str]) -> float:
        """
        Computes (1 - rho_rarity) where rho_rarity is empirical commonality prior.
        Common landmark presentations produce low penalty (near 0.15).
        Atypical/rare presentations produce higher penalty (up to 0.70).
        """
        if not symptoms:
            # Neutral prior when no symptoms extracted
            return 0.20

        common_matches = sum(
            1 for s in symptoms
            if any(c in str(s).lower() or str(s).lower() in c for c in self.LANDMARK_COMMON_SYMPTOMS)
        )
        common_ratio = common_matches / len(symptoms)
        # rho in [0.30, 0.85] -> (1 - rho) in [0.15, 0.70]
        rho = 0.30 + (0.55 * common_ratio)
        return round(1.0 - rho, 4)

    def compute_differential_overlap(self, domain_scores: Dict[str, float]) -> float:
        """
        Computes differential overlap O_diff = s_(2) / s_(1) strictly across specialist domains.
        When a secondary specialty competes with primary, O_diff approaches 1.0.
        When only one specialty is relevant, O_diff is 0.0.
        """
        specialist_scores = [
            float(domain_scores[k]) for k in ["cardio", "pulmo", "neuro"] if k in domain_scores
        ]
        scores = sorted(specialist_scores, reverse=True)
        if not scores or scores[0] <= 0.0:
            return 0.0
        s_top = scores[0]
        s_sec = scores[1] if len(scores) > 1 else 0.0
        ratio = s_sec / s_top
        return round(min(1.0, max(0.0, ratio)), 4)

    def evaluate(
        self,
        symptoms: List[str],
        domain_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Evaluates full ambiguity score R(x) and maps to activation tier.
        """
        h_norm, domain_counts = self.compute_symptom_domain_entropy(symptoms)
        rarity_penalty = self.compute_rarity_score(symptoms)
        diff_overlap = self.compute_differential_overlap(domain_scores)

        r_score = (
            self.gamma_1 * h_norm
            + self.gamma_2 * rarity_penalty
            + self.gamma_3 * diff_overlap
        )
        r_score = round(min(1.0, max(0.0, r_score)), 4)

        if r_score < self.r_low:
            tier = "LOW"
            abstain_eligible = False
            retrieval_enabled = False
        elif r_score < self.r_high:
            tier = "MEDIUM"
            abstain_eligible = False
            retrieval_enabled = True
        else:
            tier = "HIGH"
            abstain_eligible = True
            retrieval_enabled = True

        return {
            "ambiguity_score": r_score,
            "routing_tier": tier,
            "abstain_eligible": abstain_eligible,
            "retrieval_enabled": retrieval_enabled,
            "h_norm": h_norm,
            "rarity_penalty": rarity_penalty,
            "diff_overlap": diff_overlap,
            "domain_symptom_counts": domain_counts,
        }
