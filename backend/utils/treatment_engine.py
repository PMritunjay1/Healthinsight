import os
import json
from typing import Dict, Any

class TreatmentEngine:
    def __init__(self, rules_path: str = None):
        if rules_path is None:
            # Default location
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            rules_path = os.path.join(base_dir, 'config', 'treatment_rules.json')
        
        self.rules_path = rules_path
        self.rules: Dict[str, Any] = {}
        self.load_rules()

    def load_rules(self):
        if not os.path.exists(self.rules_path):
            raise FileNotFoundError(f"Treatment rules configuration not found at {self.rules_path}")
        
        with open(self.rules_path, 'r') as f:
            self.rules = json.load(f)
        
        # Enforce basic validation
        self.validate_schema()

    def validate_schema(self):
        required_keys = {
            "recommended_treatment", "medication_category", "lifestyle_recommendations",
            "contraindications", "red_flags", "emergency_referral_conditions",
            "specialist_referral", "follow_up", "guideline_version", "last_updated",
            "evidence_source", "rationale"
        }
        
        for disease, data in self.rules.items():
            missing = required_keys - set(data.keys())
            if missing:
                raise ValueError(f"Schema violation for disease '{disease}': Missing required fields {missing}")
            
            # recommended_treatment must be a dictionary with risk-stratified entries
            recs = data["recommended_treatment"]
            if not isinstance(recs, dict):
                raise TypeError(f"Schema violation for disease '{disease}': 'recommended_treatment' must be a risk-stratified dictionary.")
            
            risk_keys = {"Low", "Medium", "High"}
            missing_risks = risk_keys - set(recs.keys())
            if missing_risks:
                raise ValueError(f"Schema violation for disease '{disease}': 'recommended_treatment' missing risk levels {missing_risks}")

    def get_guidelines(self, disease: str, risk_level: str = "Medium") -> Dict[str, Any]:
        # Normalize disease name matching (case insensitive strip)
        target_key = None
        for key in self.rules.keys():
            if key.strip().lower() == disease.strip().lower():
                target_key = key
                break
                
        # If no exact match, try intelligent substring/keyword matching
        if target_key is None:
            disease_lower = disease.strip().lower()
            # Normalize common variants
            for key in self.rules.keys():
                key_lower = key.strip().lower()
                # Split multi-word diseases to check for core keywords (e.g. 'asthma', 'gerd', 'reflux')
                key_words = [w for w in key_lower.replace("'", "").replace("-", " ").split() if len(w) > 3]
                if any(w in disease_lower for w in key_words) or key_lower in disease_lower or disease_lower in key_lower:
                    target_key = key
                    break
        
        # Normalize risk level
        norm_risk = risk_level.strip().capitalize()
        if norm_risk == "Critical":
            norm_risk = "High"
        if norm_risk not in ("Low", "Medium", "High"):
            norm_risk = "Medium"
            
        if target_key is None:
            # Safe clinical fallback guidelines
            return {
                "diagnosis": disease,
                "risk_level": risk_level,
                "recommended_treatment": "Consult a General Practitioner for formal clinical diagnosis and management.",
                "medication_category": "Triage referral only; no immediate pharmaceuticals advised.",
                "lifestyle_recommendations": "Rest, stay hydrated, and monitor vitals closely.",
                "contraindications": "Avoid self-medication or taking unprescribed antibiotics.",
                "red_flag_symptoms": "High fever, difficulty breathing, altered mental status, severe localized pain.",
                "emergency_referral_conditions": "Development of any red flag warning symptoms.",
                "specialist_referral": "General Practitioner",
                "follow_up_advice": "Consult a primary care physician within 24 to 48 hours.",
                "evidence_source": "World Health Organization (WHO) Basic Triage Protocol",
                "guideline_version": "WHO Triage v1.0",
                "confidence": 1.0,
                "rationale": "An unrecognized medical condition cannot be mapped to specific guideline rules. Safe referral is the standard protocol."
            }
            
        data = self.rules[target_key]
        
        # Get risk-stratified recommendation
        rec_actions = data["recommended_treatment"].get(norm_risk, data["recommended_treatment"]["Medium"])
        
        return {
            "diagnosis": target_key,
            "risk_level": norm_risk,
            "recommended_treatment": rec_actions,
            "medication_category": data["medication_category"],
            "lifestyle_recommendations": data["lifestyle_recommendations"],
            "contraindications": data["contraindications"],
            "red_flag_symptoms": data["red_flags"],
            "emergency_referral_conditions": data["emergency_referral_conditions"],
            "specialist_referral": data["specialist_referral"],
            "follow_up_advice": data["follow_up"],
            "evidence_source": data["evidence_source"],
            "guideline_version": data["guideline_version"],
            "confidence": 1.0,
            "rationale": data["rationale"]
        }
