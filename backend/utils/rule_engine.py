import os
import re
import json

class RuleEngine:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                'config', 
                'intent_rules.json'
            )
        self.config_path = config_path
        self.rules = self.load_and_validate_rules()
        
        # Load rules into fields for fast lookup
        self.synonyms = self.rules['synonyms']
        self.domain_mappings = self.rules['domain_mappings']
        self.urgency_rules = self.rules['urgency_rules']
        self.task_rules = self.rules['task_rules']
        self.stopwords = self.rules['stopwords']
        self.negation_patterns = self.rules['negation_patterns']

    def load_and_validate_rules(self):
        """Loads and performs schema validation, duplicate detection, and conflict detection."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                rules = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in configuration file: {str(e)}")
            
        # 1. Verify required sections exist
        required_sections = [
            "synonyms", "domain_mappings", "urgency_rules", 
            "task_rules", "stopwords", "negation_patterns"
        ]
        for section in required_sections:
            if section not in rules:
                raise ValueError(f"Required configuration section '{section}' is missing.")
                
        # 2. Detect duplicate keywords in urgency lists
        urgency_kws = {}
        for level, keywords in rules['urgency_rules'].items():
            for kw in keywords:
                kw_clean = kw.lower().strip()
                if kw_clean in urgency_kws:
                    # Conflicting or duplicate check
                    old_level = urgency_kws[kw_clean]
                    if old_level == level:
                        raise ValueError(f"Duplicate keyword found in urgency level '{level}': '{kw}'")
                    else:
                        raise ValueError(f"Conflict: Keyword '{kw}' mapped to both '{old_level}' and '{level}' urgency.")
                urgency_kws[kw_clean] = level
                
        # 3. Detect duplicate and conflicting keywords in task rules
        task_kws = {}
        for task_name, content in rules['task_rules'].items():
            keywords = content.get('keywords', [])
            for kw in keywords:
                kw_clean = kw.lower().strip()
                if kw_clean in task_kws:
                    old_task = task_kws[kw_clean]
                    if old_task == task_name:
                        raise ValueError(f"Duplicate keyword found in task '{task_name}': '{kw}'")
                    else:
                        raise ValueError(f"Conflict: Keyword '{kw}' mapped to both '{old_task}' and '{task_name}' tasks.")
                task_kws[kw_clean] = task_name
                
        return rules

    def is_negated(self, text: str, keyword: str) -> bool:
        """Determines if a matched keyword is negated in the text."""
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        idx = text_lower.find(keyword_lower)
        if idx == -1:
            return False
            
        # Extract preceding phrase (up to 30 chars before the keyword)
        prefix = text_lower[max(0, idx - 30):idx].strip()
        
        # Check if the prefix contains any negation patterns with word boundary
        for neg in self.negation_patterns:
            pattern = r'\b' + re.escape(neg) + r'\b'
            if re.search(pattern, prefix):
                return True
        return False

    def apply_synonym_mapping(self, text: str) -> str:
        """Applies synonym normalizations specified in the config rules."""
        if not isinstance(text, str):
            return ""
        text_lower = text.lower()
        for synonym_key, replacement in self.synonyms.items():
            # Match word boundary for synonym keys
            pattern = r'\b' + re.escape(synonym_key) + r'\b'
            text_lower = re.sub(pattern, replacement, text_lower)
        return text_lower

    def map_domain(self, specialty: str, text: str) -> str:
        """Maps medical specialty raw value to standard domain specialty."""
        if not isinstance(specialty, str) or not isinstance(text, str):
            return "general"
            
        spec_lower = specialty.lower().strip()
        text_lower = text.lower()
        
        # Traverse mappings
        for mapping in self.domain_mappings:
            specialty_keywords = mapping.get('specialty_keywords', [])
            
            # Check if specialty matches
            if any(kw in spec_lower for kw in specialty_keywords):
                # Check for disambiguation rule
                disambiguation = mapping.get('disambiguation')
                if disambiguation:
                    keywords = disambiguation.get('keywords', [])
                    match_domain = disambiguation.get('match_domain')
                    default_domain = disambiguation.get('default_domain')
                    
                    if any(kw in text_lower for kw in keywords):
                        return match_domain
                    return default_domain
                else:
                    return mapping.get('target_domain', 'general')
                    
        return "general"

    def map_urgency(self, text: str) -> str:
        """Classifies urgency level based on deterministic keyword matching and negation checks."""
        if not isinstance(text, str):
            return "low"
            
        text_lower = text.lower()
        
        # Priority 1: Critical
        for kw in self.urgency_rules.get('critical', []):
            if kw in text_lower:
                if not self.is_negated(text_lower, kw):
                    return "critical"
                    
        # Priority 2: High
        for kw in self.urgency_rules.get('high', []):
            if kw in text_lower:
                if not self.is_negated(text_lower, kw):
                    return "high"
                    
        # Priority 3: Medium
        for kw in self.urgency_rules.get('medium', []):
            if kw in text_lower:
                if not self.is_negated(text_lower, kw):
                    return "medium"
                    
        return "low"

    def map_task_type(self, specialty: str, text: str) -> str:
        """Determines clinical task type from text severity and clinical keywords."""
        if not isinstance(specialty, str) or not isinstance(text, str):
            return "general"
            
        spec_lower = specialty.lower()
        text_lower = text.lower()
        
        # Priority 1: Emergency (Critical triage)
        if self.map_urgency(text) == "critical":
            return "emergency"
            
        # Priority 2: Treatment
        treatment_cfg = self.task_rules.get('treatment', {})
        treatment_kws = treatment_cfg.get('keywords', [])
        treatment_specs = treatment_cfg.get('specialty_keywords', [])
        
        if any(spec in spec_lower for spec in treatment_specs) or any(kw in text_lower for kw in treatment_kws):
            return "treatment"
            
        # Priority 3: Diagnosis
        diagnosis_cfg = self.task_rules.get('diagnosis', {})
        diagnosis_kws = diagnosis_cfg.get('keywords', [])
        diagnosis_specs = diagnosis_cfg.get('specialty_keywords', [])
        
        if any(spec in spec_lower for spec in diagnosis_specs) or any(kw in text_lower for kw in diagnosis_kws):
            return "diagnosis"
            
        return "general"
