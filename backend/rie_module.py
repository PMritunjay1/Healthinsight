import os
import re
import json
import logging
from typing import Dict, List, Any, Set
from backend.models import RIEOutput

logger = logging.getLogger("RIE_Module")

class RIEModule:
    def __init__(self, threshold: int = 2):
        self.threshold = threshold
        
        # Load dictionaries
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(base_dir, 'config')
        
        self.abbr_path = os.path.join(config_dir, 'abbreviation_dictionary.json')
        self.neg_path = os.path.join(config_dir, 'negation_patterns.json')
        self.syn_path = os.path.join(config_dir, 'synonym_dictionary.json')
        
        self.abbreviations: Dict[str, str] = {}
        self.negation_patterns: List[str] = []
        self.synonyms: Dict[str, List[str]] = {}
        self.spell_corrections: Dict[str, str] = {
            "pan": "pain", "paain": "pain", "paine": "pain",
            "chect": "chest", "ches": "chest", "chst": "chest",
            "athma": "asthma", "ashtma": "asthma", "astma": "asthma",
            "headeche": "headache", "hedache": "headache", "headach": "headache",
            "feever": "fever", "fver": "fever", "fevr": "fever",
            "coughh": "cough", "couggh": "cough", "couhg": "cough",
            "nauseus": "nausea", "nauseaa": "nausea", "nausia": "nausea",
            "diabete": "diabetes", "diabtes": "diabetes", "diabetis": "diabetes",
            "arthritiss": "arthritis", "sweling": "swelling",
            "swolling": "swelling", "sweeling": "swelling",
            "anxity": "anxiety", "htnn": "htn",
            "sobb": "sob", "utii": "uti",
            "breth": "breath", "brething": "breathing", "shoortness": "shortness",
            "presure": "pressure", "radiatin": "radiating", "radiatng": "radiating",
            "palpitaton": "palpitations", "palpitation": "palpitations",
            "weaknes": "weakness", "weaknss": "weakness", "wickness": "weakness",
            "droop": "droop", "droopin": "droop", "strok": "stroke", "strk": "stroke"
        }
        
        self.last_corrected_targets: Set[str] = set()
        self.last_abbr_expanded_count = 0
        
        self.load_dictionaries()

    def load_dictionaries(self):
        # Load abbreviation dictionary
        if os.path.exists(self.abbr_path):
            with open(self.abbr_path, 'r') as f:
                self.abbreviations = json.load(f)
        else:
            logger.warning(f"Abbreviation file not found at {self.abbr_path}")

        # Load negation patterns
        if os.path.exists(self.neg_path):
            with open(self.neg_path, 'r') as f:
                self.negation_patterns = json.load(f)
        else:
            logger.warning(f"Negation patterns not found at {self.neg_path}")

        # Load synonym dictionary
        if os.path.exists(self.syn_path):
            with open(self.syn_path, 'r') as f:
                self.synonyms = json.load(f)
        else:
            logger.warning(f"Synonym dictionary not found at {self.syn_path}")

    def clean_text(self, text: str) -> str:
        # Lowercase
        t = text.lower().strip()
        # Clean double spaces
        t = re.sub(r'\s+', ' ', t)
        return t

    def apply_spelling_corrections(self, text: str) -> str:
        self.last_corrected_targets = set()
        words = text.split()
        corrected_words = []
        for w in words:
            # Strip punctuation from word for lookup
            clean_w = re.sub(r'[^\w]', '', w)
            if clean_w in self.spell_corrections:
                corrected_term = self.spell_corrections[clean_w]
                corrected = w.replace(clean_w, corrected_term)
                corrected_words.append(corrected)
                # Keep track of standard spelling of the corrected entity
                self.last_corrected_targets.add(corrected_term)
            else:
                corrected_words.append(w)
        return " ".join(corrected_words)

    def expand_abbreviations(self, text: str) -> str:
        expanded = text
        self.last_abbr_expanded_count = 0
        for abbr, full in self.abbreviations.items():
            pattern = re.compile(rf'\b{abbr}\b', re.IGNORECASE)
            matches = list(pattern.finditer(expanded))
            if matches:
                self.last_abbr_expanded_count += len(matches)
                expanded = pattern.sub(full, expanded)
        return expanded

    def parse_numerical_values(self, text: str) -> Dict[str, Any]:
        nums = {}
        
        # 1. Blood Pressure: e.g. 120/80
        bp_match = re.search(r'\b(\d{2,3})/(\d{2,3})\b', text)
        if bp_match:
            nums["blood_pressure"] = f"{bp_match.group(1)}/{bp_match.group(2)}"
            
        # 2. Heart Rate: e.g. 72 bpm, HR 72, heart rate of 72
        hr_match = re.search(r'\b(\d{2,3})\s*(?:bpm|beats per minute|beats/min)\b', text)
        if not hr_match:
            hr_match = re.search(r'\b(?:heart rate|hr)\s*(?:of|is|:)?\s*(\d{2,3})\b', text)
        if hr_match:
            nums["heart_rate"] = int(hr_match.group(1))

        # 3. Temperature: e.g. 98.6 f, 37.2 c, temperature 99
        temp_match = re.search(r'\b(\d{2,3}(?:\.\d+)?)\s*(?:c|f|fahrenheit|celsius|degree|degrees|°c|°f)\b', text)
        if not temp_match:
            temp_match = re.search(r'\b(?:temperature|temp)\s*(?:of|is|:)?\s*(\d{2,3}(?:\.\d+)?)\b', text)
        if temp_match:
            nums["temperature"] = float(temp_match.group(1))

        # 4. Oxygen Saturation: e.g. 98%, spo2 of 95, o2 sat 97
        spo2_match = re.search(r'\b(\d{2,3})\s*(?:%|percent|o2|spo2|oxygen saturation)\b', text)
        if not spo2_match:
            spo2_match = re.search(r'\b(?:spo2|o2 sat)\s*(?:of|is|:)?\s*(\d{2,3})\b', text)
        if spo2_match:
            nums["oxygen_saturation"] = int(spo2_match.group(1))

        # 5. Respiratory Rate: e.g. 18 breaths/min, RR 20, respiratory rate of 16
        rr_match = re.search(r'\b(\d{1,2})\s*(?:breaths per minute|breaths/min|pm|bpm)\b', text)
        if not rr_match:
            rr_match = re.search(r'\b(?:respiratory rate|rr)\s*(?:of|is|:)?\s*(\d{1,2})\b', text)
        if rr_match:
            nums["respiratory_rate"] = int(rr_match.group(1))

        # 6. Glucose: e.g. 110 mg/dl, blood sugar of 140
        gl_match = re.search(r'\b(\d{2,3})\s*(?:mg/dl|mg/dL|glucose|blood sugar)\b', text)
        if not gl_match:
            gl_match = re.search(r'\b(?:glucose|blood sugar)\s*(?:of|is|:)?\s*(\d{2,3})\b', text)
        if gl_match:
            nums["glucose"] = int(gl_match.group(1))

        # 7. BMI: e.g. BMI of 28.5
        bmi_match = re.search(r'\bbmi\s*(?:of|is|:)?\s*(\d{1,2}(?:\.\d+)?)\b', text)
        if bmi_match:
            nums["bmi"] = float(bmi_match.group(1))

        return nums

    def detect_negation(self, text: str, word_start: int) -> bool:
        # Look back up to 4 words (approx. 25 chars) before the entity start index
        lookback_start = max(0, word_start - 30)
        lookback_window = text[lookback_start:word_start]
        
        # Check if any negation phrase occurs in the window before the word
        for pattern in self.negation_patterns:
            regex = re.compile(rf'\b{pattern}\b', re.IGNORECASE)
            if regex.search(lookback_window):
                return True
        return False

    def extract_entities(self, text: str, mode: str = "all") -> Dict[str, Dict[str, Any]]:
        """
        Extracts standardized entities and tracks their source and confidence.
        Returns dict: { 'std_name': { 'source': 'Regex/Dictionary/etc', 'confidence': float, 'start_idx': int } }
        """
        extracted = {}
        
        # If regex_only mode, we bypass abbreviations/synonyms and search for exact matching base keys
        if mode == "regex_only":
            for standard_key in self.synonyms.keys():
                pattern = re.compile(rf'\b{standard_key}\b', re.IGNORECASE)
                for match in pattern.finditer(text):
                    extracted[standard_key] = {
                        "source": "Regex",
                        "confidence": 1.0,
                        "start_idx": match.start()
                    }
            return extracted

        # Mode: regex_dict or all (full pipeline)
        # Scan synonym dictionary
        for standard_key, variations in self.synonyms.items():
            # Check standard key first
            pattern_std = re.compile(rf'\b{standard_key}\b', re.IGNORECASE)
            for match in pattern_std.finditer(text):
                # Standard key gets a confidence of 1.0, or 0.8 if it was a spelling corrected word
                conf = 0.8 if standard_key in self.last_corrected_targets else 1.0
                extracted[standard_key] = {
                    "source": "Dictionary",
                    "confidence": conf,
                    "start_idx": match.start()
                }
            
            # Check synonyms
            for variant in variations:
                pattern_var = re.compile(rf'\b{variant}\b', re.IGNORECASE)
                for match in pattern_var.finditer(text):
                    if standard_key not in extracted:
                        # Determine source and baseline confidence
                        if variant in self.abbreviations:
                            source_label = "Abbreviation Expansion"
                            base_conf = 0.95
                        else:
                            source_label = "Synonym Mapping"
                            base_conf = 0.9
                            
                        # If the variant or standard term was spelling corrected, lower confidence to 0.8
                        # We check if variant matches any of our correction lookups
                        is_corrected = False
                        for typo, corr in self.spell_corrections.items():
                            if typo in variant or corr in self.last_corrected_targets:
                                is_corrected = True
                                break
                        
                        conf = 0.8 if is_corrected else base_conf
                        
                        extracted[standard_key] = {
                            "source": source_label,
                            "confidence": conf,
                            "start_idx": match.start()
                        }
                        
        return extracted

    async def run(self, patient_text: str, history: str = "", mode: str = "all") -> RIEOutput:
        """
        Runs RIE pipeline.
        mode can be: 'regex_only', 'regex_dict', or 'all' (standard with negation)
        """
        raw_combined = patient_text + " " + history
        
        # 1. Clean Text
        cleaned = self.clean_text(raw_combined)
        
        # Preprocessing additions if not in regex_only
        if mode != "regex_only":
            # Apply spelling corrections
            cleaned = self.apply_spelling_corrections(cleaned)
            # Expand abbreviations
            cleaned = self.expand_abbreviations(cleaned)
            
        # 2. Entity Extraction
        entities = self.extract_entities(cleaned, mode=mode)
        
        # 3. Parse Numerical Values
        numerical = self.parse_numerical_values(cleaned)
        
        ss = []
        mh = []
        rf = []
        meta = {}
        
        negated_count = 0
        
        # Categorize
        risk_keys = {"smoker", "hypertension", "diabetic", "obese", "alcohol", "elderly"}
        history_keys = {"smoker", "hypertension", "diabetic", "asthma", "obese", "alcohol", "autoimmune"}
        
        for std_key, info in entities.items():
            start_idx = info["start_idx"]
            
            # 4. Negation Detection (skip in regex_only/regex_dict mode)
            is_neg = False
            if mode == "all":
                is_neg = self.detect_negation(cleaned, start_idx)
                
            if is_neg:
                logger.info(f"Filtered negated entity: {std_key}")
                negated_count += 1
                continue
                
            # Log entity metadata source and confidence
            meta[std_key] = {
                "source": info["source"],
                "confidence": info["confidence"]
            }
            
            # Populate categorizations
            if std_key in risk_keys:
                rf.append(std_key)
            if std_key in history_keys:
                mh.append(std_key)
            if std_key not in risk_keys and std_key not in history_keys:
                ss.append(std_key)
                
        # Append numerical metadata
        for num_k in numerical.keys():
            meta[num_k] = {
                "source": "Numeric Parser",
                "confidence": 1.0
            }
            
        # Safe default confidence score
        confidence = 1.0 if (len(ss) + len(mh) + len(rf) + len(numerical) > 0) else 0.8
        
        # Build Stats Dictionary
        stats = {
            "total_entities_extracted": len(ss) + len(mh) + len(rf),
            "symptoms": len(ss),
            "diseases": len(mh),
            "vitals": len(numerical),
            "negated_entities": negated_count,
            "abbreviations_expanded": self.last_abbr_expanded_count,
            "spelling_corrections": len(self.last_corrected_targets)
        }
        
        return RIEOutput(
            symptoms=sorted(list(set(ss))),
            medical_history=sorted(list(set(mh))),
            risk_factors=sorted(list(set(rf))),
            key_findings=[],
            confidence=confidence,
            numerical_values=numerical,
            metadata=meta,
            stats=stats
        )
