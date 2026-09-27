import os
import re
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from backend.agents.base_agent import BaseAgent
from backend.models import PatientInput, AgentResponse, RiskOutput
from backend.rie_module import RIEModule

class RiskAgent(BaseAgent):
    def __init__(self, domain: str = "general"):
        super().__init__(name="RiskAgent", response_model=RiskOutput, domain=domain)
        
        # Load local ONNX session and tokenizer
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', 'risk_model_Bio_ClinicalBERT_quant.onnx')
        
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT', local_files_only=True)
        
        # Instantiate RIE for template preprocessing
        self.rie = RIEModule()
        
        # Target classes (mapped to ESI 1-4 triage levels)
        self.classes = ["Low", "Medium", "High", "Critical"]
        self.temperature = 1.0824 # Optimal validation temperature scaling parameter

    def _get_system_prompt(self) -> str:
        return "Deterministic local Bio_ClinicalBERT ONNX risk assessment."

    def _synthesize_note(self, text: str, history: str, rie_output) -> str:
        text_comb = text.lower() + " " + history.lower()
        
        # 1. Age
        age_match = re.search(r'\b(\d{1,2})\s*-?\s*year\s*-?\s*old\b', text_comb)
        age_str = f"{float(age_match.group(1)):.1f}" if age_match else "unknown"
        if age_str == "unknown" and "elderly" in text_comb:
            age_str = "75.0"
            
        # 2. Arrival Mode
        arrival = "walk in"
        if "ambulance" in text_comb or "ems" in text_comb:
            arrival = "ambulance"
            
        # 3. Pain Score
        pain = "unknown"
        pain_match = re.search(r'\bpain\s*(?:score|level)?\s*(?:of|is|:)?\s*(\d{1,2})\b', text_comb)
        if pain_match:
            pain = pain_match.group(1)
        elif any(k in text_comb for k in ("severe pain", "crushing", "extreme discomfort")):
            pain = "8"
        elif "pain" in text_comb or "discomfort" in text_comb:
            pain = "5"
            
        # 4. Extract vitals from RIE
        hr = "unknown"
        bp = "unknown"
        o2 = "unknown"
        temp = "unknown"
        chronics = "0"
        visits = "0"
        
        if rie_output:
            nums = rie_output.numerical_values
            if "heart_rate" in nums:
                hr = f"{float(nums['heart_rate']):.1f}"
            if "blood_pressure" in nums:
                parts = str(nums["blood_pressure"]).split('/')
                if parts:
                    bp = f"{float(parts[0]):.1f}" # Systolic blood pressure
            if "oxygen_saturation" in nums:
                o2 = f"{float(nums['oxygen_saturation']):.1f}"
            if "temperature" in nums:
                t_val = float(nums['temperature'])
                # Convert Fahrenheit to Celsius if it looks like F
                if t_val > 50:
                    t_val = (t_val - 32) * 5/9
                temp = f"{t_val:.1f}"
                
            # Chronic conditions count
            chronics = str(len(rie_output.medical_history))
            # ER visits estimation
            if "chronic" in text_comb or len(rie_output.medical_history) > 1:
                visits = "2"
                
        note = (
            f"patient is a {age_str}-year-old who presented as a {arrival}. "
            f"chief complaint is severe discomfort with a pain score of {pain}/10. "
            f"vitals on admission: pulse rate {hr} bpm, systolic blood pressure {bp} mmhg, "
            f"oxygen saturation {o2}%, and body temperature {temp} c. "
            f"past medical history is significant for {chronics} chronic conditions and {visits} previous emergency department visits."
        )
        return note

    async def run(self, input_data: PatientInput) -> AgentResponse:
        start_time = time.time()
        error_msg = None
        output = None
        
        try:
            patient_text = input_data.patient_text
            
            # 1. Run local RIE to extract parameters
            rie_out = await self.rie.run(patient_text, input_data.history)
            
            # 2. Synthesize note matching the training distribution template
            note = self._synthesize_note(patient_text, input_data.history, rie_out)
            
            # 3. Run Tokenizer
            inputs = self.tokenizer(note, padding='max_length', truncation=True, max_length=128, return_tensors="np")
            input_ids = inputs["input_ids"].astype(np.int64)
            attention_mask = inputs["attention_mask"].astype(np.float32)
            
            # 4. Run ONNX Session
            outputs = self.session.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })
            logits = outputs[0] # Shape [1, 4]
            
            # 5. Apply Temperature Scaling Calibration
            calibrated_logits = logits / self.temperature
            
            # 6. Compute Softmax Probabilities
            exp_logits = np.exp(calibrated_logits - np.max(calibrated_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            probs = probs[0]
            
            # 7. Resolve predicted class
            pred_idx = int(np.argmax(probs))
            risk_level = self.classes[pred_idx]
            
            # 8. Extract critical warning flags if risk is High or Critical
            critical_flags = []
            if risk_level in ("High", "Critical"):
                text_lower = patient_text.lower()
                flags_map = {
                    "chest pain": "Suspected myocardial ischemia / cardiac pathology",
                    "angina": "Suspected myocardial ischemia / cardiac pathology",
                    "difficulty breathing": "Impending respiratory distress",
                    "breathless": "Impending respiratory distress",
                    "sob": "Impending respiratory distress",
                    "silent chest": "Acute respiratory compromise / severe asthma",
                    "cyanosis": "Peripheral/Central oxygen saturation depletion",
                    "confusion": "Altered mental status / acute metabolic encephalopathy",
                    "unconscious": "Loss of airway protective reflexes",
                    "sepsis": "Systemic inflammatory response syndrome",
                    "hemorrhage": "Acute cardiovascular volume depletion",
                    "bleeding": "Acute cardiovascular volume depletion"
                }
                for kw, flag in flags_map.items():
                    if kw in text_lower:
                        critical_flags.append(flag)
                
                if not critical_flags:
                    critical_flags.append("High clinical acuity triage indicators flagged by Bio_ClinicalBERT")
            
            explanation = (
                f"Triage category resolved to '{risk_level}' with {probs[pred_idx]*100:.2f}% model confidence. "
                f"Fine-tuned on NHAMCS Emergency Department data using Bio_ClinicalBERT. "
                f"Risk indicators prioritize safety and immediate critical care assessments."
            )
            
            output = RiskOutput(
                risk_level=risk_level,
                critical_flags=critical_flags,
                explanation=explanation
            )
        except Exception as e:
            error_msg = f"RiskAgent failed: {str(e)}"
            
        latency_ms = (time.time() - start_time) * 1000
        
        return AgentResponse(
            output=output,
            latency_ms=latency_ms,
            error=error_msg
        )
