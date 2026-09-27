import os
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from backend.agents.base_agent import BaseAgent
from backend.models import PatientInput, AgentResponse, DiagnosisOutput

class DiagnosisAgent(BaseAgent):
    def __init__(self, domain: str = "general"):
        super().__init__(name="DiagnosisAgent", response_model=DiagnosisOutput, domain=domain)
        
        # Load local ONNX session and tokenizer
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', 'diagnosis_model_BiomedNLP-PubMedBERT-base-uncased-abstract_quant.onnx')
        
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract', local_files_only=True)
        
        # Exact Label List (alphabetically sorted matching LabelEncoder output)
        self.classes = [
            'Acne', 'Arthritis', 'Bronchial Asthma', 'Cervical spondylosis', 'Chicken pox', 
            'Common Cold', 'Dengue', 'Dimorphic Hemorrhoids', 'Fungal infection', 'Hypertension', 
            'Impetigo', 'Jaundice', 'Malaria', 'Migraine', 'Pneumonia', 'Psoriasis', 
            'Typhoid', 'Varicose Veins', 'allergy', 'diabetes', 'drug reaction', 
            'gastroesophageal reflux disease', 'peptic ulcer disease', 'urinary tract infection'
        ]
        
        self.temperature = 0.2646 # Optimal validation temperature scaling parameter

    def _get_system_prompt(self) -> str:
        return "Deterministic local PubMedBERT ONNX diagnosis inference."

    async def run(self, input_data: PatientInput) -> AgentResponse:
        start_time = time.time()
        error_msg = None
        output = None
        
        try:
            patient_text = input_data.patient_text
            
            # 1. Run Tokenizer with exact padding to match static ONNX graph size (256)
            inputs = self.tokenizer(patient_text, padding='max_length', truncation=True, max_length=256, return_tensors="np")
            input_ids = inputs["input_ids"].astype(np.int64)
            attention_mask = inputs["attention_mask"].astype(np.int64)
            
            # 2. Run ONNX Session
            outputs = self.session.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })
            logits = outputs[0] # Shape [1, 24]
            
            # 3. Apply Temperature Scaling Calibration
            calibrated_logits = logits / self.temperature
            
            # 4. Compute Softmax Probabilities
            exp_logits = np.exp(calibrated_logits - np.max(calibrated_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            probs = probs[0] # Flatten to 1D array
            
            # 5. Resolve predictions (Top-1 and differentials)
            sorted_indices = np.argsort(probs)[::-1]
            
            primary_idx = sorted_indices[0]
            primary_diag = self.classes[primary_idx]
            primary_conf = float(probs[primary_idx])
            
            # Differential diagnoses: top 2nd and 3rd choices
            differentials = [self.classes[idx] for idx in sorted_indices[1:4]]
            
            output = DiagnosisOutput(
                primary_diagnosis=primary_diag,
                differential_diagnoses=differentials,
                confidence=round(primary_conf, 4)
            )
        except Exception as e:
            error_msg = f"DiagnosisAgent failed: {str(e)}"
            
        latency_ms = (time.time() - start_time) * 1000
        
        return AgentResponse(
            output=output,
            latency_ms=latency_ms,
            error=error_msg
        )
