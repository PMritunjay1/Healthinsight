import os
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from backend.agents.base_agent import BaseAgent
from backend.models import PatientInput, AgentResponse, IPMOutput

class IntentAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="IntentAgent", response_model=IPMOutput)
        self.domain = "general"
        
        # Load local ONNX session and tokenizer
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', 'intent_model_single_quant.onnx')
        
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', local_files_only=True)
        
        # Load rule engine for clinical urgency and task type matching
        from backend.utils.rule_engine import RuleEngine
        self.rule_engine = RuleEngine()

    def _get_system_prompt(self) -> str:
        return "Deterministic local DistilBERT ONNX intent processing."

    async def run(self, input_data: PatientInput) -> AgentResponse:
        start_time = time.time()
        error_msg = None
        output = None
        
        try:
            patient_text = input_data.patient_text
            history = input_data.history
            combined_text = patient_text + " " + history
            
            # 1. Run Tokenizer with exact padding to match static ONNX graph size (512)
            inputs = self.tokenizer(patient_text, padding='max_length', truncation=True, max_length=512, return_tensors="np")
            input_ids = inputs["input_ids"].astype(np.int64)
            attention_mask = inputs["attention_mask"].astype(np.int64)
            
            # 2. Run ONNX Session for domain classification
            outputs = self.session.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })
            logits = outputs[0]
            pred_idx = int(np.argmax(logits, axis=1)[0])
            
            # Map index to standard domain specialty
            domains = ["general", "gastro", "neuro", "cardio", "pulmo"]
            predicted_domain = domains[pred_idx]
            
            # 3. Apply spelling normalization and clinical rules for urgency/task type
            normalized_text = self.rule_engine.apply_synonym_mapping(combined_text)
            
            urgency_level = self.rule_engine.map_urgency(normalized_text)
            task_type = self.rule_engine.map_task_type(predicted_domain, normalized_text)
            
            # Standard entity mapping
            extracted_entities = []
            
            output = IPMOutput(
                medical_domain=predicted_domain,
                urgency_level=urgency_level,
                task_type=task_type,
                extracted_entities=extracted_entities
            )
        except Exception as e:
            error_msg = f"IntentAgent failed: {str(e)}"
            
        latency_ms = (time.time() - start_time) * 1000
        
        return AgentResponse(
            output=output,
            latency_ms=latency_ms,
            error=error_msg
        )
