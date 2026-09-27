import os
import time
import logging
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Optional
from transformers import AutoTokenizer

from backend.agents.base_agent import BaseAgent
from backend.models import PatientInput, AgentResponse, SpecialistOutput

logger = logging.getLogger("SpecialistAgents")


class BaseSpecialistAgent(BaseAgent):
    """
    Unified execution engine for frozen multi-task PubMedBERT clinical specialists.
    Processes clinical text and outputs:
    1. Multi-label disease probabilities with frozen thresholding
    2. Chief complaint triage prediction
    3. Clinical urgency risk level
    """
    def __init__(
        self,
        specialist_type: str,
        domain: str,
        onnx_model_file: str,
        disease_classes: List[str],
        complaint_classes: List[str],
        urgency_classes: List[str],
        frozen_thresholds: Dict[str, float],
        max_length: int = 384,
    ):
        super().__init__(name=f"{specialist_type.capitalize()}SpecialistAgent", response_model=SpecialistOutput, domain=domain)
        self.specialist_type = specialist_type
        self.disease_classes = disease_classes
        self.complaint_classes = complaint_classes
        self.urgency_classes = urgency_classes
        self.frozen_thresholds = frozen_thresholds
        self.max_length = max_length

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'models', onnx_model_file)
        
        # Fallback to unquantized FP32 if quant file does not exist
        if not os.path.exists(model_path) and onnx_model_file.endswith('_quant.onnx'):
            fp32_file = onnx_model_file.replace('_quant.onnx', '.onnx')
            alt_path = os.path.join(base_dir, 'models', fp32_file)
            if os.path.exists(alt_path):
                model_path = alt_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Specialist ONNX model not found at {model_path}")

        # Initialize ONNX inference session strictly on local CPU
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=['CPUExecutionProvider'])

        # Load local tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
                local_files_only=True
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
            )

        logger.info(f"Loaded {self.specialist_type} ONNX model ({os.path.basename(model_path)})")

    def _get_system_prompt(self) -> str:
        return f"Deterministic local PubMedBERT multi-task {self.specialist_type} specialist inference."

    async def run(self, input_data: PatientInput) -> AgentResponse:
        start_time = time.time()
        error_msg = None
        output = None

        try:
            # Combine clinical report with history
            patient_text = input_data.patient_text or ""
            history = input_data.history or ""
            combined_text = f"{patient_text} {history}".strip()
            if not combined_text:
                combined_text = "No clinical narrative provided."

            # 1. Tokenize input matching model sequence length
            inputs = self.tokenizer(
                combined_text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors="np"
            )
            input_ids = inputs["input_ids"].astype(np.int64)
            attention_mask = inputs["attention_mask"].astype(np.int64)

            # 2. Run ONNX Session
            outputs = self.session.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })
            # Outputs: [disease_logits, complaint_logits, urgency_logits]
            disease_logits = outputs[0][0]     # Shape [6]
            complaint_logits = outputs[1][0]   # Shape [11]
            urgency_logits = outputs[2][0]     # Shape [3]

            # 3. Multi-label sigmoid probabilities
            disease_probs = 1.0 / (1.0 + np.exp(-disease_logits))
            prob_dict = {
                cls_name: round(float(disease_probs[i]), 4)
                for i, cls_name in enumerate(self.disease_classes)
            }

            # 4. Apply frozen validation decision thresholds
            active_diagnoses = [
                cls_name for i, cls_name in enumerate(self.disease_classes)
                if disease_probs[i] >= self.frozen_thresholds.get(cls_name, 0.5)
            ]

            # 5. Determine primary diagnosis and differentials
            sorted_indices = np.argsort(disease_probs)[::-1]
            if active_diagnoses:
                # Primary is top disease among active diagnoses (or highest overall)
                active_indices = [i for i in sorted_indices if self.disease_classes[i] in active_diagnoses]
                primary_idx = active_indices[0] if active_indices else sorted_indices[0]
            else:
                primary_idx = sorted_indices[0]

            primary_diag = self.disease_classes[primary_idx]
            primary_conf = float(disease_probs[primary_idx])

            # Differential diagnoses: next highest candidates
            diff_indices = [i for i in sorted_indices if i != primary_idx]
            differentials = [self.disease_classes[i] for i in diff_indices[:3]]

            # 6. Predict chief complaint (argmax of 11 classes)
            pred_comp_idx = int(np.argmax(complaint_logits))
            if 0 <= pred_comp_idx < len(self.complaint_classes):
                pred_complaint = self.complaint_classes[pred_comp_idx]
            else:
                pred_complaint = self.complaint_classes[0]

            # 7. Predict clinical urgency (argmax of 3 classes)
            pred_urg_idx = int(np.argmax(urgency_logits))
            if 0 <= pred_urg_idx < len(self.urgency_classes):
                pred_urgency = self.urgency_classes[pred_urg_idx]
            else:
                pred_urgency = "Urgent"

            output = SpecialistOutput(
                specialist_type=self.specialist_type,
                primary_diagnosis=primary_diag,
                differential_diagnoses=differentials,
                disease_probabilities=prob_dict,
                active_diagnoses=active_diagnoses,
                predicted_complaint=pred_complaint,
                predicted_urgency=pred_urgency,
                confidence=round(primary_conf, 4),
                thresholds_applied=self.frozen_thresholds
            )

        except Exception as e:
            error_msg = f"{self.name} inference failed: {str(e)}"
            logger.exception(error_msg)

        latency_ms = (time.time() - start_time) * 1000

        return AgentResponse(
            output=output,
            latency_ms=latency_ms,
            error=error_msg
        )


class CardiologistSpecialistAgent(BaseSpecialistAgent):
    """Authoritative Frozen Cardiologist Specialist Agent."""
    DISEASE_CLASSES = [
        'angina',
        'arrhythmia',
        'atrial fibrillation',
        'coronary artery disease',
        'heart failure',
        'hypertension'
    ]
    COMPLAINT_CLASSES = [
        'cardiac assessment',
        'chest pain',
        'cough',
        'dyspnea',
        'edema',
        'fatigue',
        'headache',
        'hypertensive symptoms',
        'nausea',
        'palpitations',
        'syncope'
    ]
    URGENCY_CLASSES = ['Emergency', 'Urgent', 'Routine']
    FROZEN_THRESHOLDS = {
        'angina': 0.32,
        'arrhythmia': 0.44,
        'atrial fibrillation': 0.56,
        'coronary artery disease': 0.56,
        'heart failure': 0.56,
        'hypertension': 0.42
    }

    def __init__(self):
        super().__init__(
            specialist_type="cardiologist",
            domain="cardio",
            onnx_model_file="cardiologist_mimic_final_quant.onnx",
            disease_classes=self.DISEASE_CLASSES,
            complaint_classes=self.COMPLAINT_CLASSES,
            urgency_classes=self.URGENCY_CLASSES,
            frozen_thresholds=self.FROZEN_THRESHOLDS,
            max_length=384
        )


class PulmonologistSpecialistAgent(BaseSpecialistAgent):
    """Authoritative Frozen Pulmonologist Specialist Agent."""
    DISEASE_CLASSES = [
        'asthma',
        'copd',
        'pneumonia',
        'respiratory_failure',
        'pleural_effusion',
        'pulmonary_embolism'
    ]
    COMPLAINT_CLASSES = [
        'dyspnea',
        'cough',
        'wheezing',
        'chest pain',
        'fever',
        'hemoptysis',
        'sputum production',
        'fatigue',
        'altered mental status',
        'syncope',
        'respiratory assessment'
    ]
    URGENCY_CLASSES = ['Emergency', 'Urgent', 'Routine']
    FROZEN_THRESHOLDS = {
        'asthma': 0.38,
        'copd': 0.48,
        'pneumonia': 0.46,
        'respiratory_failure': 0.52,
        'pleural_effusion': 0.56,
        'pulmonary_embolism': 0.84
    }

    def __init__(self):
        super().__init__(
            specialist_type="pulmonologist",
            domain="pulmo",
            onnx_model_file="pulmonologist_final_quant.onnx",
            disease_classes=self.DISEASE_CLASSES,
            complaint_classes=self.COMPLAINT_CLASSES,
            urgency_classes=self.URGENCY_CLASSES,
            frozen_thresholds=self.FROZEN_THRESHOLDS,
            max_length=384
        )


class NeurologistSpecialistAgent(BaseSpecialistAgent):
    """Authoritative Frozen Neurologist Specialist Agent."""
    DISEASE_CLASSES = [
        'stroke_ischemic',
        'intracranial_hemorrhage',
        'epilepsy_seizures',
        'altered_mental_status_encephalopathy',
        'transient_ischemic_attack',
        'neuropathy_neurodegenerative'
    ]
    COMPLAINT_CLASSES = [
        'altered_mental_status',
        'confusion_memory',
        'dizziness_vertigo',
        'headache_migraine',
        'loss_of_consciousness_syncope',
        'numbness_paresthesia',
        'other_neurological',
        'seizure_convulsion',
        'speech_difficulty',
        'vision_disturbance',
        'weakness_paralysis'
    ]
    URGENCY_CLASSES = ['Emergency', 'Urgent', 'Routine']
    FROZEN_THRESHOLDS = {
        'stroke_ischemic': 0.48,
        'intracranial_hemorrhage': 0.66,
        'epilepsy_seizures': 0.60,
        'altered_mental_status_encephalopathy': 0.42,
        'transient_ischemic_attack': 0.46,
        'neuropathy_neurodegenerative': 0.42
    }

    def __init__(self):
        super().__init__(
            specialist_type="neurologist",
            domain="neuro",
            onnx_model_file="neurologist_final_quant.onnx",
            disease_classes=self.DISEASE_CLASSES,
            complaint_classes=self.COMPLAINT_CLASSES,
            urgency_classes=self.URGENCY_CLASSES,
            frozen_thresholds=self.FROZEN_THRESHOLDS,
            max_length=384
        )
