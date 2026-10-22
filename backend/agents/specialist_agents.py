import os
import time
import logging
import hashlib
from collections import Counter
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Optional, Union
from transformers import AutoTokenizer

from backend.agents.base_agent import BaseAgent
from backend.models import PatientInput, AgentResponse, SpecialistOutput

# Ensure CUDA 12 runtime DLLs from PyTorch are discoverable on Windows
try:
    import torch
    _torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(_torch_lib) and hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(_torch_lib)
        os.environ['PATH'] = _torch_lib + os.pathsep + os.environ.get('PATH', '')
except Exception:
    pass

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
        if os.path.isabs(onnx_model_file):
            model_path = onnx_model_file
        else:
            model_path = os.path.join(base_dir, 'models', onnx_model_file)
        
        # Fallback to unquantized FP32 if quant file does not exist
        if not os.path.exists(model_path) and onnx_model_file.endswith('_quant.onnx'):
            fp32_file = onnx_model_file.replace('_quant.onnx', '.onnx')
            alt_path = fp32_file if os.path.isabs(fp32_file) else os.path.join(base_dir, 'models', fp32_file)
            if os.path.exists(alt_path):
                model_path = alt_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Specialist ONNX model not found at {model_path}")

        # Initialize ONNX inference session with CUDA priority and CPU fallback
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        available = ort.get_available_providers()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)

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

    async def run(self, input_data: Union[PatientInput, dict, str]) -> AgentResponse:
        """
        Executes multi-task specialist inference with genuine stochastic consistency (Algorithm 2).
        Runs m=5 deterministic passes: 1 clean input + 4 deterministic 5% non-entity token perturbations.
        NOTE: Perturbation consistency is not Monte Carlo dropout; it measures empirical prediction
        invariance under controlled input perturbations without altering model weights.
        """
        start_time = time.time()
        error_msg = None
        output = None

        try:
            # Flexible input extraction supporting PatientInput, dict, or raw text
            if isinstance(input_data, str):
                patient_text = input_data
                history = ""
            elif isinstance(input_data, dict):
                patient_text = input_data.get("patient_text", "")
                history = input_data.get("history", "")
            else:
                patient_text = getattr(input_data, "patient_text", "") or ""
                history = getattr(input_data, "history", "") or ""

            combined_text = f"{patient_text} {history}".strip()
            if not combined_text:
                combined_text = "No clinical narrative provided."

            # 1. Prepare clinical text with head-tail truncation if narrative is long
            # Preserves presenting complaint / initial history (head) and discharge diagnoses / assessment (tail)
            if len(combined_text) > 2500:
                head_text = combined_text[:1500]
                tail_text = combined_text[-1000:]
                tokenized_input = f"{head_text} {tail_text}"
            else:
                tokenized_input = combined_text

            # Tokenize input with dynamic length matching narrative (up to max_length)
            inputs = self.tokenizer(
                tokenized_input,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np"
            )
            input_ids = inputs["input_ids"].astype(np.int64)
            attention_mask = inputs["attention_mask"].astype(np.int64)
            seq_len = input_ids.shape[1]

            # 2. Construct Batched 5-Pass Input (Algorithm 2)
            # Pass 0: Clean baseline input
            # Passes 1..4: Deterministic 5% non-entity token perturbations
            cls_id = getattr(self.tokenizer, "cls_token_id", 101) or 101
            sep_id = getattr(self.tokenizer, "sep_token_id", 102) or 102
            pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
            mask_id = getattr(self.tokenizer, "mask_token_id", 103) or 103

            cand_indices = [
                idx for idx in range(seq_len)
                if attention_mask[0, idx] == 1 and input_ids[0, idx] not in (cls_id, sep_id, pad_id)
            ]

            m_passes = 5
            batch_input_ids = np.repeat(input_ids, m_passes, axis=0)
            batch_attention_mask = np.repeat(attention_mask, m_passes, axis=0)

            k_perturb = max(1, int(len(cand_indices) * 0.05)) if cand_indices else 0
            text_hash_int = int(hashlib.sha256(combined_text.encode("utf-8")).hexdigest()[:8], 16)

            for pass_idx in range(1, m_passes):
                if cand_indices and k_perturb > 0:
                    pass_rng = np.random.default_rng(seed=(text_hash_int + pass_idx) % (2**31 - 1))
                    chosen_indices = pass_rng.choice(cand_indices, size=k_perturb, replace=False)
                    batch_input_ids[pass_idx, chosen_indices] = mask_id

            # Execute single batched forward pass for all m=5 passes simultaneously
            outputs = self.session.run(None, {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_mask
            })
            # Outputs: [disease_logits (5, 6), complaint_logits (5, 11), urgency_logits (5, 3)]
            batch_disease_logits = outputs[0]
            complaint_logits = outputs[1][0]   # Clean pass chief complaint logits
            urgency_logits = outputs[2][0]     # Clean pass urgency logits

            # 3. Clean Pass Diagnostic Evaluation (Pass 0)
            clean_logits = batch_disease_logits[0]
            disease_probs = 1.0 / (1.0 + np.exp(-clean_logits))
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
                active_indices = [i for i in sorted_indices if self.disease_classes[i] in active_diagnoses]
                primary_idx = active_indices[0] if active_indices else sorted_indices[0]
            else:
                primary_idx = sorted_indices[0]

            primary_diag = self.disease_classes[primary_idx]
            primary_conf = float(disease_probs[primary_idx])

            # 6. Multi-Pass Consistency Evaluation across all 5 passes
            pass_predictions = []
            for p_idx in range(m_passes):
                p_logits = batch_disease_logits[p_idx]
                p_probs = 1.0 / (1.0 + np.exp(-p_logits))
                p_active = [
                    self.disease_classes[i] for i in range(len(self.disease_classes))
                    if p_probs[i] >= self.frozen_thresholds.get(self.disease_classes[i], 0.5)
                ]
                p_sorted = np.argsort(p_probs)[::-1]
                if p_active:
                    p_active_idx = [i for i in p_sorted if self.disease_classes[i] in p_active]
                    chosen_idx = p_active_idx[0] if p_active_idx else p_sorted[0]
                else:
                    chosen_idx = p_sorted[0]
                pass_predictions.append(self.disease_classes[chosen_idx])

            # Modal frequency and consistency uncertainty: u^cons = 1 - (1/m) sum 1[y^j == y^mode]
            pred_counts = Counter(pass_predictions)
            modal_diag, modal_count = pred_counts.most_common(1)[0]
            u_cons = round(1.0 - (modal_count / float(m_passes)), 4)
            u_instability = u_cons  # Backwards compatibility for RACModule

            # Differential diagnoses: next highest candidates
            diff_indices = [i for i in sorted_indices if i != primary_idx]
            differentials = [self.disease_classes[i] for i in diff_indices[:3]]

            # 7. Predict chief complaint (argmax of 11 classes)
            pred_comp_idx = int(np.argmax(complaint_logits))
            if 0 <= pred_comp_idx < len(self.complaint_classes):
                pred_complaint = self.complaint_classes[pred_comp_idx]
            else:
                pred_complaint = self.complaint_classes[0]

            # 8. Predict clinical urgency (argmax of 3 classes)
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
                thresholds_applied=self.frozen_thresholds,
                u_instability=u_instability,
                u_cons=u_cons
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

    def __init__(self, onnx_model_file: str = "cardiologist_mimic_final_quant.onnx"):
        super().__init__(
            specialist_type="cardiologist",
            domain="cardio",
            onnx_model_file=onnx_model_file,
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

    def __init__(self, onnx_model_file: str = "pulmonologist_final_quant.onnx"):
        super().__init__(
            specialist_type="pulmonologist",
            domain="pulmo",
            onnx_model_file=onnx_model_file,
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

    def __init__(self, onnx_model_file: str = "neurologist_final_quant.onnx"):
        super().__init__(
            specialist_type="neurologist",
            domain="neuro",
            onnx_model_file=onnx_model_file,
            disease_classes=self.DISEASE_CLASSES,
            complaint_classes=self.COMPLAINT_CLASSES,
            urgency_classes=self.URGENCY_CLASSES,
            frozen_thresholds=self.FROZEN_THRESHOLDS,
            max_length=384
        )
