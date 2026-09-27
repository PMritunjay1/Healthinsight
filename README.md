# HealthInsight

HealthInsight is an intelligent, agent-based clinical diagnosis support system. It processes unstructured patient text and historical clinical notes to predict primary diagnoses, identify clinical chief complaints, and assess urgency levels using locally executed neural models.

## Project Overview

The current implementation provides local/offline inference when the required model weights are supplied. By adopting an agentic collaboration model, HealthInsight parses input, extracts relevant medical symptoms and histories, and assigns the case to a specialized diagnostic model. Models are executed on CPU via ONNX Runtime.

## Current Architecture

The core processing pipeline involves a linear multi-agent workflow triggered via the FastAPI backend:

1. **OCR Module**: Extracts text from unstructured uploads (images, PDFs) using local parsing tools.
2. **RIE (Rule-Based Information Extraction)**: Cleans text and maps terms to standard medical concepts.
3. **Dynamic Specialist Routing**: An intent agent evaluates the clinical history to determine which domain specialist (e.g., Cardiologist, Pulmonologist, Neurologist) is best suited for the case.
4. **Local Specialist Model**: A domain-specific multi-task PubMedBERT model (executed via ONNX Runtime) processes the text to produce raw diagnosis, complaint, and urgency predictions.
5. **RAC (Reliability & Calibration) Module**: Calibrates the multi-task outputs using temperature scaling to ensure calibrated confidence scores.
6. **Consensus/Orchestration**: The orchestrator fuses the calibrated outputs and rules to construct the final patient response JSON.

## Repository Structure

The current codebase strictly reflects the production implementation:
- `backend/`: FastAPI application, orchestrator, agents, and pipeline modules (`ocr_module.py`, `rie_module.py`, `rac_module.py`, `specialist_router.py`).
- `src/`, `services/`: React/Vite frontend UI code (`App.tsx`, `index.tsx`, `vite.config.ts`).
- `tests/`: Integration and unit tests (`test_ocr.py`, `test_specialist_integration.py`).
- `requirements.txt`: Python backend dependencies.
- `package.json`: Node.js frontend dependencies.
- `main.py`: Entrypoint for running offline evaluations/benchmarks.

## Installation & Setup

### Prerequisites
- Python 3.10+
- Node.js 18+

### 1. Backend Setup
```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Frontend Setup
```bash
# Install Node dependencies
npm install
```

## Running the Application

### Start the Backend (FastAPI)
```bash
python backend/server.py
# The API will be available at http://127.0.0.1:8000
```

### Start the Frontend (React/Vite)
```bash
npm run dev
# The UI will be available on the localhost port provided by Vite.
```

## Running Tests

To run the backend tests (OCR and integration):
```bash
pytest tests/
```

## Trained Model Setup

The trained model weights are intentionally not included in GitHub because of their large size. The trained ONNX model package will be provided separately to the mentor.

The system expects the following exact folder structure and filenames because the current implementation loads these paths directly:

```text
HealthInsight/
└── backend/
    └── models/
        ├── cardiologist_mimic_final_quant.onnx
        ├── pulmonologist_final_quant.onnx
        ├── neurologist_final_quant.onnx
        ├── intent_model_single_quant.onnx
        ├── risk_model_Bio_ClinicalBERT_quant.onnx
        └── diagnosis_model_BiomedNLP-PubMedBERT-base-uncased-abstract_quant.onnx
```

**Instructions:**
1. Clone the GitHub repository.
2. Create the `backend/models/` directory if it does not already exist.
3. Copy the separately provided trained model files into that directory.
4. Keep each filename exactly as specified.
5. Install the required dependencies.
6. Start the backend/frontend using the documented commands.

The GitHub repository contains the implementation/code, while the separately supplied model package contains the trained weights required for full local inference.

| Model | Purpose | Location |
|---|---|---|
| Cardiologist | Local cardiology specialist | `backend/models/` |
| Pulmonologist | Local pulmonology specialist | `backend/models/` |
| Neurologist | Local neurology specialist | `backend/models/` |
| Intent model | Chief-complaint/intent classification | `backend/models/` |
| Risk model | Risk/urgency prediction | `backend/models/` |
| Diagnosis model | Diagnosis/fallback prediction | `backend/models/` |

## Reproducibility Limitations

- The implementation relies on local model files that must be downloaded separately. 
- Generalization is constrained to the clinical entities observed in the MIMIC-IV demo subset used for training the specialist models. Unseen complaints may yield low-confidence predictions.
- The pipeline represents the current local offline system; no external cloud dependencies or API keys are required.
