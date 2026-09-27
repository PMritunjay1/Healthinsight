# Final Dependency Audit Report

## CURRENT RUNTIME DEPENDENCY CHAIN
The runtime uses a strict, unidirectional data flow entirely disconnected from the historical datasets and evaluation scripts.

- **Frontend**: `index.html` -> `index.tsx` -> `App.tsx` -> `services/analysisService.ts` -> POST `/api/analyze`
- **FastAPI Backend**: `backend/server.py`
- **Orchestration**: `server.py` -> `backend/orchestrator.py`
- **OCR Pipeline**: `server.py` -> `backend/ocr_module.py`
- **Rules Extraction**: `orchestrator.py` -> `backend/rie_module.py` -> loads `backend/config/` (abbreviations, synonyms, negations)
- **Routing**: `orchestrator.py` -> `backend/specialist_router.py`
- **Agents**: `orchestrator.py` -> `backend/agents/intent_agent.py`, `specialist_agents.py`, `diagnosis_agent.py`, `risk_agent.py`, `treatment_agent.py`
- **Models**: `specialist_agents.py` -> directly loads ONNX from `backend/models/`
- **Consensus**: `orchestrator.py` -> `backend/rac_module.py`

## FILES RETAINED

filename/path | ACTION | reason
---|---|---
`README.md` | KEEP | Current documentation required for mentor.
`PROJECT_STRUCTURE.md` | KEEP | Current architecture documentation required for mentor.
`backend/server.py` | KEEP | FastAPI runtime entrypoint.
`backend/orchestrator.py` | KEEP | Core execution controller.
`backend/models.py` | KEEP | Pydantic data models used across all runtime scripts.
`backend/ocr_module.py` | KEEP | Runtime file parsing.
`backend/rie_module.py` | KEEP | Runtime extraction preprocessing.
`backend/rac_module.py` | KEEP | Runtime Reliability and Calibration logic.
`backend/specialist_router.py` | KEEP | Runtime logic to select specialists.
`backend/agents/*.py` | KEEP | Implementations for all 5 runtime agents.
`backend/utils/rule_engine.py` | KEEP | Required by `intent_agent.py`.
`backend/utils/treatment_engine.py` | KEEP | Required by `treatment_agent.py`.
`backend/config/*.json` (abbrev, synonym, intent, treatment, negation) | KEEP | Reference dictionaries loaded at runtime.
`App.tsx`, `index.*`, `services/`, `vite.config.ts`, `package.json`, `tsconfig.json`, `types.ts` | KEEP | React/Vite Frontend source code.
`requirements.txt` | KEEP | Production Python dependencies.
`tests/` | KEEP | Required unit/integration tests (`test_ocr.py`, `test_specialist_integration.py`).

## LEGACY / NOT USED BY CURRENT APPLICATION (Archived)

filename/path | ACTION | reason
---|---|---
`backend/datasets.py` | ARCHIVE | Only used for evaluations on benchmark subsets. Not used by server/orchestrator.
`backend/evaluation.py` | ARCHIVE | IEEE metrics computation pipeline. Not used at runtime.
`backend/graphs.py` | ARCHIVE | Generated plots from evaluation data.
`main.py` | ARCHIVE | Older evaluation execution entrypoint.
`backend/training/` | ARCHIVE | Training scripts for all agents. Not required for deployment.
`backend/utils/build_master_dataset.py`, `dataset_downloader.py`, `preprocess.py` | ARCHIVE | Dataset preparation scripts.
`backend/config/cardio_icd_mapping.json`, `clinical_demo_cases.json` | ARCHIVE | Historical test data not used by runtime.
`src/` | ARCHIVE | Complete source code for historical benchmarks, Mimic data ingestion, and model training.
`data/` | ARCHIVE | JSON subsets of test and validation cases.

## MODEL PACKAGE

filename/path | ACTION | reason
---|---|---
`cardiologist_mimic_final_quant.onnx` | KEEP | Required for deployment.
`pulmonologist_final_quant.onnx` | KEEP | Required for deployment.
`neurologist_final_quant.onnx` | KEEP | Required for deployment.
`intent_model_single_quant.onnx` | KEEP | Required for deployment.
`risk_model_Bio_ClinicalBERT_quant.onnx` | KEEP | Required for deployment.
`diagnosis_model_BiomedNLP-PubMedBERT-base-uncased-abstract_quant.onnx` | KEEP | Required for deployment.
`*.onnx` (FP32) | ARCHIVE | Unquantized intermediaries.
`*.onnx.data` | ARCHIVE | FP32 external data chunks. Not used by quantized models.
Other model variants | ARCHIVE | Older versions (e.g. `multitask`, `v3`).
