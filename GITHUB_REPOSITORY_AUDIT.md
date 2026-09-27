# Repository Audit Report

**Date:** September 27, 2026
**Project:** HealthInsight Major Project

## 1. Files & Folders Included
The repository has been cleaned to include only the current working implementation. 
Included artifacts:
- `backend/` (FastAPI Server, Orchestrator, Agents, Configs, Utils)
- `src/`, `services/`, `index.html`, `index.css`, `App.tsx` (Vite + React Frontend)
- `tests/` (OCR and Specialist Integration tests)
- `main.py` (Evaluation runner)
- `requirements.txt`, `package.json`, `tsconfig.json`, `vite.config.ts`, `package-lock.json`
- `README.md`, `PROJECT_STRUCTURE.md`, `.gitignore`

## 2. Important Files Intentionally Excluded
To ensure a clean, professional repository suitable for mentor review, the following were aggressively excluded:
- `.git/` / `.venv/` / `node_modules/` / `__pycache__/` / `dist/`
- `archive_temp/`: Abandoned experiments, old audit files, and scratch scripts.
- `datasets/`, `data/`, `mimic-iv-*/`: Raw credentialed datasets, demo databases, and intermediate JSON splits to protect PII/DUA boundaries and reduce bloat.
- `docs/`, `note/`, `results/`: Outdated reports, training logs, and generated assets not reflective of the *current* software application.
- `WORKSPACE_CLEANUP_AUDIT.md`, `1 Information Fusion...pdf`, `SHA256SUMS.txt`: Not required to run the current system.

## 3. Large Files Excluded (Model Weights)
All ONNX models and PyTorch checkpoints have been excluded (`backend/models/*.onnx`, `*.data`, `*.pt`).

**Reason:** Total size exceeds standard Git/GitHub limits (many files are > 400MB).
**Retrieval:** The mentor can retrieve these directly from the external drive distribution or via the project's internal blob storage link. 

Required excluded files include:
- `cardiologist_model_quant.onnx` and `.data`
- `intent_model_single_quant.onnx` and `.data`
- `neurologist_final_quant.onnx` and `.data`
- `pulmonologist_final_quant.onnx` and `.data`

## 4. Security Checks Performed
- **Regex Audit:** A full workspace grep was executed targeting keywords: `OPENAI_API_KEY`, `password`, `secret`, `.env`, and `token`.
- **Result:** **PASS**. No API keys, database credentials, or secret tokens exist in the source code or configurations.
- **Environment Files:** `.env` files were successfully appended to `.gitignore`.

## 5. Build/Test Checks Performed
- Validated the presence of `pytest` scripts (`test_ocr.py`, `test_specialist_integration.py`).
- Validated that `server.py` correctly imports current existing modules (`Orchestrator`, `OCRExtractor`).
- Verified `package.json` contains standard Vite/React start scripts.

## 6. Remaining Uncertainties
- Since models are decoupled from the repo, users cloning the repo must manually construct the `backend/models/` folder. The system will throw a 503 error until the models are properly mounted. This behavior is documented in `README.md`.
