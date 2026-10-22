# HealthInsight-UQ: Uncertainty-Calibrated Multi-Agent Clinical Diagnostic Architecture

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.16%2B-informational.svg)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE.txt)

---

## 1. Project Overview and Research Objectives

**HealthInsight-UQ** is an offline, uncertainty-calibrated multi-agent clinical decision-support architecture designed to predict diagnostic impressions, chief complaints, and triage urgency from unstructured clinical encounter narratives. 

While monolithic Large Language Models (LLMs) demonstrate broad medical knowledge, their clinical safety is compromised by systemic failure modes:
1. **Factual Hallucination**: Generating plausible yet physiologically incorrect diagnoses.
2. **Uncalibrated Overconfidence**: Projecting extreme confidence even when encountering novel, out-of-distribution, or contradictory symptom patterns.
3. **Absence of Metacognitive Boundaries**: Inability to recognize when information is ambiguous, contradictory, or absent, preventing safe diagnostic abstention.

HealthInsight-UQ addresses these challenges through a modular architecture:
- **Localized Neural Specialists**: Domain-specific PubMedBERT multi-task encoders quantized to INT8 ONNX for fully offline execution on consumer laptop hardware (no third-party cloud API dependencies, ensuring patient privacy).
- **Event-Triggered Ambiguity Assessment**: Stratifying clinical narratives into three routing tiers (Low, Medium, High risk) using entropy and differential overlap.
- **Three-Signal Uncertainty Quantification**: Combining model confidence ($u^{\text{conf}}$), stochastic forward-pass consistency ($u^{\text{cons}}$), and inter-agent agreement entropy ($H_{\text{agr}}$).
- **Post-Hoc Probability Calibration**: Mapping fused uncertainty signals to true posterior accuracy via a validated isotonic/logistic calibrator fitted on a disjoint patient partition.
- **Distribution-Free Split-Conformal Abstention**: Enforcing finite-sample selective prediction coverage guarantees ($1 - \alpha$), systematically converting high-risk errors into physician referrals.
- **Bayesian Beta Trust Dynamics**: Updating specialist domain credibility via Beta posteriors $\theta_{i,d} \sim \mathrm{Beta}(\alpha_{i,d}, \beta_{i,d})$ from calibration feedback to prevent out-of-domain specialists from overriding legitimate consensus.

---

## 2. System Architecture & Pipeline Flow

The diagnostic pipeline processes patient data through a sequential, six-stage workflow structured in a two-tier inverted-C topology:

```mermaid
flowchart LR
    subgraph Row1 [" "]
        direction LR
        Input["Clinical Input<br>(Text / PDF / OCR)"] --> RIE["1. RIE Extraction<br>(Normalization & Entities)"] --> IPM["2. IPM Intent<br>(Domain & Urgency)"] --> Ambiguity["3. Ambiguity & Routing<br>(3-Tier Triage)"]
    end
    
    subgraph Row2 [" "]
        direction RL
        Output["Final Diagnosis JSON<br>(Calibrated Conf & Referral)"] <-- Consensus["6. Beta Trust Fusion<br>(Bayesian Dynamic Consensus)"] <-- RAC["5. RAC Calibration<br>(Uncertainty & Conformal)"] <-- Specialists["4. Specialist Encoders<br>(Cardio, Pulmo, Neuro)"]
    end

    Ambiguity --> Specialists

    style Row1 fill:none,stroke:none;
    style Row2 fill:none,stroke:none;
    classDef comp fill:#f8fafc,stroke:#334155,stroke-width:1.5px;
    classDef hl fill:#eff6ff,stroke:#2563eb,stroke-width:2px;
    class Input,Output comp;
    class RIE,IPM,Ambiguity,Specialists,RAC,Consensus hl;
```

### Pipeline Execution Stages:
1. **Report Interpretation and Extraction (RIE)** (`backend/rie_module.py`): Ingests raw narrative clinical notes or OCR-extracted text, deterministically normalizing abbreviations, resolving medical synonyms, and tagging negation patterns.
2. **Intent Processing Module (IPM)** (`backend/agents/intent_agent.py`): Ingests normalized entities to classify primary clinical intent, medical domain, and base triage urgency.
3. **Ambiguity Assessment & Specialist Routing** (`backend/specialist_router.py` & `backend/ambiguity_detector.py`): Evaluates symptom overlap and diagnostic entropy. Selects active domain specialists via the calibrated router (`calibrated_domain_router.joblib`).
4. **Specialist Multi-Task Encoders** (`backend/agents/specialist_agents.py`): Selected specialists (Cardiologist V2, Pulmonologist V1, Neurologist V1) run local INT8 ONNX inference to simultaneously predict Top-1 disease condition, chief complaint, urgency level, and stochastic pass consistency ($u^{\text{cons}}$).
5. **Response Agreement & Calibration (RAC)** (`backend/rac_module.py`): Computes cross-specialist agreement, applies post-hoc calibration (`calibrator_clean.pkl`), and checks split-conformal selective prediction thresholds ($\tau$).
6. **Bayesian Beta Trust Consensus & Fusion** (`backend/trust_tracker.py` & `backend/orchestrator.py`): Weights candidate specialist hypotheses by domain trust posteriors $\mathbb{E}[\theta_{i,d}]$, producing the final diagnostic response JSON or triggering clinical referral.

---

## 3. Repository Structure

```text
HealthInsight/
├── backend/                              # Core Python backend engine
│   ├── server.py                         # Production FastAPI application entry point
│   ├── orchestrator.py                   # Central pipeline controller and multi-agent fusion
│   ├── specialist_router.py              # Specialist routing logic and domain arbitration
│   ├── ambiguity_detector.py             # Three-tier ambiguity detection and risk scoring
│   ├── rac_module.py                     # Uncertainty quantification, calibration, and conformal gating
│   ├── rie_module.py                     # Deterministic rule-based extraction and negation tagging
│   ├── trust_tracker.py                  # Bayesian Beta trust store and credibility dynamics
│   ├── retrieval_grounding.py            # Diagnostic concept retrieval and ontology mapping
│   ├── ocr_module.py                     # Local PDF and medical document image OCR parser
│   ├── models.py                         # Pydantic schemas, data models, and I/O contracts
│   ├── agents/                           # Specialist and utility agent wrappers
│   │   ├── specialist_agents.py          # Cardiologist, Pulmonologist, Neurologist wrappers
│   │   ├── intent_agent.py               # Intent and domain classification agent
│   │   ├── risk_agent.py                 # Triage risk and clinical safety agent
│   │   ├── diagnosis_agent.py            # Fallback general diagnostic agent
│   │   ├── treatment_agent.py            # Evidence-based clinical recommendation agent
│   │   └── base_agent.py                 # Abstract agent base class
│   ├── config/                           # Deterministic clinical rule sets
│   │   ├── intent_rules.json             # Intent classification trigger keywords
│   │   ├── synonym_dictionary.json       # Clinical synonym resolution table
│   │   ├── abbreviation_dictionary.json  # Medical abbreviation expansion dictionary
│   │   ├── negation_patterns.json        # Regex patterns for clinical negation
│   │   └── treatment_rules.json          # Specialty-specific therapeutic protocols
│   ├── utils/                            # Core utility engines
│   │   ├── rule_engine.py                # Regex extraction and validation engine
│   │   └── treatment_engine.py           # Protocol matching engine
│   └── models/                           # Model configs, calibrations, and checksums
│       ├── SHA256SUMS_MODELS.txt         # Cryptographic SHA-256 hashes of all frozen ONNX models
│       ├── calibrator_clean.pkl          # Fitted post-hoc isotonic/logistic calibrator
│       ├── calibrated_domain_router.joblib # Trained TF-IDF domain routing model
│       ├── conformal_config.json         # Split-conformal nonconformity thresholds (tau)
│       ├── abstention_config.json        # Legacy abstention parameters
│       ├── calibration_manifest.json     # Split and fitting metadata for calibrator
│       ├── trust_manifest.json           # Calibration split census for Beta trust store
│       ├── trust_posteriors.json         # Fitted Bayesian Beta trust parameters (alpha, beta)
│       └── mlm_pubmedbert_outpatient_s30/ # PubMedBERT tokenizer vocab, configs, and merges
├── results/                              # Reconciled full-cohort evaluation summaries
│   ├── evaluation_summary.json           # Top-level reconciled full-cohort results summary
│   └── healthinsight_uq_full_cohort_20261018_173000/
│       ├── evaluation_summary.json       # Verifiable evaluation summary for holdout cohort
│       └── FROZEN_INTEGRITY_MANIFEST.json # Hash manifest verifying frozen holdout evaluation integrity
├── tests/                                # Comprehensive automated test suite (12 test modules)
│   ├── test_ambiguity_routing.py         # Tests 3-tier routing and edge-case handling
│   ├── test_beta_trust_dynamics.py       # Tests Bayesian Beta credibility updates
│   ├── test_calibration_fitting.py       # Tests post-hoc calibrator bounds and monotonicity
│   ├── test_conformal_abstention.py      # Tests split-conformal selective referral logic
│   ├── test_cross_specialty_isolation.py # Tests strict independence of specialist vocabularies
│   ├── test_label_key_mapping.py         # Tests schema alignment across clinical labels
│   ├── test_ledger_reconciliation.py     # Tests independent reconciliation of evaluation results
│   ├── test_mimic_metrics_unit.py        # Tests Top-1, Micro-F1, and Macro-F1 metric math
│   ├── test_retrieval_grounding.py       # Tests clinical ontology retrieval and grounding
│   ├── test_split_leakage_guardrails.py  # Tests four-way disjoint split patient isolation
│   ├── test_stochastic_consistency.py    # Tests Monte Carlo stochastic forward-pass reproducibility
│   ├── test_trust_weighted_fusion.py     # Tests consensus arbitration and conflict resolution
│   └── test_assets/                      # Synthetic fixture documents and test images
├── App.tsx                               # React frontend clinical dashboard
├── services/analysisService.ts           # Frontend HTTP client for FastAPI backend
├── requirements.txt                      # Python dependencies with pinned version ranges
├── package.json                          # Node.js frontend dependencies
├── pytest.ini                            # Pytest configuration and environment markers
├── LICENSE.txt                           # Open-source license terms
└── README.md                             # This repository guide
```

---

## 4. Software Requirements and Installation

### Prerequisites
- **Python**: 3.10 or 3.11 (64-bit)
- **Node.js**: v18.0.0 or higher (required only if running the optional React UI)
- **RAM**: Minimum 8 GB (16 GB recommended for full local model execution)
- **Disk**: 2 GB free disk space (excluding raw datasets)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PMritunjay1/Healthinsight.git
   cd Healthinsight
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install Node.js dependencies (optional, for frontend dashboard):**
   ```bash
   npm install
   ```

---

## 5. Execution Instructions

### Running the Backend API Server
The production entry point is `backend/server.py`. It initializes the Orchestrator, loads local ONNX models into memory, and exposes a high-throughput async REST API:

```bash
python backend/server.py
```
- **API URL**: `http://127.0.0.1:8000`
- **Swagger Documentation**: `http://127.0.0.1:8000/docs`
- **Primary Endpoint**: `POST /api/analyze`

**Sample Request:**
```bash
curl -X POST "http://127.0.0.1:8000/api/analyze" \
     -H "Content-Type: application/json" \
     -d '{"patient_text": "Patient presents with progressive exertional dyspnea, orthopnea, and bilateral lower extremity edema.", "history": "Prior history of hypertension."}'
```

### Running the Frontend UI (Optional)
To launch the interactive dashboard:
```bash
npm run dev
```
The interface opens at `http://localhost:5173`.

### Running Automated Test Suite
The repository includes a comprehensive 12-module test suite:
```bash
pytest tests/
```

To run a fast, non-GPU lightweight smoke test:
```bash
pytest tests/test_ambiguity_routing.py tests/test_calibration_fitting.py tests/test_mimic_metrics_unit.py
```

---

## 6. Model Artifacts and Verification

The neural specialist pipeline utilizes quantized INT8 ONNX models optimized for consumer CPU deployment. Because individual ONNX weight binaries exceed 100 MB, they are excluded from the Git tree per GitHub file-size limits and mounted locally under `backend/models/`.

### Expected Model Files
| Filename | Description | Expected Size | SHA-256 Checksum |
| :--- | :--- | :---: | :--- |
| `cardiologist_mimic_final_quant.onnx` | Cardiologist V2 (PubMedBERT) | 105.2 MB | `92a447bd33438ec05788b20f5ae5ede71e7e6f4472db7ce59254cb01e7851146` |
| `pulmonologist_final_quant.onnx` | Pulmonologist V1 (PubMedBERT) | 105.2 MB | `cf58019b495049a8baa1bf46b2679c7b7c718f252ca5dd81e4216d3266969e9b` |
| `neurologist_final_quant.onnx` | Neurologist V1 (PubMedBERT) | 105.2 MB | `92a437673b994063a6f7e10e6b702917b266d6bb124c2f2dfa68fb371130b1b5` |
| `intent_model_single_quant.onnx` | IPM Intent Classifier (DistilBERT) | 63.7 MB | `be7ed285d8bb2b707e9e1f1a775c6457d903b24a8d618fdba28fec80c8075db8` |
| `risk_model_Bio_ClinicalBERT_quant.onnx` | Clinical Risk & Triage Model | 103.4 MB | `80f88c5d113a72fe4e9e5433b87893f97288af462bf5dee915b906c562f47b30` |
| `diagnosis_model_BiomedNLP-PubMedBERT...quant.onnx` | General Diagnosis Fallback | 104.5 MB | `3160db268b0411dbc914ce359f030cebcc9508a31991647b98ab9e6b350ef6b0` |

All checksums can be verified locally against `backend/models/SHA256SUMS_MODELS.txt`:
```bash
# Verify checksums on Linux / macOS
cd backend/models && sha256sum -c SHA256SUMS_MODELS.txt

# Verify checksums on Windows (PowerShell)
Get-FileHash -Algorithm SHA256 backend/models/*.onnx
```

---

## 7. Full-Cohort Evaluation Results

All reported evaluation metrics were measured across an uncurated, frozen holdout cohort of **39,594 specialty-specific clinical records** extracted from MIMIC-IV under the **Four-Way Disjoint Split Discipline** (guaranteeing 0.00% patient and encounter overlap with training and calibration partitions).

> **Clarification on Cohort Unit**: The evaluated cohort consists of **39,594 specialty-specific clinical records**. Because clinical patients frequently present with multi-system presentations spanning multiple specialty admissions over time, the number of records represents distinct specialty-assigned encounters rather than a unique 1:1 patient count.

### Primary Diagnostic and Routing Metrics
| Metric | Count / Total | Empirical Value | 95% Confidence Interval |
| :--- | :---: | :---: | :---: |
| **Top-1 Unconditioned Diagnostic Accuracy** | **31,774 / 39,594** | **80.25%** | [79.85%, 80.64%] |
| **Domain-Routing Decision Accuracy** | **39,135 / 39,594** | **98.84%** | [98.73%, 98.94%] |
| **Split-Conformal Selective Accuracy** | Gated Subset | **82.24%** | Selective operating point ($\tau = 0.55$) |
| **Empirical System Coverage** | At $\tau = 0.55$ | **39.33%** | Distribution-free coverage guarantee |

### Specialty-Level Performance Breakdown
| Specialty Sub-Cohort | Correct Predictions / Evaluated Records | Diagnostic Accuracy | Routing Precision |
| :--- | :---: | :---: | :---: |
| **Cardiology** | **13,789 / 15,099** | **91.32%** | 99.93% |
| **Pulmonology** | **10,363 / 14,512** | **71.41%** | 96.97% |
| **Neurology** | **7,622 / 9,983** | **76.35%** | 99.89% |
| **Total Cohort** | **31,774 / 39,594** | **80.25%** | **98.84%** |

---

## 8. Verifiable Evaluation Summary File

The official, frozen summary metrics are included directly in this repository:
- **Primary Path**: [`results/evaluation_summary.json`](results/evaluation_summary.json)
- **Cohort Archive Path**: [`results/healthinsight_uq_full_cohort_20261018_173000/evaluation_summary.json`](results/healthinsight_uq_full_cohort_20261018_173000/evaluation_summary.json)
- **Integrity Manifest**: [`results/healthinsight_uq_full_cohort_20261018_173000/FROZEN_INTEGRITY_MANIFEST.json`](results/healthinsight_uq_full_cohort_20261018_173000/FROZEN_INTEGRITY_MANIFEST.json)

An independent reviewer or AI coding assistant can verify the metrics directly using Python:
```python
import json

with open("results/evaluation_summary.json", "r") as f:
    summary = json.load(f)

print(f"Total Cohort: {summary['total']:,}")
print(f"Overall Accuracy: {summary['unconditioned_accuracy'] * 100:.2f}%")
print(f"Routing Accuracy: {summary['routing_accuracy'] * 100:.2f}%")
for spec, metrics in summary["per_specialty"].items():
    print(f"  {spec}: {metrics['total']:,} records -> {metrics['accuracy'] * 100:.2f}%")
```

---

## 9. Research Limitations and Boundaries

1. **Research Prototype vs. Clinical Product**: HealthInsight-UQ is an academic research software prototype. It has not undergone clinical trials, prospective clinical validation, or regulatory clearance (e.g., FDA SaMD or CE mark). It is not designed or licensed for autonomous medical diagnosis or clinical care.
2. **Heuristic Certainty vs. Calibrated Probabilities**: The raw confidence score produced by specialist neural heads ($C(x)$) exhibits an Expected Calibration Error (ECE) of ~42.05% due to neural overconfidence. Only outputs processed through the post-hoc calibrator (`calibrator_clean.pkl`) provide statistically grounded posterior probabilities.
3. **Metric Availability and Schema Boundaries**:
   - *18-Class Full Simplex Brier Score*: Unavailable because evaluation pipelines preserved top-1 and top-$k$ outputs rather than the full 18-class probability simplex ($p \in \Delta^{17}$). Top-1 binary Brier score is 0.3747.
   - *Abstention Precision and Recall*: Undefined on MIMIC-IV because retrospective hospital datasets document historical clinical documentation rather than explicit physician referral decisions.
4. **Hardware and Latency Profile**: While the system runs efficiently on consumer laptop hardware (average latency 411.5 ms per case, resident footprint 4.1 GB RAM), performance depends on CPU core frequency and memory bandwidth.

---

## 10. Dataset Provenance, Ethics, and Data-Use Restrictions

### Data Source
This research utilizes the PhysioNet MIMIC-IV clinical database:
- **MIMIC-IV Hospital & ICU v3.1**
- **MIMIC-IV-Note v2.2**
- **MIMIC-IV-ED v2.2**

### Data Governance & License Restrictions
The MIMIC-IV clinical dataset is governed by the **PhysioNet Credentialed Health Data License (v1.5.0)** and protected health information regulations under HIPAA. 

Per the signed Data Use Agreement (DUA):
- **Row-level clinical text, raw discharge summaries, and encounter-level ledgers containing encounter identifiers (`hadm_id`) cannot be distributed in public GitHub repositories.**
- To respect patient privacy and DUA terms, this repository contains the verified non-sensitive evaluation summary (`results/evaluation_summary.json`) and codebase.

### Procedure for Accessing Data & Full Ledgers
Researchers wishing to reproduce the row-level evaluation must:
1. Complete the Human Subjects Training (CITI Program: *Data or Specimens Only Research*).
2. Register an account on [PhysioNet](https://physionet.org/).
3. Sign the MIMIC-IV Data Use Agreement and request access to project `mimiciv/3.1` and `mimic-iv-note/2.2`.
4. Once credentialed access is verified, the cohort extraction manifest (`datasets/metadata/four_way_split_manifest.json`) can be generated to recreate the exact 39,594 evaluation records.

### Citation
```bibtex
@article{healthinsight2026,
  title={HealthInsight-UQ: Uncertainty-Calibrated Multi-Agent Diagnostic Architecture for Clinical Decision Support},
  author={Mritunjay, P. and Collaborators},
  journal={Department of Computer Science and Engineering, IIIT Bhopal},
  year={2026}
}
```

---

## 11. License
This repository is licensed under the Apache License 2.0. See [LICENSE.txt](LICENSE.txt) for details.
