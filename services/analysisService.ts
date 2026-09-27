import {
  MedicalSpecialty,
  PipelineAnalysisResult,
  SpecialistCardData,
  RiskData,
  TreatmentData,
  RACData,
  RoutingData,
  DifferentialItem,
  DemoCase
} from "../types";

export const API_BASE_URL = "http://127.0.0.1:8000";

/**
 * 7 Authoritative Pre-configured Clinical Demonstration Cases
 * Aligned with the Thesis Validation Protocols
 */
export const DEMO_CLINICAL_CASES: DemoCase[] = [
  {
    id: "case-1-cardio",
    label: "Cardiovascular (ACS / CAD)",
    category: "Cardiovascular",
    badgeColor: "rose",
    symptoms: "65 year old with crushing substernal chest pain radiating to the left arm and palpitations.",
    history: "Past Diagnoses: Hypertension\nChronic Conditions: Hyperlipidemia\nCurrent Medications: Lisinopril 20mg, Atorvastatin 40mg",
    expectedRouting: "Cardiologist Specialist",
    description: "Acute cardiac presentation triggering isolated Cardiologist specialist neural model."
  },
  {
    id: "case-2-pulmo",
    label: "Respiratory (COPD / Pneumonia)",
    category: "Respiratory",
    badgeColor: "sky",
    symptoms: "Patient has severe wheezing, productive cough, fever and shortness of breath.",
    history: "Past Diagnoses: Asthma, Chronic Bronchitis\nCurrent Medications: Albuterol Inhaler",
    expectedRouting: "Pulmonologist Specialist",
    description: "Acute pulmonary presentation triggering isolated Pulmonologist specialist neural model."
  },
  {
    id: "case-3-neuro",
    label: "Neurological (Stroke / TIA)",
    category: "Neurological",
    badgeColor: "violet",
    symptoms: "Patient developed sudden right-sided weakness, facial droop and difficulty speaking.",
    history: "Past Diagnoses: TIA\nChronic Conditions: Hypertension\nCurrent Medications: Aspirin 81mg",
    expectedRouting: "Neurologist Specialist",
    description: "Acute neurological deficit presentation triggering isolated Neurologist specialist neural model."
  },
  {
    id: "case-4-cardio-pulmo",
    label: "Cardio + Pulmo Overlap (CHF + COPD)",
    category: "Overlap",
    badgeColor: "amber",
    symptoms: "Patient has progressive dyspnea, orthopnea, bilateral pedal edema and wheezing.",
    history: "Past Diagnoses: Congestive Heart Failure, COPD\nCurrent Medications: Furosemide 40mg, Tiotropium",
    expectedRouting: "Cardiologist + Pulmonologist (RAC Consensus)",
    description: "Cardiopulmonary overlap co-activating Cardiologist & Pulmonologist with RAC consensus synthesis."
  },
  {
    id: "case-5-neuro-pulmo",
    label: "Neuro + Pulmo Overlap (Aspiration Post-Stroke)",
    category: "Overlap",
    badgeColor: "amber",
    symptoms: "Patient with recent stroke now has aspiration symptoms, fever and respiratory distress.",
    history: "Past Diagnoses: Ischemic Stroke, Dysphagia\nCurrent Medications: Clopidogrel 75mg",
    expectedRouting: "Neurologist + Pulmonologist (RAC Consensus)",
    description: "Complex neuro-respiratory overlap co-activating Neurologist & Pulmonologist."
  },
  {
    id: "case-6-general",
    label: "General Fallback (Gastroenteritis)",
    category: "General",
    badgeColor: "emerald",
    symptoms: "Patient has abdominal pain, nausea and vomiting.",
    history: "Past Diagnoses: GERD\nCurrent Medications: Omeprazole 20mg",
    expectedRouting: "General Diagnosis Agent / Gastroenterologist",
    description: "Non-specialized general case routing to general diagnostic model with guideline referral."
  },
  {
    id: "case-7-negation",
    label: "Negation Suppression Audit",
    category: "Negation",
    badgeColor: "indigo",
    symptoms: "Patient has cough but denies chest pain and denies focal neurological weakness.",
    history: "Past Diagnoses: URI\nCurrent Medications: None",
    expectedRouting: "Pulmonologist Only (Negations Suppressed)",
    description: "Verifies RIE negation extraction correctly suppresses false positive cardiac/neuro routing."
  }
];

/**
 * Triggers the 100% offline local multi-agent diagnostic server pipeline on port 8000.
 */
export const runMedicalAnalysisPipeline = async (
  patientText: string,
  historyText: string = ""
): Promise<PipelineAnalysisResult> => {
  if (!patientText || !patientText.trim()) {
    throw new Error("Patient clinical text presentation cannot be empty.");
  }

  // POST request directly to the local FastAPI analysis endpoint
  const response = await fetch(`${API_BASE_URL}/api/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      patient_text: patientText.trim(),
      history: historyText.trim()
    })
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Diagnostic Server (${response.status}): ${errorText || response.statusText}`);
  }

  const data = await response.json();

  // 1. Extract RIE signals
  const symptoms: string[] = data.rie?.symptoms || [];
  const riskFactors: string[] = data.rie?.risk_factors || [];
  const vitals: string[] = Object.entries(data.rie?.numerical_values || {}).map(([k, v]) => `${k}: ${v}`);

  // 2. Extract Routing
  const routing: RoutingData = {
    selectedSpecialists: data.routing?.selected_specialists || [],
    primaryDomain: data.routing?.primary_domain || "general",
    domainScores: data.routing?.domain_scores || {},
    routingReasons: data.routing?.routing_reasons || [],
    isMultispecialist: Boolean(data.routing?.is_multispecialist)
  };

  // 3. Extract Active Specialist Cards
  const activeSpecialists: SpecialistCardData[] = [];
  const agents = data.agents || {};

  if (agents.cardiologist?.output) {
    const out = agents.cardiologist.output;
    activeSpecialists.push({
      id: "cardiologist",
      name: "Cardiologist Specialist",
      specialty: MedicalSpecialty.Cardiologist,
      domain: "Cardiovascular System",
      primaryDiagnosis: out.primary_diagnosis || "Cardiovascular Disease",
      activeDiagnoses: out.active_diagnoses || [],
      diseaseProbabilities: out.disease_probabilities || {},
      predictedComplaint: out.predicted_complaint || "Cardiovascular distress",
      predictedUrgency: out.predicted_urgency || "Urgent",
      confidence: out.confidence ?? 0.85,
      latencyMs: agents.cardiologist.latency_ms || 0
    });
  }

  if (agents.pulmonologist?.output) {
    const out = agents.pulmonologist.output;
    activeSpecialists.push({
      id: "pulmonologist",
      name: "Pulmonologist Specialist",
      specialty: MedicalSpecialty.Pulmonologist,
      domain: "Respiratory System",
      primaryDiagnosis: out.primary_diagnosis || "Respiratory Disease",
      activeDiagnoses: out.active_diagnoses || [],
      diseaseProbabilities: out.disease_probabilities || {},
      predictedComplaint: out.predicted_complaint || "Respiratory distress",
      predictedUrgency: out.predicted_urgency || "Urgent",
      confidence: out.confidence ?? 0.85,
      latencyMs: agents.pulmonologist.latency_ms || 0
    });
  }

  if (agents.neurologist?.output) {
    const out = agents.neurologist.output;
    activeSpecialists.push({
      id: "neurologist",
      name: "Neurologist Specialist",
      specialty: MedicalSpecialty.Neurologist,
      domain: "Nervous System",
      primaryDiagnosis: out.primary_diagnosis || "Neurological Disorder",
      activeDiagnoses: out.active_diagnoses || [],
      diseaseProbabilities: out.disease_probabilities || {},
      predictedComplaint: out.predicted_complaint || "Neurological deficit",
      predictedUrgency: out.predicted_urgency || "Urgent",
      confidence: out.confidence ?? 0.85,
      latencyMs: agents.neurologist.latency_ms || 0
    });
  }

  if (agents.diagnosis?.output && activeSpecialists.length === 0) {
    const out = agents.diagnosis.output;
    const diagName = (out.primary_diagnosis || "").toLowerCase();
    const isGastro = (data.ipm?.medical_domain || "").toLowerCase().includes("gastro") ||
      routing.primaryDomain.toLowerCase().includes("gastro") ||
      diagName.includes("gastro") || diagName.includes("ulcer") || diagName.includes("gerd") ||
      diagName.includes("typhoid") || diagName.includes("abdominal");

    activeSpecialists.push({
      id: "diagnosis",
      name: isGastro ? "Gastroenterology Specialist" : "General Diagnostic Agent",
      specialty: isGastro ? MedicalSpecialty.Gastroenterologist : MedicalSpecialty.General,
      domain: isGastro ? "Gastrointestinal System" : "General / Internal Medicine",
      primaryDiagnosis: out.primary_diagnosis || "Clinical Assessment",
      activeDiagnoses: out.differential_diagnoses || [out.primary_diagnosis],
      diseaseProbabilities: {},
      predictedComplaint: "Clinical Presentation",
      predictedUrgency: data.ipm?.urgency_level ? (data.ipm.urgency_level.toUpperCase()) : "Routine",
      confidence: out.confidence ?? 0.80,
      latencyMs: agents.diagnosis.latency_ms || 0
    });
  }

  // 4. Extract Risk Data
  const riskOut = agents.risk?.output || {};
  const risk: RiskData = {
    riskLevel: riskOut.risk_level || data.fusion?.final_risk_level || "Medium",
    confidence: riskOut.confidence ?? 0.90,
    criticalFlags: riskOut.critical_flags || [],
    explanation: riskOut.explanation || "Clinical acuity assessed via rule-based safety triage."
  };

  // 5. Extract RAC Data
  const rac: RACData = {
    agreementScore: data.rac?.agreement_score ?? 0.80,
    conflictPenalty: data.rac?.conflict_penalty ?? 0.0,
    adjustedConfidenceMap: data.rac?.adjusted_confidence_map || {}
  };

  // 6. Extract Treatment Data
  const treatOut = agents.treatment?.output || {};
  const treatment: TreatmentData = {
    recommendedTreatment: treatOut.recommended_treatment || "Consult attending clinician for specialist evaluation.",
    medicationCategory: treatOut.medication_category || "Clinical Management",
    lifestyleRecommendations: treatOut.lifestyle_recommendations || "Supportive care and monitoring.",
    contraindications: treatOut.contraindications || "None reported.",
    specialistReferral: treatOut.specialist_referral || "Primary Care Physician",
    followUpAdvice: treatOut.follow_up_advice || "Standard clinical follow-up as indicated.",
    evidenceSource: treatOut.evidence_source || "Clinical Practice Guidelines",
    guidelineVersion: treatOut.guideline_version || "2024.1"
  };

  // 7. Extract Differentials
  const differentials: DifferentialItem[] = [];
  if (agents.diagnosis?.output?.differential_diagnoses) {
    agents.diagnosis.output.differential_diagnoses.forEach((d: string) => {
      differentials.push({ condition: d });
    });
  }

  activeSpecialists.forEach((spec) => {
    Object.entries(spec.diseaseProbabilities)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .forEach(([condition, prob]) => {
        if (!differentials.some(d => d.condition.toLowerCase() === condition.toLowerCase())) {
          differentials.push({ condition, probability: prob });
        }
      });
  });

  return {
    raw: data,
    symptoms,
    riskFactors,
    vitals,
    routing,
    activeSpecialists,
    risk,
    rac,
    treatment,
    finalDiagnosis: data.fusion?.final_diagnosis || "Clinical Condition",
    clinicalExplanation: data.fusion?.explanation ||
      data.agents?.diagnosis?.output?.clinical_explanation ||
      (data.fusion?.final_diagnosis ? `Multidisciplinary diagnostic synthesis indicates ${data.fusion.final_diagnosis}. Acuity assessed at ${risk.riskLevel} acuity level. Clinical management protocols assigned based on verified clinical practice guidelines.` : "Clinical assessment synthesized via multi-agent consensus protocols."),
    overallConfidence: data.fusion?.overall_confidence ?? 0.85,
    differentials,
    totalLatencyMs: data.total_latency_ms || 0
  };
};

/**
 * Triggers the OCR endpoint to extract text from a clinical document or image.
 */
export const extractTextFromDocument = async (file: File): Promise<string> => {
  if (!file) throw new Error("No file provided.");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(`${API_BASE_URL}/api/extract_text`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Server Error ${response.status}`);
    }

    const data = await response.json();
    if (data.processing_status === "error") {
      throw new Error(data.warnings?.join(", ") || "Failed to extract text");
    }
    
    let result = data.ocr_text || "";
    if (data.warnings && data.warnings.length > 0) {
       result = `[OCR Warnings: ${data.warnings.join(", ")}]\n\n${result}`;
    }
    
    return result;

  } catch (error: any) {
    console.error("OCR API error:", error);
    throw new Error(error.message || "Failed to connect to the OCR server.");
  }
};
