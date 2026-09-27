/**
 * Health Insights: Multi-Specialist Clinical Diagnostic System Types
 * 100% Offline Local Architecture Data Contracts
 */

export const MedicalSpecialty = {
  Cardiologist: 'Cardiologist',
  Pulmonologist: 'Pulmonologist',
  Neurologist: 'Neurologist',
  Gastroenterologist: 'Gastroenterologist',
  General: 'General Diagnostic Specialist',
  Psychologist: 'Psychologist',
  Endocrinologist: 'Endocrinologist',
  Immunologist: 'Immunologist',
  Nephrologist: 'Nephrologist',
  Hematologist: 'Hematologist',
  Oncologist: 'Oncologist',
  Radiologist: 'Radiologist'
} as const;

export type MedicalSpecialty = (typeof MedicalSpecialty)[keyof typeof MedicalSpecialty];

export interface SpecialistCardData {
  id: string;
  name: string;
  specialty: MedicalSpecialty;
  domain: string;
  primaryDiagnosis: string;
  activeDiagnoses: string[];
  diseaseProbabilities: Record<string, number>;
  predictedComplaint: string;
  predictedUrgency: string;
  confidence: number;
  latencyMs: number;
}

export interface RiskData {
  riskLevel: 'Low' | 'Medium' | 'High' | string;
  confidence: number;
  criticalFlags: string[];
  explanation: string;
}

export interface TreatmentData {
  recommendedTreatment: string;
  medicationCategory: string;
  lifestyleRecommendations: string;
  contraindications: string;
  specialistReferral: string;
  followUpAdvice: string;
  evidenceSource: string;
  guidelineVersion: string;
}

export interface RACData {
  agreementScore: number;
  conflictPenalty: number;
  adjustedConfidenceMap: Record<string, number>;
}

export interface RoutingData {
  selectedSpecialists: string[];
  primaryDomain: string;
  domainScores: Record<string, number>;
  routingReasons: string[];
  isMultispecialist: boolean;
}

export interface DifferentialItem {
  condition: string;
  probability?: number;
}

export interface PipelineAnalysisResult {
  raw: any;
  symptoms: string[];
  riskFactors: string[];
  vitals: string[];
  routing: RoutingData;
  activeSpecialists: SpecialistCardData[];
  risk: RiskData;
  rac: RACData;
  treatment: TreatmentData;
  finalDiagnosis: string;
  clinicalExplanation: string;
  overallConfidence: number;
  differentials: DifferentialItem[];
  totalLatencyMs: number;
}

export interface DemoCase {
  id: string;
  label: string;
  category: 'Cardiovascular' | 'Respiratory' | 'Neurological' | 'Overlap' | 'General' | 'Negation';
  badgeColor: string;
  symptoms: string;
  history: string;
  expectedRouting: string;
  description: string;
}

export interface PatientHistory {
  pastDiagnoses: string;
  chronicConditions: string;
  allergies: string;
  currentMedications: string;
  familyHistory: string;
  lifestyleFactors: string;
}

// Backward-compatible legacy interfaces for any existing consumers
export interface StructuredCase {
  patientSummary: string;
  symptoms: string[];
  vitalSigns: string[];
  labResults: string[];
  imagingFindings: string[];
  riskFactors: string[];
}

export interface MedicalIntent {
  primaryIntent: string;
  secondaryIntents: string[];
}

export interface SpecialistAnalysis {
  summary: string;
  keyFindings: string[];
  potentialConditions: string[];
  recommendations: string[];
  confidenceScore: number;
}

export interface SpecialistReport {
  specialty: MedicalSpecialty;
  analysis: SpecialistAnalysis | null;
  status: 'pending' | 'loading' | 'complete' | 'error';
}

export interface FinalDiagnosisDetails {
  final_diagnosis: string;
  clinical_explanation: string;
  criticalIssues?: string[];
  recommendedActions?: string[];
  differentials: string[];
  confidence: number;
  latency_ms?: number;
}

export interface FinalReport {
  summary: string;
  details?: FinalDiagnosisDetails;
  confidence: number;
  status: 'pending' | 'loading' | 'complete' | 'error';
}

export interface RefinedReport {
  resolutions: string[];
  unifiedInsights: string[];
}

export interface PipelineResult {
  structuredCase: StructuredCase;
  intent: MedicalIntent;
  preDiagnosis: any;
  specialistReports: Record<string, SpecialistAnalysis>;
  refinedReports: RefinedReport;
  finalDiagnosis: FinalDiagnosisDetails;
  executionTime?: number;
}