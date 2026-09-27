from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal

class RIEOutput(BaseModel):
    symptoms: List[str] = Field(default_factory=list, description="Extracted symptoms.")
    medical_history: List[str] = Field(default_factory=list, description="Extracted past medical history.")
    risk_factors: List[str] = Field(default_factory=list, description="Extracted risk factors.")
    key_findings: List[str] = Field(default_factory=list, description="Other key clinical findings.")
    confidence: float = Field(default=1.0, description="Extraction confidence score.")
    numerical_values: Dict[str, Any] = Field(default_factory=dict, description="Extracted numerical values.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata containing extraction source and confidence per entity.")
    stats: Dict[str, int] = Field(default_factory=dict, description="Extraction statistics.")

class IPMOutput(BaseModel):
    medical_domain: str = Field(..., description="E.g., Cardiology, Neurology, Pulmonology, General, etc.")
    urgency_level: Literal["low", "medium", "high", "critical"] = Field(..., description="Urgency of the case.")
    task_type: Literal["diagnosis", "treatment", "emergency", "general"] = Field(..., description="Action required.")
    extracted_entities: List[str] = Field(default_factory=list, description="Key extracted entities from intent fallback.")

class RACOutput(BaseModel):
    agreement_score: float = Field(..., description="Ratio of agreement.")
    conflict_penalty: float = Field(..., description="Penalty applied due to conflict.")
    adjusted_confidence_map: Dict[str, float] = Field(..., description="Adjusted confidence per agent.")

class PatientInput(BaseModel):
    patient_text: str = Field(..., description="The main clinical report or narrative.")
    history: str = Field(default="", description="Patient medical history and background.")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata like age, sex, etc.")

class IntentOutput(BaseModel):
    primary_intent: str = Field(..., description="The main intent of the clinical text.")
    secondary_intents: List[str] = Field(default_factory=list, description="Other intents detected.")
    extracted_entities: List[str] = Field(default_factory=list, description="Key extracted medical entities.")

class DiagnosisOutput(BaseModel):
    primary_diagnosis: str = Field(..., description="The single most likely primary diagnosis.")
    differential_diagnoses: List[str] = Field(..., description="List of possible differential diagnoses.")
    confidence: float = Field(..., description="Confidence score from 0.0 to 1.0.")

class TreatmentOutput(BaseModel):
    diagnosis: str = Field(..., description="The predicted disease condition.")
    risk_level: str = Field(..., description="The input clinical risk level (Low, Medium, High, Critical).")
    recommended_treatment: str = Field(..., description="Concrete recommended actions or steps to take.")
    medication_category: str = Field(..., description="Medication categories advised for first-line treatment.")
    lifestyle_recommendations: str = Field(..., description="Non-pharmacological lifestyle and self-care recommendations.")
    contraindications: str = Field(..., description="Contraindicated drugs, actions, or foods to avoid.")
    red_flag_symptoms: str = Field(..., description="Red flag warning signs for the condition.")
    emergency_referral_conditions: str = Field(..., description="Conditions requiring immediate emergency department referral.")
    specialist_referral: str = Field(..., description="Recommended specialist for follow-up.")
    follow_up_advice: str = Field(..., description="Recommended follow-up and monitoring interval.")
    evidence_source: str = Field(..., description="Evidence standard or source authority (e.g. WHO, NICE, CDC).")
    guideline_version: str = Field(..., description="Guideline version identifier.")
    confidence: float = Field(default=1.0, description="Guideline confidence score (1.0).")
    rationale: str = Field(..., description="Clinical justification for this recommendation.")

class RiskOutput(BaseModel):
    risk_level: str = Field(..., description="Criticality of the case (e.g., Low, Medium, High, Critical).")
    critical_flags: List[str] = Field(default_factory=list, description="Specific life-threatening or severe flags.")
    explanation: str = Field(..., description="Detailed explanation of the risk assessment.")

class SpecialistOutput(BaseModel):
    specialist_type: str = Field(..., description="Specialist domain: 'cardiologist', 'pulmonologist', or 'neurologist'")
    primary_diagnosis: str = Field(..., description="Top predicted disease from specialist multi-label head")
    differential_diagnoses: List[str] = Field(default_factory=list, description="Other detected or ranked differential diagnoses")
    disease_probabilities: Dict[str, float] = Field(default_factory=dict, description="Calibrated sigmoid probabilities for all specialty diseases")
    active_diagnoses: List[str] = Field(default_factory=list, description="Diseases meeting or exceeding the frozen decision threshold")
    predicted_complaint: str = Field(..., description="Predicted chief complaint from specialist triage head")
    predicted_urgency: str = Field(..., description="Predicted clinical urgency: 'Emergency', 'Urgent', or 'Routine'")
    confidence: float = Field(..., description="Confidence score for primary diagnosis")
    thresholds_applied: Dict[str, float] = Field(default_factory=dict, description="Authoritative frozen validation thresholds applied")

class SpecialistRoutingDecision(BaseModel):
    selected_specialists: List[str] = Field(default_factory=list, description="List of selected specialist agent names (e.g. 'cardiologist', 'pulmonologist', 'neurologist')")
    primary_domain: str = Field(..., description="Dominant medical domain identified")
    domain_scores: Dict[str, float] = Field(default_factory=dict, description="Relevance scores per clinical domain")
    routing_reasons: List[str] = Field(default_factory=list, description="Clinical justifications for the routing decision")
    is_multispecialist: bool = Field(default=False, description="True if multiple specialist domains were co-activated")

class FinalFusionOutput(BaseModel):
    final_diagnosis: str
    final_recommendation: List[str]
    final_risk_level: str
    overall_confidence: float
    active_specialists: Optional[List[str]] = Field(default_factory=list, description="Specialists that contributed to the final diagnosis")
    multispecialist_findings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Per-specialist clinical summary findings")

class AgentResponse(BaseModel):
    output: Any
    latency_ms: float
    error: Optional[str] = None
