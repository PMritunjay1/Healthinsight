import time
from backend.agents.base_agent import BaseAgent
from backend.models import PatientInput, AgentResponse, TreatmentOutput
from backend.utils.treatment_engine import TreatmentEngine

class TreatmentAgent(BaseAgent):
    def __init__(self, domain: str = "general"):
        super().__init__(name="TreatmentAgent", response_model=TreatmentOutput, domain=domain)
        self.engine = TreatmentEngine()

    def _get_system_prompt(self) -> str:
        return "Deterministic clinical guideline rule lookup."

    async def run(self, input_data: PatientInput) -> AgentResponse:
        start_time = time.time()
        error_msg = None
        output = None
        
        try:
            # Retrieve predictions populated by the orchestrator from prior parallel passes
            predicted_diag = input_data.metadata.get("predicted_diagnosis", "Unknown")
            predicted_risk = input_data.metadata.get("predicted_risk_level", "Medium")
            
            # Run local rule engine lookup
            guidelines = self.engine.get_guidelines(predicted_diag, predicted_risk)
            
            # Parse into validated Pydantic model
            output = TreatmentOutput(**guidelines)
        except Exception as e:
            error_msg = f"TreatmentAgent rule lookup failed: {str(e)}"
            
        latency_ms = (time.time() - start_time) * 1000
        
        return AgentResponse(
            output=output,
            latency_ms=latency_ms,
            error=error_msg
        )
