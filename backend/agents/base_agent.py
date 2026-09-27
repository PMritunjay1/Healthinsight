import time
from typing import Type
from pydantic import BaseModel
from backend.models import PatientInput, AgentResponse

class BaseAgent:
    """Base class for HealthInsight local diagnostic agents."""
    def __init__(self, name: str, response_model: Type[BaseModel], domain: str = "general", timeout: int = 15):
        self.name = name
        self.response_model = response_model
        self.domain = domain
        self.timeout = timeout

    async def run(self, input_data: PatientInput) -> AgentResponse:
        """Execute agent task. Must be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement run method.")
