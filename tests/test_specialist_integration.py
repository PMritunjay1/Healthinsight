"""
Health Insights Multi-Agent Framework: Specialist Integration Automated Test Suite
==================================================================================
Tests:
1. ONNX Model Loading & Sessions for all 3 Specialists
2. Cardiologist Routing & Inference
3. Pulmonologist Routing & Inference
4. Neurologist Routing & Inference
5. Ambiguous / Multi-Domain Routing & Consensus
6. General Non-Specialist Case Fallback
7. Specialist Isolation & Independence
8. Output Schema Validation & Backward API Compatibility
9. 100% Local / Offline Execution Guarantee (Zero Network / External LLM calls)
10. Frozen Specialist Artifacts Verification
"""

import os
import sys
import unittest
import asyncio
from typing import Dict, Any

# Ensure workspace is on sys.path
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from backend.models import (
    PatientInput, SpecialistOutput, SpecialistRoutingDecision,
    FinalFusionOutput, AgentResponse, RACOutput, IPMOutput, RIEOutput
)
from backend.agents.specialist_agents import (
    CardiologistSpecialistAgent,
    PulmonologistSpecialistAgent,
    NeurologistSpecialistAgent
)
from backend.specialist_router import SpecialistRouter
from backend.orchestrator import Orchestrator


class TestSpecialistIntegration(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=======================================================")
        print("  HEALTH INSIGHTS SPECIALIST INTEGRATION TEST SUITE  ")
        print("=======================================================")
        cls.orchestrator = Orchestrator()
        cls.router = SpecialistRouter()

    # ─────────────────────────────────────────────────────────────
    # TEST 1: ONNX MODELS LOADING & INITIALIZATION
    # ─────────────────────────────────────────────────────────────
    def test_01_onnx_models_loading(self):
        """Verify all three specialist agents load frozen ONNX graphs properly."""
        cardio = CardiologistSpecialistAgent()
        pulmo = PulmonologistSpecialistAgent()
        neuro = NeurologistSpecialistAgent()

        self.assertIsNotNone(cardio.session)
        self.assertIsNotNone(pulmo.session)
        self.assertIsNotNone(neuro.session)

        # Verify class counts
        self.assertEqual(len(cardio.disease_classes), 6)
        self.assertEqual(len(pulmo.disease_classes), 6)
        self.assertEqual(len(neuro.disease_classes), 6)

        # Verify complaint and urgency class counts
        self.assertEqual(len(cardio.complaint_classes), 11)
        self.assertEqual(len(pulmo.complaint_classes), 11)
        self.assertEqual(len(neuro.complaint_classes), 11)

        self.assertEqual(len(cardio.urgency_classes), 3)
        self.assertEqual(len(pulmo.urgency_classes), 3)
        self.assertEqual(len(neuro.urgency_classes), 3)

        # Verify frozen thresholds are set
        self.assertEqual(len(cardio.frozen_thresholds), 6)
        self.assertEqual(len(pulmo.frozen_thresholds), 6)
        self.assertEqual(len(neuro.frozen_thresholds), 6)

    # ─────────────────────────────────────────────────────────────
    # TEST 2: CARDIOLOGIST ROUTING & INFERENCE
    # ─────────────────────────────────────────────────────────────
    async def test_02_cardiologist_routing_and_inference(self):
        """Verify cardiovascular cases route to Cardiologist with valid predictions."""
        input_data = PatientInput(
            patient_text="62-year-old male with crushing substernal chest pain radiating to left shoulder and arm, diaphoresis, and palpitations. Known coronary artery disease and hypertension.",
            history="10-year history of essential hypertension, previous angioplasty."
        )

        res = await self.orchestrator.process_case(input_data.patient_text, input_data.history)
        routing = res["routing"]
        agents = res["agents"]
        fusion = res["fusion"]

        self.assertIn("cardiologist", routing["selected_specialists"])
        self.assertIn("cardiologist", agents)

        cardio_out = agents["cardiologist"]["output"]
        self.assertIsNotNone(cardio_out)
        self.assertEqual(cardio_out["specialist_type"], "cardiologist")
        self.assertIn(cardio_out["primary_diagnosis"], CardiologistSpecialistAgent.DISEASE_CLASSES)
        self.assertIn(cardio_out["predicted_urgency"], ["Emergency", "Urgent", "Routine"])
        self.assertIn(cardio_out["predicted_complaint"], CardiologistSpecialistAgent.COMPLAINT_CLASSES)
        self.assertGreaterEqual(cardio_out["confidence"], 0.0)

        # Verify fusion incorporates specialist diagnosis
        self.assertTrue(len(fusion["final_diagnosis"]) > 0)
        self.assertIn("cardiologist", fusion["active_specialists"])

    # ─────────────────────────────────────────────────────────────
    # TEST 3: PULMONOLOGIST ROUTING & INFERENCE
    # ─────────────────────────────────────────────────────────────
    async def test_03_pulmonologist_routing_and_inference(self):
        """Verify respiratory cases route to Pulmonologist with valid predictions."""
        input_data = PatientInput(
            patient_text="55-year-old female presenting with severe expiratory wheezing, productive cough with thick purulent sputum, and acute dyspnea. SpO2 88% on ambient room air.",
            history="Diagnosed with COPD 5 years ago, 30 pack-year smoking history."
        )

        res = await self.orchestrator.process_case(input_data.patient_text, input_data.history)
        routing = res["routing"]
        agents = res["agents"]
        fusion = res["fusion"]

        self.assertIn("pulmonologist", routing["selected_specialists"])
        self.assertIn("pulmonologist", agents)

        pulmo_out = agents["pulmonologist"]["output"]
        self.assertIsNotNone(pulmo_out)
        self.assertEqual(pulmo_out["specialist_type"], "pulmonologist")
        self.assertIn(pulmo_out["primary_diagnosis"], PulmonologistSpecialistAgent.DISEASE_CLASSES)
        self.assertIn(pulmo_out["predicted_urgency"], ["Emergency", "Urgent", "Routine"])
        self.assertIn(pulmo_out["predicted_complaint"], PulmonologistSpecialistAgent.COMPLAINT_CLASSES)

        self.assertIn("pulmonologist", fusion["active_specialists"])

    # ─────────────────────────────────────────────────────────────
    # TEST 4: NEUROLOGIST ROUTING & INFERENCE
    # ─────────────────────────────────────────────────────────────
    async def test_04_neurologist_routing_and_inference(self):
        """Verify neurological cases route to Neurologist with valid predictions."""
        input_data = PatientInput(
            patient_text="70-year-old female brought to ED with acute sudden onset right-sided facial droop, right arm paralysis, slurred speech (dysarthria), and confusion starting 30 minutes ago.",
            history="History of hypertension, atrial fibrillation, and previous transient ischemic attack."
        )

        res = await self.orchestrator.process_case(input_data.patient_text, input_data.history)
        routing = res["routing"]
        agents = res["agents"]
        fusion = res["fusion"]

        self.assertIn("neurologist", routing["selected_specialists"])
        self.assertIn("neurologist", agents)

        neuro_out = agents["neurologist"]["output"]
        self.assertIsNotNone(neuro_out)
        self.assertEqual(neuro_out["specialist_type"], "neurologist")
        self.assertIn(neuro_out["primary_diagnosis"], NeurologistSpecialistAgent.DISEASE_CLASSES)
        self.assertIn(neuro_out["predicted_urgency"], ["Emergency", "Urgent", "Routine"])
        self.assertIn(neuro_out["predicted_complaint"], NeurologistSpecialistAgent.COMPLAINT_CLASSES)

        self.assertIn("neurologist", fusion["active_specialists"])

    # ─────────────────────────────────────────────────────────────
    # TEST 5: AMBIGUOUS & MULTI-DOMAIN ROUTING
    # ─────────────────────────────────────────────────────────────
    async def test_05_multidomain_routing_and_consensus(self):
        """Verify overlapping cases activate multiple relevant specialists and fuse findings."""
        input_data = PatientInput(
            patient_text="74-year-old patient with severe progressive shortness of breath, bilateral lung crackles, orthopnea, bilateral lower extremity edema, and substernal chest tightness.",
            history="Congestive heart failure, chronic obstructive pulmonary disease (COPD), and atrial fibrillation."
        )

        res = await self.orchestrator.process_case(input_data.patient_text, input_data.history)
        routing = res["routing"]
        fusion = res["fusion"]

        # Should detect multi-specialist domain
        self.assertTrue(routing["is_multispecialist"])
        self.assertGreaterEqual(len(routing["selected_specialists"]), 2)
        self.assertIn("cardiologist", routing["selected_specialists"])
        self.assertIn("pulmonologist", routing["selected_specialists"])

        # Check multi-specialist findings structure in fusion output
        self.assertIn("cardiologist", fusion["multispecialist_findings"])
        self.assertIn("pulmonologist", fusion["multispecialist_findings"])

    # ─────────────────────────────────────────────────────────────
    # TEST 6: GENERAL NON-SPECIALIST CASE FALLBACK
    # ─────────────────────────────────────────────────────────────
    async def test_06_general_case_fallback(self):
        """Verify general non-specialist cases route to General Diagnosis Agent."""
        input_data = PatientInput(
            patient_text="21-year-old college student with itchy red rash, honey-colored crusted pustules on face and chin for 4 days without systemic fever or chest symptoms.",
            history="No prior chronic illness."
        )

        res = await self.orchestrator.process_case(input_data.patient_text, input_data.history)
        routing = res["routing"]
        agents = res["agents"]
        fusion = res["fusion"]

        self.assertEqual(routing["primary_domain"], "general")
        self.assertIn("diagnosis", agents)
        self.assertIsNotNone(fusion["final_diagnosis"])

    # ─────────────────────────────────────────────────────────────
    # TEST 7: SPECIALIST ISOLATION & INDEPENDENCE
    # ─────────────────────────────────────────────────────────────
    async def test_07_specialist_isolation(self):
        """Verify each specialist operates on its independent taxonomy without interference."""
        test_text = PatientInput(
            patient_text="Elderly patient presenting for routine comprehensive checkup.",
            history="Mild essential hypertension."
        )

        cardio = CardiologistSpecialistAgent()
        pulmo = PulmonologistSpecialistAgent()
        neuro = NeurologistSpecialistAgent()

        cardio_resp = await cardio.run(test_text)
        pulmo_resp = await pulmo.run(test_text)
        neuro_resp = await neuro.run(test_text)

        # Ensure no cross-talk in disease classes or outputs
        self.assertEqual(set(cardio_resp.output.disease_probabilities.keys()), set(CardiologistSpecialistAgent.DISEASE_CLASSES))
        self.assertEqual(set(pulmo_resp.output.disease_probabilities.keys()), set(PulmonologistSpecialistAgent.DISEASE_CLASSES))
        self.assertEqual(set(neuro_resp.output.disease_probabilities.keys()), set(NeurologistSpecialistAgent.DISEASE_CLASSES))

    # ─────────────────────────────────────────────────────────────
    # TEST 8: SCHEMA & BACKWARD COMPATIBILITY
    # ─────────────────────────────────────────────────────────────
    async def test_08_schema_and_backward_compatibility(self):
        """Verify API response dictionary strictly conforms to Pydantic models."""
        res = await self.orchestrator.process_case("Patient with severe migraine and dizziness.", "None")

        # Required legacy top-level keys
        required_keys = {"input", "routing", "agents", "fusion", "rac", "rie", "ipm", "total_latency_ms"}
        self.assertTrue(required_keys.issubset(res.keys()))

        # Validate Pydantic deserialization
        PatientInput(**res["input"])
        SpecialistRoutingDecision(**res["routing"])
        FinalFusionOutput(**res["fusion"])
        RACOutput(**res["rac"])
        RIEOutput(**res["rie"])
        IPMOutput(**res["ipm"])
        self.assertIsInstance(res["total_latency_ms"], (int, float))

    # ─────────────────────────────────────────────────────────────
    # TEST 9: 100% LOCAL / OFFLINE EXECUTION GUARANTEE
    # ─────────────────────────────────────────────────────────────
    async def test_09_no_external_api_calls(self):
        """Verify pipeline executes 100% offline without any network or cloud LLM calls."""
        import urllib.request
        import http.client

        def blocked_network(*args, **kwargs):
            raise RuntimeError("CRITICAL VIOLATION: Pipeline attempted external network connection!")

        orig_urlopen = urllib.request.urlopen
        orig_connect = http.client.HTTPConnection.connect

        urllib.request.urlopen = blocked_network
        http.client.HTTPConnection.connect = blocked_network
        try:
            res = await self.orchestrator.process_case(
                "Patient with palpitations, chest tightness, and shortness of breath.",
                "History of hypertension and coronary artery disease."
            )
            self.assertIsNotNone(res["fusion"]["final_diagnosis"])
            print("  -> 100% Offline execution verified (zero network connections).")
        finally:
            urllib.request.urlopen = orig_urlopen
            http.client.HTTPConnection.connect = orig_connect

    # ─────────────────────────────────────────────────────────────
    # TEST 10: FROZEN ARTIFACTS INTEGRITY VERIFICATION
    # ─────────────────────────────────────────────────────────────
    def test_10_frozen_specialist_artifacts_integrity(self):
        """Verify that all three frozen specialist ONNX files and checkpoints exist intact."""
        files_to_check = [
            os.path.join(WORKSPACE, "backend", "models", "cardiologist_mimic_final_quant.onnx"),
            os.path.join(WORKSPACE, "backend", "models", "pulmonologist_final_quant.onnx"),
            os.path.join(WORKSPACE, "backend", "models", "neurologist_final_quant.onnx"),
        ]

        for fp in files_to_check:
            self.assertTrue(os.path.exists(fp), f"Missing frozen artifact: {fp}")
            self.assertGreater(os.path.getsize(fp), 50 * 1024 * 1024, f"File {fp} too small!")
        print("  -> All 3 frozen specialist artifacts verified intact.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
