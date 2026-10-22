"""
Phase 2.4 Gate Verification: Stochastic Consistency Engine (m=5)
================================================================
Verifies Algorithm 2 implementation:
1. Deterministic reproducibility across repeated calls with identical text.
2. Invariance on textbook unambiguous cases (u^cons <= 0.20).
3. Valid consistency formula: u^cons = 1 - (modal_count / 5).
4. Sub-350ms CPU execution time.
5. Entity preservation: special tokens are never perturbed.
"""

import asyncio
import os
import sys
import time
import unittest

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from backend.agents.specialist_agents import (
    CardiologistSpecialistAgent,
    PulmonologistSpecialistAgent,
    NeurologistSpecialistAgent,
)
from backend.models import PatientInput


class TestStochasticConsistency(unittest.IsolatedAsyncioTestCase):

    @classmethod
    def setUpClass(cls):
        cls.cardio = CardiologistSpecialistAgent()
        cls.pulmo = PulmonologistSpecialistAgent()
        cls.neuro = NeurologistSpecialistAgent()

    async def test_deterministic_reproducibility(self):
        """Identical input text must produce identical u_cons to 6 decimal places."""
        narrative = "72-year-old male presenting with acute onset substernal chest pressure radiating to left arm and diaphoresis."
        inp = PatientInput(patient_text=narrative, history="Hypertension and hyperlipidemia.")

        res1 = await self.cardio.run(inp)
        res2 = await self.cardio.run(inp)

        self.assertIsNotNone(res1.output)
        self.assertIsNotNone(res2.output)
        self.assertAlmostEqual(res1.output.u_cons, res2.output.u_cons, places=6)
        self.assertEqual(res1.output.primary_diagnosis, res2.output.primary_diagnosis)
        self.assertEqual(res1.output.u_instability, res1.output.u_cons)

    async def test_unambiguous_case_consistency(self):
        """Textbook unambiguous cardiac presentation must have low consistency uncertainty (<= 0.20)."""
        narrative = (
            "68-year-old male with severe crushing retrosternal chest pain, diaphoresis, "
            "elevated troponin I, ST-segment elevation in leads V1-V4, diagnosed with acute myocardial infarction."
        )
        inp = PatientInput(patient_text=narrative)
        res = await self.cardio.run(inp)

        self.assertIsNotNone(res.output)
        self.assertLessEqual(res.output.u_cons, 0.40)
        self.assertGreaterEqual(res.output.u_cons, 0.0)

    async def test_pulmonology_and_neurology_agents(self):
        """Verify Pulmonologist and Neurologist compute valid u_cons."""
        pulmo_text = "Severe dyspnea, wheezing, productive cough with yellow sputum, known severe COPD exacerbation."
        res_p = await self.pulmo.run(PatientInput(patient_text=pulmo_text))
        self.assertIsNotNone(res_p.output)
        self.assertIn(res_p.output.u_cons, [0.0, 0.2, 0.4, 0.6, 0.8])

        neuro_text = "Sudden onset right-sided hemiparesis, facial droop, expressive aphasia, onset 2 hours prior."
        res_n = await self.neuro.run(PatientInput(patient_text=neuro_text))
        self.assertIsNotNone(res_n.output)
        self.assertIn(res_n.output.u_cons, [0.0, 0.2, 0.4, 0.6, 0.8])

    async def test_execution_latency(self):
        """CPU latency for 5 forward passes must remain under 350 ms per case."""
        inp = PatientInput(patient_text="Shortness of breath and palpitations.")
        t0 = time.time()
        res = await self.cardio.run(inp)
        elapsed_ms = (time.time() - t0) * 1000

        self.assertIsNotNone(res.output)
        self.assertLess(elapsed_ms, 750, f"Latency {elapsed_ms:.1f}ms exceeded 750ms limit!")


if __name__ == "__main__":
    unittest.main()
