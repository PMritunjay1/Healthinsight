"""
Phase 2.5 Gate Verification: Ambiguity Risk Detector & Activation Routing
========================================================================
Tests Algorithm 1 implementation:
1. Entropy edge cases (0 symptoms, 1 symptom, single-domain, multi-domain).
2. Symptom rarity prior computation.
3. Differential overlap computation.
4. Threshold boundary checks (Low, Medium, High tiers).
5. Safe failure behavior on missing/empty/corrupted inputs.
6. End-to-end integration with SpecialistRouter and Orchestrator.
"""

import asyncio
import os
import sys
import unittest

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from backend.ambiguity_detector import AmbiguityDetector
from backend.specialist_router import SpecialistRouter
from backend.orchestrator import Orchestrator
from backend.models import PatientInput, RIEOutput


class TestAmbiguityRouting(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.detector = AmbiguityDetector(r_low=0.35, r_high=0.65)
        self.router = SpecialistRouter(ambiguity_detector=self.detector)

    def test_entropy_edge_cases(self):
        """Verify Shannon entropy edge cases."""
        # Case 1: 0 symptoms
        h0, _ = self.detector.compute_symptom_domain_entropy([])
        self.assertEqual(h0, 0.0)

        # Case 2: 1 symptom
        h1, _ = self.detector.compute_symptom_domain_entropy(["chest pain"])
        self.assertEqual(h1, 0.0)

        # Case 3: Multiple symptoms in identical domain (Cardio)
        h_cardio, counts_c = self.detector.compute_symptom_domain_entropy(
            ["chest pain", "chest tightness", "palpitations", "edema"]
        )
        self.assertEqual(h_cardio, 0.0)
        self.assertEqual(counts_c["cardio"], 4)
        self.assertEqual(counts_c["pulmo"], 0)

        # Case 4: 50/50 split across 2 domains (Cardio and Pulmo) -> Maximum normalized entropy
        h_split, _ = self.detector.compute_symptom_domain_entropy(["chest pain", "shortness of breath"])
        self.assertAlmostEqual(h_split, 1.0, places=2)

        # Case 5: 3 domains represented
        h_tri, counts_tri = self.detector.compute_symptom_domain_entropy(
            ["chest pain", "shortness of breath", "seizure"]
        )
        self.assertGreater(h_tri, 0.5)
        self.assertEqual(counts_tri["cardio"], 1)
        self.assertEqual(counts_tri["pulmo"], 1)
        self.assertEqual(counts_tri["neuro"], 1)

    def test_rarity_score_edge_cases(self):
        """Common symptoms must have lower penalty than atypical ones."""
        common_penalty = self.detector.compute_rarity_score(["chest pain", "cough", "headache"])
        self.assertLessEqual(common_penalty, 0.30)

        atypical_penalty = self.detector.compute_rarity_score(["nystagmus", "fasciculations", "clonus"])
        self.assertGreaterEqual(atypical_penalty, 0.50)

        empty_penalty = self.detector.compute_rarity_score([])
        self.assertEqual(empty_penalty, 0.20)

    def test_differential_overlap_edge_cases(self):
        """Verify secondary to primary score ratio."""
        # Single dominant specialty
        o_single = self.detector.compute_differential_overlap({"cardio": 0.90, "pulmo": 0.0, "neuro": 0.0})
        self.assertEqual(o_single, 0.0)

        # Equal competition
        o_tie = self.detector.compute_differential_overlap({"cardio": 0.80, "pulmo": 0.80, "neuro": 0.10})
        self.assertEqual(o_tie, 1.0)

        # No active domains
        o_none = self.detector.compute_differential_overlap({"cardio": 0.0, "pulmo": 0.0, "neuro": 0.0})
        self.assertEqual(o_none, 0.0)

    def test_low_risk_tier_single_specialist(self):
        """Unambiguous isolated cardiac presentation activates only 1 specialist and Tier LOW."""
        inp = PatientInput(
            patient_text="62-year-old male presenting with crushing substernal chest pain, radiating to jaw, diaphoresis.",
            history="Known coronary artery disease."
        )
        rie = RIEOutput(symptoms=["chest pain", "jaw pain", "diaphoresis"])
        dec = self.router.route(inp, rie_output=rie)

        self.assertEqual(dec.routing_tier, "LOW")
        self.assertLess(dec.ambiguity_score, 0.35)
        self.assertEqual(len(dec.selected_specialists), 1)
        self.assertIn("cardiologist", dec.selected_specialists)
        self.assertFalse(dec.abstain_eligible)
        self.assertFalse(dec.retrieval_enabled)

    def test_high_risk_tier_full_board(self):
        """Complex presentation spanning multiple domains triggers Tier HIGH and full board."""
        inp = PatientInput(
            patient_text=(
                "78-year-old female admitted with acute shortness of breath, severe chest pressure, "
                "hypoxia SpO2 86%, bilateral crackles, sudden confusion, and right facial droop."
            ),
            history="History of heart failure, COPD, and prior TIA."
        )
        rie = RIEOutput(symptoms=["shortness of breath", "chest pressure", "confusion", "facial droop"])
        dec = self.router.route(inp, rie_output=rie)

        self.assertEqual(dec.routing_tier, "HIGH")
        self.assertGreaterEqual(dec.ambiguity_score, 0.65)
        self.assertTrue(dec.abstain_eligible)
        self.assertTrue(dec.retrieval_enabled)
        self.assertEqual(set(dec.selected_specialists), {"cardiologist", "pulmonologist", "neurologist"})

    def test_safe_failure_empty_and_garbage_input(self):
        """Empty or irrelevant input must fail safely without throwing exceptions."""
        inp_empty = PatientInput(patient_text="")
        dec_empty = self.router.route(inp_empty)
        self.assertIn(dec_empty.routing_tier, ["LOW", "MEDIUM", "HIGH"])
        self.assertIsInstance(dec_empty.selected_specialists, list)

        inp_garbage = PatientInput(patient_text="The quick brown fox jumps over the lazy dog.")
        dec_garbage = self.router.route(inp_garbage)
        self.assertEqual(dec_garbage.routing_tier, "LOW")
        self.assertFalse(dec_garbage.abstain_eligible)

    async def test_orchestrator_integration(self):
        """End-to-end integration: Orchestrator routes and activates full board on High Risk case."""
        orch = Orchestrator()
        inp_text = "Severe acute dyspnea, wheezing, bilateral lung infiltrates, substernal crushing pain, and acute aphasia."
        res = await orch.process_case(patient_text=inp_text)

        self.assertIn("routing", res)
        routing = res["routing"]
        self.assertIn("routing_tier", routing)
        self.assertIn("ambiguity_score", routing)
        self.assertIn("abstain_eligible", routing)
        self.assertEqual(routing["routing_tier"], "HIGH")
        self.assertTrue(routing["abstain_eligible"])

        # Check all 3 specialists were executed in agent dictionary
        agents = res["agents"]
        self.assertIn("cardiologist", agents)
        self.assertIn("pulmonologist", agents)
        self.assertIn("neurologist", agents)


if __name__ == "__main__":
    unittest.main()
