"""
Phase 2.3 Gate Verification: Cross-Specialty Leakage Registry
============================================================
Tests:
1. Leakage index file exists.
2. Verified denominators match exact cohort numbers:
   - Full test cohort = 39,594
   - Clean encounter subset = 34,900
   - Clean patient subset = 31,200
   - Encounter-leaked cases = 4,694 (2,394 pulmo->neuro + 2,300 neuro->pulmo)
3. CrossSpecialtyRegistry query methods return correct classification.
"""

import json
import os
import sys
import unittest

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from scripts.utils.cross_specialty_registry import CrossSpecialtyRegistry, INDEX_PATH


class TestCrossSpecialtyIsolation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.registry = CrossSpecialtyRegistry()
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            cls.index = json.load(f)

    def test_denominators(self):
        """Verify exact denominators recorded in the registry."""
        denoms = self.index["cohort_denominators"]
        self.assertEqual(denoms["full_test_cohort"], 39594)
        self.assertEqual(denoms["clean_encounter_subset"], 34900)
        self.assertEqual(denoms["clean_patient_subset"], 31200)
        self.assertEqual(denoms["encounter_leaked_count"], 4694)
        self.assertEqual(denoms["patient_leaked_count"], 8394)

    def test_breakdown_patient_counts(self):
        """Verify 1,851 and 1,894 overlapping patient counts."""
        bk = self.index["breakdown"]
        self.assertEqual(bk["pulmo_train_to_neuro_test"]["overlapping_patients"], 1851)
        self.assertEqual(bk["pulmo_train_to_neuro_test"]["overlapping_encounters"], 2394)
        self.assertEqual(bk["neuro_train_to_pulmo_test"]["overlapping_patients"], 1894)
        self.assertEqual(bk["neuro_train_to_pulmo_test"]["overlapping_encounters"], 2300)
        self.assertEqual(bk["cardiology_cross_leakage"]["overlapping_patients"], 0)

    def test_registry_methods(self):
        """Verify registry lookup queries."""
        leaked_enc_list = self.index["encounter_leaked_ids"]
        self.assertGreater(len(leaked_enc_list), 0)
        sample_leaked = leaked_enc_list[0]

        self.assertTrue(self.registry.is_encounter_leaked(sample_leaked))
        self.assertFalse(self.registry.is_clean_encounter(sample_leaked))

        # Non-leaked ID test
        dummy_id = "non_existent_encounter_12345"
        self.assertFalse(self.registry.is_encounter_leaked(dummy_id))
        self.assertTrue(self.registry.is_clean_encounter(dummy_id))


if __name__ == "__main__":
    unittest.main()
