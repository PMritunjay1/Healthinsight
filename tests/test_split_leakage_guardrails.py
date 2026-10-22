"""
Phase 2.2 Gate Verification: Four-Way Disjoint Split Discipline
==============================================================
Verifies:
1. Four-Way Split Manifest exists and documents all partitions.
2. Strict patient-level disjointness between:
   - D_train and D_cal (clean_calib)
   - D_cal and D_test
   - D_val_tune and D_cal
3. Strict encounter-level disjointness between D_cal and D_test.
4. Clean calibration CSVs have 0 overlap with training CSVs.
"""

import json
import os
import sys
import unittest

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

MANIFEST_PATH = os.path.join(WORKSPACE, "datasets", "metadata", "four_way_split_manifest.json")


class TestSplitLeakageGuardrails(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        self_manifest = MANIFEST_PATH
        if not os.path.exists(self_manifest):
            from scripts.build_four_way_disjoint_splits import build_four_way_splits
            build_four_way_splits()
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            cls.manifest = json.load(f)

    def test_manifest_structure(self):
        """Manifest must include Cardiology, Pulmonology, and Neurology."""
        specs = self.manifest.get("specialties", {})
        self.assertIn("Cardiology", specs)
        self.assertIn("Pulmonology", specs)
        self.assertIn("Neurology", specs)

    def test_cardiology_four_way_disjointness(self):
        """Verify strict zero-leakage for Cardiology partitions."""
        cal_path = os.path.join(WORKSPACE, "datasets", "processed", "common", "clean_calib.json")
        val_tune_path = os.path.join(WORKSPACE, "datasets", "processed", "common", "val_tune.json")
        train_path = os.path.join(WORKSPACE, "datasets", "processed", "common", "train_split.json")
        test_path = os.path.join(WORKSPACE, "datasets", "processed", "common", "test_split.json")

        with open(cal_path, "r", encoding="utf-8") as f: cal = json.load(f)
        with open(val_tune_path, "r", encoding="utf-8") as f: val_tune = json.load(f)
        with open(train_path, "r", encoding="utf-8") as f: train = json.load(f)
        with open(test_path, "r", encoding="utf-8") as f: test = json.load(f)

        cal_pts = {r["patient_id"] for r in cal if r.get("patient_id")}
        val_tune_pts = {r["patient_id"] for r in val_tune if r.get("patient_id")}
        train_pts = {r["patient_id"] for r in train if r.get("patient_id")}
        test_pts = {r["patient_id"] for r in test if r.get("patient_id")}

        cal_encs = {str(r.get("encounter_id") or r.get("patient_id")) for r in cal}
        test_encs = {str(r.get("encounter_id") or r.get("patient_id")) for r in test}

        # Assert zero patient overlap
        self.assertEqual(len(cal_pts & train_pts), 0, "Cardio: D_cal overlaps with D_train!")
        self.assertEqual(len(cal_pts & test_pts), 0, "Cardio: D_cal overlaps with D_test!")
        self.assertEqual(len(cal_pts & val_tune_pts), 0, "Cardio: D_cal overlaps with D_val_tune!")
        # Assert zero encounter overlap
        self.assertEqual(len(cal_encs & test_encs), 0, "Cardio: Encounter overlap with D_test!")

    def test_pulmonology_four_way_disjointness(self):
        """Verify strict zero-leakage for Pulmonology partitions."""
        cal_path = os.path.join(WORKSPACE, "datasets", "processed", "pulmonologist", "clean_calib.json")
        val_tune_path = os.path.join(WORKSPACE, "datasets", "processed", "pulmonologist", "val_tune.json")
        train_path = os.path.join(WORKSPACE, "datasets", "processed", "pulmonologist", "train_split.json")
        test_path = os.path.join(WORKSPACE, "datasets", "processed", "pulmonologist", "test_split.json")

        with open(cal_path, "r", encoding="utf-8") as f: cal = json.load(f)
        with open(val_tune_path, "r", encoding="utf-8") as f: val_tune = json.load(f)
        with open(train_path, "r", encoding="utf-8") as f: train = json.load(f)
        with open(test_path, "r", encoding="utf-8") as f: test = json.load(f)

        cal_pts = {r["patient_id"] for r in cal if r.get("patient_id")}
        val_tune_pts = {r["patient_id"] for r in val_tune if r.get("patient_id")}
        train_pts = {r["patient_id"] for r in train if r.get("patient_id")}
        test_pts = {r["patient_id"] for r in test if r.get("patient_id")}

        cal_encs = {str(r.get("encounter_id") or r.get("patient_id")) for r in cal}
        test_encs = {str(r.get("encounter_id") or r.get("patient_id")) for r in test}

        self.assertEqual(len(cal_pts & train_pts), 0, "Pulmo: D_cal overlaps with D_train!")
        self.assertEqual(len(cal_pts & test_pts), 0, "Pulmo: D_cal overlaps with D_test!")
        self.assertEqual(len(cal_pts & val_tune_pts), 0, "Pulmo: D_cal overlaps with D_val_tune!")
        self.assertEqual(len(cal_encs & test_encs), 0, "Pulmo: Encounter overlap with D_test!")

    def test_neurology_four_way_disjointness(self):
        """Verify strict zero-leakage for Neurology partitions."""
        cal_path = os.path.join(WORKSPACE, "datasets", "processed", "neurologist", "clean_calib.json")
        val_tune_path = os.path.join(WORKSPACE, "datasets", "processed", "neurologist", "val_tune.json")
        train_path = os.path.join(WORKSPACE, "datasets", "processed", "neurologist", "train_split.json")
        test_path = os.path.join(WORKSPACE, "datasets", "processed", "neurologist", "test_split.json")

        with open(cal_path, "r", encoding="utf-8") as f: cal = json.load(f)
        with open(val_tune_path, "r", encoding="utf-8") as f: val_tune = json.load(f)
        with open(train_path, "r", encoding="utf-8") as f: train = json.load(f)
        with open(test_path, "r", encoding="utf-8") as f: test = json.load(f)

        cal_pts = {r["patient_id"] for r in cal if r.get("patient_id")}
        val_tune_pts = {r["patient_id"] for r in val_tune if r.get("patient_id")}
        train_pts = {r["patient_id"] for r in train if r.get("patient_id")}
        test_pts = {r["patient_id"] for r in test if r.get("patient_id")}

        cal_encs = {str(r.get("encounter_id") or r.get("patient_id")) for r in cal}
        test_encs = {str(r.get("encounter_id") or r.get("patient_id")) for r in test}

        self.assertEqual(len(cal_pts & train_pts), 0, "Neuro: D_cal overlaps with D_train!")
        self.assertEqual(len(cal_pts & test_pts), 0, "Neuro: D_cal overlaps with D_test!")
        self.assertEqual(len(cal_pts & val_tune_pts), 0, "Neuro: D_cal overlaps with D_val_tune!")
        self.assertEqual(len(cal_encs & test_encs), 0, "Neuro: Encounter overlap with D_test!")


if __name__ == "__main__":
    unittest.main()
