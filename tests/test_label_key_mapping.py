"""
Unit Tests for Authoritative Label Key Mapping and Extraction
=============================================================
Verifies that ground-truth labels across Cardiology, Pulmonology,
and Neurology are reliably extracted from dataset records and that the
historical bug ('pulmo_disease_labels' / 'neuro_disease_labels') is resolved.
"""

import json
import os
import sys
import unittest

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from scripts.utils.label_extractor import (
    LABEL_KEY_MAP,
    SPECIALTY_LABEL_MAP,
    ALL_DISEASE_CLASSES,
    extract_ground_truth_labels,
    normalize_specialty,
    get_specialty_key,
    get_allowed_labels,
)


class TestLabelKeyMapping(unittest.TestCase):

    def test_specialty_normalization(self):
        """Test normalization for diverse role/name formats."""
        self.assertEqual(normalize_specialty("Cardiology"), "Cardiology")
        self.assertEqual(normalize_specialty("cardiologist"), "Cardiology")
        self.assertEqual(normalize_specialty("cardio"), "Cardiology")

        self.assertEqual(normalize_specialty("Pulmonology"), "Pulmonology")
        self.assertEqual(normalize_specialty("pulmonologist"), "Pulmonology")
        self.assertEqual(normalize_specialty("pulmo"), "Pulmonology")
        self.assertEqual(normalize_specialty("respiratory"), "Pulmonology")

        self.assertEqual(normalize_specialty("Neurology"), "Neurology")
        self.assertEqual(normalize_specialty("neurologist"), "Neurology")
        self.assertEqual(normalize_specialty("neuro"), "Neurology")

        with self.assertRaises(ValueError):
            normalize_specialty("dermatology")

    def test_schema_mapping_integrity(self):
        """Verify the exact required dataset keys for each specialty."""
        self.assertEqual(LABEL_KEY_MAP["Cardiology"], "cardio_disease_labels")
        self.assertEqual(LABEL_KEY_MAP["Pulmonology"], "pulmonary_disease_labels")
        self.assertEqual(LABEL_KEY_MAP["Neurology"], "neurology_disease_labels")

        self.assertEqual(get_specialty_key("cardio"), "cardio_disease_labels")
        self.assertEqual(get_specialty_key("pulmonologist"), "pulmonary_disease_labels")
        self.assertEqual(get_specialty_key("neuro"), "neurology_disease_labels")

    def test_vocabulary_cardinality(self):
        """Ensure 6 diseases per domain, totaling 18 canonical classes."""
        self.assertEqual(len(SPECIALTY_LABEL_MAP["Cardiology"]), 6)
        self.assertEqual(len(SPECIALTY_LABEL_MAP["Pulmonology"]), 6)
        self.assertEqual(len(SPECIALTY_LABEL_MAP["Neurology"]), 6)
        self.assertEqual(len(ALL_DISEASE_CLASSES), 18)
        self.assertEqual(len(set(ALL_DISEASE_CLASSES)), 18)

    def test_extraction_on_live_cardiology_split(self):
        """Verify label extraction on Cardiology test split."""
        path = os.path.join(WORKSPACE, "datasets", "processed", "common", "test_split.json")
        self.assertTrue(os.path.exists(path), f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        self.assertGreater(len(records), 0)
        # Sample first 20 records
        for r in records[:20]:
            labels = extract_ground_truth_labels(r, "Cardiology")
            self.assertIsInstance(labels, list)
            self.assertGreater(len(labels), 0, f"Empty labels for record {r.get('encounter_id')}")
            for l in labels:
                self.assertIn(l, SPECIALTY_LABEL_MAP["Cardiology"])

    def test_extraction_on_live_pulmonology_split(self):
        """Verify label extraction on Pulmonology test split using true key 'pulmonary_disease_labels'."""
        path = os.path.join(WORKSPACE, "datasets", "processed", "pulmonologist", "test_split.json")
        self.assertTrue(os.path.exists(path), f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        self.assertGreater(len(records), 0)
        # Sample first 20 records
        for r in records[:20]:
            # Old buggy access returns empty
            self.assertEqual(r.get("pulmo_disease_labels", []), [])
            # Correct extractor retrieves valid labels
            labels = extract_ground_truth_labels(r, "Pulmonology")
            self.assertIsInstance(labels, list)
            self.assertGreater(len(labels), 0, f"Empty labels for record {r.get('encounter_id')}")
            for l in labels:
                self.assertIn(l, SPECIALTY_LABEL_MAP["Pulmonology"])

    def test_extraction_on_live_neurology_split(self):
        """Verify label extraction on Neurology test split using true key 'neurology_disease_labels'."""
        path = os.path.join(WORKSPACE, "datasets", "processed", "neurologist", "test_split.json")
        self.assertTrue(os.path.exists(path), f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        self.assertGreater(len(records), 0)
        # Sample first 20 records
        for r in records[:20]:
            # Old buggy access returns empty
            self.assertEqual(r.get("neuro_disease_labels", []), [])
            # Correct extractor retrieves valid labels
            labels = extract_ground_truth_labels(r, "Neurology")
            self.assertIsInstance(labels, list)
            self.assertGreater(len(labels), 0, f"Empty labels for record {r.get('encounter_id')}")
            for l in labels:
                self.assertIn(l, SPECIALTY_LABEL_MAP["Neurology"])


if __name__ == "__main__":
    unittest.main()
