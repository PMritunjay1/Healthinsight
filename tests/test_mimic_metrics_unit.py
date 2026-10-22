import unittest
import sys
import os

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from scripts.mimic_evaluation_metrics import compute_case_outcome, aggregate_cohort_metrics

class TestMIMICMetricsEngine(unittest.TestCase):
    def test_01_correct_single_label(self):
        """Case 1: Correct prediction with one ground-truth label."""
        res = compute_case_outcome(
            predicted_diagnosis="heart failure",
            ground_truth_labels=["heart failure"]
        )
        self.assertTrue(res["usable"])
        self.assertTrue(res["any_label_match"])
        self.assertTrue(res["is_single_label"])
        self.assertTrue(res["single_label_match"])
        self.assertTrue(res["strict_exact_set_match"])

    def test_02_matching_one_of_several_labels(self):
        """
        Case 2: Prediction matching one of several ground-truth labels.
        MUST count as any-label match but MUST NOT count as strict multilabel exact set match!
        """
        res = compute_case_outcome(
            predicted_diagnosis="heart failure",
            ground_truth_labels=["heart failure", "hypertension"]
        )
        self.assertTrue(res["usable"])
        self.assertTrue(res["any_label_match"])
        self.assertFalse(res["is_single_label"])
        self.assertFalse(res["single_label_match"])
        self.assertFalse(res["strict_exact_set_match"], "Matching 1 of 2 labels cannot be exact set match!")

    def test_03_matching_none(self):
        """Case 3: A prediction matching none of the ground-truth labels."""
        res = compute_case_outcome(
            predicted_diagnosis="angina",
            ground_truth_labels=["asthma"]
        )
        self.assertTrue(res["usable"])
        self.assertFalse(res["any_label_match"])
        self.assertFalse(res["single_label_match"])
        self.assertFalse(res["strict_exact_set_match"])

    def test_04_abstention(self):
        """Case 4: An abstention."""
        res = compute_case_outcome(
            predicted_diagnosis="Uncertain — refer to clinician",
            ground_truth_labels=["copd"],
            is_abstained=True
        )
        self.assertFalse(res["usable"])
        self.assertTrue(res["is_abstained"])
        self.assertFalse(res["any_label_match"])

    def test_05_inference_exception(self):
        """Case 5: An inference exception."""
        res = compute_case_outcome(
            predicted_diagnosis=None,
            ground_truth_labels=["pneumonia"],
            is_exception=True
        )
        self.assertFalse(res["usable"])
        self.assertTrue(res["is_exception"])
        self.assertFalse(res["any_label_match"])

    def test_06_invalid_unmapped_label(self):
        """Case 6: An invalid or unmapped label."""
        res = compute_case_outcome(
            predicted_diagnosis="Unknown Condition",
            ground_truth_labels=["stroke_ischemic"]
        )
        self.assertTrue(res["usable"])
        self.assertEqual(res["status"], "INVALID_PREDICTION")
        self.assertFalse(res["any_label_match"])

    def test_07_empty_ground_truth(self):
        """Case 7: An empty ground-truth label list."""
        res = compute_case_outcome(
            predicted_diagnosis="hypertension",
            ground_truth_labels=[]
        )
        self.assertTrue(res["usable"])
        self.assertFalse(res["any_label_match"])
        self.assertFalse(res["is_single_label"])

    def test_08_duplicate_ground_truth_labels(self):
        """Case 8: Duplicate labels in ground truth should be deduplicated to single-label."""
        res = compute_case_outcome(
            predicted_diagnosis="epilepsy_seizures",
            ground_truth_labels=["epilepsy_seizures", "epilepsy_seizures"]
        )
        self.assertTrue(res["usable"])
        self.assertTrue(res["any_label_match"])
        self.assertTrue(res["is_single_label"], "Duplicate labels must be deduplicated to length 1")
        self.assertTrue(res["single_label_match"])
        self.assertTrue(res["strict_exact_set_match"])

    def test_09_leakage_excluded(self):
        """Case 9: A case excluded due to verified leakage."""
        res = compute_case_outcome(
            predicted_diagnosis="heart failure",
            ground_truth_labels=["heart failure"],
            is_leakage_excluded=True
        )
        self.assertFalse(res["eligible"])
        self.assertEqual(res["status"], "EXCLUDED_LEAKAGE")
        self.assertFalse(res["usable"])
        self.assertFalse(res["any_label_match"])

    def test_10_cohort_aggregation(self):
        """Verify explicit denominator accounting on an aggregated cohort."""
        cases = [
            compute_case_outcome("heart failure", ["heart failure"]), # 1 usable match, single
            compute_case_outcome("heart failure", ["heart failure", "hypertension"]), # 1 usable match, multi
            compute_case_outcome("angina", ["asthma"]), # 1 usable non-match, single
            compute_case_outcome("Uncertain — refer to clinician", ["copd"], is_abstained=True), # 1 abstention
            compute_case_outcome(None, ["pneumonia"], is_exception=True), # 1 exception
            compute_case_outcome("heart failure", ["heart failure"], is_leakage_excluded=True), # 1 excluded
        ]
        agg = aggregate_cohort_metrics(cases)
        self.assertEqual(agg["total_source_encounters"], 6)
        self.assertEqual(agg["leakage_excluded_encounters"], 1)
        self.assertEqual(agg["eligible_cohort"], 5)
        self.assertEqual(agg["usable_encounters"], 3)
        self.assertEqual(agg["abstentions"], 1)
        self.assertEqual(agg["exceptions"], 1)
        self.assertEqual(agg["coverage_pct"], 60.0) # 3/5
        self.assertEqual(agg["abstention_rate_pct"], 20.0) # 1/5
        self.assertEqual(agg["exception_rate_pct"], 20.0) # 1/5
        self.assertEqual(agg["any_label_matches"], 2)
        self.assertEqual(agg["selective_any_label_accuracy_pct"], 66.67) # 2/3
        self.assertEqual(agg["full_cohort_any_label_accuracy_pct"], 40.0) # 2/5
        self.assertEqual(agg["strict_single_label_eligible"], 4)
        self.assertEqual(agg["strict_single_label_usable"], 2)
        self.assertEqual(agg["strict_single_label_matches"], 1)
        self.assertEqual(agg["strict_single_label_selective_acc_pct"], 50.0) # 1/2
        self.assertEqual(agg["strict_single_label_full_cohort_acc_pct"], 25.0) # 1/4
        self.assertEqual(agg["strict_multilabel_exact_set_matches"], 1) # Only case 1 is strict set match

if __name__ == "__main__":
    unittest.main()
