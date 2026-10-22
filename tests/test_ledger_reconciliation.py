"""
Phase 2.1 Gate Verification: Independent MIMIC Ledger Reconciliation
===================================================================
Tests that:
1. The audit certificate exists and reports 'VERIFIED_AND_REPRODUCED'.
2. The cohort size is exactly 39,594.
3. Selective accuracy is reproduced at 48.17% (+/- 0.05%).
4. Coverage is verified at 50.61% (+/- 0.05%).
5. Zero duplicate record IDs or corrupted rows exist.
"""

import json
import os
import sys
import unittest

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

CERT_PATH = os.path.join(
    WORKSPACE,
    "results",
    "reconciled_baseline_audit_20261018",
    "ledger_audit_certificate.json",
)


class TestLedgerReconciliation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(CERT_PATH):
            from scripts.audit_and_reconcile_ledger import audit_and_reconcile
            audit_and_reconcile()
        with open(CERT_PATH, "r", encoding="utf-8") as f:
            cls.cert = json.load(f)

    def test_reconciliation_verdict(self):
        """Must confirm VERIFIED_AND_REPRODUCED status."""
        self.assertEqual(self.cert.get("verification_status"), "VERIFIED_AND_REPRODUCED")

    def test_cohort_integrity(self):
        """Assert exactly 39,594 rows, zero duplicates, zero corrupted."""
        ledger_meta = self.cert["input_ledger"]
        self.assertEqual(ledger_meta["total_rows"], 39594)
        self.assertEqual(ledger_meta["unique_records"], 39594)
        self.assertEqual(ledger_meta["corrupted_rows"], 0)

    def test_specialty_breakdown(self):
        """Assert exact counts across Cardiology (15,099), Pulmonology (14,512), Neurology (9,983)."""
        counts = self.cert["specialty_breakdown"]
        self.assertEqual(counts["Cardiology"], 15099)
        self.assertEqual(counts["Pulmonology"], 14512)
        self.assertEqual(counts["Neurology"], 9983)
        self.assertEqual(sum(counts.values()), 39594)

    def test_selective_accuracy_reproduced(self):
        """Assert selective accuracy matches 48.17% +/- 0.05%."""
        metrics = self.cert["metrics_reproduced"]
        sel_acc = metrics["measured_selective_accuracy"]
        self.assertAlmostEqual(sel_acc, 48.17, delta=0.05)
        self.assertEqual(metrics["matches"], 9653)
        self.assertEqual(metrics["usable_cases"], 20038)
        self.assertEqual(metrics["abstained_cases"], 19556)
        self.assertEqual(metrics["total_cases"], 39594)

    def test_coverage_pct(self):
        """Assert coverage is 50.61% +/- 0.05%."""
        metrics = self.cert["metrics_reproduced"]
        self.assertAlmostEqual(metrics["coverage_pct"], 50.61, delta=0.05)


if __name__ == "__main__":
    unittest.main()
