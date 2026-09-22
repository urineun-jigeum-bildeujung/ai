"""P1 runner regression tests using only a temporary output directory."""
from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path[:0] = [str(SCRIPTS), str(SCRIPTS / "nutrition")]

from close_gold_fallback_evidence_p1 import run  # noqa: E402
from gold_evidence_intake import INTAKE_FIELDS  # noqa: E402


class GoldFallbackEvidenceP1RunnerTests(unittest.TestCase):
    """A missing human intake must remain an explicit zero-evidence state."""

    def test_missing_intake_writes_zero_operational_artifact_and_does_not_run_real_e2e(self):
        source = ROOT / "data" / "eval"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for name in (
                "gold_product_candidates.csv",
                "gold_product_gap_analysis.csv",
                "blocked_report_gold_v1.csv",
            ):
                shutil.copy2(source / name, target / name)

            summary = run(eval_dir=target)

            self.assertEqual(summary["human_intake_rows"], 0)
            self.assertEqual(summary["accepted"], 0)
            self.assertEqual(summary["rejected"], 0)
            self.assertEqual(summary["verified_cat"], 0)
            self.assertEqual(summary["verified_dog"], 0)
            self.assertEqual(summary["runtime_eligible"], 0)
            self.assertFalse(summary["e2e_executed"])
            self.assertFalse((target / "human_gold_evidence_intake_v1.csv").exists())

            artifact = json.loads((target / "verified_evidence_gold_v1.json").read_text(encoding="utf-8"))
            self.assertTrue(artifact["input_missing"])
            self.assertEqual(artifact["operational_evidence"], [])
            self.assertEqual(artifact["runtime_eligible_count"], 0)
            with (target / "verified_evidence_gold_v1.csv").open(encoding="utf-8", newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])
            with (target / "gold_evidence_intake_manifest_template_v1.csv").open(encoding="utf-8", newline="") as handle:
                self.assertEqual(tuple(csv.DictReader(handle).fieldnames or ()), INTAKE_FIELDS)

            e2e = json.loads((target / "gold_persisted_product_e2e_v1.json").read_text(encoding="utf-8"))
            self.assertFalse(e2e["executed"])
            self.assertEqual(e2e["candidate_results"], [])


if __name__ == "__main__":
    unittest.main()
