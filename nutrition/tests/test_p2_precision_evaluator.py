import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "scripts"
sys.path[:0] = [str(S), str(S / "nutrition")]
from evaluate_allergen_precision import evaluate, text_class


def row(label="", source="OPFF", method="CANONICAL_ALIAS", text="chicken"):
    return {"product_id": "p", "source_dataset": source, "raw_ingredient_text": text, "segmented_text": f"['{text}']", "mapping_method": method, "review_label": label}


class PrecisionEvaluator(unittest.TestCase):
    def test_blank_labels_are_incomplete(self):
        result = evaluate([row(), row("CORRECT")])
        self.assertEqual((result["status"], result["precision"], result["overall"]["remaining_rows"]), ("INCOMPLETE", None, 1))

    def test_invalid_label_is_rejected(self):
        with self.assertRaises(ValueError): evaluate([row("AUTO_APPROVED")])

    def test_strict_precision(self):
        self.assertEqual(evaluate([row("CORRECT"), row("INCORRECT")])["overall"]["strict_precision"], 0.5)

    def test_determinate_precision(self):
        self.assertEqual(evaluate([row("CORRECT"), row("AMBIGUOUS"), row("INCORRECT")])["overall"]["determinate_precision"], 0.5)

    def test_ambiguous_and_incorrect_rates(self):
        metrics = evaluate([row("CORRECT"), row("AMBIGUOUS"), row("INCORRECT"), row("INCORRECT")])["overall"]
        self.assertEqual((metrics["ambiguous_rate"], metrics["incorrect_rate"]), (0.25, 0.5))

    def test_source_stratification(self):
        result = evaluate([row("CORRECT", "OPFF"), row("INCORRECT", "OEM"), row("AMBIGUOUS", "GLOBAL")])
        self.assertEqual(set(result["breakdown"]["source_dataset"]), {"OPFF", "OEM", "GLOBAL"})

    def test_mapping_method_stratification(self):
        result = evaluate([row("CORRECT", method="CANONICAL_ALIAS"), row("INCORRECT", method="CANONICAL_EXACT")])
        self.assertEqual(set(result["breakdown"]["mapping_method"]), {"CANONICAL_ALIAS", "CANONICAL_EXACT"})

    def test_wilson_interval_includes_sample_size(self):
        metrics = evaluate([row("CORRECT") for _ in range(4)])["overall"]
        self.assertEqual(metrics["n"], 4)
        self.assertLess(metrics["ci95_lower"], 1.0)
        self.assertEqual(metrics["ci95_upper"], 1.0)

    def test_incomplete_audit_cannot_pass_precision_gate(self):
        self.assertEqual(evaluate([row()])["gates"]["GATE_F_HUMAN_PRECISION"], "WAITING")

    def test_candidate_file_never_auto_approves_turkey(self):
        with (ROOT / "data/eval/allergen_dictionary_candidate_review_p2.csv").open(encoding="utf-8") as handle:
            labels = [r["review_label"] for r in csv.DictReader(handle)]
        self.assertTrue(labels)
        self.assertTrue(all(label == "" for label in labels))

    def test_coverage_and_precision_are_separate(self):
        result = evaluate([row("CORRECT")])
        self.assertIn("coverage", result)
        self.assertNotIn("coverage", result["overall"])

    def test_text_class_is_deterministic_and_can_be_unknown(self):
        self.assertEqual(text_class(row(text="poulet et dinde")), "COMPOUND")
        self.assertEqual(text_class(row(text="")), "UNKNOWN")
