"""Parity P0 regressions for raw NIAS applicability preservation."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SCRIPTS), str(SCRIPTS / "nutrition")]

from nutrition.reference_parity import build_artifact, select_reference, validate_rows
from pipeline_p1c_v1 import compare_nias


class NutritionReferenceParityP0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = build_artifact()
        cls.rows = cls.artifact["rows"]

    def select(self, **kwargs):
        return select_reference(self.rows, **kwargs)

    def test_generator_invariants(self):
        validate_rows(self.rows)
        self.assertEqual(self.artifact["artifact_version"], "NIAS_2024_PARITY_P0_V1")

    def test_cat_taurine_adult_dry_and_canned_dm_are_distinct(self):
        dry = self.select(species="CAT", life_stage="ADULT_MAINTENANCE", nutrient_code="TAURINE", basis="DRY_MATTER", reference_form="DRY")
        canned = self.select(species="CAT", life_stage="ADULT_MAINTENANCE", nutrient_code="TAURINE", basis="DRY_MATTER", reference_form="CANNED")
        self.assertEqual((dry["status"], dry["min_value"]), ("SELECTED", 0.1))
        self.assertEqual((canned["status"], canned["min_value"]), ("SELECTED", 0.2))

    def test_cat_taurine_growth_dry_and_canned_dm_are_distinct(self):
        dry = self.select(species="CAT", life_stage="GROWTH_REPRODUCTION", nutrient_code="TAURINE", basis="DRY_MATTER", reference_form="DRY")
        canned = self.select(species="CAT", life_stage="GROWTH_REPRODUCTION", nutrient_code="TAURINE", basis="DRY_MATTER", reference_form="CANNED")
        self.assertEqual((dry["min_value"], canned["min_value"]), (0.1, 0.2))

    def test_cat_taurine_energy_dry_and_canned_are_distinct(self):
        dry = self.select(species="CAT", life_stage="ADULT_MAINTENANCE", nutrient_code="TAURINE", basis="PER_1000KCAL", reference_form="DRY")
        canned = self.select(species="CAT", life_stage="ADULT_MAINTENANCE", nutrient_code="TAURINE", basis="PER_1000KCAL", reference_form="CANNED")
        self.assertEqual((dry["min_value"], canned["min_value"]), (0.25, 0.5))

    def test_dm_and_energy_rules_do_not_overwrite_each_other(self):
        dm = self.select(species="CAT", life_stage="GROWTH_REPRODUCTION", nutrient_code="TAURINE", basis="DRY_MATTER", reference_form="CANNED")
        energy = self.select(species="CAT", life_stage="GROWTH_REPRODUCTION", nutrient_code="TAURINE", basis="PER_1000KCAL", reference_form="CANNED")
        self.assertEqual((dm["min_value"], energy["min_value"]), (0.2, 0.5))

    def test_cat_copper_dry_and_canned_are_distinct(self):
        dry = self.select(species="CAT", life_stage="GROWTH_REPRODUCTION", nutrient_code="COPPER", basis="DRY_MATTER", reference_form="DRY")
        canned = self.select(species="CAT", life_stage="GROWTH_REPRODUCTION", nutrient_code="COPPER", basis="DRY_MATTER", reference_form="CANNED")
        self.assertEqual((dry["min_value"], canned["min_value"]), (1.0, 0.84))

    def test_unknown_form_fails_closed_when_form_specific_rule_exists(self):
        result = self.select(species="CAT", life_stage="ADULT_MAINTENANCE", nutrient_code="TAURINE", basis="DRY_MATTER", reference_form="UNKNOWN")
        self.assertEqual((result["status"], result["reason_code"]), ("NO_REF", "REFERENCE_FORM_REQUIRED"))

    def test_generic_nutrient_is_not_blocked_by_product_form(self):
        result = self.select(species="CAT", life_stage="ADULT_MAINTENANCE", nutrient_code="CALCIUM", basis="DRY_MATTER", reference_form="UNKNOWN")
        self.assertEqual(result["status"], "SELECTED")

    def test_dog_early_and_late_growth_are_independent_minima(self):
        early = self.select(species="DOG", life_stage="GROWTH_REPRODUCTION", life_stage_detail="EARLY_GROWTH_AND_REPRODUCTION", nutrient_code="CRUDE_PROTEIN", basis="DRY_MATTER", reference_form="UNKNOWN")
        late = self.select(species="DOG", life_stage="GROWTH_REPRODUCTION", life_stage_detail="LATE_GROWTH", nutrient_code="CRUDE_PROTEIN", basis="DRY_MATTER", reference_form="UNKNOWN")
        self.assertEqual((early["min_value"], early["max_value"]), (22.5, None))
        self.assertEqual((late["min_value"], late["max_value"]), (20.0, None))

    def test_dog_growth_without_detail_fails_closed(self):
        result = self.select(species="DOG", life_stage="GROWTH_REPRODUCTION", nutrient_code="CRUDE_PROTEIN", basis="DRY_MATTER", reference_form="UNKNOWN")
        self.assertEqual((result["status"], result["reason_code"]), ("NO_REF", "REFERENCE_LIFE_STAGE_DETAIL_REQUIRED"))

    def test_minimum_and_maximum_semantics_are_separate(self):
        selected = self.select(species="DOG", life_stage="ADULT_MAINTENANCE", nutrient_code="MOISTURE", basis="AS_FED", reference_form="DRY")
        self.assertEqual(selected["status"], "SELECTED")
        self.assertEqual({row["threshold_type"] for row in selected["rows"]}, {"MINIMUM", "MAXIMUM"})
        self.assertLess(selected["min_value"], selected["max_value"])

    def test_compare_nias_uses_explicit_form_before_moisture_form(self):
        item = {"product_id": "cat", "nutrient_code": "TAURINE", "basis_normalized_value": 0.15,
                "basis_normalized_basis": "DRY_MATTER", "product_form": "WET"}
        result, _, _ = compare_nias([item], self.rows, {"cat": "CAT"}, {"cat": "ADULT_MAINTENANCE"}, None, {"cat": "DRY_FOOD"})
        self.assertEqual((result[0]["reference_form"], result[0]["nias_compare_status"]), ("DRY", "IN_RANGE"))

    def test_compare_nias_unknown_form_is_no_ref_not_out_of_range(self):
        item = {"product_id": "cat", "nutrient_code": "TAURINE", "basis_normalized_value": 0.15,
                "basis_normalized_basis": "DRY_MATTER", "product_form": "UNKNOWN"}
        result, _, _ = compare_nias([item], self.rows, {"cat": "CAT"}, {"cat": "ADULT_MAINTENANCE"})
        self.assertEqual((result[0]["nias_compare_status"], result[0]["reference_reason_code"]), ("NO_REF", "REFERENCE_FORM_REQUIRED"))

    def test_internal_pipeline_e2e_reference_fixtures(self):
        """Synthetic internal fixtures; they are not persisted production products."""
        cases = [
            ("cat-dry", "CAT", "ADULT_MAINTENANCE", None, "TAURINE", "DRY", 0.15, "IN_RANGE"),
            ("cat-canned", "CAT", "ADULT_MAINTENANCE", None, "TAURINE", "CANNED", 0.15, "OUT_OF_RANGE"),
            ("cat-unknown", "CAT", "ADULT_MAINTENANCE", None, "TAURINE", None, 0.15, "NO_REF"),
            ("dog-early", "DOG", "GROWTH_REPRODUCTION", "EARLY_GROWTH_AND_REPRODUCTION", "CRUDE_PROTEIN", None, 21.0, "OUT_OF_RANGE"),
            ("dog-late", "DOG", "GROWTH_REPRODUCTION", "LATE_GROWTH", "CRUDE_PROTEIN", None, 21.0, "IN_RANGE"),
        ]
        for pid, species, stage, detail, nutrient, explicit_form, value, expected in cases:
            item = {"product_id": pid, "nutrient_code": nutrient, "basis_normalized_value": value,
                    "basis_normalized_basis": "DRY_MATTER", "product_form": "UNKNOWN"}
            compared, _, _ = compare_nias([item], self.rows, {pid: species}, {pid: stage},
                                            {pid: detail} if detail else None,
                                            {pid: explicit_form} if explicit_form else None)
            self.assertEqual(compared[0]["nias_compare_status"], expected, pid)


if __name__ == "__main__":
    unittest.main()
