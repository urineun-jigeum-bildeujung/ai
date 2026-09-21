import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SCRIPTS), str(SCRIPTS / "nutrition")]

from nutrition_readiness import (
    evaluate_nutrition_coverage,
    evaluate_nutrition_readiness,
    resolve_product_target_stage,
    resolve_reference_stage,
)
from api_nutrition import AnalyzeRequest, NutritionItemIn, PetIn, ProductIn, _analyze_product
from pipeline_p1c_v1 import compute_nutrition_comparison_status


class NutritionReadinessTests(unittest.TestCase):
    def test_cat_runtime_requirement_includes_taurine(self):
        readiness = evaluate_nutrition_readiness(
            {"category": "food", "target_species": "cat", "aafco_life_stage": "ADULT",
             "nutrition_items": [
                 {"nutrient_code": x, "value": 1}
                 for x in ["CRUDE_PROTEIN", "CRUDE_FAT", "MOISTURE", "CALCIUM", "PHOSPHORUS"]
             ]},
            "cat", resolve_reference_stage({"life_stage": "adult", "age_years": 3}),
        )
        self.assertEqual(readiness["input_readiness"], "PARTIAL")
        self.assertEqual(readiness["missing_nutrients"], ["TAURINE"])

    def test_matrix_coverage_is_descriptive_and_does_not_upgrade_readiness(self):
        product = {"category": "food", "target_species": "dog", "aafco_life_stage": "ADULT",
                   "nutrition_items": [
                       {"nutrient_code": x, "value": 1, "unit": "PERCENT", "basis": "AS_FED"}
                       for x in ["CRUDE_PROTEIN", "CRUDE_FAT", "MOISTURE", "CALCIUM", "PHOSPHORUS"]
                   ]}
        stage = resolve_reference_stage({"life_stage": "adult", "age_years": 3})
        coverage = evaluate_nutrition_coverage(product, "dog", stage)
        self.assertEqual(coverage["nutrition_coverage"], "STANDARD")
        self.assertIn("ARGININE", coverage["missing_nutrients"])
        self.assertEqual(evaluate_nutrition_readiness(product, "dog", stage)["input_readiness"], "READY")

    def test_senior_is_not_silently_resolved_as_adult_reference(self):
        result = resolve_reference_stage({"life_stage": "senior", "age_years": 10})
        self.assertEqual(result["status"], "UNSUPPORTED")
        self.assertIsNone(result["stage"])

    def test_product_stage_is_label_claim_not_pet_reference_stage(self):
        self.assertEqual(resolve_product_target_stage("ADULT")["stage"], "ADULT_MAINTENANCE")
        self.assertEqual(resolve_product_target_stage(None)["status"], "UNKNOWN")
        self.assertEqual(resolve_product_target_stage("SENIOR")["status"], "UNSUPPORTED")

    def test_senior_request_is_unknown_not_adult_comparison(self):
        result = _analyze_product(AnalyzeRequest(
            pet=PetIn(id="senior", species="dog", age_years=10, weight_kg=10, life_stage="senior"),
            product=ProductIn(
                id="food", name="food", category="food", target_species="dog", aafco_life_stage="ADULT",
                nutrition_items=[NutritionItemIn(nutrient_code=x, value=10, basis="AS_FED") for x in
                                 ["CRUDE_PROTEIN", "CRUDE_FAT", "MOISTURE", "CALCIUM", "PHOSPHORUS"]],
            ),
        ))
        self.assertEqual(result["pet_reference_stage"]["status"], "UNSUPPORTED")
        self.assertEqual(result["nutrition_comparison_status"], "UNKNOWN")
        self.assertEqual(result["analysis_status"], "INSUFFICIENT_DATA")

    def test_explicit_unsupported_reference_never_uses_dog_adult_essentials(self):
        result = compute_nutrition_comparison_status(
            [{"product_id": "senior", "nutrient_code": "CRUDE_PROTEIN", "nias_compare_status": "IN_RANGE"}],
            {"senior": "DOG"}, {"senior": "SENIOR"},
        )["senior"]
        self.assertEqual(result["nutrition_comparison_status"], "UNKNOWN")
        self.assertEqual(result["warning_codes"], ["UNSUPPORTED_REFERENCE_COMBINATION"])
        self.assertEqual(result["essential_nutrients"], [])


if __name__ == "__main__":
    unittest.main()
