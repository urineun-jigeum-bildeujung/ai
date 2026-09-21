"""Regression tests for P0 nutrition safety invariants."""
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "nutrition"))

from match_v1_1_category import match_by_category
from match_v1 import load_seed, match_allergens
from api_nutrition import AnalyzeRequest, PetIn, ProductIn, NutritionItemIn, _analyze_product
from pipeline_p1c_v1 import (
    _canonical_status_8step,
    classify_dry_wet,
    compute_nutrition_comparison_status,
    normalize_basis,
)


class P0NutritionSafetyTests(unittest.TestCase):
    def test_allergen_filter_requires_normalized_exact_match(self):
        seed = load_seed("seed_feed_codes.json")
        persona = {"allergies": ["chicken"]}
        exact = match_allergens(["chicken"], seed, persona)
        partial = match_allergens(["chick"], seed, persona)
        self.assertTrue(exact["excluded"])
        self.assertTrue(partial["excluded"])
        self.assertEqual(partial["unmapped_ingredients"], ["chick"])

    def test_unmapped_ingredients_block_allergy_profile_as_insufficient_data(self):
        result = _analyze_product(AnalyzeRequest(
            pet=PetIn(id="dog", species="dog", age_years=3, weight_kg=10, allergies=["chicken"]),
            product=ProductIn(id="unknown-ingredient", name="unknown ingredient", ingredient_list=["unlisted-protein"], target_species="dog", aafco_life_stage="MAINTENANCE"),
        ))
        self.assertTrue(result["excluded"])
        self.assertEqual(result["safety_status"], "SAFETY_DATA_INSUFFICIENT")
        self.assertEqual(result["allergy_check_status"], "INSUFFICIENT_DATA")
        self.assertEqual(result["unmapped_ingredients"], ["unlisted-protein"])

    def test_fully_mapped_non_conflicting_ingredients_keep_existing_policy(self):
        result = _analyze_product(AnalyzeRequest(
            pet=PetIn(id="dog", species="dog", age_years=3, weight_kg=10, allergies=["chicken"]),
            product=ProductIn(id="mapped-food", name="mapped food", ingredient_list=["salmon", "rice"], target_species="dog", aafco_life_stage="MAINTENANCE"),
        ))
        self.assertFalse(result["excluded"])
        self.assertEqual(result["safety_status"], "NO_CONFLICT_DETECTED")
        self.assertEqual(result["unmapped_ingredients"], [])

    def test_no_allergy_profile_keeps_existing_policy_for_unmapped_ingredients(self):
        result = _analyze_product(AnalyzeRequest(
            pet=PetIn(id="dog", species="dog", age_years=3, weight_kg=10),
            product=ProductIn(id="unmapped-no-allergy", name="unmapped", ingredient_list=["unlisted-protein"], target_species="dog", aafco_life_stage="MAINTENANCE"),
        ))
        self.assertFalse(result["excluded"])
        self.assertEqual(result["safety_status"], "NOT_APPLICABLE")

    def test_food_api_uses_p0d_pipeline_not_legacy_matcher(self):
        result = _analyze_product(AnalyzeRequest(
            pet=PetIn(id="dog", species="dog", age_years=3, weight_kg=10),
            product=ProductIn(
                id="labelled-food",
                name="labelled food",
                ingredient_list=["salmon"],
                target_species="dog",
                aafco_life_stage="MAINTENANCE",
                nutrition_items=[
                    NutritionItemIn(nutrient_code="CRUDE_PROTEIN", value=25),
                    NutritionItemIn(nutrient_code="CRUDE_FAT", value=12),
                    NutritionItemIn(nutrient_code="MOISTURE", value=10),
                    NutritionItemIn(nutrient_code="CALCIUM", value=1),
                    NutritionItemIn(nutrient_code="PHOSPHORUS", value=0.8),
                ],
            ),
        ))
        self.assertEqual(result["analysis_engine"], "pipeline_p1c_v1")
        self.assertTrue(result["p0_d_applied"])
        self.assertEqual(result["nutrition_comparison_status"], "TRUE")

    def test_missing_ingredients_are_safety_data_insufficient_not_clear(self):
        result = _analyze_product(AnalyzeRequest(
            pet=PetIn(id="dog", species="dog", age_years=3, weight_kg=10, allergies=["chicken"]),
            product=ProductIn(id="unknown-food", name="unknown food", ingredient_list=[], target_species="dog", aafco_life_stage="MAINTENANCE"),
        ))
        self.assertTrue(result["excluded"])
        self.assertEqual(result["safety_status"], "SAFETY_DATA_INSUFFICIENT")
        self.assertEqual(result["allergy_check_status"], "INSUFFICIENT_DATA")

    def test_empty_analysis_never_returns_aafco_pass(self):
        result = match_by_category(
            {"id": "pet", "species": "dog", "age_years": 3, "weight_kg": 10},
            {"name": "empty", "category": "food", "guaranteed_analysis": {}, "aafco_life_stage": "MAINTENANCE"},
        )
        self.assertIsNone(result["aafco_pass"])
        self.assertEqual(result["nutrition_comparison_status"], "UNKNOWN")

    def test_dog_taurine_is_not_applicable_using_species_map(self):
        row = {"product_id": "dog-product", "nutrient_code": "TAURINE", "nias_compare_status": "NO_REF"}
        self.assertEqual(_canonical_status_8step(row, {"dog-product": "DOG"}), "NOT_APPLICABLE")

    def test_missing_moisture_mineral_is_invalid_before_range_result(self):
        row = {
            "product_id": "x", "nutrient_code": "CALCIUM", "nias_compare_status": "IN_RANGE",
            "basis_normalization_status": "SKIPPED_NO_MOISTURE",
        }
        self.assertEqual(_canonical_status_8step(row), "INVALID")

    def test_invalid_moisture_never_produces_a_dry_matter_value(self):
        for moisture in (-1, 100, 101):
            rows = [
                {"product_id": "x", "nutrient_code": "MOISTURE", "aligned_value": moisture, "basis": "AS_FED"},
                {"product_id": "x", "nutrient_code": "CALCIUM", "aligned_value": 1.0, "basis": "AS_FED"},
            ]
            self.assertEqual(classify_dry_wet(rows)[0]["product_form"], "UNKNOWN")
            normalized = normalize_basis(rows)
            self.assertIsNone(normalized[1]["basis_normalized_value"])
            self.assertTrue(normalized[1]["basis_invalid"])

    def test_unknown_product_status_remains_none_in_legacy_boolean(self):
        result = compute_nutrition_comparison_status([
            {"product_id": "x", "nutrient_code": "CRUDE_PROTEIN", "nias_compare_status": "NO_VALUE"},
        ])
        self.assertEqual(result["x"]["nutrition_comparison_status"], "UNKNOWN")
        self.assertIsNone(result["x"]["aafco_pass"])


if __name__ == "__main__":
    unittest.main()
