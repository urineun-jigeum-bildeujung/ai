"""P0 domain-contract regressions: safety outcomes cannot vary by entry path."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path[:0] = [str(SCRIPTS), str(SCRIPTS / "nutrition")]

from allergen_service import canonicalize_profile, evaluate_safety, product_allergen_refs
from api_nutrition import AnalyzeRequest, PetIn, ProductIn, NutritionItemIn, _analyze_product
from match_v1_1_category import match_by_category
from pipeline_p1c_v1 import compute_nutrition_comparison_status, normalize_basis


def pet(**overrides):
    return {"id": "p", "species": "dog", "age_years": 3, "weight_kg": 10,
            "allergies": ["chicken"], "allergy_profile_status": "KNOWN_LIST", **overrides}


def food(**overrides):
    return {"id": "f", "name": "f", "category": "food", "target_species": "dog",
            "aafco_life_stage": "MAINTENANCE", "ingredient_list": ["salmon"], **overrides}


class P0Contracts(unittest.TestCase):
    def test_supported_profile_aliases_are_one_code_and_block(self):
        for alias in ["chicken", "CHICKEN", "닭고기", "poulet"]:
            self.assertEqual(canonicalize_profile([alias], "KNOWN_LIST")["codes"], ["chicken"])
            self.assertEqual(evaluate_safety(pet(allergies=[alias]), food(ingredient_list=["chicken"]))["safety_status"], "SAFETY_BLOCKED")

    def test_profile_states_fail_closed(self):
        self.assertEqual(evaluate_safety(pet(allergy_profile_status="UNKNOWN"), food())["safety_status"], "SAFETY_DATA_INSUFFICIENT")
        self.assertEqual(evaluate_safety(pet(allergies=[], allergy_profile_status="KNOWN_NONE"), food(ingredient_list=["unlisted"]))["safety_status"], "NOT_APPLICABLE")
        self.assertEqual(evaluate_safety(pet(allergies=["not-a-code"]), food())["safety_status"], "SAFETY_DATA_INSUFFICIENT")

    def test_compound_phrase_preserves_chicken_and_unresolved_turkey_evidence(self):
        refs = product_allergen_refs("p", ["protéines déshydratées de poulet et de dinde"])
        self.assertEqual({r["allergen_code"] for r in refs}, {"chicken", "turkey"})
        self.assertEqual(next(r for r in refs if r["allergen_code"] == "chicken")["mapping_method"], "CANONICAL_ALIAS")
        self.assertEqual(next(r for r in refs if r["allergen_code"] == "turkey")["mapping_method"], "UNRESOLVED")

    def test_api_and_direct_matcher_have_same_safety(self):
        p, f = pet(), food(ingredient_list=["unlisted"])
        direct = match_by_category(p, f)
        api = _analyze_product(AnalyzeRequest(pet=PetIn(**p), product=ProductIn(**f)))
        self.assertEqual(direct["safety_status"], api["safety_status"])
        self.assertEqual(direct["excluded"], api["excluded"])

    def test_species_and_life_stage_fail_closed(self):
        self.assertEqual(evaluate_safety(pet(), food(target_species="cat"))["exclude_reasons"], ["SPECIES_MISMATCH"])
        self.assertEqual(evaluate_safety(pet(), food(target_species=None))["safety_status"], "SAFETY_DATA_INSUFFICIENT")
        self.assertEqual(evaluate_safety(pet(age_years=.5), food())["exclude_reasons"], ["LIFE_STAGE_MISMATCH"])
        self.assertEqual(evaluate_safety(pet(), food(aafco_life_stage=None))["safety_status"], "SAFETY_DATA_INSUFFICIENT")

    def test_duplicate_status_is_order_invariant(self):
        base = [{"product_id": "x", "nutrient_code": x, "nias_compare_status": "IN_RANGE", "basis_normalized_value": 1}
                for x in ["CRUDE_PROTEIN", "CRUDE_FAT", "MOISTURE", "CALCIUM", "PHOSPHORUS"]]
        conflict = {"product_id": "x", "nutrient_code": "CRUDE_PROTEIN", "nias_compare_status": "OUT_OF_RANGE", "basis_normalized_value": 9}
        a = compute_nutrition_comparison_status(base + [conflict])["x"]["nutrition_comparison_status"]
        b = compute_nutrition_comparison_status([conflict] + base)["x"]["nutrition_comparison_status"]
        self.assertEqual((a, b), ("UNKNOWN", "UNKNOWN"))

    def test_as_fed_requires_moisture_for_dm(self):
        rows = [{"product_id": "x", "nutrient_code": "MOISTURE", "aligned_value": 10, "basis": "AS_FED"},
                {"product_id": "x", "nutrient_code": "CRUDE_PROTEIN", "aligned_value": 18, "basis": "AS_FED"}]
        self.assertEqual(normalize_basis(rows)[1]["basis_normalized_value"], 20)
        self.assertIsNone(normalize_basis(rows[1:])[0]["basis_normalized_value"])
