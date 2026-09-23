"""Regression coverage for verified PR #100 runtime review findings."""
from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path[:0] = [str(SCRIPTS), str(SCRIPTS / "nutrition")]

import allergen_repository
from allergen_service import evaluate_safety
from api_nutrition import AnalyzeRequest, PetIn, ProductIn, _analyze_product
from match_v1 import load_seed, match_nutrients
from match_v1_1_category import _pet_to_persona
from nutrition_readiness import evaluate_nutrition_coverage, resolve_reference_stage


class _FailingConnection:
    def __init__(self) -> None:
        self.closed = False

    def execute(self, *_args, **_kwargs):
        raise sqlite3.OperationalError("broken schema")

    def close(self) -> None:
        self.closed = True


class ReviewRuntimeRegressionTests(unittest.TestCase):
    def test_non_food_has_all_status_axes_without_claiming_nutrition_coverage(self):
        result = _analyze_product(AnalyzeRequest(
            pet=PetIn(id="pet", species="dog", age_years=3, weight_kg=10),
            product=ProductIn(id="treat", name="treat", category="treat", ingredient_list=["salmon"]),
        ))
        self.assertEqual(result["input_readiness"], {
            "input_readiness": "UNSUPPORTED",
            "reason_codes": ["CATEGORY_NOT_FOOD"],
        })
        self.assertEqual(result["nutrition_coverage"], {
            "nutrition_coverage": "UNKNOWN",
            "reason_codes": ["CATEGORY_NOT_FOOD"],
        })
        self.assertEqual(result["nutrition_comparison_status"], "NOT_APPLICABLE")
        self.assertEqual(result["analysis_status"], "READY")

    def test_non_food_safety_block_prevents_ready(self):
        result = _analyze_product(AnalyzeRequest(
            pet=PetIn(
                id="pet", species="dog", age_years=3, weight_kg=10,
                allergies=["chicken"], allergy_profile_status="KNOWN_LIST",
            ),
            product=ProductIn(id="treat", name="treat", category="treat", ingredient_list=["chicken"]),
        ))
        self.assertEqual(result["safety_status"], "SAFETY_BLOCKED")
        self.assertEqual(result["analysis_status"], "INSUFFICIENT_DATA")

    def test_senior_uses_adult_reference_but_puppy_uses_growth_reference(self):
        seed = load_seed("seed_nutrition_standard.json")
        senior = match_nutrients({"단백질_g": 19.0}, {"species": "dog", "life_stage": "senior"}, seed)
        puppy = match_nutrients({"단백질_g": 19.0}, {"species": "dog", "life_stage": "puppy"}, seed)
        self.assertEqual(senior["details"][0]["min"], 18.0)
        self.assertEqual(puppy["details"][0]["min"], 20.0)

    def test_persona_adapter_preserves_life_stage_key_consumed_by_matcher(self):
        persona = _pet_to_persona({"id": "puppy", "species": "dog", "age_years": 0.5, "weight_kg": 3})
        self.assertEqual(persona["life_stage"], "puppy")
        self.assertEqual(persona["lifestage"], "puppy")

    def test_product_allergies_cannot_override_pet_profile(self):
        result = evaluate_safety(
            {"allergies": [], "allergy_profile_status": "KNOWN_NONE"},
            {"id": "treat", "category": "treat", "allergies": ["chicken"], "ingredient_list": []},
        )
        self.assertFalse(result["excluded"])
        self.assertEqual(result["safety_status"], "NOT_APPLICABLE")

    def test_coverage_matrix_read_error_returns_unknown_not_exception(self):
        product = {
            "category": "food", "target_species": "dog", "aafco_life_stage": "ADULT",
            "nutrition_items": [],
        }
        stage = resolve_reference_stage({"life_stage": "adult", "age_years": 3})
        with patch("nutrition_readiness._comprehensive_matrix_rows", side_effect=OSError("unreadable")):
            result = evaluate_nutrition_coverage(product, "dog", stage)
        self.assertEqual(result["nutrition_coverage"], "UNKNOWN")
        self.assertEqual(result["reason_codes"], ["COMPREHENSIVE_MATRIX_REFERENCE_UNRESOLVED"])

    def test_allergen_repository_closes_connection_when_query_fails(self):
        connection = _FailingConnection()
        with patch.object(allergen_repository, "connect", return_value=connection):
            result = allergen_repository.get_refs("product", "dictionary", "pipeline")
        self.assertEqual(result, ([], "LINEAGE_INTEGRITY_ERROR"))
        self.assertTrue(connection.closed)
