import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path[:0] = [str(SCRIPTS), str(SCRIPTS / "nutrition")]

from api_nutrition import app
from nutrition.product_input_adapter import load_product_input


class PersistedProductAdapterTests(unittest.TestCase):
    fixture_id = "0064992280178"

    def test_fixture_is_loaded_without_imputation(self):
        loaded = load_product_input(self.fixture_id)
        product = loaded["product"]
        self.assertEqual(product["id"], self.fixture_id)
        self.assertEqual(loaded["provenance"]["product_source_dataset"], "OPFF")
        self.assertEqual(loaded["provenance"]["nutrition_item_count"], 4)
        self.assertTrue(loaded["provenance"]["moisture_present"])
        self.assertTrue(loaded["provenance"]["target_species_present"])
        self.assertFalse(loaded["provenance"]["life_stage_present"])
        self.assertIsNone(product["aafco_life_stage"])
        self.assertEqual(loaded["provenance"]["unit_basis_contract"], "EXPLICIT_PER_ROW")
        self.assertEqual(loaded["provenance"]["unit_basis_missing_rows"], 0)

    def test_product_id_route_preserves_missing_life_stage_as_insufficient(self):
        from fastapi.testclient import TestClient
        client = TestClient(app)
        response = client.post("/api/nutrition/analyze/by-product-id", json={
            "pet": {"id": "fixture-cat", "species": "cat", "age_years": 3, "weight_kg": 4,
                    "allergies": [], "allergy_profile_status": "KNOWN_NONE", "life_stage": "adult"},
            "product_id": self.fixture_id,
        })
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["analysis_engine"], "pipeline_p1c_v1")
        self.assertEqual(result["nutrition_comparison_status"], "UNKNOWN")
        self.assertEqual(result["safety_status"], "SAFETY_DATA_INSUFFICIENT")
        self.assertEqual(result["analysis_status"], "INSUFFICIENT_DATA")
        self.assertFalse(result["input_provenance"]["life_stage_present"])

    def test_oem_without_source_confirmed_category_fail_closes(self):
        from fastapi.testclient import TestClient
        response = TestClient(app).post("/api/nutrition/analyze/by-product-id", json={
            "pet": {"id": "dog", "species": "dog", "age_years": 3, "weight_kg": 10},
            "product_id": "OFF_KR_0723633808309",
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["source_validation_status"], "INVALID_SOURCE_DATA")
        self.assertEqual(response.json()["analysis_status"], "INSUFFICIENT_DATA")

    def test_invalid_loaded_records_are_structured_not_500(self):
        from fastapi.testclient import TestClient
        base = {
            "id": "bad", "name": "bad", "category": "food", "ingredient_list": [],
            "nutrition_items": [{"nutrient_code": "CRUDE_PROTEIN", "value": 20, "unit": "PERCENT", "basis": "AS_FED"}],
            "target_species": "dog", "aafco_life_stage": "ADULT",
        }
        invalid_variants = [
            {**base, "nutrition_items": [{**base["nutrition_items"][0], "nutrient_code": "bad-code"}]},
            {**base, "nutrition_items": [{**base["nutrition_items"][0], "basis": "UNKNOWN"}]},
            {**base, "target_species": "fox"},
            {**base, "name": None},
        ]
        client = TestClient(app)
        for product in invalid_variants:
            with self.subTest(product=product), patch("api_nutrition.load_product_input", return_value={
                "product": product, "provenance": {"product_source_dataset": "TEST"},
            }):
                response = client.post("/api/nutrition/analyze/by-product-id", json={
                    "pet": {"id": "dog", "species": "dog", "age_years": 3, "weight_kg": 10},
                    "product_id": "bad",
                })
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["source_validation_status"], "INVALID_SOURCE_DATA")
                self.assertEqual(response.json()["analysis_status"], "INSUFFICIENT_DATA")


if __name__ == "__main__":
    unittest.main()
