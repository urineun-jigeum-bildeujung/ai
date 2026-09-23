"""HTTP-boundary regressions for the current local Nutrition runtime.

These tests intentionally exercise FastAPI through ``TestClient``.  They do
not claim AWS Service DB, result-store, or FE production E2E coverage.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path[:0] = [str(SCRIPTS), str(SCRIPTS / "nutrition")]

from api_nutrition import app


RUNTIME_ARTIFACTS = (
    ROOT / "data" / "raw" / "seed_feed_codes.json",
    ROOT / "data" / "raw" / "seed_13_guaranteed_analysis.json",
    ROOT / "data" / "raw" / "seed_14_nutrition_reference_v5.json",
    ROOT / "data" / "raw" / "seed_9_placeholder_feed_opff.json",
    ROOT / "data" / "raw" / "seed_9b_off_korean_oem.json",
    ROOT / "data" / "raw" / "seed_9_global_brands_v2.json",
    ROOT / "data" / "processed" / "allergen_evidence_catalog_p2.db",
    ROOT / "data" / "processed" / "required_nutrient_matrix_v2.csv",
    ROOT / "data" / "eval" / "verified_evidence_gold_v1.json",
)


def dog_pet(**overrides: object) -> dict[str, object]:
    return {
        "id": "e2e-dog",
        "species": "dog",
        "age_years": 3,
        "weight_kg": 10,
        "allergies": [],
        "allergy_profile_status": "KNOWN_NONE",
        "life_stage": "adult",
        **overrides,
    }


def minimum_dog_items() -> list[dict[str, object]]:
    return [
        {"nutrient_code": "CRUDE_PROTEIN", "value": 25, "unit": "PERCENT", "basis": "AS_FED"},
        {"nutrient_code": "CRUDE_FAT", "value": 12, "unit": "PERCENT", "basis": "AS_FED"},
        {"nutrient_code": "MOISTURE", "value": 10, "unit": "PERCENT", "basis": "AS_FED"},
        {"nutrient_code": "CALCIUM", "value": 1, "unit": "PERCENT", "basis": "AS_FED"},
        {"nutrient_code": "PHOSPHORUS", "value": 0.8, "unit": "PERCENT", "basis": "AS_FED"},
    ]


def dog_food(**overrides: object) -> dict[str, object]:
    return {
        "id": "e2e-dog-food",
        "name": "E2E dog food",
        "category": "food",
        "ingredient_list": ["salmon"],
        "target_species": "dog",
        "aafco_life_stage": "ADULT",
        "nutrition_items": minimum_dog_items(),
        **overrides,
    }


class RuntimeE2ETests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def post_analyze(self, pet: dict[str, object], product: dict[str, object]) -> dict[str, object]:
        response = self.client.post("/api/nutrition/analyze", json={"pet": pet, "product": product})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def assert_status_axes(self, body: dict[str, object]) -> None:
        self.assertIn("input_readiness", body)
        self.assertIn("nutrition_coverage", body)
        self.assertIn("nutrition_comparison_status", body)
        self.assertIn("safety_status", body)
        self.assertIn("analysis_status", body)

    def test_health_lists_available_and_future_routes_separately(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        registered_paths = {route.path for route in app.routes}
        self.assertEqual(body["service"], "nutrition")
        self.assertIn("POST /api/nutrition/analyze", body["endpoints"])
        self.assertNotIn("POST /api/nutrition/compare", body["endpoints"])
        self.assertTrue({
            "/health",
            "/api/nutrition/analyze",
            "/api/nutrition/analyze/by-product-id",
            "/api/nutrition/safety",
            "/api/nutrition/report",
            "/api/nutrition/compare",
        }.issubset(registered_paths))
        self.assertEqual(body["future_endpoints"], [{
            "method": "POST", "path": "/api/nutrition/compare",
            "status": "NOT_IMPLEMENTED", "http_status": 501,
        }])

    def test_food_analysis_returns_all_five_status_axes(self) -> None:
        body = self.post_analyze(dog_pet(), dog_food())
        self.assert_status_axes(body)
        self.assertTrue(body["p0_d_applied"])
        self.assertEqual(body["analysis_engine"], "pipeline_p1c_v1")
        self.assertEqual(body["input_readiness"]["input_readiness"], "READY")

    def test_missing_nutrition_evidence_never_returns_ready_or_pass(self) -> None:
        body = self.post_analyze(dog_pet(), dog_food(id="e2e-missing", nutrition_items=[]))
        self.assert_status_axes(body)
        self.assertEqual(body["input_readiness"]["input_readiness"], "INSUFFICIENT_DATA")
        self.assertEqual(body["nutrition_comparison_status"], "UNKNOWN")
        self.assertEqual(body["analysis_status"], "INSUFFICIENT_DATA")
        self.assertIsNone(body["aafco_pass"])

    def test_allergy_conflict_blocks_despite_nutrition_result(self) -> None:
        body = self.post_analyze(
            dog_pet(allergies=["chicken"], allergy_profile_status="KNOWN_LIST"),
            dog_food(id="e2e-allergy-conflict", ingredient_list=["chicken"]),
        )
        self.assert_status_axes(body)
        self.assertEqual(body["safety_status"], "SAFETY_BLOCKED")
        self.assertEqual(body["analysis_status"], "INSUFFICIENT_DATA")

    def test_missing_allergy_evidence_fail_closes(self) -> None:
        body = self.post_analyze(
            dog_pet(allergies=["chicken"], allergy_profile_status="KNOWN_LIST"),
            dog_food(id="e2e-allergy-insufficient", ingredient_list=["unlisted-protein"]),
        )
        self.assert_status_axes(body)
        self.assertEqual(body["safety_status"], "SAFETY_DATA_INSUFFICIENT")
        self.assertEqual(body["analysis_status"], "INSUFFICIENT_DATA")

    def test_species_mismatch_is_safety_blocked(self) -> None:
        body = self.post_analyze(dog_pet(), dog_food(id="e2e-cat-food", target_species="cat"))
        self.assert_status_axes(body)
        self.assertEqual(body["input_readiness"]["species_status"], "MISMATCH")
        self.assertEqual(body["safety_status"], "SAFETY_BLOCKED")
        self.assertEqual(body["analysis_status"], "INSUFFICIENT_DATA")

    def test_life_stage_paths_keep_growth_and_senior_fail_close_distinct(self) -> None:
        puppy = self.post_analyze(
            dog_pet(id="e2e-puppy", age_years=0.5, life_stage="puppy"),
            dog_food(id="e2e-puppy-food", aafco_life_stage="GROWTH"),
        )
        kitten_items = minimum_dog_items() + [
            {"nutrient_code": "TAURINE", "value": 0.2, "unit": "PERCENT", "basis": "AS_FED"},
        ]
        kitten = self.post_analyze(
            dog_pet(id="e2e-kitten", species="cat", age_years=0.5, life_stage="kitten"),
            dog_food(
                id="e2e-kitten-food", target_species="cat", aafco_life_stage="GROWTH",
                nutrition_items=kitten_items,
            ),
        )
        senior = self.post_analyze(
            dog_pet(id="e2e-senior", age_years=10, life_stage="senior"),
            dog_food(id="e2e-senior-food"),
        )
        self.assertEqual(puppy["pet_reference_stage"]["stage"], "GROWTH_REPRODUCTION")
        self.assertEqual(kitten["pet_reference_stage"]["stage"], "GROWTH_REPRODUCTION")
        # P0-D has no senior reference.  Its documented current behavior is
        # fail-close, not an implicit adult/maintenance substitution.
        self.assertEqual(senior["pet_reference_stage"]["status"], "UNSUPPORTED")
        self.assertEqual(senior["nutrition_comparison_status"], "UNKNOWN")
        self.assertEqual(senior["analysis_status"], "INSUFFICIENT_DATA")

    def test_non_food_has_stable_not_applicable_contract(self) -> None:
        body = self.post_analyze(dog_pet(), dog_food(id="e2e-treat", category="treat"))
        self.assert_status_axes(body)
        self.assertFalse(body["p0_d_applied"])
        self.assertEqual(body["nutrition_comparison_status"], "NOT_APPLICABLE")
        self.assertEqual(body["input_readiness"]["input_readiness"], "UNSUPPORTED")
        self.assertEqual(body["nutrition_coverage"]["nutrition_coverage"], "UNKNOWN")

    def test_persisted_local_product_and_unknown_id_routes(self) -> None:
        response = self.client.post("/api/nutrition/analyze/by-product-id", json={
            "pet": dog_pet(id="e2e-persisted", species="cat", weight_kg=4),
            "product_id": "0064992280178",
        })
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assert_status_axes(body)
        self.assertEqual(body["analysis_engine"], "pipeline_p1c_v1")
        self.assertIn("input_provenance", body)

        missing = self.client.post("/api/nutrition/analyze/by-product-id", json={
            "pet": dog_pet(), "product_id": "does-not-exist",
        })
        self.assertEqual(missing.status_code, 404)

    def test_invalid_persisted_source_returns_structured_fail_close(self) -> None:
        response = self.client.post("/api/nutrition/analyze/by-product-id", json={
            "pet": dog_pet(), "product_id": "OFF_KR_0723633808309",
        })
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["source_validation_status"], "INVALID_SOURCE_DATA")
        self.assertEqual(body["input_readiness"]["input_readiness"], "INSUFFICIENT_DATA")
        self.assertEqual(body["analysis_status"], "INSUFFICIENT_DATA")

    def test_same_http_input_has_same_decision_fields(self) -> None:
        payload = {"pet": dog_pet(), "product": dog_food(id="e2e-deterministic")}
        responses = [self.client.post("/api/nutrition/analyze", json=payload) for _ in range(3)]
        for response in responses:
            self.assertEqual(response.status_code, 200, response.text)
        decisions = [
            {
                "input_readiness": response.json()["input_readiness"],
                "nutrition_coverage": response.json()["nutrition_coverage"],
                "nutrition_comparison_status": response.json()["nutrition_comparison_status"],
                "safety_status": response.json()["safety_status"],
                "analysis_status": response.json()["analysis_status"],
                "aafco_pass": response.json()["aafco_pass"],
            }
            for response in responses
        ]
        self.assertEqual(decisions[0], decisions[1])
        self.assertEqual(decisions[1], decisions[2])

    def test_supporting_endpoints_and_runtime_artifacts_are_available(self) -> None:
        self.assertTrue(all(path.is_file() for path in RUNTIME_ARTIFACTS))
        request = {"pet": dog_pet(), "product": dog_food(id="e2e-supporting")}
        safety = self.client.post("/api/nutrition/safety", json=request)
        report = self.client.post("/api/nutrition/report", json=request)
        compare = self.client.post("/api/nutrition/compare")
        self.assertEqual(safety.status_code, 200, safety.text)
        self.assertEqual(report.status_code, 200, report.text)
        self.assertEqual(compare.status_code, 501)
        self.assertIn("NOT_IMPLEMENTED", compare.json()["detail"])


if __name__ == "__main__":
    unittest.main()
