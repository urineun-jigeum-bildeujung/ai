"""Additive Service-ID Feeding: policy gaps never become a dose."""
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/nutrition")]
import api_nutrition as api
from mer_coefficient_policy import resolve_mer_coefficient
from mock_feeding_fixture import mock_energy
from mock_integration_fixture import build_mock_fixture


def pet(**changes):
    return {"id": 10, "species": "DOG", "age": 2, "weight": 8,
            "allergies": [], "allergy_profile_status": "KNOWN_NONE", "life_stage": "adult",
            "bcs": 3, "is_neutered": True, **changes}


def product(**changes):
    return {"id": 1, "sku": "MOCK-0001", "product_name": "synthetic",
            "category_code": "FOOD", "subcategory_code": "DRY_FOOD", "target_age_group": "ADULT",
            "target_species": ["DOG"], "ingredient_codes": [], "allergen_flags": [],
            "caution_codes": [], **changes}


@pytest.mark.parametrize("species", ["DOG", "CAT"])
def test_unknown_coefficient_preserves_ready_nutrition_and_reports_rer(species):
    result = api.analyze_service_records(pet(species=species), product(target_species=[species]))
    assert result["nutrition_comparison_status"] == "TRUE"
    assert result["aafco_pass"] is True and result["safety_status"] == "NOT_APPLICABLE"
    feeding = result["feeding"]
    assert feeding["status"] == "INSUFFICIENT_DATA"
    assert feeding["rer_kcal_per_day"] == 333.0
    assert feeding["mer_kcal_per_day"] is None and feeding["daily_serving_g"] is None
    assert feeding["coefficient"] is None and feeding["coefficient_version"] is None
    assert feeding["reason_codes"] == ["MER_COEFFICIENT_UNRESOLVED"]
    assert feeding["energy_basis"] == "AS_FED"
    assert feeding["energy_source_type"] == "MOCK_INTEGRATION_FIXTURE"


def test_mock_age_fallback_cannot_resolve_feeding_stage():
    result = api.analyze_service_records(pet(life_stage=None), product())
    assert result["nutrition_comparison_status"] == "TRUE"
    assert result["input_provenance"]["pet_reference_stage_source"] == "AGE_RULE_ELIGIBLE"
    assert "PET_LIFE_STAGE_UNRESOLVED" in result["feeding"]["reason_codes"]
    assert result["feeding"]["daily_serving_g"] is None


@pytest.mark.parametrize("code", ["CHOCOLATE_CACAO", "XYLITOL"])
def test_safety_block_keeps_nutrition_true_and_hides_dose(code):
    result = api.analyze_service_records(pet(), product(caution_codes=[code]))
    assert result["nutrition_comparison_status"] == "TRUE"
    assert result["safety_status"] == "SAFETY_BLOCKED" and result["excluded"]
    assert result["feeding"]["status"] == "BLOCKED"
    assert result["feeding"]["daily_serving_g"] is None
    assert "FEEDING_SAFETY_EXCLUDED" in result["feeding"]["reason_codes"]


@pytest.mark.parametrize("product_id,expected", [(1, "FIXTURE_READY"), (11, "FIXTURE_PARTIAL"), (29, "FIXTURE_UNAVAILABLE")])
def test_fixture_completeness_is_not_changed_by_feeding(product_id, expected):
    result = api.analyze_service_records(pet(), product(id=product_id, sku=f"MOCK-{product_id:04d}"))
    assert result["input_provenance"]["fixture_status"] == expected
    assert result["feeding"]["daily_serving_g"] is None
    if product_id == 29:
        assert result["feeding"]["energy_density_kcal_per_kg"] is None
        assert "ENERGY_DENSITY_MISSING" in result["feeding"]["reason_codes"]


def test_request_numbers_and_planning_ranges_cannot_authorize_coefficient():
    result = resolve_mer_coefficient(pet(coefficient=1.6, mer_coefficient=1.6, coefficient_range=[1.4, 1.6]))
    assert result["coefficient"] is None
    assert result["reason_codes"] == ["MER_COEFFICIENT_UNRESOLVED"]


def test_energy_mismatched_identity_and_corrupt_artifact_fail_close(monkeypatch, tmp_path):
    import mock_feeding_fixture as fixture
    source = product()
    provenance = build_mock_fixture(source)["nutrition_source"]
    assert mock_energy(source, {**provenance, "service_sku": "MOCK-OTHER"}) is None
    assert mock_energy(source, {**provenance, "service_product_id": "999"}) is None
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    assert mock_energy(source, provenance) is None
    (tmp_path / "mock_energy_v1.json").write_text("[]")
    assert mock_energy(source, provenance) is None


def test_same_service_input_is_deterministic_and_not_mutated():
    source_pet, source_product = pet(), product()
    original = deepcopy((source_pet, source_product))
    results = [api.analyze_service_records(source_pet, source_product) for _ in range(3)]
    assert results[0] == results[1] == results[2]
    assert (source_pet, source_product) == original


def test_authenticated_service_endpoint_adds_feeding_and_forbids_client_coefficient(monkeypatch):
    for key in ("MEMBER_DATABASE_URL", "PRODUCT_DATABASE_URL", "INTERNAL_GATEWAY_SECRET"):
        monkeypatch.setenv(key, "synthetic-test-value")
    monkeypatch.setattr(api.service_repository, "get_pet", lambda pid, mid: pet(id=pid))
    monkeypatch.setattr(api.service_repository, "get_product", lambda pid: product(id=pid))
    headers = {"X-Internal-Secret": "synthetic-test-value", "X-Member-Id": "42"}
    with TestClient(api.app) as client:
        response = client.post("/api/nutrition/analyze/by-service-id", json={"pet_id": 10, "product_id": 1}, headers=headers)
        assert response.status_code == 200
        assert response.json()["feeding"]["daily_serving_g"] is None
        assert "MER_COEFFICIENT_UNRESOLVED" in response.json()["feeding"]["reason_codes"]
        assert client.post("/api/nutrition/analyze/by-service-id", json={"pet_id": 10, "product_id": 1, "coefficient": 1.6}, headers=headers).status_code == 422
