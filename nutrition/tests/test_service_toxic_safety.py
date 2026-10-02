"""Synthetic Service source regressions; no live toxic evidence or DB writes."""
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/nutrition")]
import api_nutrition as api
import service_db_adapter as adapter
from service_caution_policy import BACKEND_REVISION, CAUTION_POLICY, EVIDENCE_SOURCE


def pet(species="DOG", profile="KNOWN_NONE", allergies=None):
    return {"id": 10, "species": species, "age": 2, "weight": 8,
            "allergies": allergies or [], "allergy_profile_status": profile, "life_stage": None}


def product(cautions=None, species="DOG", product_id=1):
    return {"id": product_id, "sku": f"MOCK-{product_id:04d}", "product_name": "synthetic",
            "category_code": "FOOD", "subcategory_code": "DRY_FOOD", "target_age_group": "ADULT",
            "target_species": [species], "ingredient_codes": [], "allergen_flags": [],
            "caution_codes": cautions or []}


@pytest.mark.parametrize("species", ["DOG", "CAT"])
def test_chocolate_blocks_even_known_none_and_ready(species):
    source = product(["CHOCOLATE_CACAO"], species)
    before = api.analyze_service_records(pet(species), product(species=species))
    result = api.analyze_service_records(pet(species), source)
    assert before["safety_status"] == "NOT_APPLICABLE"
    assert result["input_readiness"]["input_readiness"] == "READY"
    assert result["nutrition_comparison_status"] == before["nutrition_comparison_status"] == "TRUE"
    assert result["aafco_pass"] is before["aafco_pass"] is True
    assert result["safety_status"] == "SAFETY_BLOCKED" and result["excluded"] is True
    assert result["analysis_status"] == "INSUFFICIENT_DATA"  # existing API semantics
    assert result["allergy_check_status"] == before["allergy_check_status"]
    assert "TOXIC_INGREDIENT" in result["safety_reason_codes"]
    assert "TOXIC_INGREDIENT" in result["exclude_reasons"]
    assert result["conflicting_toxic_ingredients"] == [{
        "caution_code": "CHOCOLATE_CACAO", "caution_level": "TOXIC",
        "applicable_species": ["DOG", "CAT"], "policy_status": "BLOCKED",
        "evidence_source": EVIDENCE_SOURCE, "policy_source": "BACKEND_CautionIngredientCode",
        "policy_version": BACKEND_REVISION,
    }]
    assert "독성" in result["consumer_card"]
    assert result["input_provenance"]["service_cautions"]["codes"] == ["CHOCOLATE_CACAO"]


# Independently asserted species matrix extracted from the Backend enum.
@pytest.mark.parametrize("code,applicable", [
    ("XYLITOL", {"DOG"}), ("CHOCOLATE_CACAO", {"DOG", "CAT"}),
    ("GRAPE_RAISIN", {"DOG", "CAT"}), ("ONION", {"DOG", "CAT"}),
    ("GARLIC", {"DOG", "CAT"}), ("ALLIUM", {"DOG", "CAT"}),
    ("MACADAMIA", {"DOG"}), ("ALCOHOL", {"DOG", "CAT"}),
    ("CAFFEINE", {"DOG", "CAT"}), ("AVOCADO", {"DOG"}),
    ("FRUIT_PITS", {"DOG"}), ("NUTMEG_SPICE", {"DOG"}),
    ("RAW_YEAST_DOUGH", {"DOG", "CAT"}), ("CITRUS", {"CAT"}),
])
@pytest.mark.parametrize("species", ["DOG", "CAT"])
def test_backend_species_matrix(code, applicable, species):
    result = api.analyze_service_records(pet(species), product([code], species))
    blocked = species in applicable
    assert ("TOXIC_INGREDIENT" in result["safety_reason_codes"]) is blocked
    assert result["excluded"] is blocked
    assert result["safety_status"] == ("SAFETY_BLOCKED" if blocked else "NOT_APPLICABLE")
    assert result["caution_evaluations"][0]["policy_status"] == ("BLOCKED" if blocked else "NOT_APPLICABLE")


@pytest.mark.parametrize("profile,allergies", [("UNKNOWN", []), ("KNOWN_LIST", ["CHICKEN"])])
def test_toxic_preserves_existing_allergy_reasons(profile, allergies):
    source = product(["CHOCOLATE_CACAO"])
    source["allergen_flags"] = ["CHICKEN"]
    before = api.analyze_service_records(pet(profile=profile, allergies=allergies),
                                         {**source, "caution_codes": []})
    result = api.analyze_service_records(pet(profile=profile, allergies=allergies), source)
    assert result["safety_status"] == "SAFETY_BLOCKED"
    assert set(before["safety_reason_codes"]) <= set(result["safety_reason_codes"])
    assert set(before["exclude_reasons"]) <= set(result["exclude_reasons"])
    assert result["allergy_check_status"] == before["allergy_check_status"]
    assert result["conflicting_allergens"] == before["conflicting_allergens"]
    if profile == "KNOWN_LIST":
        assert "SERVICE_ALLERGEN_CONFLICT" in result["exclude_reasons"]


@pytest.mark.parametrize("code", [c for c, (level, _) in CAUTION_POLICY.items() if level != "TOXIC"]
                         + ["UNKNOWN_CAUTION"])
def test_non_toxic_has_no_new_block_policy(code):
    before = api.analyze_service_records(pet("CAT"), product(species="CAT"))
    result = api.analyze_service_records(pet("CAT"), product([code], "CAT"))
    for key in ("safety_status", "excluded", "safety_reason_codes", "exclude_reasons", "analysis_status"):
        assert result[key] == before[key]
    assert result["caution_evaluations"][0]["policy_status"] == "POLICY_NOT_DEFINED"
    assert result["conflicting_toxic_ingredients"] == []


@pytest.mark.parametrize("product_id,status", [(1, "FIXTURE_READY"), (11, "FIXTURE_PARTIAL"), (29, "FIXTURE_UNAVAILABLE")])
def test_absent_and_empty_cautions_preserve_whole_response(product_id, status):
    source = product(product_id=product_id)
    absent = deepcopy(source)
    absent.pop("caution_codes")
    result = api.analyze_service_records(pet(), source)
    assert result == api.analyze_service_records(pet(), absent)
    assert result["input_provenance"]["fixture_status"] == status
    assert "caution_evaluations" not in result
    blocked = api.analyze_service_records(pet(), {**source, "caution_codes": ["CHOCOLATE_CACAO"]})
    assert blocked["safety_status"] == "SAFETY_BLOCKED" and blocked["excluded"]
    assert blocked["nutrition_items"] == result["nutrition_items"]
    assert blocked["nutrition_comparison_status"] == result["nutrition_comparison_status"]


def test_adapter_keeps_caution_namespace_separate_and_deterministic():
    source = product(["HIGH_FAT", "CHOCOLATE_CACAO", "CHOCOLATE_CACAO"])
    loaded = adapter.adapt_product(source)
    assert loaded["product"]["service_caution_codes"] == ["CHOCOLATE_CACAO", "HIGH_FAT"]
    assert loaded["product"]["service_ingredient_codes"] == []
    assert loaded["product"]["service_allergen_flags"] == []
    assert source["caution_codes"] == ["HIGH_FAT", "CHOCOLATE_CACAO", "CHOCOLATE_CACAO"]
    assert api.analyze_service_records(pet(), source) == api.analyze_service_records(pet(), source)


@pytest.mark.parametrize("codes", ["CHOCOLATE_CACAO", [None], [14]])
def test_malformed_cautions_are_rejected(codes):
    with pytest.raises(adapter.ServiceInputError, match="SERVICE_CODE_LIST_INVALID"):
        adapter.adapt_product({**product(), "caution_codes": codes})


def test_unmapped_ingredient_does_not_claim_toxic_detection():
    source = product()
    source["ingredient_codes"] = ["UNKNOWN-INGREDIENT"]
    result = api.analyze_service_records(pet(profile="KNOWN_LIST", allergies=["CHICKEN"]), source)
    assert "UNMAPPED_INGREDIENT" in result["safety_reason_codes"]
    assert "TOXIC_INGREDIENT" not in result["safety_reason_codes"]


@pytest.mark.parametrize("species", ["DOG", "CAT"])
def test_service_http_preserves_toxic_evidence(monkeypatch, species):
    import service_repository as repo
    for key in ("MEMBER_DATABASE_URL", "PRODUCT_DATABASE_URL"):
        monkeypatch.setenv(key, "test-only-dsn")
    monkeypatch.setenv("INTERNAL_GATEWAY_SECRET", "test-only-secret")
    monkeypatch.setattr(repo, "get_pet", lambda *_: pet(species))
    monkeypatch.setattr(repo, "get_product", lambda *_: product(["CHOCOLATE_CACAO"], species))
    with TestClient(api.app) as client:
        response = client.post("/api/nutrition/analyze/by-service-id",
                               json={"pet_id": 10, "product_id": 1},
                               headers={"X-Internal-Secret": "test-only-secret", "X-Member-Id": "42"})
    assert response.status_code == 200
    body = response.json()
    assert body["safety_status"] == "SAFETY_BLOCKED" and body["excluded"] is True
    assert body["conflicting_toxic_ingredients"][0]["evidence_source"] == EVIDENCE_SOURCE
    assert body["analysis_status"] == "INSUFFICIENT_DATA"
