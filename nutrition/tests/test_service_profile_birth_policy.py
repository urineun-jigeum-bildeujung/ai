"""FE declaration versus SELECT rows; date and namespace policy regressions."""
from datetime import date, datetime
from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from test_service_repository import configure, connection, HEADERS, PATH
import api_nutrition as api
import service_repository as repo
import service_db_adapter as adapter
from service_caution_policy import CAUTION_POLICY


def pet(**changes):
    return {"id": 123, "species": "DOG", "age": 8, "weight": 10,
            "allergies": [], "allergy_profile_status": "KNOWN_NONE", "life_stage": None,
            "birth_date": date(2025, 10, 2), "bcs": 3, "is_neutered": True, **changes}


def product(**changes):
    return {"id": 1, "sku": "MOCK-0001", "category_code": "FOOD", "subcategory_code": "DRY_FOOD",
            "target_species": ["DOG"], "target_age_group": "ADULT", "ingredient_codes": ["닭고기"],
            "allergen_flags": [], "caution_codes": [], **changes}


@pytest.mark.parametrize("declared,codes,profile", [
    (None, [], "UNKNOWN"), (None, ["CHICKEN"], "UNKNOWN"),
    ("UNKNOWN", [], "UNKNOWN"), ("UNKNOWN", ["CHICKEN"], "UNKNOWN"),
    ("KNOWN_NONE", [], "KNOWN_NONE"), ("KNOWN_NONE", ["CHICKEN"], "UNKNOWN"),
    ("KNOWN_LIST", [], "UNKNOWN"), ("KNOWN_LIST", ["CHICKEN"], "KNOWN_LIST"),
])
def test_http_declaration_checked_against_repository_rows(monkeypatch, declared, codes, profile):
    """Verify declared allergy status is checked against stored rows before safety and feeding."""
    configure(monkeypatch)
    # Run the actual repository SELECT aggregation, including row-count checks.
    connection(monkeypatch, (123, "DOG", 8, 10, 3, True, date(2025, 10, 2)),
               [[(code,) for code in codes]])
    monkeypatch.setattr(repo, "get_product", lambda _: product())
    monkeypatch.setattr(adapter, "_today", lambda: date(2026, 10, 2))
    original = api.adapt_pet
    canonical = []
    def capture(source):
        result = original(source)
        canonical.append(result)
        return result
    monkeypatch.setattr(api, "adapt_pet", capture)
    body = {"pet_id": 123, "product_id": 1}
    if declared is not None:
        body["allergy_profile_status"] = declared
    with TestClient(api.app) as client:
        response = client.post(PATH, json=body, headers=HEADERS)
    assert response.status_code == 200
    assert canonical[0]["allergy_profile_status"] == profile
    result = response.json()
    if profile == "UNKNOWN":
        assert result["safety_status"] == "SAFETY_DATA_INSUFFICIENT" and result["excluded"]
        assert "ALLERGY_PROFILE_UNKNOWN" in result["safety_reason_codes"]
        assert result["feeding"]["status"] == "BLOCKED"
    elif profile == "KNOWN_LIST":
        assert result["safety_status"] == "SAFETY_BLOCKED"
    else:
        assert result["safety_status"] == "NOT_APPLICABLE" and not result["excluded"]
    assert (result["feeding"]["daily_serving_g"] is None) == (profile != "KNOWN_NONE")


@pytest.mark.parametrize("value", ["KNOWN", "known_none", "", 0, True, [], {}])
def test_invalid_profile_rejected_before_source_lookup(monkeypatch, value):
    monkeypatch.setattr(repo, "get_pet", lambda *_: pytest.fail("invalid request read DB"))
    with TestClient(api.app) as client:
        assert client.post(PATH, json={"pet_id": 123, "product_id": 1,
                                      "allergy_profile_status": value}).status_code == 422


@pytest.mark.parametrize("today,birth,stage", [
    (date(2026, 10, 1), date(2025, 10, 2), "GROWTH_REPRODUCTION"),
    (date(2026, 10, 2), "2025-10-02", "ADULT_MAINTENANCE"),
    (date(2026, 10, 3), date(2025, 10, 2), "ADULT_MAINTENANCE"),
    (date(2026, 10, 2), date(2026, 10, 2), "GROWTH_REPRODUCTION"),
    (date(2025, 2, 27), date(2024, 2, 29), "GROWTH_REPRODUCTION"),
    (date(2025, 2, 28), date(2024, 2, 29), "ADULT_MAINTENANCE"),
])
@pytest.mark.parametrize("species", ["DOG", "CAT"])
def test_birth_date_precedes_stale_age_and_stage(monkeypatch, today, birth, stage, species):
    """Verify birth dates override stale age and stage without mutating Service inputs."""
    monkeypatch.setattr(adapter, "_today", lambda: today)
    source = pet(birth_date=birth, age=None, species=species, life_stage="pregnant", life_stage_detail="LACTATION")
    original = deepcopy(source)
    canonical = adapter.adapt_pet(source)
    assert canonical["life_stage"] == stage and canonical["life_stage_detail"] is None
    assert (canonical["age_years"] < 1) == (stage == "GROWTH_REPRODUCTION")
    assert source == original
    result = api.analyze_service_records(source, product(target_species=[species]))
    assert result["pet_reference_stage"]["stage"] == stage
    assert "PET_LIFE_STAGE_UNRESOLVED" not in result["feeding"]["reason_codes"]
    assert (result["feeding"]["daily_serving_g"] is None) == (stage == "GROWTH_REPRODUCTION")
    if stage == "GROWTH_REPRODUCTION":
        assert result["excluded"] and "LIFE_STAGE_MISMATCH" in result["safety_reason_codes"]


@pytest.mark.parametrize("birth", ["invalid", "2026-02-30", "20261002", "2026-10-02T00:00:00",
    date(2026, 10, 3), datetime(2025, 10, 2), 1, True, ""])
def test_invalid_birth_never_falls_back_to_age(monkeypatch, birth):
    monkeypatch.setattr(adapter, "_today", lambda: date(2026, 10, 2))
    with pytest.raises(adapter.ServiceInputError, match="^PET_BIRTH_DATE_INVALID$"):
        api.analyze_service_records(pet(birth_date=birth), product())


def test_absent_birth_keeps_production_fail_close():
    canonical = adapter.adapt_pet(pet(birth_date=None))
    assert canonical["life_stage"] == "UNKNOWN"
    result = api.analyze_service_records(pet(birth_date=None), product(sku=None))
    assert result["excluded"] and "PET_LIFE_STAGE_UNSUPPORTED" in result["safety_reason_codes"]


@pytest.mark.parametrize("code", sorted(adapter.SPECIFIC_FISH_CODES))
def test_specific_fish_does_not_expand_to_parent(monkeypatch, code):
    assert adapter.service_allergen_codes(code) == ["SERVICE_CODE:" + code]
    ref = adapter.structured_refs("1", [code])[0]
    assert ref["allergen_code"] is None and ref["mapping_method"] == "UNRESOLVED"
    dictionary = deepcopy(adapter.DICTIONARY)
    specific = code.casefold()
    dictionary["entries"][specific] = {"allergen_code": specific}
    dictionary["aliases"][specific] = specific
    monkeypatch.setattr(adapter, "DICTIONARY", dictionary)
    assert adapter.service_allergen_codes(code) == [specific]
    ref = adapter.structured_refs("1", [code])[0]
    assert ref["allergen_code"] == specific and ref["mapping_method"] == "STRUCTURED_SOURCE"


@pytest.mark.parametrize("code", sorted(CAUTION_POLICY))
def test_caution_code_cannot_become_allergy_evidence(code):
    canonical = adapter.adapt_pet(pet(allergies=[code], allergy_profile_status="KNOWN_LIST"))
    assert canonical["allergies"] == ["SERVICE_CODE:" + code]
    assert adapter.structured_refs("1", [code])[0]["mapping_method"] == "UNRESOLVED"
    loaded = adapter.adapt_product(product(allergen_flags=[code]), index={})["product"]
    result = adapter.evaluate_service_safety(canonical, loaded)
    assert result["safety_status"] == "SAFETY_DATA_INSUFFICIENT"
    assert "ALLERGY_CONFLICT" not in result["safety_reason_codes"]
