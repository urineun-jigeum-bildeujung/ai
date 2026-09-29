"""테스트 더블은 입력 경계 검증 전용이며 운영 identity 검증 근거가 아니다."""
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/nutrition")]
import api_nutrition as api
import service_db_adapter as adapter
from audit_service_identity import product_identifiers


def pet(**changes):
    return {"id": 123, "species": "DOG", "age": 3, "weight": 10,
            "allergies": [], "allergy_profile_status": "KNOWN_NONE", "life_stage": "adult", **changes}


def product(**changes):
    return {"id": 456, "sku": None, "product_name": "test source", "category_code": "FOOD",
            "target_species": ["DOG"], "target_age_group": "ADULT",
            "ingredient_codes": ["CHKN-MEAT"], "allergen_flags": [], **changes}


@pytest.mark.parametrize("species", ["DOG", "CAT"])
def test_pet_units_and_species_are_explicit(species):
    result = adapter.adapt_pet(pet(species=species))
    assert (result["id"], result["species"], result["age_years"], result["weight_kg"]) == ("123", species.lower(), 3, 10)


@pytest.mark.parametrize("status,codes,expected", [
    (None, [], "UNKNOWN"), (None, ["CHICKEN"], "UNKNOWN"), ("UNKNOWN", [], "UNKNOWN"),
    ("KNOWN_NONE", [], "KNOWN_NONE"), ("KNOWN_NONE", ["CHICKEN"], "UNKNOWN"),
    ("KNOWN_LIST", [], "UNKNOWN"), ("KNOWN_LIST", ["CHICKEN"], "KNOWN_LIST"),
])
def test_service_profile_never_promotes_unknown(status, codes, expected):
    canonical = adapter.adapt_pet(pet(allergy_profile_status=status, allergies=codes))
    assert canonical["allergy_profile_status"] == expected
    if expected == "UNKNOWN":
        result = adapter.evaluate_service_safety(canonical, adapter.adapt_product(product(), index={})["product"])
        assert result["safety_status"] == "SAFETY_DATA_INSUFFICIENT" and result["excluded"]


@pytest.mark.parametrize("field,value", [("age", None), ("age", -1), ("weight", None), ("weight", 0),
                                        ("weight", float("nan")), ("weight", True), ("species", None)])
def test_missing_pet_data_is_not_filled(field, value):
    with pytest.raises(adapter.ServiceInputError):
        adapter.adapt_pet(pet(**{field: value}))


def test_missing_stage_stays_unknown_despite_adult_age():
    canonical = adapter.adapt_pet(pet(life_stage=None))
    result = adapter.evaluate_service_safety(canonical, adapter.adapt_product(product(), index={})["product"])
    assert canonical["life_stage"] == "UNKNOWN"
    assert result["safety_reason_codes"] == ["PET_LIFE_STAGE_UNSUPPORTED"]


@pytest.mark.parametrize("category,expected", [("FOOD", "food"), ("TREAT", "treat"), ("SUPPLEMENT", "supplement")])
def test_service_category_and_no_fabricated_nutrients(category, expected):
    result = adapter.adapt_product(product(category_code=category), index={})["product"]
    assert result["category"] == expected and result["nutrition_items"] == []


def test_missing_category_rejected_species_and_stage_not_defaulted():
    with pytest.raises(adapter.ServiceInputError):
        adapter.adapt_product(product(category_code=None), index={})
    result = adapter.adapt_product(product(target_species=None, target_age_group=None), index={})["product"]
    assert result["target_species"] is None and result["aafco_life_stage"] is None
    assert adapter.adapt_product(product(target_species=["DOG", "CAT"]), index={})["product"]["target_species"] == "both"


@pytest.mark.parametrize("target_age_group", ["ADULT", "GROWTH", "SENIOR", None])
def test_marketing_age_group_is_not_aafco_label_evidence(target_age_group):
    loaded = adapter.adapt_product(product(target_age_group=target_age_group), index={})
    canonical = loaded["product"]
    assert canonical["aafco_life_stage"] is None
    assert canonical["service_target_age_group"] == target_age_group
    assert loaded["provenance"]["aafco_life_stage_evidence_status"] == "UNKNOWN"
    result = adapter.evaluate_service_safety(adapter.adapt_pet(pet()), canonical)
    assert result["safety_status"] == "SAFETY_DATA_INSUFFICIENT" and result["excluded"]
    assert result["safety_reason_codes"] == ["PRODUCT_LIFE_STAGE_UNKNOWN"]


def test_exact_identity_does_not_prove_aafco_claim(monkeypatch):
    monkeypatch.setattr(adapter, "load_product_input", lambda _: {
        "product": {"category": "food", "target_species": "dog", "aafco_life_stage": "GROWTH", "nutrition_items": []},
        "provenance": {"metadata_input_source": "RAW_PRODUCT_LABEL"},
    })
    index = {"036000291452": [{"product_id": "OFF_036000291452"}]}
    loaded = adapter.adapt_product(product(sku="036000291452", target_age_group="ADULT", aafco_life_stage="ADULT"), index=index)
    assert loaded["provenance"]["identity_bridge"]["status"] == "MATCHED"
    assert loaded["product"]["aafco_life_stage"] is None
    assert loaded["product"]["service_target_age_group"] == "ADULT"


@pytest.mark.parametrize("sku", [None, 4006381333931, "4006381333932", " 4006381333931", "4006-381333931", "４００６３８１３３３９３１", "1234567890"])
def test_invalid_sku_not_normalized_into_identity(sku):
    assert adapter.bridge_sku(sku, {})["status"] == "INVALID_SKU"


@pytest.mark.parametrize("sku", ["96385074", "036000291452", "4006381333931", "10012345678902"])
def test_all_four_gtin_lengths(sku):
    assert adapter.bridge_sku(sku, {})["status"] == "UNMATCHED"


def test_exact_identity_and_ambiguity():
    entry = {"product_id": "OFF_036000291452", "source_dataset": "OPFF"}
    assert adapter.bridge_sku("0036000291452", {"036000291452": [entry]})["status"] == "MATCHED"
    assert adapter.bridge_sku("036000291452", {"036000291452": [entry, entry]})["status"] == "AMBIGUOUS"


@pytest.mark.parametrize("sku,index", [(None, {}), ("036000291452", {}),
    ("036000291452", {"036000291452": [{"product_id": "a"}, {"product_id": "b"}]})])
def test_nonmatch_never_loads_nutrition(monkeypatch, sku, index):
    monkeypatch.setattr(adapter, "load_product_input", lambda _: pytest.fail("nonmatch read"))
    assert adapter.adapt_product(product(sku=sku), index=index)["product"]["nutrition_items"] == []


def test_match_reuses_adapter_without_unit_basis_default(monkeypatch):
    items = [{"nutrient_code": "CALCIUM", "value": None, "unit": None, "basis": None}]
    local = {"product": {"category": "food", "target_species": "dog", "aafco_life_stage": "ADULT",
                         "nutrition_items": items}, "provenance": {"test_only": True}}
    monkeypatch.setattr(adapter, "load_product_input", lambda pid: deepcopy(local))
    index = {"036000291452": [{"product_id": "OFF_036000291452"}]}
    loaded = adapter.adapt_product(product(sku="036000291452"), index=index)
    assert loaded["product"]["id"] == "456" and loaded["product"]["nutrition_items"] == items
    assert loaded["provenance"]["operating_identity_verified"] is False
    local["product"]["target_species"] = "cat"
    loaded = adapter.adapt_product(product(sku="036000291452"), index=index)
    assert loaded["provenance"]["identity_bridge"]["status"] == "AMBIGUOUS"
    assert loaded["product"]["nutrition_items"] == []


def test_structured_codes_exact_unknown_unresolved():
    refs = adapter.structured_refs("456", ["CHKN-MEAT", "CHKN-FAT", "BEEF-MEAT", "chkn-meat", "CHKN"])
    assert [r["mapping_method"] for r in refs] == ["STRUCTURED_SOURCE"] * 3 + ["UNRESOLVED"] * 2
    assert [r["allergen_code"] for r in refs[:3]] == ["chicken", "chicken", "beef"]


def test_service_salmon_never_becomes_fish_but_exact_flag_blocks():
    canonical = adapter.adapt_pet(pet(allergies=["SALMON"], allergy_profile_status="KNOWN_LIST"))
    assert "fish" not in canonical["allergies"] and canonical["service_allergy_codes"] == ["SALMON"]
    data = adapter.adapt_product(product(allergen_flags=["FISH"]), index={})["product"]
    assert adapter.evaluate_service_safety(canonical, data)["safety_status"] == "SAFETY_DATA_INSUFFICIENT"
    data["service_allergen_flags"] = ["SALMON"]
    result = adapter.evaluate_service_safety(canonical, data)
    assert result["safety_status"] == "SAFETY_BLOCKED" and result["excluded"]
    assert result["conflicting_allergens"][0]["namespace"] == "SERVICE"


def test_no_flag_intersection_still_requires_ingredient_evidence():
    canonical = adapter.adapt_pet(pet(allergies=["CHICKEN"], allergy_profile_status="KNOWN_LIST"))
    data = adapter.adapt_product(product(ingredient_codes=["UNKNOWN-CODE"], allergen_flags=[]), index={})["product"]
    # 원료 gate만 분리 검증하는 canonical 테스트 입력. 운영 label evidence가 아니다.
    data["aafco_life_stage"] = "ADULT"
    assert adapter.evaluate_service_safety(canonical, data)["safety_status"] == "SAFETY_DATA_INSUFFICIENT"
    data = adapter.adapt_product(product(), index={})["product"]
    data["aafco_life_stage"] = "ADULT"
    assert adapter.evaluate_service_safety(canonical, data)["safety_status"] == "SAFETY_BLOCKED"


def test_internal_service_path_is_real_engine_not_local_id_lookup(monkeypatch):
    monkeypatch.setattr(api, "_allergy_gate", lambda *_: pytest.fail("service ID must not query local catalog"))
    result = api.analyze_service_records(pet(), product())
    assert result["analysis_engine"] == "pipeline_p1c_v1"
    assert result["product_id"] == "456" and result["pet_id"] == "123"
    assert result["nutrition_comparison_status"] == "UNKNOWN"
    assert result["excluded"] and "PRODUCT_LIFE_STAGE_UNKNOWN" in result["safety_reason_codes"]
    # Nutrition-only 부족 분기는 명시적 canonical 테스트 label로 별도 검증한다.
    # Service source의 target_age_group 승격이나 운영 evidence 생성이 아니다.
    loaded = adapter.adapt_product(product(), index={})
    loaded["product"]["aafco_life_stage"] = "ADULT"
    monkeypatch.setattr(api, "adapt_product", lambda _: deepcopy(loaded))
    result = api.analyze_service_records(pet(), product())
    assert result["analysis_status"] == "INSUFFICIENT_DATA" and not result["excluded"]


def test_service_endpoint_never_uses_mock_or_untrusted_header(monkeypatch):
    monkeypatch.setattr(api, "analyze_service_records", lambda *_: pytest.fail("no configured source"))
    monkeypatch.setattr(api, "load_product_input", lambda *_: pytest.fail("local fallback prohibited"))
    monkeypatch.setattr(adapter, "load_product_input", lambda *_: pytest.fail("local fallback prohibited"))
    monkeypatch.setattr(api, "adapt_product", lambda *_: pytest.fail("adapter fallback prohibited"))
    with TestClient(api.app) as client:
        response = client.post("/api/nutrition/analyze/by-service-id", json={"pet_id": 123, "product_id": 456})
        assert response.status_code == 503 and response.json() == {"detail": "SERVICE_SOURCE_NOT_CONFIGURED"}
        response = client.post("/api/nutrition/analyze/by-service-id", json={"pet_id": 123, "product_id": 456},
                               headers={"X-Member-Id": "999"})
        assert response.status_code == 503 and response.json() == {"detail": "SERVICE_SOURCE_NOT_CONFIGURED"}
        assert client.post("/api/nutrition/analyze/by-service-id", json={"pet_id": 123, "product_id": 456, "member_id": 999}).status_code == 422
        assert client.post("/api/nutrition/analyze/by-service-id", json={"pet_id": True, "product_id": 456}).status_code == 422


def test_coverage_is_counted_not_assumed():
    index = {"036000291452": [{"product_id": "a"}]}
    assert adapter.identity_coverage([{"sku": "036000291452"}, {"sku": None}], index) == {
        "products": 2, "sku_present": 1, "valid_gtin": 1, "MATCHED": 1, "UNMATCHED": 0, "AMBIGUOUS": 0, "INVALID_SKU": 1}


def test_dump_reader_projects_identifiers_only():
    rows = product_identifiers("COPY public.products (id, product_name, sku) FROM stdin;\n1\tunused\t036000291452\n2\tunused\t\\N\n\\.\n")
    assert rows == [{"id": "1", "sku": "036000291452"}, {"id": "2", "sku": None}]


@pytest.mark.parametrize("sql", ["", "COPY public.products (id, sku) FROM stdin;\n1\t123\n",
    "COPY public.products (id, sku) FROM stdin;\n1\t123\n1\t123\n\\.\n"])
def test_incomplete_or_duplicate_dump_not_counted_as_valid(sql):
    with pytest.raises(ValueError):
        product_identifiers(sql)
