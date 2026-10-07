"""Stored profile and exact Service identities, with unknown evidence preserved."""
from datetime import date
import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from test_service_repository import configure, connection, HEADERS, PATH
from test_service_profile_birth_policy import pet, product
import api_nutrition as api
import service_db_adapter as adapter
import service_repository as repo


@pytest.mark.parametrize("stored,codes,expected", [
    ("KNOWN_NONE", [], "KNOWN_NONE"), ("UNKNOWN", [], "UNKNOWN"),
    ("KNOWN_LIST", ["TURKEY", "TUNA"], "KNOWN_LIST"),
    ("KNOWN_NONE", ["CHICKEN"], "UNKNOWN"),
    ("KNOWN_LIST", [], "UNKNOWN"), ("INVALID", [], "UNKNOWN"),
])
def test_persisted_profile_is_read_and_consistency_checked(monkeypatch, stored, codes, expected):
    connection(monkeypatch, (123, "DOG", 3, 10, 3, True, date(2023, 1, 1), "SMALL", stored),
               [[(code,) for code in codes]])
    loaded = repo.get_pet(123, 42)
    assert adapter.adapt_pet(loaded)["allergy_profile_status"] == expected


@pytest.mark.parametrize("path,ids", [
    (PATH, {"product_id": 1}), ("/api/nutrition/compare", {"product_ids": [1, 2]}),
])
@pytest.mark.parametrize("declaration,expected", [
    ({}, "NOT_APPLICABLE"), ({"allergy_profile_status": None}, "SAFETY_DATA_INSUFFICIENT"),
    ({"allergy_profile_status": "KNOWN_LIST"}, "SAFETY_DATA_INSUFFICIENT"),
])
def test_endpoints_use_stored_none_without_request_override(monkeypatch, path, ids, declaration, expected):
    configure(monkeypatch)
    monkeypatch.setattr(repo, "get_pet", lambda *_: pet(birth_date=date(2023, 1, 1)))
    monkeypatch.setattr(repo, "get_product", lambda _: product())
    observed = []
    original = api.analyze_service_records
    def capture(source, item):
        result = original(source, item)
        observed.append(result["safety_status"])
        return result
    monkeypatch.setattr(api, "analyze_service_records", capture)
    with TestClient(api.app) as client:
        response = client.post(path, json={"pet_id": 123, **ids, **declaration}, headers=HEADERS)
    assert response.status_code == 200
    assert observed and set(observed) == {expected}


@pytest.mark.parametrize("code", sorted(adapter.SERVICE_IDENTITIES))
def test_declared_backend_identity_has_exact_ingredient_conflict(code):
    source = pet(allergies=[code], allergy_profile_status="KNOWN_LIST")
    canonical = adapter.adapt_pet(source)
    display = adapter.SERVICE_IDENTITIES[code][1]
    item = adapter.adapt_product(product(ingredient_codes=[display]), index={})["product"]
    result = adapter.evaluate_service_safety(canonical, item)
    assert result["safety_status"] == "SAFETY_BLOCKED"
    assert "ALLERGY_CONFLICT" in result["safety_reason_codes"]


def test_tuna_is_not_salmon_and_generic_fish_is_not_clearance():
    canonical = adapter.adapt_pet(pet(allergies=["TUNA"], allergy_profile_status="KNOWN_LIST"))
    salmon = adapter.adapt_product(product(ingredient_codes=["연어"]), index={})["product"]
    assert adapter.evaluate_service_safety(canonical, salmon)["safety_status"] == "NO_CONFLICT_DETECTED"
    fish = adapter.adapt_product(product(ingredient_codes=["fish"]), index={})["product"]
    assert adapter.evaluate_service_safety(canonical, fish)["safety_status"] == "SAFETY_DATA_INSUFFICIENT"
    broad = adapter.adapt_pet(pet(allergies=["FISH"], allergy_profile_status="KNOWN_LIST"))
    assert adapter.evaluate_service_safety(broad, salmon)["safety_status"] == "SAFETY_BLOCKED"


def test_product141_identity_does_not_invent_non_allergen_evidence():
    refs = adapter.structured_refs("141", ["양고기", "귀리", "고구마", "당근", "비트"])
    assert [r["allergen_code"] for r in refs] == ["lamb", "oat", "potato", None, None]
    assert all(r["ingredient_resolution_status"] == "RESOLVED" for r in refs)
    assert all(r["mapping_method"] == "UNRESOLVED" for r in refs[-2:])


def test_published_service_mapping_matches_runtime():
    path = Path(__file__).resolve().parents[1] / "docs" / "service_allergen_mapping_v2.csv"
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == len({r["service_code"] for r in rows}) == 38
    for row in rows:
        assert sorted(adapter.service_allergen_codes(row["service_code"])) == sorted(row["canonical_codes"].split("|"))


@pytest.mark.parametrize("code", sorted(adapter.SERVICE_TOXIC_CODES))
def test_toxic_namespace_is_not_an_allergen_conflict(code):
    canonical = adapter.adapt_pet(pet(allergies=[code], allergy_profile_status="KNOWN_LIST"))
    item = adapter.adapt_product(product(allergen_flags=[code]), index={})["product"]
    result = adapter.evaluate_service_safety(canonical, item)
    assert "ALLERGY_CONFLICT" not in result["safety_reason_codes"]
    assert "ALLERGY_PROFILE_UNKNOWN" in result["safety_reason_codes"]
