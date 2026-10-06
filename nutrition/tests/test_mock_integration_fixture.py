from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUTRITION = ROOT / "scripts" / "nutrition"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(NUTRITION))
sys.path.insert(0, str(SCRIPTS))

import mock_integration_fixture as fixture
import service_db_adapter as adapter
import audit_mock_fixture_coverage as audit


def product(product_id=1, sku=None, category="FOOD", species=None, subcategory="DRY_FOOD", age="ADULT"):
    return {
        "id": product_id,
        "sku": sku or f"MOCK-{product_id:04d}",
        "product_name": f"integration product {product_id}",
        "category_code": category,
        "subcategory_code": subcategory,
        "target_age_group": age,
        "target_species": species or ["DOG"],
        "ingredient_codes": [],
        "allergen_flags": [],
    }


def test_mock_sku_uses_separate_identifier_namespace_and_source():
    loaded = adapter.adapt_product(product(1))
    provenance = loaded["provenance"]
    assert provenance["identity_bridge"] == {
        "status": "MATCHED",
        "identifier_type": "MOCK_SKU",
        "identifier_value": "MOCK-0001",
        "namespace": "SERVICE_INTEGRATION_FIXTURE",
    }
    assert provenance["nutrition_source"]["type"] == "MOCK_INTEGRATION_FIXTURE"
    assert provenance["nutrition_source"]["real_product_identity_claimed"] is False
    assert provenance["fixture_status"] == "FIXTURE_READY"
    assert loaded["product"]["nutrition_items"]


def test_strict_gtin_path_remains_unchanged_for_non_mock_sku():
    index = {"036000291452": [{"product_id": "OFF_036000291452", "source_dataset": "OPFF"}]}
    bridge = adapter.bridge_sku("036000291452", index)
    assert bridge["status"] == "MATCHED"
    assert bridge["canonical_gtin"] == "036000291452"
    assert fixture.build_mock_fixture(product(1, sku="036000291452")) is None


def test_registered_identity_load_failure_does_not_break_regular_or_mock_sku(monkeypatch, caplog):
    def fail_identity_load():
        raise RuntimeError("MOCK_IDENTITY_ARTIFACT_INVALID")

    monkeypatch.setattr(fixture, "_registered_identities", fail_identity_load)

    with caplog.at_level(logging.WARNING, logger=fixture.__name__):
        assert fixture.is_mock_source(product(1, sku="036000291452")) is False
    assert "MOCK_IDENTITY_ARTIFACT_INVALID" in caplog.text
    assert fixture.is_mock_source(product(1, sku="MOCK-0001")) is True


def test_unknown_non_gtin_non_mock_still_fails_closed():
    loaded = adapter.adapt_product(product(1, sku="NOT-A-REAL-ID"), index={})
    assert loaded["provenance"]["identity_bridge"]["status"] == "INVALID_SKU"
    assert loaded["provenance"]["nutrition_source"] is None
    assert loaded["product"]["nutrition_items"] == []


def test_partial_fixture_has_core_only_and_does_not_hardcode_result():
    loaded = adapter.adapt_product(product(11))
    assert loaded["provenance"]["fixture_status"] == "FIXTURE_PARTIAL"
    codes = {row["nutrient_code"] for row in loaded["product"]["nutrition_items"]}
    assert codes == {"CRUDE_PROTEIN", "CRUDE_FAT", "MOISTURE"}
    assert "nutrition_comparison_status" not in loaded["provenance"]["nutrition_source"]
    assert "aafco_pass" not in loaded["provenance"]["nutrition_source"]


def test_unavailable_fixture_is_explicit_and_source_is_not_null():
    loaded = adapter.adapt_product(product(29))
    assert loaded["provenance"]["fixture_status"] == "FIXTURE_UNAVAILABLE"
    assert loaded["provenance"]["nutrition_source"]["type"] == "MOCK_INTEGRATION_FIXTURE"
    assert loaded["product"]["nutrition_items"] == []


def test_non_food_is_processed_but_not_fabricated_as_food_evidence():
    loaded = adapter.adapt_product(product(5, category="TREAT", subcategory="JERKY_TREAT"))
    assert loaded["provenance"]["fixture_status"] == "FIXTURE_UNAVAILABLE"
    assert loaded["provenance"]["nutrition_source"]["fixture_reason"] == "CATEGORY_NOT_FOOD"
    assert loaded["product"]["nutrition_items"] == []


def test_same_input_is_identical_across_repeat_and_module_reload():
    source = product(21, species=["CAT"], subcategory="WET_FOOD", age="GROWTH")
    first = fixture.build_mock_fixture(source)
    second = fixture.build_mock_fixture(source)
    assert first == second
    reloaded = importlib.reload(fixture)
    assert reloaded.build_mock_fixture(source) == first


def test_286_contract_simulation_is_complete_and_deterministic():
    rows = [fixture.coverage_record(product(i)) for i in range(1, 287)]
    counts = {status: sum(r["fixture_status"] == status for r in rows)
              for status in ("FIXTURE_READY", "FIXTURE_PARTIAL", "FIXTURE_UNAVAILABLE")}
    assert counts == {"FIXTURE_READY": 251, "FIXTURE_PARTIAL": 26, "FIXTURE_UNAVAILABLE": 9}
    assert sum(counts.values()) == 286
    assert rows == [fixture.coverage_record(product(i)) for i in range(1, 287)]


def test_mock_ready_fixture_runs_the_real_rule_engine():
    import api_nutrition as api
    pet_source = {
        "id": 10, "species": "DOG", "age": 2.0, "weight": 8.0,
        "allergies": [], "allergy_profile_status": "KNOWN_NONE", "life_stage": None,
    }
    result = api.analyze_service_records(pet_source, product(1))
    assert result["analysis_engine"] == "pipeline_p1c_v1"
    assert result["p0_d_applied"] is True
    assert result["input_provenance"]["nutrition_source"]["source_type"] == "MOCK_INTEGRATION_FIXTURE"
    assert result["nutrition_comparison_status"] in {"TRUE", "FALSE"}
    assert result["input_readiness"]["input_readiness"] == "READY"


def test_mock_partial_fixture_keeps_existing_unknown_fail_close():
    import api_nutrition as api
    pet_source = {
        "id": 10, "species": "DOG", "age": 2.0, "weight": 8.0,
        "allergies": [], "allergy_profile_status": "KNOWN_NONE", "life_stage": None,
    }
    result = api.analyze_service_records(pet_source, product(11))
    assert result["input_provenance"]["fixture_status"] == "FIXTURE_PARTIAL"
    assert result["nutrition_comparison_status"] == "UNKNOWN"
    assert result["aafco_pass"] is None
    assert result["analysis_status"] == "INSUFFICIENT_DATA"


def test_nutrition_ready_does_not_override_unknown_pet_safety():
    import api_nutrition as api
    pet_source = {
        "id": 10, "species": "DOG", "age": 2.0, "weight": 8.0,
        "allergies": [], "allergy_profile_status": "UNKNOWN", "life_stage": None,
    }
    result = api.analyze_service_records(pet_source, product(1))
    assert result["nutrition_comparison_status"] in {"TRUE", "FALSE"}
    assert result["safety_status"] == "SAFETY_DATA_INSUFFICIENT"
    assert result["excluded"] is True
    assert result["analysis_status"] == "INSUFFICIENT_DATA"


def _audit_products(mock_count=286, registered_count=36):
    products = []
    for product_id in range(1, mock_count + 1):
        products.append(product(product_id, sku=f"MOCK-{product_id:04d}"))
    registered = sorted(fixture._registered_identities(), key=lambda row: int(row[0]))
    for product_id, sku in registered[:registered_count]:
        products.append(product(int(product_id), sku=sku))
    return products


def test_audit_accepts_all_322_mock_targets(monkeypatch):
    monkeypatch.setattr(audit.service_repository, "list_active_products", lambda: _audit_products())

    report = audit.build_report(expected_mock_total=286, expected_target_total=322)

    assert report["active_product_total"] == 322
    assert report["mock_product_total"] == 286
    assert report["registered_mock_product_total"] == 36
    assert report["mock_target_total"] == 322
    assert report["non_mock_product_total"] == 36
    assert report["non_target_product_total"] == 0
    assert report["acceptance"]["expected_mock_total_match"] is True
    assert report["acceptance"]["expected_target_total_match"] is True
    assert report["acceptance"]["all_mock_processed"] is True
    assert report["acceptance"]["all_targets_processed"] is True
    assert report["observations"]["all_mock_namespace"] is False
    assert report["observations"]["all_active_products_targeted"] is True
    monkeypatch.setattr(sys, "argv", ["audit_mock_fixture_coverage"])
    assert audit.main() == 0


def test_audit_rejects_when_target_count_is_short(monkeypatch):
    monkeypatch.setattr(audit.service_repository, "list_active_products", lambda: _audit_products(285, 36))

    report = audit.build_report(expected_mock_total=286, expected_target_total=322)

    assert report["active_product_total"] == 321
    assert report["acceptance"]["expected_mock_total_match"] is False
    assert report["acceptance"]["expected_target_total_match"] is False
    monkeypatch.setattr(sys, "argv", ["audit_mock_fixture_coverage"])
    assert audit.main() == 2


def test_audit_rejects_when_a_mock_product_has_no_fixture_status(monkeypatch):
    products = _audit_products()
    original_coverage_record = audit.coverage_record

    def missing_fixture(source):
        if source["sku"] == "MOCK-0001":
            return {
                "service_product_id": source["id"],
                "service_sku": source["sku"],
                "fixture_status": "NOT_MOCK_PRODUCT",
                "fixture_profile": None,
                "nutrition_comparison_possible": False,
            }
        return original_coverage_record(source)

    monkeypatch.setattr(audit.service_repository, "list_active_products", lambda: products)
    monkeypatch.setattr(audit, "coverage_record", missing_fixture)

    report = audit.build_report(expected_mock_total=286, expected_target_total=322)

    assert report["acceptance"]["expected_mock_total_match"] is True
    assert report["acceptance"]["expected_target_total_match"] is True
    assert report["acceptance"]["all_mock_processed"] is False
    assert report["acceptance"]["all_targets_processed"] is False
