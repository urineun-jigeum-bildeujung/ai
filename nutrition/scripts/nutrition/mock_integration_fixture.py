"""Deterministic nutrition fixtures for BE Mock integration products.

This module is integration-only.  It never claims that a ``MOCK-*`` Service
or explicitly registered Service product is an OPFF/OEM/manufacturer product, and it does not
change the strict GTIN production evidence path.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "data" / "integration" / "mock_nutrition_profiles_v1.json"
MOCK_SKU_RE = re.compile(r"^MOCK-[A-Za-z0-9_-]+$")
SOURCE_TYPE = "MOCK_INTEGRATION_FIXTURE"
FIXTURE_VERSION = "mock_nutrition_fixture_v1"
GENERATION_RULE_VERSION = "deterministic_profile_assignment_v1"


def is_mock_sku(value: object) -> bool:
    return isinstance(value, str) and bool(MOCK_SKU_RE.fullmatch(value))


@lru_cache(maxsize=1)
def _registered_identities():
    payload = json.loads((PROFILE_PATH.parent / "mock_service_identity_v1.json").read_text(encoding="utf-8"))
    if (payload.get("fixture_version") != "mock_service_identity_v1"
            or payload.get("data_generation_type") != "SCHEMA_DRIVEN_SYNTHETIC"
            or payload.get("production_evidence") is not False):
        raise RuntimeError("MOCK_IDENTITY_ARTIFACT_INVALID")
    return {(row["service_product_id"], row["service_sku"]) for row in payload["items"]}


def is_mock_source(source: dict[str, Any]) -> bool:
    if is_mock_sku(source.get("sku")):
        return True
    try:
        identities = _registered_identities()
    except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError, KeyError, TypeError, AttributeError):
        return False
    return (str(source.get("id")), source.get("sku")) in identities


def mock_ingredient_refs(refs):
    """Apply explicit synthetic identities only after mock source eligibility."""
    payload = json.loads((PROFILE_PATH.parent / "mock_ingredient_identity_v1.json").read_text(encoding="utf-8"))
    if (payload.get("fixture_version") != "mock_ingredient_identity_v1"
            or payload.get("data_generation_type") != "SCHEMA_DRIVEN_SYNTHETIC"
            or payload.get("production_evidence") is not False):
        raise RuntimeError("MOCK_INGREDIENT_ARTIFACT_INVALID")
    result = []
    for ref in refs:
        codes = payload["mappings"].get(ref["raw_text"])
        if ref["mapping_method"] != "UNRESOLVED" or not codes:
            result.append(ref)
            continue
        for code in codes:
            result.append({**ref, "allergen_code": code, "mapping_method": "STRUCTURED_SOURCE",
                           "ingredient_resolution_status": "RESOLVED", "matched_text": ref["raw_text"],
                           "source": SOURCE_TYPE, "source_version": payload["fixture_version"],
                           "dictionary_version": payload["fixture_version"],
                           "evidence_scope": "SCHEMA_DRIVEN_SYNTHETIC", "production_evidence": False})
    return result


@lru_cache(maxsize=1)
def _artifact() -> dict[str, Any]:
    payload = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    if payload.get("fixture_version") != FIXTURE_VERSION or payload.get("source_type") != SOURCE_TYPE:
        raise RuntimeError("MOCK_FIXTURE_ARTIFACT_INVALID")
    return payload


def _species_key(values: list[str]) -> str | None:
    species = tuple(sorted(set(values or [])))
    return {("DOG",): "DOG", ("CAT",): "CAT", ("CAT", "DOG"): "BOTH"}.get(species)


def _stage_key(value: object) -> str | None:
    raw = str(value or "").upper()
    if raw == "GROWTH":
        return "GROWTH"
    if raw == "ADULT":
        return "ADULT"
    # The current reference contract has no explicit SENIOR stage.  Do not
    # silently convert it into an authoritative adult label claim.
    return None


def _form_key(value: object) -> str:
    raw = str(value or "").upper()
    return "WET" if raw == "WET_FOOD" else "DRY"


def _fixture_status(source: dict[str, Any]) -> tuple[str, str | None]:
    """Choose data-completeness fixtures only; never choose Rule Engine results.

    Non-food products are intentionally unavailable to the P0-D food comparison.
    For food products, sparse deterministic slots exercise fail-close behavior.
    The rule is based only on the immutable Service product id and is stable
    across requests/restarts.  It does not inspect reference thresholds/results.
    """
    if str(source.get("category_code") or "").upper() != "FOOD":
        return "FIXTURE_UNAVAILABLE", "CATEGORY_NOT_FOOD"
    try:
        product_id = int(source["id"])
    except (KeyError, TypeError, ValueError):
        return "FIXTURE_UNAVAILABLE", "SERVICE_PRODUCT_ID_INVALID"
    if product_id % 29 == 0:
        return "FIXTURE_UNAVAILABLE", "DETERMINISTIC_UNAVAILABLE_SLOT"
    if product_id % 11 == 0:
        return "FIXTURE_PARTIAL", "DETERMINISTIC_PARTIAL_SLOT"
    return "FIXTURE_READY", None


def build_mock_fixture(source: dict[str, Any]) -> dict[str, Any] | None:
    """Return a deterministic fixture for an existing Service source row.

    The caller must have already loaded the product from Service DB.  This is
    therefore not a registry that makes a nonexistent ``MOCK-*`` SKU exist.
    """
    sku = source.get("sku")
    if not is_mock_source(source):
        return None

    product_id = str(source.get("id") or "")
    status, reason = _fixture_status(source)
    species_key = _species_key(source.get("target_species") or [])
    stage_key = _stage_key(source.get("target_age_group"))
    form_key = _form_key(source.get("subcategory_code"))

    profile = None
    items: list[dict[str, Any]] = []
    if status != "FIXTURE_UNAVAILABLE" and species_key and stage_key:
        profile = f"{species_key}_{stage_key}_{form_key}"
        base_items = _artifact()["profiles"].get(profile)
        if base_items:
            items = [
                {**item, "source": f"{SOURCE_TYPE}:{FIXTURE_VERSION}"}
                for item in base_items
            ]
            if status == "FIXTURE_PARTIAL":
                # Retain only the label core.  Missing minerals/taurine must be
                # observed by the existing readiness/comparison logic, not
                # replaced with a hard-coded result.
                items = [item for item in items if item["nutrient_code"] in {"CRUDE_PROTEIN", "CRUDE_FAT", "MOISTURE"}]
        else:
            status, reason = "FIXTURE_UNAVAILABLE", "PROFILE_NOT_DEFINED"
    elif status != "FIXTURE_UNAVAILABLE":
        status, reason = "FIXTURE_UNAVAILABLE", "PROFILE_INPUT_UNRESOLVED"

    fixture_stage = {"ADULT": "ADULT", "GROWTH": "GROWTH"}.get(stage_key)
    source_meta = {
        "type": SOURCE_TYPE,
        "source_type": SOURCE_TYPE,
        "fixture_version": FIXTURE_VERSION,
        **({"identity_registration_version": "mock_service_identity_v1"} if not is_mock_sku(sku) else {}),
        "generation_rule_version": GENERATION_RULE_VERSION,
        "service_product_id": product_id,
        "service_sku": sku,
        "product_name": source.get("product_name"),
        "service_category_code": source.get("category_code"),
        "service_subcategory_code": source.get("subcategory_code"),
        "service_target_species": list(source.get("target_species") or []),
        "service_target_age_group": source.get("target_age_group"),
        "fixture_profile": profile,
        "fixture_status": status,
        "fixture_reason": reason,
        "real_product_identity_claimed": False,
        "data_generation_type": "SCHEMA_DRIVEN_SYNTHETIC",
        "production_evidence": False,
        "schema_contract": "SERVICE_DB_COMPATIBLE",
    }
    return {
        "identifier": {
            "status": "MATCHED",
            "identifier_type": "MOCK_SKU",
            "identifier_value": sku,
            "namespace": "SERVICE_INTEGRATION_FIXTURE",
        },
        "fixture_status": status,
        "fixture_profile": profile,
        "nutrition_items": items,
        "aafco_life_stage": fixture_stage,
        "nutrition_source": source_meta,
    }


def coverage_record(source: dict[str, Any]) -> dict[str, Any]:
    fixture = build_mock_fixture(source)
    if fixture is None:
        return {
            "service_product_id": source.get("id"),
            "service_sku": source.get("sku"),
            "fixture_status": "NOT_MOCK_PRODUCT",
            "fixture_profile": None,
            "nutrition_comparison_possible": False,
        }
    possible = (
        fixture["fixture_status"] == "FIXTURE_READY"
        and str(source.get("category_code") or "").upper() == "FOOD"
        and bool(fixture["nutrition_items"])
        and fixture["aafco_life_stage"] in {"ADULT", "GROWTH"}
        and _species_key(source.get("target_species") or []) is not None
    )
    return {
        "service_product_id": source.get("id"),
        "service_sku": source.get("sku"),
        "fixture_status": fixture["fixture_status"],
        "fixture_profile": fixture["fixture_profile"],
        "nutrition_comparison_possible": possible,
        "nutrition_source": fixture["nutrition_source"],
    }


def probe_fixture() -> dict[str, object]:
    try:
        payload = _artifact()
        profiles = payload.get("profiles")
        if not isinstance(profiles, dict) or not profiles:
            return {"required": True, "status": "DOWN"}
        return {"required": True, "status": "UP"}
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return {"required": True, "status": "DOWN"}
