"""Shared input-completeness and life-stage contracts for Nutrition API.

``input_readiness`` describes whether persisted/request input is complete enough
for the current runtime rule engine.  It is deliberately separate from the
engine's final ``analysis_status``.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

from pipeline_p1c_v1 import ESSENTIAL_NUTRIENTS


ROOT = Path(__file__).resolve().parents[2]
COMPREHENSIVE_MATRIX_PATH = ROOT / "data" / "processed" / "required_nutrient_matrix_v2.csv"


@lru_cache(maxsize=1)
def _comprehensive_matrix_rows() -> tuple[dict[str, str], ...]:
    """Load coverage metadata only; this file is not a runtime rule input."""
    with COMPREHENSIVE_MATRIX_PATH.open(encoding="utf-8-sig", newline="") as handle:
        return tuple(csv.DictReader(handle))


def resolve_reference_stage(pet: dict[str, Any]) -> dict[str, str | None]:
    """Resolve a pet nutrition-reference stage without silent senior fallback."""
    raw = str(pet.get("life_stage") or "").casefold()
    if raw in {"puppy", "kitten", "growth", "growth_reproduction"}:
        return {"stage": "GROWTH_REPRODUCTION", "source": "EXPLICIT_LIFE_STAGE", "status": "RESOLVED", "reason": "EXPLICIT_GROWTH"}
    if raw in {"adult", "maintenance", "adult_maintenance"}:
        return {"stage": "ADULT_MAINTENANCE", "source": "EXPLICIT_LIFE_STAGE", "status": "RESOLVED", "reason": "EXPLICIT_ADULT"}
    if raw:
        return {"stage": None, "source": "UNSUPPORTED", "status": "UNSUPPORTED", "reason": "UNSUPPORTED_PET_LIFE_STAGE"}
    age = pet.get("age_years")
    if isinstance(age, (int, float)) and age >= 0:
        return {
            "stage": "GROWTH_REPRODUCTION" if age < 1 else "ADULT_MAINTENANCE",
            "source": "AGE_RULE",
            "status": "RESOLVED",
            "reason": "AGE_UNDER_ONE" if age < 1 else "AGE_ONE_OR_OLDER",
        }
    return {"stage": None, "source": "MISSING", "status": "UNKNOWN", "reason": "PET_REFERENCE_STAGE_MISSING"}


def resolve_product_target_stage(value: object) -> dict[str, str | None]:
    """Resolve only an explicit product label claim; no name/category inference."""
    raw = str(value or "").upper()
    mapping = {
        "ADULT": "ADULT_MAINTENANCE",
        "MAINTENANCE": "ADULT_MAINTENANCE",
        "ADULT_MAINTENANCE": "ADULT_MAINTENANCE",
        "GROWTH": "GROWTH_REPRODUCTION",
        "GROWTH_REPRODUCTION": "GROWTH_REPRODUCTION",
        "ALL_LIFE_STAGES": "ALL_LIFE_STAGES",
    }
    if raw in mapping:
        return {"stage": mapping[raw], "status": "RESOLVED", "reason": "EXPLICIT_PRODUCT_TARGET_STAGE"}
    if not raw:
        return {"stage": None, "status": "UNKNOWN", "reason": "PRODUCT_TARGET_STAGE_MISSING"}
    return {"stage": None, "status": "UNSUPPORTED", "reason": "UNSUPPORTED_PRODUCT_TARGET_STAGE"}


def evaluate_nutrition_readiness(
    product: dict[str, Any],
    pet_species: str,
    reference_stage: dict[str, str | None],
) -> dict[str, Any]:
    """Evaluate current-runtime input completeness, not nutritional success."""
    reasons: list[str] = []
    category = product.get("category")
    if category != "food":
        return {
            "input_readiness": "UNSUPPORTED",
            "required_nutrients": [], "present_nutrients": [], "missing_nutrients": [],
            "species_status": "NOT_APPLICABLE", "life_stage_status": "NOT_APPLICABLE",
            "moisture_status": "NOT_APPLICABLE", "reason_codes": ["CATEGORY_NOT_FOOD"],
        }

    target = product.get("target_species")
    if target not in {"dog", "cat", "both"}:
        species_status = "UNKNOWN"; reasons.append("PRODUCT_SPECIES_UNKNOWN")
        effective_species = None
    elif target != "both" and target != pet_species:
        species_status = "MISMATCH"; reasons.append("SPECIES_MISMATCH")
        effective_species = None
    else:
        species_status = "MATCHED"; effective_species = pet_species.upper()

    product_stage = resolve_product_target_stage(product.get("aafco_life_stage"))
    if product_stage["status"] != "RESOLVED":
        reasons.append(str(product_stage["reason"]))
    if reference_stage["status"] != "RESOLVED":
        reasons.append(str(reference_stage["reason"]))

    required = []
    if effective_species and reference_stage["stage"]:
        required = list(ESSENTIAL_NUTRIENTS.get(
            (effective_species, reference_stage["stage"]),
            [],
        ))
        if not required:
            reasons.append("UNSUPPORTED_REFERENCE_COMBINATION")

    present = {
        item.get("nutrient_code")
        for item in product.get("nutrition_items", [])
        if item.get("nutrient_code") and item.get("value") not in (None, 0, 0.0)
    }
    missing = sorted(set(required) - present)
    if missing:
        reasons.append("MISSING_REQUIRED_NUTRIENTS")
    moisture_status = "PRESENT" if "MOISTURE" in present else "MISSING"

    if not product.get("nutrition_items") or not present:
        readiness = "INSUFFICIENT_DATA"
        reasons.append("NUTRITION_ITEMS_MISSING")
    elif not required or reference_stage["status"] != "RESOLVED" or species_status != "MATCHED":
        readiness = "INSUFFICIENT_DATA"
    elif missing or product_stage["status"] != "RESOLVED":
        readiness = "PARTIAL"
    else:
        readiness = "READY"
    return {
        "input_readiness": readiness,
        "required_nutrients": required,
        "present_nutrients": sorted(present),
        "missing_nutrients": missing,
        "species_status": species_status,
        "life_stage_status": product_stage["status"],
        "reference_stage_status": reference_stage["status"],
        "reference_stage": reference_stage["stage"],
        "moisture_status": moisture_status,
        "reason_codes": sorted(set(reasons)),
    }


def evaluate_nutrition_coverage(
    product: dict[str, Any],
    pet_species: str,
    reference_stage: dict[str, str | None],
) -> dict[str, Any]:
    """Describe matrix coverage without changing minimum runtime requirements.

    ``STANDARD`` means the current minimum runtime nutrient set is structurally
    present, not that a reference comparison or nutritional adequacy succeeded.
    ``COMPREHENSIVE`` means every currently applicable tier1/tier2 matrix input
    is structurally present.  ``tier3_unavailable`` is retained as trace data,
    not used to prevent either status.
    """
    species = pet_species.upper()
    stage = reference_stage.get("stage")
    if species not in {"DOG", "CAT"} or reference_stage.get("status") != "RESOLVED" or not stage:
        return {
            "nutrition_coverage": "UNKNOWN",
            "matrix_source": str(COMPREHENSIVE_MATRIX_PATH.relative_to(ROOT)),
            "applicable_nutrients": [], "present_comparable_nutrients": [],
            "missing_nutrients": [], "unavailable_nutrients": [],
            "reason_codes": ["COMPREHENSIVE_MATRIX_REFERENCE_UNRESOLVED"],
        }

    rows = [
        row for row in _comprehensive_matrix_rows()
        if row.get("species") == species and row.get("life_stage") == stage
    ]
    # CA_P_RATIO is a derived comparison, not a label nutrient input. Tier 3 is
    # explicitly marked unavailable in the matrix and cannot be an input gate.
    applicable = sorted({
        row["nutrient_code"] for row in rows
        if row.get("tier") in {"tier1_core", "tier2_extended", "basis_conversion_input"}
        and row.get("nutrient_code") != "CA_P_RATIO"
    })
    unavailable = sorted({row["nutrient_code"] for row in rows if row.get("tier") == "tier3_unavailable"})
    structured = {
        item.get("nutrient_code")
        for item in product.get("nutrition_items", [])
        if item.get("nutrient_code")
        and isinstance(item.get("value"), (int, float))
        and item.get("value") not in (0, 0.0)
        and item.get("unit")
        and item.get("basis")
    }
    present = sorted(set(applicable) & structured)
    missing = sorted(set(applicable) - structured)
    minimum = set(ESSENTIAL_NUTRIENTS.get((species, stage), []))
    minimum_missing = sorted(minimum - structured)
    tier1 = {
        row["nutrient_code"] for row in rows
        if row.get("tier") == "tier1_core" and row.get("nutrient_code") != "CA_P_RATIO"
    }
    tier2 = {row["nutrient_code"] for row in rows if row.get("tier") == "tier2_extended"}
    if not present:
        status = "NONE"
    elif minimum_missing:
        status = "LIMITED"
    elif (tier1 | tier2) - structured:
        status = "STANDARD"
    else:
        status = "COMPREHENSIVE"
    return {
        "nutrition_coverage": status,
        "matrix_source": str(COMPREHENSIVE_MATRIX_PATH.relative_to(ROOT)),
        "matrix_reference_species": species,
        "matrix_reference_stage": stage,
        "applicable_nutrients": applicable,
        "present_comparable_nutrients": present,
        "missing_nutrients": missing,
        "unavailable_nutrients": unavailable,
        "minimum_runtime_missing_nutrients": minimum_missing,
        "reason_codes": ([] if status != "UNKNOWN" else ["COMPREHENSIVE_MATRIX_REFERENCE_UNRESOLVED"]),
    }
