"""Rank local FOOD artifacts by missing data; never fetch or infer source values."""
from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from nutrition_readiness import (  # noqa: E402
    evaluate_nutrition_coverage,
    evaluate_nutrition_readiness,
    resolve_product_target_stage,
    resolve_reference_stage,
)
from product_input_adapter import ROOT, _read, load_product_input  # noqa: E402


OUT = ROOT / "data" / "eval" / "ready_fixture_candidates_minimum_runtime_v1.json"
SOURCES = (
    ("seed_9_placeholder_feed_opff.json", "items"),
    ("seed_9_global_brands_v2.json", "products"),
)


def _ids() -> list[str]:
    return sorted({
        row["product_id"]
        for filename, key in SOURCES
        for row in _read(filename).get(key, [])
        if row.get("product_id")
    })


def _candidate(product_id: str, species: str) -> dict | None:
    loaded = load_product_input(product_id)
    product = loaded["product"]
    if product.get("category") != "food":
        return None
    if product.get("target_species") not in {species.lower(), "both"}:
        return None
    # This is an explicit audit request profile, not a product target-stage
    # inference. Product label stage remains independently reported below.
    stage = resolve_reference_stage({"life_stage": "adult", "age_years": 3})
    readiness = evaluate_nutrition_readiness(product, species.lower(), stage)
    coverage = evaluate_nutrition_coverage(product, species.lower(), stage)
    product_stage = resolve_product_target_stage(product.get("aafco_life_stage"))
    # Coverage's minimum list is defined for the explicit audit species even
    # when a product has no nutrition rows.  Do not use readiness's empty list
    # from a species-mismatch branch as a proxy for missing source fields.
    missing_fields = list(coverage["minimum_runtime_missing_nutrients"])
    if product_stage["status"] != "RESOLVED":
        missing_fields.append("PRODUCT_TARGET_STAGE")
    if not loaded["provenance"]["ingredient_count"]:
        missing_fields.append("INGREDIENTS")
    return {
        "product_id": product_id,
        "name": product["name"],
        "audit_pet_species": species,
        "coverage_reference_stage": stage["stage"],
        "coverage_stage_source": "AUDIT_REQUEST_SCENARIO_NOT_PRODUCT_CLAIM",
        "source_dataset": loaded["provenance"]["product_source_dataset"],
        "category": product["category"],
        "present_nutrients": readiness["present_nutrients"],
        "missing_minimum_nutrients": coverage["minimum_runtime_missing_nutrients"],
        "missing_comprehensive_nutrients": coverage["missing_nutrients"],
        "nutrition_coverage": coverage["nutrition_coverage"],
        "product_target_stage": product_stage,
        "ingredient_count": loaded["provenance"]["ingredient_count"],
        "provenance_completeness": {
            "category_source": loaded["provenance"]["category_source"],
            "target_species_present": loaded["provenance"]["target_species_present"],
            "life_stage_present": loaded["provenance"]["life_stage_present"],
            "moisture_present": loaded["provenance"]["moisture_present"],
            "unit_basis_missing_rows": loaded["provenance"]["unit_basis_missing_rows"],
        },
        "minimum_missing_source_fields": missing_fields,
    }


def _rank(species: str) -> list[dict]:
    rows = [row for product_id in _ids() if (row := _candidate(product_id, species))]
    return sorted(
        rows,
        key=lambda row: (
            len(row["minimum_missing_source_fields"]),
            len(row["missing_comprehensive_nutrients"]),
            -len(row["present_nutrients"]),
            row["product_id"],
        ),
    )[:10]


def main() -> None:
    dog, cat = _rank("DOG"), _rank("CAT")
    recommended = min(
        [*dog, *cat],
        key=lambda row: (
            len(row["minimum_missing_source_fields"]),
            len(row["missing_comprehensive_nutrients"]),
            # DOG wins only as a tie-breaker; it is not forced over a better CAT candidate.
            0 if row["audit_pet_species"] == "DOG" else 1,
            row["product_id"],
        ),
    )
    OUT.write_text(json.dumps({
        "scope": "local artifact candidate ranking; no source collection or value inference",
        "minimum_runtime_source": "scripts/pipeline_p1c_v1.py:ESSENTIAL_NUTRIENTS",
        "comprehensive_matrix_source": "data/processed/required_nutrient_matrix_v2.csv",
        "candidate_exclusions": ["OEM category not source-confirmed", "non-food category"],
        "dog_top_10": dog,
        "cat_top_10": cat,
        "recommended_first_fixture": recommended,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "dog_top_10_rows": len(dog), "cat_top_10_rows": len(cat),
        "recommended": {k: recommended[k] for k in ("product_id", "audit_pet_species", "minimum_missing_source_fields")},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
