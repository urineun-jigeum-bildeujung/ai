"""Persisted raw product records -> API ProductIn-compatible input.

This module only loads and reshapes existing PRODUCT_LABEL / guaranteed-analysis
records.  It never infers moisture, species, life stage, nutrient values,
reference values, units, bases, or category.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:  # Imported both as ``nutrition.product_input_adapter`` and as a script sibling.
    from .gold_evidence_gate import IDENTITY_BINDING_FIELDS, evaluate_runtime_eligibility
    from .gtin_validation import is_valid_gtin
except ImportError:  # pragma: no cover - direct script/API import compatibility
    from gold_evidence_gate import IDENTITY_BINDING_FIELDS, evaluate_runtime_eligibility
    from gtin_validation import is_valid_gtin


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
DEFAULT_GOLD_OPERATIONAL_EVIDENCE_PATH = ROOT / "data" / "eval" / "verified_evidence_gold_v1.json"
_GTIN_PREFIXES = ("OFF_KR_", "OFF_")


def _present(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _canonical_gtin_from_product_id(product_id: str) -> str | None:
    """Return a canonical local GTIN without accepting formatted identifiers.

    Product IDs in the raw sources can use one documented ``OFF`` prefix.  The
    underlying value must still be an exact ASCII GTIN with a valid check digit;
    whitespace, punctuation, and arbitrary numeric-looking IDs are not
    normalized into an identity key.  A valid leading-zero EAN-13 is canonical
    to its equivalent GTIN-12, matching the P0 cohort builder's join rule.
    """
    value = str(product_id or "")
    for prefix in _GTIN_PREFIXES:
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    if not is_valid_gtin(value):
        return None
    if len(value) == 13 and value.startswith("0") and is_valid_gtin(value[1:]):
        return value[1:]
    return value


def _artifact_path_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_gold_operational_evidence(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read only the validator-owned operational evidence layer.

    The legacy zero-row P0 JSON is intentionally treated as an empty layer.
    A malformed/non-empty artifact is never interpreted as verified evidence.
    """
    label = _artifact_path_label(path)
    if not path.exists():
        return [], {"status": "ARTIFACT_NOT_FOUND", "artifact_path": label}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], {"status": "ARTIFACT_UNREADABLE", "artifact_path": label}
    if not isinstance(payload, dict):
        return [], {"status": "ARTIFACT_SCHEMA_INVALID", "artifact_path": label}

    artifact_version = payload.get("artifact_version") or payload.get("version")
    rows = payload.get("operational_evidence")
    if rows is None and payload.get("verified_evidence_count") == 0:
        # P0's prior summary artifact had no operational rows.  This remains a
        # valid zero-evidence state during the P1 plumbing rollout.
        return [], {
            "status": "NO_OPERATIONAL_EVIDENCE",
            "artifact_path": label,
            "artifact_version": artifact_version,
        }
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return [], {"status": "ARTIFACT_SCHEMA_INVALID", "artifact_path": label}
    if not rows:
        return [], {
            "status": "NO_OPERATIONAL_EVIDENCE",
            "artifact_path": label,
            "artifact_version": artifact_version,
        }
    return [dict(row) for row in rows], {
        "status": "OPERATIONAL_EVIDENCE_LOADED",
        "artifact_path": label,
        "artifact_version": artifact_version,
    }


def _candidate_from_evidence_row(row: dict[str, Any]) -> dict[str, Any]:
    """Construct the gate input exclusively from validated operational rows."""
    fields = (
        *IDENTITY_BINDING_FIELDS,
        "gtin_valid",
        "identity_state",
        "identity_conflict",
        "identity_source_type",
        "identity_source_url_or_document",
        "identity_source_authority",
        "species",
        "life_stage",
        "life_stage_evidence_state",
        "life_stage_detail",
        "life_stage_detail_evidence_state",
        "product_form",
        "product_form_evidence_state",
    )
    return {field: row.get(field) for field in fields}


def _accepted_operational_row(row: dict[str, Any]) -> bool:
    """Require the validator's acceptance metadata before re-running the gate."""
    return (
        row.get("validation_status") == "ACCEPTED"
        and row.get("runtime_eligible") is True
        and _present(row.get("validation_version"))
        and _present(row.get("validated_at"))
        and _present(row.get("candidate_id"))
    )


def _gold_evidence_selection(product_id: str, artifact_path: Path) -> dict[str, Any]:
    """Select one fully revalidated Gold candidate for a persisted product.

    This function deliberately does not merge raw guaranteed analysis with Gold
    rows.  It either returns one full validator/gate-approved layer or nothing,
    keeping all existing raw fail-close behavior intact.
    """
    canonical_gtin = _canonical_gtin_from_product_id(product_id)
    base = {
        "canonical_gtin": canonical_gtin,
        "raw_product_id": product_id,
    }
    if canonical_gtin is None:
        return {"status": "RAW_PRODUCT_GTIN_UNRESOLVED", "rows": [], "provenance": base}

    rows, artifact_provenance = _read_gold_operational_evidence(artifact_path)
    base.update(artifact_provenance)
    if not rows:
        return {"status": artifact_provenance["status"], "rows": [], "provenance": base}

    direct_rows = [
        row for row in rows
        if row.get("product_id") == product_id and row.get("canonical_gtin") == canonical_gtin
    ]
    if not direct_rows:
        return {"status": "NO_MATCHING_OPERATIONAL_EVIDENCE", "rows": [], "provenance": base}

    candidate_ids = {row.get("candidate_id") for row in direct_rows}
    if not all(_present(candidate_id) for candidate_id in candidate_ids):
        return {"status": "ARTIFACT_CANDIDATE_ID_INVALID", "rows": [], "provenance": base}

    all_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        candidate_id = row.get("candidate_id")
        if _present(candidate_id):
            all_by_candidate[str(candidate_id)].append(row)

    approved: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]] = []
    rejected_reasons: list[str] = []
    for candidate_id in sorted(str(value) for value in candidate_ids):
        candidate_rows = all_by_candidate[candidate_id]
        # A candidate ID must never bind operational nutrients for another
        # product/GTIN.  Ignore neither side of such a collision.
        if any(
            row.get("product_id") != product_id or row.get("canonical_gtin") != canonical_gtin
            for row in candidate_rows
        ):
            rejected_reasons.append("ARTIFACT_IDENTITY_BINDING_CONFLICT")
            continue
        if not all(_accepted_operational_row(row) for row in candidate_rows):
            rejected_reasons.append("ARTIFACT_VALIDATION_NOT_ACCEPTED")
            continue
        candidate = _candidate_from_evidence_row(candidate_rows[0])
        gate = evaluate_runtime_eligibility(candidate, candidate_rows)
        if gate["runtime_eligible"]:
            approved.append((candidate, candidate_rows, gate))
        else:
            rejected_reasons.extend(gate["reason_codes"])

    if len(approved) != 1:
        status = "GOLD_GATE_REJECTED" if not approved else "AMBIGUOUS_OPERATIONAL_CANDIDATES"
        return {
            "status": status,
            "rows": [],
            "provenance": {
                **base,
                "candidate_ids": sorted(str(value) for value in candidate_ids),
                "reason_codes": sorted(set(rejected_reasons)),
            },
        }

    candidate, approved_rows, gate = approved[0]
    return {
        "status": "APPLIED",
        "candidate": candidate,
        "rows": approved_rows,
        "provenance": {
            **base,
            "candidate_id": candidate["candidate_id"],
            "runtime_gate": gate,
            "validation_versions": sorted({str(row["validation_version"]) for row in approved_rows}),
            "validated_at": sorted({str(row["validated_at"]) for row in approved_rows}),
            "evidence_rows": [
                {
                    "nutrient_code": row.get("nutrient_code"),
                    "nutrient_source_type": row.get("nutrient_source_type"),
                    "nutrient_source_url_or_document": row.get("nutrient_source_url_or_document"),
                    "nutrient_source_authority": row.get("nutrient_source_authority"),
                    "guarantee_type": row.get("guarantee_type"),
                    "verification_method": row.get("verification_method"),
                }
                for row in approved_rows
            ],
        },
    }


def _gold_nutrition_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert only already validated nutrient evidence to the API input shape."""
    return [
        {
            "nutrient_code": row.get("nutrient_code"),
            "value": row.get("value"),
            "unit": row.get("unit"),
            "basis": row.get("basis"),
            "source": row.get("nutrient_source_type"),
        }
        for row in rows
    ]


def _read(filename: str) -> dict[str, Any]:
    return json.loads((RAW / filename).read_text(encoding="utf-8"))


def _find_product(product_id: str) -> tuple[str, dict[str, Any]] | None:
    sources = (
        ("OPFF", "seed_9_placeholder_feed_opff.json", "items"),
        ("OEM", "seed_9b_off_korean_oem.json", "products"),
        ("GLOBAL", "seed_9_global_brands_v2.json", "products"),
    )
    for source_dataset, filename, key in sources:
        for product in _read(filename).get(key, []):
            if product.get("product_id") == product_id:
                return source_dataset, product
    return None


def _target_species(value: Any) -> str | None:
    """Convert only explicit source values to the API enum; never default."""
    if isinstance(value, list):
        values = {str(item).upper() for item in value}
        if values == {"DOG"}:
            return "dog"
        if values == {"CAT"}:
            return "cat"
        if {"DOG", "CAT"}.issubset(values) or "BOTH" in values:
            return "both"
        return None
    value = str(value or "").upper()
    return {"DOG": "dog", "CAT": "cat", "BOTH": "both"}.get(value)


def _ingredients(product: dict[str, Any]) -> list[str]:
    for key in ("ingredients", "ingredients_parsed"):
        value = product.get(key)
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
    for key in ("ingredients_text", "ingredients_text_raw"):
        value = product.get(key)
        if isinstance(value, str) and value.strip():
            return [value]
    return []


def _category(source_dataset: str, product: dict[str, Any]) -> tuple[str | None, str]:
    """Return food only when an explicit source field supports it."""
    if source_dataset == "OPFF" and product.get("category") == "food":
        return "food", "EXPLICIT_OPFF_CATEGORY"
    if source_dataset == "GLOBAL" and product.get("product_type") in {"DRY_FOOD", "WET_FOOD"}:
        return "food", "EXPLICIT_GLOBAL_PRODUCT_TYPE"
    # OEM category values are missing for 140/189 rows and not a closed enum.
    return None, "CATEGORY_NOT_SOURCE_CONFIRMED"


def _product_form(source_dataset: str, product: dict[str, Any]) -> tuple[str | None, str]:
    """Forward only documented source product type; moisture remains a fallback."""
    if source_dataset == "GLOBAL":
        value = product.get("product_type")
        if value in {"DRY_FOOD", "WET_FOOD"}:
            return str(value), "EXPLICIT_GLOBAL_PRODUCT_TYPE"
    return None, "PRODUCT_FORM_NOT_SOURCE_CONFIRMED"


def _nutrition_items(product_id: str) -> list[dict[str, Any]]:
    rows = _read("seed_13_guaranteed_analysis.json").get("items", [])
    # Preserve zero/null observations: pipeline normalization decides their status.
    return [
        {
            "nutrient_code": row.get("nutrient_code"),
            "value": row.get("value"),
            # seed_13 currently has explicit unit/basis in every row.  Do not
            # manufacture defaults if a future source row omits either value.
            "unit": row.get("unit"),
            "basis": row.get("basis"),
            "source": row.get("source"),
        }
        for row in rows
        if row.get("product_id") == product_id and row.get("nutrient_code")
    ]


def load_product_input(
    product_id: str,
    *,
    operational_evidence_path: Path | None = None,
) -> dict[str, Any]:
    """Return a canonical request product plus explicit provenance.

    Raises LookupError only when no persisted product record exists.  Missing
    nutrition, ingredients, species, or life stage remain missing in the result.

    A human-validated Gold evidence artifact is an optional, separate layer.
    It replaces raw nutrition and Gold-scoped metadata only after the adapter
    re-runs the shared strict gate against an exact product-ID/GTIN binding.
    An absent, malformed, mismatched, or rejected artifact leaves the existing
    raw PRODUCT_LABEL / seed_13 path unchanged.
    """
    found = _find_product(product_id)
    if found is None:
        raise LookupError(product_id)
    source_dataset, product = found
    raw_nutrients = _nutrition_items(product_id)
    ingredients = _ingredients(product)
    category, category_source = _category(source_dataset, product)
    raw_target_species = _target_species(product.get("target_species"))
    raw_product_form, raw_product_form_source = _product_form(source_dataset, product)
    selection = _gold_evidence_selection(
        product_id,
        Path(operational_evidence_path) if operational_evidence_path is not None else DEFAULT_GOLD_OPERATIONAL_EVIDENCE_PATH,
    )

    if selection["status"] == "APPLIED":
        candidate = selection["candidate"]
        nutrients = _gold_nutrition_items(selection["rows"])
        target_species = _target_species(candidate.get("species"))
        life_stage = candidate.get("life_stage")
        product_form = candidate.get("product_form")
        nutrient_source = "VERIFIED_GOLD_OPERATIONAL_EVIDENCE"
        metadata_source = "VERIFIED_GOLD_OPERATIONAL_EVIDENCE"
    else:
        nutrients = raw_nutrients
        target_species = raw_target_species
        life_stage = product.get("target_life_stage")
        product_form = raw_product_form
        nutrient_source = "RAW_GUARANTEED_ANALYSIS"
        metadata_source = "RAW_PRODUCT_LABEL"

    return {
        "product": {
            "id": product_id,
            "name": product.get("name") or product_id,
            "category": category,
            "ingredient_list": ingredients,
            "nutrition_items": nutrients,
            "target_species": target_species,
            # Raw values are forwarded only when the Gold layer is absent.  A
            # successful Gold layer carries its own authoritative metadata;
            # neither branch uses name/category keyword inference.
            "aafco_life_stage": life_stage,
            "product_form": product_form,
            "ingredient_source": "PRODUCT_LABEL",
            "ingredient_source_version": source_dataset,
        },
        "provenance": {
            "product_source_dataset": source_dataset,
            "product_source_file": {
                "OPFF": "seed_9_placeholder_feed_opff.json",
                "OEM": "seed_9b_off_korean_oem.json",
                "GLOBAL": "seed_9_global_brands_v2.json",
            }[source_dataset],
            "nutrition_source_file": "seed_13_guaranteed_analysis.json",
            "nutrition_item_count": len(nutrients),
            "raw_nutrition_item_count": len(raw_nutrients),
            "nutrition_input_source": nutrient_source,
            "ingredient_count": len(ingredients),
            "target_species_present": target_species is not None,
            "life_stage_present": bool(life_stage),
            "moisture_present": any(item["nutrient_code"] == "MOISTURE" and item["value"] is not None for item in nutrients),
            "category_source": category_source,
            "product_form_source": (
                metadata_source if selection["status"] == "APPLIED" else raw_product_form_source
            ),
            "metadata_input_source": metadata_source,
            "unit_basis_contract": "EXPLICIT_PER_ROW",
            "unit_basis_missing_rows": sum(not item["unit"] or not item["basis"] for item in nutrients),
            "gold_operational_evidence": {
                **selection["provenance"],
                # Selection is the effective runtime decision.  Artifact read
                # state remains available in the surrounding provenance, but
                # cannot overwrite a gate rejection or an applied decision.
                "status": selection["status"],
            },
        },
    }
