"""NIAS 2024 applicability-preserving canonical reference (Parity P0).

This module deliberately reads the already extracted NIAS table artifact, not
the PDF and not a manually copied value list.  A canonical row represents one
raw threshold: it therefore never encodes two applicability-specific minima as
a ``min_value``/``max_value`` range.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RAW_TABLES = ROOT / "data" / "raw" / "seed_46_nias_2024_nutrient_tables.json"
LEGACY_V5 = ROOT / "data" / "raw" / "seed_14_nutrition_reference_v5.json"
ARTIFACT = ROOT / "data" / "processed" / "nutrition_reference_nias_2024_parity_p0_v1.json"
BEFORE_AFTER_AUDIT = ROOT / "data" / "eval" / "nutrition_reference_parity_p0_before_after_v1.json"
ARTIFACT_VERSION = "NIAS_2024_PARITY_P0_V1"

# page number in the extracted JSON (human-visible PDF page number) -> context
TABLE_CONTEXT = {
    42: ("DOG", "GROWTH_REPRODUCTION", "DRY_MATTER", "2-17a", "DOG_GROWTH_DETAIL"),
    43: ("DOG", "GROWTH_REPRODUCTION", "DRY_MATTER", "2-17a", "DOG_GROWTH_DETAIL"),
    44: ("DOG", "ADULT_MAINTENANCE", "DRY_MATTER", "2-17b", "STANDARD"),
    45: ("DOG", "ADULT_MAINTENANCE", "DRY_MATTER", "2-17b", "STANDARD"),
    46: ("DOG", "GROWTH_REPRODUCTION", "PER_1000KCAL", "2-17c", "DOG_GROWTH_DETAIL"),
    47: ("DOG", "GROWTH_REPRODUCTION", "PER_1000KCAL", "2-17c", "DOG_GROWTH_DETAIL"),
    48: ("DOG", "ADULT_MAINTENANCE", "PER_1000KCAL", "2-17d", "STANDARD"),
    49: ("DOG", "ADULT_MAINTENANCE", "PER_1000KCAL", "2-17d", "STANDARD"),
    50: ("CAT", "GROWTH_REPRODUCTION", "DRY_MATTER", "2-17e", "CAT_GROWTH_DETAIL"),
    51: ("CAT", "GROWTH_REPRODUCTION", "DRY_MATTER", "2-17e", "CAT_GROWTH_DETAIL"),
    52: ("CAT", "ADULT_MAINTENANCE", "DRY_MATTER", "2-17f", "STANDARD"),
    53: ("CAT", "ADULT_MAINTENANCE", "DRY_MATTER", "2-17f", "STANDARD"),
    54: ("CAT", "GROWTH_REPRODUCTION", "PER_1000KCAL", "2-17g", "CAT_GROWTH_DETAIL"),
    55: ("CAT", "GROWTH_REPRODUCTION", "PER_1000KCAL", "2-17g", "CAT_GROWTH_DETAIL"),
    56: ("CAT", "ADULT_MAINTENANCE", "PER_1000KCAL", "2-17h", "STANDARD"),
    57: ("CAT", "ADULT_MAINTENANCE", "PER_1000KCAL", "2-17h", "STANDARD"),
}

NUTRIENT_MAP = {
    "단백질": "CRUDE_PROTEIN", "조단백질": "CRUDE_PROTEIN", "지방": "CRUDE_FAT", "조지방": "CRUDE_FAT",
    "조섬유": "CRUDE_FIBER", "수분": "MOISTURE", "칼슘": "CALCIUM", "인": "PHOSPHORUS",
    "나트륨": "SODIUM", "칼륨": "POTASSIUM", "마그네슘": "MAGNESIUM", "철": "IRON",
    "구리": "COPPER", "아연": "ZINC", "비타민 A": "VITAMIN_A", "비타민 D": "VITAMIN_D",
    "비타민 E": "VITAMIN_E", "비타민 B1": "VITAMIN_B1", "비타민 B2": "VITAMIN_B2",
    "타우린": "TAURINE", "아르지닌": "ARGININE", "아르기닌": "ARGININE", "라이신": "LYSINE",
    "메티오닌": "METHIONINE", "메치오닌": "METHIONINE", "메티오닌+시스틴": "METHIONINE_CYSTINE",
    "메치오닌+시스틴": "METHIONINE_CYSTINE", "트레오닌": "THREONINE", "트립토판": "TRYPTOPHAN",
    "히스티딘": "HISTIDINE", "페닐알라닌": "PHENYLALANINE", "리놀레산": "LINOLEIC_ACID",
    "알파-리놀렌산": "ALPHA_LINOLENIC_ACID", "아라키돈산": "ARACHIDONIC_ACID", "EPA+DHA": "EPA_DHA",
}
UNIT_MAP = {"g": "GRAM", "mg": "MILLIGRAM", "IU": "IU", "%": "PERCENT", "μg": "MICROGRAM", "µg": "MICROGRAM", "㎍": "MICROGRAM"}


def _number(value: Any) -> float | None:
    if value is None:
        return None
    value = str(value).strip().replace(",", "").replace(" ", "")
    if value in {"", "-", "—", "NR", "ND"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _label_parts(raw_label: str) -> tuple[str | None, str | None, str | None]:
    """Return nutrient identity, source form term, and project form policy."""
    label = raw_label.replace("\n", " ").strip()
    form_raw = None
    form = None
    if "건사료" in label:
        form_raw, form = "건사료", "DRY"
    elif "통조림 사료" in label:
        form_raw, form = "통조림 사료", "CANNED"
    base = re.sub(r"\([^)]*\)", "", label).strip()
    return NUTRIENT_MAP.get(base), form_raw, form


def _cat_detail_values(value: Any) -> list[tuple[str | None, float]]:
    """Only materialize CAT detail where the raw cell explicitly labels it."""
    raw = str(value or "").replace(" ", "")
    matches = re.findall(r"([0-9.,]+)\((성장|번식)\)", raw)
    if not matches:
        parsed = _number(value)
        return [] if parsed is None else [(None, parsed)]
    out = []
    for number, label in matches:
        out.append(("GROWTH" if label == "성장" else "REPRODUCTION", float(number.replace(",", ""))))
    return out


def _threshold_row(*, species: str, life_stage: str, life_stage_detail: str | None,
                   nutrient_code: str, basis: str, unit: str, value: float,
                   threshold_type: str, raw_label: str, form_raw: str | None,
                   form: str | None, table_id: str, page: int,
                   fediaf: Any, aafco: Any) -> dict[str, Any]:
    return {
        "species": species, "life_stage": life_stage, "life_stage_detail": life_stage_detail,
        "nutrient_code": nutrient_code, "basis": basis, "unit": unit,
        "threshold_type": threshold_type, "threshold_value": value,
        # Compatibility projection; selector uses threshold_type/value, never numeric ordering.
        "min_value": value if threshold_type == "MINIMUM" else None,
        "max_value": value if threshold_type == "MAXIMUM" else None,
        "source_name_original": raw_label, "reference_form_raw": form_raw,
        "reference_form_canonical": form, "source_table": table_id, "source_page": page,
        "table_id": table_id, "authority": "국립축산과학원", "edition": "2024",
        "source_version": "NIAS_2024", "fediaf_value": _number(fediaf), "aafco_value": _number(aafco),
    }


def _iter_data_rows(table: list[list[Any]]) -> list[list[Any]]:
    return [row for row in table if row and row[0] and str(row[0]).strip() not in {"영양소", "미네랄", "비타민"}]


def build_rows(raw_tables: dict[str, Any], legacy: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page_data in raw_tables["tables"]:
        page = page_data.get("page")
        if page not in TABLE_CONTEXT:
            continue
        species, stage, basis, table_id, mode = TABLE_CONTEXT[page]
        for table in page_data.get("tables", []):
            for source_row in _iter_data_rows(table):
                label = str(source_row[0]).replace("\n", " ").strip()
                code, form_raw, form = _label_parts(label)
                if code is None:
                    continue
                unit = UNIT_MAP.get(str(source_row[1]).strip() if len(source_row) > 1 else "", "GRAM")
                fediaf = source_row[2] if len(source_row) > 2 else None
                aafco = source_row[3] if len(source_row) > 3 else None
                if mode == "DOG_GROWTH_DETAIL":
                    # The header explicitly names col 4/5 as two aNIAS minima.
                    details = (("EARLY_GROWTH_AND_REPRODUCTION", source_row[4] if len(source_row) > 4 else None),
                               ("LATE_GROWTH", source_row[5] if len(source_row) > 5 else None))
                    for detail, raw_value in details:
                        value = _number(raw_value)
                        if value is not None:
                            rows.append(_threshold_row(species=species, life_stage=stage, life_stage_detail=detail,
                                nutrient_code=code, basis=basis, unit=unit, value=value, threshold_type="MINIMUM",
                                raw_label=label, form_raw=form_raw, form=form, table_id=table_id, page=page, fediaf=fediaf, aafco=aafco))
                    # In this table the last column is the only actual maximum.
                    value = _number(source_row[6] if len(source_row) > 6 else None)
                    if value is not None:
                        rows.append(_threshold_row(species=species, life_stage=stage, life_stage_detail=None,
                            nutrient_code=code, basis=basis, unit=unit, value=value, threshold_type="MAXIMUM",
                            raw_label=label, form_raw=form_raw, form=form, table_id=table_id, page=page, fediaf=fediaf, aafco=aafco))
                else:
                    min_raw = source_row[4] if len(source_row) > 4 else None
                    detail_values = _cat_detail_values(min_raw) if mode == "CAT_GROWTH_DETAIL" else [(None, _number(min_raw))]
                    for detail, value in detail_values:
                        if value is not None:
                            rows.append(_threshold_row(species=species, life_stage=stage, life_stage_detail=detail,
                                nutrient_code=code, basis=basis, unit=unit, value=value, threshold_type="MINIMUM",
                                raw_label=label, form_raw=form_raw, form=form, table_id=table_id, page=page, fediaf=fediaf, aafco=aafco))
                    value = _number(source_row[5] if len(source_row) > 5 else None)
                    if value is not None:
                        rows.append(_threshold_row(species=species, life_stage=stage, life_stage_detail=None,
                            nutrient_code=code, basis=basis, unit=unit, value=value, threshold_type="MAXIMUM",
                            raw_label=label, form_raw=form_raw, form=form, table_id=table_id, page=page, fediaf=fediaf, aafco=aafco))

    # V5 contains the project-specific moisture and branded compatibility rules
    # that do not originate in table 2-17. Retain them in the new single artifact.
    for legacy_row in legacy.get("rows", []):
        # Moisture DRY/WET applicability is a project compatibility rule stored
        # only in v5, even though its source_version is NIAS_2024.
        if (legacy_row.get("source_version") == "NIAS_2024"
                and legacy_row.get("table_id") not in {"MOISTURE-DRY", "MOISTURE-WET"}):
            continue
        for threshold_type, key in (("MINIMUM", "min_value"), ("MAXIMUM", "max_value")):
            value = legacy_row.get(key)
            if value is None:
                continue
            form = {"MOISTURE-DRY": "DRY", "MOISTURE-WET": "CANNED"}.get(legacy_row.get("table_id"))
            rows.append({**legacy_row, "life_stage_detail": None, "threshold_type": threshold_type,
                "threshold_value": value, "source_name_original": None, "reference_form_raw": None,
                "reference_form_canonical": form, "source_table": legacy_row.get("table_id"), "source_page": None,
                "authority": "LEGACY_COMPATIBILITY", "edition": legacy_row.get("source_version"),
                "min_value": value if threshold_type == "MINIMUM" else None,
                "max_value": value if threshold_type == "MAXIMUM" else None})
    return rows


def applicability_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(key) for key in ("species", "life_stage", "life_stage_detail", "nutrient_code", "basis", "threshold_type", "reference_form_canonical", "authority", "edition"))


def validate_rows(rows: list[dict[str, Any]]) -> None:
    keys = [applicability_key(row) for row in rows]
    duplicate = [key for key, count in Counter(keys).items() if count > 1]
    if duplicate:
        raise ValueError(f"duplicate applicability keys: {duplicate[:3]}")
    required_forms = [row for row in rows if row["source_name_original"] and ("건사료" in row["source_name_original"] or "통조림 사료" in row["source_name_original"])]
    if any(row.get("reference_form_canonical") is None for row in required_forms):
        raise ValueError("form-specific raw source lost its canonical form")
    dog_detail = [row for row in rows if row["species"] == "DOG" and row["life_stage"] == "GROWTH_REPRODUCTION"]
    if not {"EARLY_GROWTH_AND_REPRODUCTION", "LATE_GROWTH"}.issubset({row.get("life_stage_detail") for row in dog_detail}):
        raise ValueError("dog growth detail lost")


def build_artifact() -> dict[str, Any]:
    raw = json.loads(RAW_TABLES.read_text(encoding="utf-8"))
    legacy = json.loads(LEGACY_V5.read_text(encoding="utf-8"))
    rows = build_rows(raw, legacy)
    validate_rows(rows)
    return {
        "artifact_version": ARTIFACT_VERSION, "source_version": "NIAS_2024", "source_artifacts": [str(RAW_TABLES.relative_to(ROOT)), str(LEGACY_V5.relative_to(ROOT))],
        "canonical_contract": {"required_selection_dimensions": ["species", "life_stage", "nutrient_code", "basis", "threshold_type"],
            "conditional_selection_dimensions": ["reference_form_canonical", "life_stage_detail"],
            "threshold_semantics": "RAW_HEADER_DEFINED"},
        "rows": rows,
    }


def write_artifact(path: Path = ARTIFACT) -> dict[str, Any]:
    artifact = build_artifact()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return artifact


def write_before_after_audit(artifact: dict[str, Any], path: Path = BEFORE_AFTER_AUDIT) -> dict[str, Any]:
    """Record applicability restoration, not a misleading raw-row-growth score."""
    legacy = json.loads(LEGACY_V5.read_text(encoding="utf-8"))["rows"]
    rows = artifact["rows"]
    focus = [
        ("CAT", "ADULT_MAINTENANCE", None, "TAURINE", "DRY_MATTER"),
        ("CAT", "GROWTH_REPRODUCTION", None, "TAURINE", "DRY_MATTER"),
        ("CAT", "ADULT_MAINTENANCE", None, "TAURINE", "PER_1000KCAL"),
        ("CAT", "GROWTH_REPRODUCTION", None, "TAURINE", "PER_1000KCAL"),
        ("CAT", "GROWTH_REPRODUCTION", None, "COPPER", "DRY_MATTER"),
        ("CAT", "GROWTH_REPRODUCTION", None, "COPPER", "PER_1000KCAL"),
        ("DOG", "GROWTH_REPRODUCTION", "EARLY_GROWTH_AND_REPRODUCTION", "CRUDE_PROTEIN", "DRY_MATTER"),
        ("DOG", "GROWTH_REPRODUCTION", "LATE_GROWTH", "CRUDE_PROTEIN", "DRY_MATTER"),
    ]
    entries = []
    for species, stage, detail, nutrient, basis in focus:
        old = [r for r in legacy if r.get("species") == species and r.get("life_stage") == stage and r.get("nutrient_code") == nutrient and r.get("basis") == basis]
        new = [r for r in rows if r.get("species") == species and r.get("life_stage") == stage and r.get("life_stage_detail") == detail and r.get("nutrient_code") == nutrient and r.get("basis") == basis]
        entries.append({"nutrient": nutrient, "species": species, "life_stage": stage,
            "life_stage_detail": detail, "basis": basis,
            "reference_form": sorted({r.get("reference_form_canonical") for r in new if r.get("reference_form_canonical")}),
            "old_rule_count": len(old), "new_rule_count": len(new),
            "information_loss_fixed": bool(new) and (bool(detail) or len({r.get("reference_form_canonical") for r in new}) > 1 or not old)})
    audit = {"artifact_version": artifact["artifact_version"], "legacy_source": str(LEGACY_V5.relative_to(ROOT)),
        "new_source": str(ARTIFACT.relative_to(ROOT)), "comparison_unit": "applicability-specific canonical threshold rows", "entries": entries}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def canonical_reference_form(explicit_form: str | None, moisture_form: str | None) -> tuple[str, str]:
    """Prefer explicit, documented product type; MID/unknown never become DRY/CANNED."""
    explicit = (explicit_form or "").upper()
    if explicit in {"DRY", "DRY_FOOD"}:
        return "DRY", "EXPLICIT_PRODUCT_FORM"
    if explicit in {"CANNED", "WET", "WET_FOOD"}:
        return "CANNED", "EXPLICIT_PRODUCT_FORM"
    inferred = (moisture_form or "").upper()
    if inferred == "DRY":
        return "DRY", "MOISTURE_DERIVED"
    if inferred == "WET":
        return "CANNED", "MOISTURE_DERIVED"
    return "UNKNOWN", "FORM_UNRESOLVED"


def select_reference(rows: list[dict[str, Any]], *, species: str, life_stage: str,
                     nutrient_code: str, basis: str, reference_form: str = "UNKNOWN",
                     life_stage_detail: str | None = None) -> dict[str, Any]:
    """Resolve a threshold bundle, failing closed on unresolved applicability."""
    candidates = [r for r in rows if r.get("species") in {species, "BOTH"} and r.get("life_stage") in {life_stage, "ALL_LIFE_STAGES"}
                  and r.get("nutrient_code") == nutrient_code and r.get("basis") == basis]
    if not candidates:
        return {"status": "NO_REF", "reason_code": "REFERENCE_NOT_FOUND"}
    details = {r.get("life_stage_detail") for r in candidates if r.get("life_stage_detail")}
    if details:
        if not life_stage_detail:
            return {"status": "NO_REF", "reason_code": "REFERENCE_LIFE_STAGE_DETAIL_REQUIRED"}
        exact = [r for r in candidates if r.get("life_stage_detail") == life_stage_detail]
        generic = [r for r in candidates if r.get("life_stage_detail") is None]
        candidates = exact or generic
        if not candidates:
            return {"status": "NO_REF", "reason_code": "REFERENCE_LIFE_STAGE_DETAIL_NOT_FOUND"}
    forms = {r.get("reference_form_canonical") for r in candidates if r.get("reference_form_canonical")}
    if forms:
        if reference_form not in forms:
            return {"status": "NO_REF", "reason_code": "REFERENCE_FORM_REQUIRED" if reference_form == "UNKNOWN" else "REFERENCE_FORM_NOT_FOUND"}
        candidates = [r for r in candidates if r.get("reference_form_canonical") == reference_form]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[str(row.get("threshold_type"))].append(row)
    if any(len(values) != 1 for values in grouped.values()):
        return {"status": "NO_REF", "reason_code": "REFERENCE_AMBIGUOUS"}
    if not grouped:
        return {"status": "NO_REF", "reason_code": "REFERENCE_NOT_FOUND"}
    selected = [values[0] for values in grouped.values()]
    return {"status": "SELECTED", "reason_code": None, "rows": selected,
            "min_value": next((r["threshold_value"] for r in selected if r["threshold_type"] == "MINIMUM"), None),
            "max_value": next((r["threshold_value"] for r in selected if r["threshold_type"] == "MAXIMUM"), None),
            "reference_form": reference_form, "life_stage_detail": life_stage_detail}


if __name__ == "__main__":
    artifact = write_artifact()
    write_before_after_audit(artifact)
    print(f"{ARTIFACT}: {len(artifact['rows'])} rows")
