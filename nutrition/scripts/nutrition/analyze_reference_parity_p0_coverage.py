"""Read-only applicability/coverage analysis for Nutrition Reference Parity P0.

It creates versioned evaluation artifacts from existing local product files.  It
does not write raw sources, alter the active reference, or infer form/stage
metadata that is absent from product records.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from pipeline_p1c_v1 import (  # noqa: E402
    TARGET_NUTRIENTS_18, align_units, classify_dry_wet, detect_outliers, normalize,
    normalize_basis,
)
from nutrition.product_input_adapter import ROOT, _read, load_product_input  # noqa: E402
from nutrition.nutrition_readiness import evaluate_nutrition_readiness, resolve_product_target_stage  # noqa: E402
from nutrition.reference_parity import ARTIFACT, LEGACY_V5, canonical_reference_form, select_reference  # noqa: E402


OUT = ROOT / "data" / "eval"
VERSION = "reference_parity_p0_coverage_v1"
SOURCES = (("OPFF", "seed_9_placeholder_feed_opff.json", "items"),
           ("OEM", "seed_9b_off_korean_oem.json", "products"),
           ("GLOBAL", "seed_9_global_brands_v2.json", "products"))
RUNTIME = {
    "DOG": ["CRUDE_PROTEIN", "CRUDE_FAT", "MOISTURE", "CALCIUM", "PHOSPHORUS"],
    "CAT": ["CRUDE_PROTEIN", "CRUDE_FAT", "MOISTURE", "CALCIUM", "PHOSPHORUS", "TAURINE"],
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _inventory() -> list[dict[str, Any]]:
    rows = []
    for source, filename, key in SOURCES:
        for record in _read(filename).get(key, []):
            if record.get("product_id"):
                rows.append({"source_dataset": source, "raw": record})
    return rows


def _product_species(product: dict[str, Any]) -> str | None:
    value = product.get("target_species")
    if value == "dog": return "DOG"
    if value == "cat": return "CAT"
    return None


def _pipeline_items(product: dict[str, Any]) -> list[dict[str, Any]]:
    raw = [{"product_id": product["id"], "nutrient_code": item.get("nutrient_code"), "value": item.get("value"),
            "unit": item.get("unit"), "basis": item.get("basis"), "source": item.get("source")}
           for item in product.get("nutrition_items", [])]
    return normalize_basis(classify_dry_wet(detect_outliers(align_units(normalize(raw)))))


def _form_info(product: dict[str, Any], pipeline_items: list[dict[str, Any]]) -> tuple[str, str, str]:
    moisture_form = next((row.get("product_form") for row in pipeline_items if row.get("nutrient_code") == "MOISTURE"), "UNKNOWN")
    canonical, source = canonical_reference_form(product.get("product_form"), moisture_form)
    if product.get("product_form"):
        status = "EXPLICIT"
    elif moisture_form == "MID":
        status = "MID_OR_AMBIGUOUS"
    elif canonical == "UNKNOWN":
        status = "UNKNOWN"
    else:
        status = "MOISTURE_DERIVED"
    return canonical, source, status


def _reference_group(reason: str | None, item: dict[str, Any], references: list[dict[str, Any]], species: str | None, stage: str | None) -> str:
    if item.get("basis_invalid"):
        return "INVALID_VALUE"
    if item.get("basis_normalized_value") is None and item.get("aligned_value") is None:
        return "NO_VALUE"
    if not species or not stage:
        return "NO_REF_STAGE_OR_SPECIES_UNKNOWN"
    if reason in {"REFERENCE_FORM_REQUIRED", "REFERENCE_FORM_NOT_FOUND"}:
        return "NO_REF_FORM_UNKNOWN"
    if reason in {"REFERENCE_LIFE_STAGE_DETAIL_REQUIRED", "REFERENCE_LIFE_STAGE_DETAIL_NOT_FOUND"}:
        return "NO_REF_STAGE_DETAIL_UNKNOWN"
    code, basis = item.get("nutrient_code"), item.get("basis_normalized_basis") or item.get("basis")
    same_nutrient = [r for r in references if r.get("species") in {species, "BOTH"} and r.get("nutrient_code") == code]
    if same_nutrient and not any(r.get("basis") == basis for r in same_nutrient):
        return "NO_REF_BASIS"
    return "NO_REF_NUTRIENT"


def _legacy_select(rows: list[dict[str, Any]], *, species: str, stage: str, code: str, basis: str, form: str) -> dict[str, Any] | None:
    """Faithful compact representation of the legacy v5 lookup order for audit only."""
    candidates = [r for r in rows if r.get("species") in {species, "BOTH"} and r.get("nutrient_code") == code]
    def form_ok(row: dict[str, Any]) -> bool:
        table = row.get("table_id")
        if table == "MOISTURE-DRY": return form in {"DRY", "MID", "UNKNOWN"}
        if table == "MOISTURE-WET": return form in {"CANNED", "MID", "UNKNOWN"}
        return True
    candidates = [r for r in candidates if form_ok(r)]
    for target in (stage, "ALL_LIFE_STAGES"):
        scoped = [r for r in candidates if r.get("life_stage") == target]
        if scoped:
            exact = [r for r in scoped if r.get("basis") == basis]
            return (exact or scoped)[0]
    return None


def _result(value: float | None, reference: dict[str, Any] | None) -> str:
    if value is None or reference is None: return "NOT_COMPARABLE"
    if reference.get("min_value") is not None and value < float(reference["min_value"]): return "OUT_OF_RANGE"
    if reference.get("max_value") is not None and value > float(reference["max_value"]): return "OUT_OF_RANGE"
    return "IN_RANGE"


def _reconciliation(legacy: list[dict[str, Any]], new: list[dict[str, Any]]) -> dict[str, Any]:
    old_rows, unmatched_new = [], set(range(len(new)))
    counts = Counter()
    unmatched_by_nutrient = Counter()
    for old in legacy:
        candidates = [i for i, row in enumerate(new) if all(row.get(key) == old.get(key) for key in ("species", "life_stage", "nutrient_code", "basis"))]
        for index in candidates: unmatched_new.discard(index)
        if not candidates:
            kind = "REMOVED_OR_UNMATCHED"
        elif len(candidates) > 1:
            kind = "SPLIT_OR_APPLICABILITY_RESTORED"
        else:
            candidate = new[candidates[0]]
            kind = "SEMANTIC_CHANGED" if candidate.get("min_value") != old.get("min_value") or candidate.get("max_value") != old.get("max_value") else "MAPPED"
        counts[kind] += 1
        if kind == "REMOVED_OR_UNMATCHED":
            unmatched_by_nutrient[str(old.get("nutrient_code"))] += 1
        old_rows.append({"legacy_key": {key: old.get(key) for key in ("species", "life_stage", "nutrient_code", "basis", "table_id")}, "mapping": kind, "new_match_count": len(candidates)})
    return {"legacy_rows": len(legacy), "new_threshold_rows": len(new), "legacy_mapping_counts": dict(counts),
            "new_rows_without_legacy_four_dimension_match": len(unmatched_new),
            "unmatched_legacy_by_nutrient": dict(sorted(unmatched_by_nutrient.items())),
            "old_rows": old_rows,
            "explanation": "new rows are threshold-level rules; form/detail splits can increase rules while unsupported legacy nutrient mappings can reduce total rows"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    legacy = json.loads(LEGACY_V5.read_text(encoding="utf-8"))["rows"]
    new_artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    references = new_artifact["rows"]
    _write_json(OUT / f"{VERSION}_reference_reconciliation.json", _reconciliation(legacy, references))

    detail_count = sum(bool(row.get("life_stage_detail")) for row in references)
    reference_coverage = {"threshold_rules": len(references), "unique_nutrients": len({r["nutrient_code"] for r in references}),
        "by_species": dict(Counter(r.get("species") for r in references)), "by_life_stage": dict(Counter(r.get("life_stage") for r in references)),
        "by_basis": dict(Counter(r.get("basis") for r in references)), "by_threshold_type": dict(Counter(r.get("threshold_type") for r in references)),
        "by_reference_form": dict(Counter(r.get("reference_form_canonical") or "GENERIC" for r in references)),
        "by_authority": dict(Counter(r.get("authority") for r in references)), "by_edition": dict(Counter(r.get("edition") for r in references)),
        "life_stage_detail_specific_rules": detail_count, "form_specific_rules": sum(bool(r.get("reference_form_canonical")) for r in references)}

    coverage_rows, app_rows, transition_rows = [], [], []
    nutrient_stats: dict[str, Counter] = defaultdict(Counter)
    form_counts, stage_counts, detail_counts, evaluability, reasons = Counter(), Counter(), Counter(), Counter(), Counter()
    source_stats: dict[str, Counter] = defaultdict(Counter)
    raw_records = _inventory()
    source_inventory: dict[str, Counter] = defaultdict(Counter)
    for source_row in raw_records:
        product_id = source_row["raw"]["product_id"]
        loaded = load_product_input(product_id)
        product, provenance, source = loaded["product"], loaded["provenance"], source_row["source_dataset"]
        species = _product_species(product)
        product_stage = resolve_product_target_stage(product.get("aafco_life_stage"))
        stage = product_stage.get("stage") if product_stage.get("status") == "RESOLVED" else None
        pipeline = _pipeline_items(product)
        canonical_form, form_source, form_status = _form_info(product, pipeline)
        inventory = source_inventory[source]
        inventory["product_count"] += 1
        inventory[f"category:{product.get('category') or 'UNKNOWN'}"] += 1
        inventory[f"species:{species or 'UNKNOWN'}"] += 1
        inventory[f"life_stage:{stage or 'UNKNOWN'}"] += 1
        inventory["nutrition_items_present"] += bool(product.get("nutrition_items"))
        inventory["ingredients_present"] += bool(provenance["ingredient_count"])
        inventory["moisture_present"] += any(row.get("nutrient_code") == "MOISTURE" and row.get("value") is not None for row in product.get("nutrition_items", []))
        inventory["explicit_product_form_present"] += bool(product.get("product_form"))
        form_counts[("ALL", form_status, canonical_form)] += 1
        if product.get("category") == "food": form_counts[("FOOD", form_status, canonical_form)] += 1
        stage_counts[stage or "UNKNOWN"] += 1
        if stage == "GROWTH_REPRODUCTION":
            detail_counts["UNKNOWN"] += 1  # no persisted source currently forwards stage-detail evidence
        nutrition_codes = {r.get("nutrient_code") for r in product.get("nutrition_items", [])}
        valid_codes = {r.get("nutrient_code") for r in pipeline if r.get("basis_normalized_value") is not None and not r.get("basis_invalid")}
        coverage_groups = ["ALL_PRODUCTS"] + ([species] if species in RUNTIME else [])
        for group in coverage_groups:
            required_codes = sorted(set(RUNTIME["DOG"]) | set(RUNTIME["CAT"])) if group == "ALL_PRODUCTS" else RUNTIME[group]
            for code in required_codes:
                stat = nutrient_stats[f"{group}:{code}"]
                stat["present"] += code in nutrition_codes
                stat["valid"] += code in valid_codes
                stat["missing"] += code not in nutrition_codes
                stat["unit_missing"] += any(r.get("nutrient_code") == code and not r.get("unit") for r in product.get("nutrition_items", []))
                stat["basis_missing"] += any(r.get("nutrient_code") == code and not r.get("basis") for r in product.get("nutrition_items", []))
                stat["invalid"] += any(r.get("nutrient_code") == code and r.get("basis_invalid") for r in pipeline)
        audit_species = species.lower() if species else "dog"
        stage_for_readiness = {"stage": stage, "status": "RESOLVED" if stage else "UNKNOWN", "reason": "PERSISTED_PRODUCT_METADATA"}
        readiness = evaluate_nutrition_readiness(product, audit_species, stage_for_readiness)
        status = readiness["input_readiness"]
        evaluability[status] += 1
        source_stats[source][status] += 1
        reasons.update(readiness["reason_codes"])
        coverage_rows.append({"product_id": product_id, "source_dataset": source, "category": product.get("category") or "UNKNOWN",
            "target_species": species or "UNKNOWN", "life_stage": stage or "UNKNOWN", "life_stage_detail": "UNKNOWN",
            "product_form": canonical_form, "product_form_status": form_status, "product_form_source": form_source,
            "nutrition_item_count": len(product.get("nutrition_items", [])), "ingredient_count": provenance["ingredient_count"],
            "moisture_present": "MOISTURE" in nutrition_codes, "input_readiness": status,
            "reason_codes": readiness["reason_codes"]})
        if product.get("category") != "food":
            continue
        for item in pipeline:
            code = item.get("nutrient_code")
            if not code: continue
            basis = item.get("basis_normalized_basis") or item.get("basis")
            selected = (select_reference(references, species=species, life_stage=stage, nutrient_code=code, basis=basis,
                                         reference_form=canonical_form) if species and stage else {"status": "NO_REF", "reason_code": "PERSISTED_SPECIES_OR_STAGE_UNKNOWN"})
            group = "REFERENCE_RESOLVED" if selected["status"] == "SELECTED" else _reference_group(selected.get("reason_code"), item, references, species, stage)
            value = item.get("basis_normalized_value")
            comparable = selected["status"] == "SELECTED" and value is not None and not item.get("basis_invalid")
            old = _legacy_select(legacy, species=species, stage=stage, code=code, basis=basis, form=canonical_form) if species and stage else None
            old_result = _result(value, old)
            new_reference = {"min": selected.get("min_value"), "max": selected.get("max_value"), "form": canonical_form, "detail": selected.get("life_stage_detail")} if selected["status"] == "SELECTED" else None
            new_result = _result(value, {"min_value": selected.get("min_value"), "max_value": selected.get("max_value")} if comparable else None)
            if old_result == "NOT_COMPARABLE" and new_result == "NOT_COMPARABLE": change = "NOT_COMPARABLE"
            elif old is not None and selected["status"] != "SELECTED": change = "OLD_RESOLVED_NEW_NO_REF"
            elif old is None and selected["status"] == "SELECTED": change = "NEWLY_SUPPORTED"
            elif old and selected["status"] == "SELECTED" and (old.get("min_value"), old.get("max_value")) != (selected.get("min_value"), selected.get("max_value")):
                change = "OLD_GENERIC_NEW_FORM_OR_DETAIL_SPECIFIC" if selected.get("reference_form") or selected.get("life_stage_detail") else "REFERENCE_CHANGED"
            else: change = "UNCHANGED"
            app_rows.append({"product_id": product_id, "source_dataset": source, "species": species or "UNKNOWN", "life_stage": stage or "UNKNOWN",
                "product_form": canonical_form, "nutrient_code": code, "basis": basis, "value": value,
                "reference_status_group": "COMPARABLE" if comparable else group, "reference_reason_code": selected.get("reason_code"),
                "reference_selected": selected["status"] == "SELECTED"})
            transition_rows.append({"product_id": product_id, "species": species or "UNKNOWN", "life_stage": stage or "UNKNOWN", "product_form": canonical_form,
                "nutrient_code": code, "old_reference": {"min": old.get("min_value"), "max": old.get("max_value")} if old else None,
                "new_reference": new_reference, "old_result": old_result, "new_result": new_result, "change_type": change})

    _write_csv(OUT / f"{VERSION}_product_coverage.csv", coverage_rows)
    _write_csv(OUT / f"{VERSION}_product_reference_applicability.csv", app_rows)
    _write_csv(OUT / f"{VERSION}_legacy_vs_new_transition.csv", transition_rows)
    reason_rows = []
    by_reason_ids: dict[str, list[str]] = defaultdict(list)
    for row in coverage_rows:
        for reason in json.loads(row["reason_codes"] if isinstance(row["reason_codes"], str) else json.dumps(row["reason_codes"])):
            by_reason_ids[reason].append(row["product_id"])
    for reason, ids in sorted(by_reason_ids.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        reason_rows.append({"reason_code": reason, "count": len(ids), "percentage_of_products": round(len(ids) / len(coverage_rows) * 100, 2),
                            "affected_product_ids": ids, "source_distribution": dict(Counter(next(r["source_dataset"] for r in coverage_rows if r["product_id"] == pid) for pid in ids))})
    _write_csv(OUT / f"{VERSION}_reason_code_distribution.csv", reason_rows)

    ca_p = Counter()
    for row in coverage_rows:
        pid = row["product_id"]
        loaded = load_product_input(pid); product = loaded["product"]
        pipeline = _pipeline_items(product)
        ca = next((x for x in pipeline if x.get("nutrient_code") == "CALCIUM"), None)
        ph = next((x for x in pipeline if x.get("nutrient_code") == "PHOSPHORUS"), None)
        ca_p["total_products"] += 1
        ca_p["calcium_present"] += ca is not None; ca_p["phosphorus_present"] += ph is not None
        ca_p["both_present"] += ca is not None and ph is not None
        valid = bool(ca and ph and ca.get("basis_normalized_value") is not None and ph.get("basis_normalized_value") is not None and not ca.get("basis_invalid") and not ph.get("basis_invalid"))
        ca_p["both_valid"] += valid
        same = valid and (ca.get("basis_normalized_basis") or ca.get("basis")) == (ph.get("basis_normalized_basis") or ph.get("basis"))
        ca_p["same_basis"] += bool(same); ca_p["ratio_computable"] += bool(same)
        ca_p["reference_resolvable"] += bool(same and row["category"] == "food" and row["target_species"] in {"DOG", "CAT"} and row["life_stage"] != "UNKNOWN")
    _write_json(OUT / f"{VERSION}_pr_cs_applicability.json", dict(ca_p))
    _write_json(OUT / f"{VERSION}_product_evaluability.json", {"scope": "persisted local product metadata; no pet profile or defaults", "product_count": len(coverage_rows), "counts": dict(evaluability), "by_source": {k: dict(v) for k, v in source_stats.items()}, "reason_code_counts": dict(reasons)})

    metrics = {"total_products": len(coverage_rows), "food_products": sum(r["category"] == "food" for r in coverage_rows), "form_counts": {"|".join(map(str, k)): v for k, v in form_counts.items()}, "life_stage_counts": dict(stage_counts), "life_stage_detail_counts": dict(detail_counts), "source_inventory": {key: dict(value) for key, value in source_inventory.items()},
        "nutrient_coverage": {key: dict(value) for key, value in nutrient_stats.items()}, "reference_applicability": dict(Counter(r["reference_status_group"] for r in app_rows)), "transition_counts": dict(Counter(r["change_type"] for r in transition_rows)), "evaluability": dict(evaluability), "reference_coverage": reference_coverage, "pr_cs": dict(ca_p)}
    _write_json(OUT / f"{VERSION}_summary.json", metrics)
    chart_rows = ([{"chart": "evaluability", "label": key, "count": value} for key, value in evaluability.items()]
                  + [{"chart": "reference_applicability", "label": key, "count": value} for key, value in Counter(r["reference_status_group"] for r in app_rows).items()]
                  + [{"chart": "legacy_new_transition", "label": key, "count": value} for key, value in Counter(r["change_type"] for r in transition_rows).items()]
                  + [{"chart": "life_stage", "label": key, "count": value} for key, value in stage_counts.items()]
                  + [{"chart": "product_form_all", "label": "|".join(map(str, key[1:])), "count": value} for key, value in form_counts.items() if key[0] == "ALL"])
    _write_csv(OUT / f"{VERSION}_chart_data.csv", chart_rows)
    reconciliation = _reconciliation(legacy, references)
    report = f"""# Nutrition Reference Parity P0 — persisted-product applicability 분석

## 실측 범위

- local product records: {metrics['total_products']}
- source-confirmed FOOD: {metrics['food_products']}
- 이 분석은 raw 데이터를 수정하거나 누락값을 보정하지 않았다.

## Reference reconciliation

`{json.dumps({key: value for key, value in reconciliation.items() if key != 'old_rows'}, ensure_ascii=False)}`

## Source inventory

`{json.dumps(metrics['source_inventory'], ensure_ascii=False)}`

## Form / life-stage coverage

`{json.dumps(metrics['form_counts'], ensure_ascii=False)}`

`{json.dumps(metrics['life_stage_counts'], ensure_ascii=False)}`

## Nutrient coverage

`{json.dumps(metrics['nutrient_coverage'], ensure_ascii=False)}`

## Evaluability

`{json.dumps(dict(evaluability), ensure_ascii=False)}`

## Reference applicability and transition

`{json.dumps(metrics['reference_applicability'], ensure_ascii=False)}`

`{json.dumps(metrics['transition_counts'], ensure_ascii=False)}`

## PR-CS 사전 조건

`{json.dumps(dict(ca_p), ensure_ascii=False)}`

## 해석 제한

- product form 또는 life-stage detail이 없는 경우 임의 추정하지 않아 `NO_REF` 계열로 남긴다.
- 결과는 local artifact의 metadata completeness이며, 임상적 적합성 또는 전체 상품 population을 의미하지 않는다.
"""
    (OUT / f"{VERSION}_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"products": len(coverage_rows), "food": metrics["food_products"], "evaluability": dict(evaluability), "pr_cs": dict(ca_p)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
