"""Read-only exact-identity join projection for local Nutrition product sources."""
from __future__ import annotations

import csv
import json
import re
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SCRIPTS), str(SCRIPTS / "nutrition")]

from nutrition.product_input_adapter import ROOT  # noqa: E402
from nutrition.danawa_adapter import danawa_load_products  # noqa: E402


RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "eval"
PREFIX = "product_identity_projection_v1"
SOURCE_FILES = {
    "OPFF": (RAW / "seed_9_placeholder_feed_opff.json", "items"),
    "OEM": (RAW / "seed_9b_off_korean_oem.json", "products"),
    "GLOBAL": (RAW / "seed_9_global_brands_v2.json", "products"),
}
GA = RAW / "seed_13_guaranteed_analysis.json"
DANAWA = (RAW / "seed_25_danawa_dog_v1.json", RAW / "seed_25_danawa_cat_v1.json")
LEGACY_DB = ROOT / "data" / "processed" / "verify_P2_v1.db"
MINIMUM = {"DOG": {"CRUDE_PROTEIN", "CRUDE_FAT", "MOISTURE", "CALCIUM", "PHOSPHORUS"},
           "CAT": {"CRUDE_PROTEIN", "CRUDE_FAT", "MOISTURE", "CALCIUM", "PHOSPHORUS", "TAURINE"}}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        out = csv.DictWriter(handle, fieldnames=keys); out.writeheader()
        for row in rows:
            out.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def identity_key(value: object) -> str | None:
    """Known source prefixes normalise to barcode only when remainder is numeric."""
    value = str(value or "")
    for prefix in ("OFF_KR_", "OFF_"):
        if value.startswith(prefix): value = value[len(prefix):]
    return value if value.isdigit() and len(value) >= 8 else None


def text_key(value: object) -> str:
    return re.sub(r"[^\w가-힣]", "", unicodedata.normalize("NFKC", str(value or "")).casefold())


def species_value(value: object) -> str | None:
    values = {str(item).upper() for item in value} if isinstance(value, list) else {str(value or "").upper()}
    if values == {"DOG"}: return "DOG"
    if values == {"CAT"}: return "CAT"
    if values in ({"DOG", "CAT"}, {"BOTH"}): return "BOTH"
    return None


def stage_value(value: object) -> str | None:
    raw = str(value or "").upper()
    return {"ADULT": "ADULT_MAINTENANCE", "MAINTENANCE": "ADULT_MAINTENANCE", "GROWTH": "GROWTH_REPRODUCTION", "GROWTH_REPRODUCTION": "GROWTH_REPRODUCTION", "ALL_LIFE_STAGES": "ALL_LIFE_STAGES"}.get(raw)


def form_value(value: object) -> str | None:
    return {"DRY_FOOD": "DRY", "WET_FOOD": "CANNED", "DRY": "DRY", "CANNED": "CANNED"}.get(str(value or "").upper())


def product_records() -> list[dict[str, Any]]:
    records = []
    for source, (path, list_key) in SOURCE_FILES.items():
        for row in read_json(path).get(list_key, []):
            identifier = row.get("product_id")
            records.append({"source": source, "source_id": identifier, "identity_key": identity_key(identifier),
                "brand": row.get("brand"), "name": row.get("name"), "name_key": text_key(row.get("name")),
                "species": species_value(row.get("target_species")), "life_stage": stage_value(row.get("target_life_stage")),
                "product_form": form_value(row.get("product_type")),
                "ingredients_present": bool(row.get("ingredients") or row.get("ingredients_parsed") or row.get("ingredients_text") or row.get("ingredients_text_raw")),
                "inline_nutrition_present": bool(row.get("nutrients") or row.get("guaranteed_analysis")), "raw": row})
    return records


def guaranteed_analysis() -> dict[str, list[dict[str, Any]]]:
    rows = read_json(GA).get("items", [])
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("product_id"): out[str(row["product_id"])].append(row)
    return out


def pairs(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    l_by_id, r_by_id = defaultdict(list), defaultdict(list)
    for row in left:
        if row["identity_key"]: l_by_id[row["identity_key"]].append(row)
    for row in right:
        if row["identity_key"]: r_by_id[row["identity_key"]].append(row)
    exact, candidates = [], []
    for key in sorted(set(l_by_id) & set(r_by_id)):
        for a in l_by_id[key]:
            for b in r_by_id[key]:
                conflict = bool(a["brand"] and b["brand"] and text_key(a["brand"]) != text_key(b["brand"]))
                metadata_conflicts = [field for field in ("species", "life_stage", "product_form") if a.get(field) and b.get(field) and a.get(field) != b.get(field)]
                exact.append({"left_source": a["source"], "left_id": a["source_id"], "right_source": b["source"], "right_id": b["source_id"],
                    "identity_key": key, "method": "BARCODE_EXACT", "ambiguous_identity": len(l_by_id[key]) > 1 or len(r_by_id[key]) > 1,
                    "brand_conflict": conflict, "metadata_conflicts": metadata_conflicts, "left_name": a["name"], "right_name": b["name"]})
    # Candidate-only: identical normalized brand + name but no shared barcode.
    right_names = defaultdict(list)
    for row in right:
        if row["name_key"]: right_names[(text_key(row["brand"]), row["name_key"])].append(row)
    for a in left:
        key = (text_key(a["brand"]), a["name_key"])
        for b in right_names.get(key, []):
            if a["identity_key"] != b["identity_key"]:
                candidates.append({"left_source": a["source"], "left_id": a["source_id"], "right_source": b["source"], "right_id": b["source_id"],
                    "method": "NORMALIZED_BRAND_NAME_CANDIDATE_ONLY", "left_name": a["name"], "right_name": b["name"]})
    return exact, candidates, {"left_records": len(left), "right_records": len(right), "exact_identity_keys": len(set(l_by_id) & set(r_by_id)),
        "exact_pairs": len(exact), "ambiguous_exact_pairs": sum(x["ambiguous_identity"] for x in exact), "brand_conflicts": sum(x["brand_conflict"] for x in exact), "metadata_conflict_pairs": sum(bool(x["metadata_conflicts"]) for x in exact),
        "candidate_name_pairs_not_approved": len(candidates), "unmatched_left": len(left) - len({x["left_id"] for x in exact}), "unmatched_right": len(right) - len({x["right_id"] for x in exact})}


def danawa_inventory() -> dict[str, Any]:
    products = [product for path in DANAWA for product in danawa_load_products(str(path))]
    return {"products": len(products), "stable_source_id": "pcode", "barcode_or_gtin_present": 0,
        "species_from_source_split": dict(Counter("DOG" if product.pcode else "UNKNOWN" for product in products[:298])) | {"CAT": len(products[298:])},
        "life_stage_known": sum(product.life_stage_status == "KNOWN" for product in products),
        "nutrition_nonempty": sum(bool(product.raw_guaranteed_analysis) for product in products),
        "calcium_present": sum("칼슘_g" in product.raw_guaranteed_analysis for product in products),
        "phosphorus_present": sum("인_g" in product.raw_guaranteed_analysis for product in products),
        "moisture_present": sum(product.moisture_percent is not None for product in products),
        "ingredients_present": sum(bool(product.ingredient_list) for product in products),
        "form_explicit": "not a separate adapter field; inferred from description and therefore not identity evidence"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = product_records(); by_source = {source: [r for r in records if r["source"] == source] for source in SOURCE_FILES}
    ga = guaranteed_analysis()
    exact_rows, candidate_rows, matrices = [], [], []
    sources = list(SOURCE_FILES)
    for index, left_name in enumerate(sources):
        for right_name in sources[index + 1:]:
            exact, candidates, matrix = pairs(by_source[left_name], by_source[right_name])
            exact_rows += exact; candidate_rows += candidates
            matrices.append({"left_source": left_name, "right_source": right_name, **matrix})
    write_csv(OUT / f"{PREFIX}_product_identity_exact_matches.csv", exact_rows)
    write_csv(OUT / f"{PREFIX}_product_identity_candidates.csv", candidate_rows)
    write_csv(OUT / f"{PREFIX}_cross_source_join_matrix.csv", matrices)

    # Exact barcode clusters.  No source row is modified; this is only a union
    # of observable field-presence for projected coverage.
    cluster = defaultdict(list)
    for row in records:
        cluster[row["identity_key"] or f"{row['source']}:{row['source_id']}"].append(row)
    current = Counter(); projected = Counter(); blockers = Counter(); ready_candidates = []
    projected_rows = []
    for key, members in cluster.items():
        primary = members[0]
        current_species = primary["species"]
        current_stage = primary["life_stage"]
        current_form = primary["product_form"]
        current_nutrition = bool(ga.get(str(primary["source_id"])))
        species = next((m["species"] for m in members if m["species"]), None)
        stage = next((m["life_stage"] for m in members if m["life_stage"]), None)
        form = next((m["product_form"] for m in members if m["product_form"]), None)
        ingredients = any(m["ingredients_present"] for m in members)
        nutrition = []
        for member in members: nutrition.extend(ga.get(str(member["source_id"]), []))
        codes = {row.get("nutrient_code") for row in nutrition if row.get("value") not in (None, 0, 0.0)}
        current["products"] += 1; projected["products"] += 1
        for prefix, values in (("current", (current_species, current_stage, current_form, current_nutrition, primary["ingredients_present"])),
                               ("projected", (species, stage, form, bool(nutrition), ingredients))):
            for label, value in zip(("species", "stage", "form", "nutrition", "ingredients"), values):
                if value: (current if prefix == "current" else projected)[label] += 1
        required = MINIMUM.get(species, MINIMUM["DOG"] if species == "BOTH" else set())
        missing = sorted(required - codes) if required else []
        candidate_status = "PROJECTED_READY_CANDIDATE" if required and not missing and stage and form and ingredients else "NOT_READY"
        if candidate_status == "PROJECTED_READY_CANDIDATE":
            ready_candidates.append({"identity_key": key, "source_combination": sorted({m["source"] for m in members}), "species": species, "stage": stage, "form": form, "nutrition_items": sorted(codes), "ingredients": ingredients, "missing_fields": missing, "expected_readiness": candidate_status})
        for reason, condition in (("NO_IDENTITY_KEY", not primary["identity_key"]), ("NUTRITION_NOT_AVAILABLE", not nutrition),
                                  ("SPECIES_MISSING", not species), ("STAGE_MISSING", not stage), ("FORM_MISSING", not form),
                                  ("INGREDIENTS_MISSING", not ingredients), ("CALCIUM_MISSING", "CALCIUM" not in codes),
                                  ("PHOSPHORUS_MISSING", "PHOSPHORUS" not in codes), ("TAURINE_MISSING", str(species).upper() == "CAT" and "TAURINE" not in codes)):
            if condition: blockers[reason] += 1
        projected_rows.append({"identity_key": key, "member_sources": sorted({m["source"] for m in members}), "member_count": len(members), "species": species, "stage": stage, "form": form, "ingredients_present": ingredients, "nutrition_codes": sorted(codes), "missing_minimum": missing, "expected_readiness": candidate_status})
    write_csv(OUT / f"{PREFIX}_projected_ready_candidates.csv", ready_candidates)
    write_csv(OUT / f"{PREFIX}_unmatched_blockers.csv", [{"blocker": key, "count": value, "percentage_of_identity_clusters": round(value / len(cluster) * 100, 2)} for key, value in blockers.most_common()])
    coverage = {"scope": "exact barcode/source-id join projection only; no raw merge", "identity_clusters": len(cluster), "current_field_coverage": dict(current), "projected_field_coverage": dict(projected),
        "delta": {key: projected[key] - current[key] for key in set(current) | set(projected)}, "projected_ready_candidate_count": len(ready_candidates), "projected_rows": projected_rows}
    write_json(OUT / f"{PREFIX}_projected_coverage.json", coverage)

    legacy_mapping = {"verified_mapping_artifact": None, "legacy_evidence": ["scripts/p2_sql_v1.py:to_ean13()", "docs/canonical_data_contract_v1.md: OFF_KR prefix normalization discussion"],
        "legacy_product_master": {"path": "data/processed/verify_P2_v1.db", "product_count": None, "fields": []}}
    if LEGACY_DB.exists():
        with sqlite3.connect(LEGACY_DB) as conn:
            legacy_mapping["legacy_product_master"]["product_count"] = conn.execute("select count(*) from product_master").fetchone()[0]
            legacy_mapping["legacy_product_master"]["fields"] = [r[1] for r in conn.execute("pragma table_info(product_master)")]
    danawa = danawa_inventory()
    report = f"""# Local product identity exact-join projection

## 범위

barcode/source-id exact match만 projected join으로 사용했다. normalized brand+name match는 candidate-only이며 merge하지 않았다.

## Exact join matrix

`{json.dumps(matrices, ensure_ascii=False)}`

## Projected coverage

`{json.dumps({k: v for k, v in coverage.items() if k != 'projected_rows'}, ensure_ascii=False)}`

## Existing mapping

`{json.dumps(legacy_mapping, ensure_ascii=False)}`

## Danawa read-only inventory

`{json.dumps(danawa, ensure_ascii=False)}`

## 제한

Danawa `pcode`는 current OPFF/GLOBAL/OEM barcode와 직접 join key가 아니다. 실제 BE schema는 이 repository에 없어 docs의 schema 주장만 확인 가능하다.
"""
    (OUT / f"{PREFIX}_report.md").write_text(report, encoding="utf-8")
    write_json(OUT / f"{PREFIX}_source_inventory.json", {"sources": {key: len(value) for key, value in by_source.items()}, "guaranteed_analysis": {"rows": sum(map(len, ga.values())), "product_ids": len(ga)}, "danawa": danawa, "legacy_mapping": legacy_mapping})
    print(json.dumps({"matrix": matrices, "projected": {k: v for k, v in coverage.items() if k != 'projected_rows'}, "danawa": danawa}, ensure_ascii=False))


if __name__ == "__main__":
    main()
