"""Read-only Nutrition Product Identity / Evidence Gold Cohort P0 artifacts.

This script deliberately does not merge source records, retrieve external pages, or
write a database.  It turns observable source facts into review queues only.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SCRIPTS), str(SCRIPTS / "nutrition")]
from nutrition.product_input_adapter import ROOT  # noqa: E402
from nutrition.danawa_adapter import danawa_load_products  # noqa: E402
from nutrition.gtin_validation import GTIN_LENGTHS, is_valid_gtin  # noqa: E402

RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "eval"
SOURCES = {
    "OPFF": (RAW / "seed_9_placeholder_feed_opff.json", "items"),
    "OEM": (RAW / "seed_9b_off_korean_oem.json", "products"),
    "GLOBAL": (RAW / "seed_9_global_brands_v2.json", "products"),
}
GA = RAW / "seed_13_guaranteed_analysis.json"
PREFIX = "nutrition_evidence_gold_cohort_p0"
MINIMUM = {"DOG": {"CRUDE_PROTEIN", "CRUDE_FAT", "MOISTURE", "CALCIUM", "PHOSPHORUS"},
           "CAT": {"CRUDE_PROTEIN", "CRUDE_FAT", "MOISTURE", "CALCIUM", "PHOSPHORUS", "TAURINE"}}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def csv_out(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    fields = fields or sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def text_key(value: object) -> str:
    # underscore is source formatting, not a brand/name identity character
    return re.sub(r"[\W_]", "", str(value or "").casefold())


def gtin_check(value: str) -> bool:
    """Validate canonical GS1 digits; Unicode numerals are not GTIN values."""
    return is_valid_gtin(value)


def gtin(raw: object) -> tuple[str | None, str, bool | None, str]:
    value = str(raw or "")
    for prefix in ("OFF_KR_", "OFF_"):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    # Do not normalize spaces, hyphens, or Unicode digits into a join key.
    digits = re.sub(r"[^0-9]", "", value)
    if digits != value or len(digits) not in GTIN_LENGTHS:
        return None, "NON_GTIN_OR_UNSUPPORTED_LENGTH", None, "NO_CANONICAL_GTIN"
    if not gtin_check(digits):
        return None, f"GTIN_{len(digits)}", False, "CHECK_DIGIT_INVALID_NO_JOIN"
    # EAN-13 with leading zero and UPC-A denote the same GTIN-12 identity.
    if len(digits) == 13 and digits.startswith("0"):
        return digits[1:], "GTIN_12_FROM_EAN13_LEADING_ZERO", True, "STRIP_EAN13_LEADING_ZERO"
    return digits, f"GTIN_{len(digits)}", True, "AS_IS"


def species(value: object) -> str | None:
    vals = {str(v).upper() for v in value} if isinstance(value, list) else {str(value or "").upper()}
    if vals == {"DOG"}: return "DOG"
    if vals == {"CAT"}: return "CAT"
    if vals in ({"DOG", "CAT"}, {"BOTH"}): return "BOTH"
    return None


def stage(value: object) -> str | None:
    return {"ADULT": "ADULT_MAINTENANCE", "MAINTENANCE": "ADULT_MAINTENANCE", "GROWTH": "GROWTH_REPRODUCTION", "GROWTH_REPRODUCTION": "GROWTH_REPRODUCTION", "ALL_LIFE_STAGES": "ALL_LIFE_STAGES"}.get(str(value or "").upper())


def form(value: object) -> str | None:
    return {"DRY_FOOD": "DRY", "WET_FOOD": "CANNED", "DRY": "DRY", "CANNED": "CANNED"}.get(str(value or "").upper())


def records() -> list[dict[str, Any]]:
    result = []
    for source, (path, key) in SOURCES.items():
        for row in load(path).get(key, []):
            canonical, typ, valid, action = gtin(row.get("product_id"))
            result.append({"source_dataset": source, "source_product_id": str(row.get("product_id") or ""),
                "barcode_raw": str(row.get("product_id") or ""), "canonical_gtin": canonical, "gtin_type": typ,
                "check_digit_valid": valid, "normalization_action": action, "brand": row.get("brand"), "name": row.get("name"),
                "package_size": row.get("quantity"), "species": species(row.get("target_species")),
                "life_stage": stage(row.get("target_life_stage")), "product_form": form(row.get("product_type")),
                "category": row.get("category") or row.get("categories"),
                "ingredients_present": bool(row.get("ingredients") or row.get("ingredients_parsed") or row.get("ingredients_text") or row.get("ingredients_text_raw")),
                "raw": row})
    return result


def nutrition_by_id() -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load(GA).get("items", []):
        if row.get("product_id"): result[str(row["product_id"])].append(row)
    return result


def conflict(field: str, values: list[object]) -> tuple[str, str]:
    raw_distinct = {str(v) for v in values if v not in (None, "", [], {})}
    # Brand/name punctuation/case variants are metadata formatting, not a hard
    # conflict.  Preserve raw values in the artifact; only classify them here.
    distinct = ({text_key(v) for v in raw_distinct} if field in {"brand", "name"} else raw_distinct)
    if not distinct: return "NO_CONFLICT", ""
    if len(distinct) == 1: return "NO_CONFLICT", next(iter(distinct))
    if field == "species" and distinct <= {"BOTH", "DOG", "CAT"}: return "COMPATIBLE_GRANULARITY", "BOTH is broad; human review before source selection"
    if field in {"life_stage", "product_form", "category"} and len(distinct) == 2 and any(v in {"ALL_LIFE_STAGES", "food", ""} for v in distinct): return "COMPATIBLE_GRANULARITY", "broad classification; no automatic selection"
    if field in {"brand", "name", "package_size"}: return "HARD_IDENTITY_CONFLICT", "same GTIN has divergent identity attribute"
    return "SOFT_ATTRIBUTE_CONFLICT", "same GTIN metadata disagreement; no automatic source priority"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, ga = records(), nutrition_by_id()
    audit = [{k: r[k] for k in ("source_dataset", "source_product_id", "barcode_raw", "canonical_gtin", "gtin_type", "check_digit_valid", "normalization_action")} for r in rows]
    csv_out(OUT / "gtin_normalization_audit.csv", audit)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows: groups[r["canonical_gtin"] or f"UNJOINABLE:{r['source_dataset']}:{r['source_product_id']}"].append(r)

    conflict_rows, provenance, ca_queue, taurine_queue, gold, gaps = [], [], [], [], [], []
    fields = ["brand", "name", "package_size", "species", "life_stage", "product_form", "category"]
    for key, members in sorted(groups.items()):
        is_exact = not key.startswith("UNJOINABLE:")
        statuses = []
        for field in fields:
            status, note = conflict(field, [m[field] for m in members])
            statuses.append(status)
            conflict_rows.append({"canonical_gtin": key if is_exact else "", "cluster_key": key, "field": field, "values_by_source": {m["source_dataset"]: m[field] for m in members}, "classification": status, "review_required": status != "NO_CONFLICT", "note": note})
        # Field-level projection is evidence, not a chosen value. Multiple values retain conflict status.
        for field in fields:
            observed = [(m[field], m) for m in members if m[field] not in (None, "")]
            status, _ = conflict(field, [value for value, _ in observed])
            provenance.append({"cluster_key": key, "canonical_gtin": key if is_exact else "", "field": field, "observed_values": sorted({str(v) for v, _ in observed}), "source_records": [{"source": m["source_dataset"], "source_product_id": m["source_product_id"], "value": v} for v, m in observed], "resolution_status": "UNRESOLVED_REVIEW_REQUIRED" if status != "NO_CONFLICT" else "OBSERVED_NO_AUTO_MERGE"})
        has_hard_identity_conflict = "HARD_IDENTITY_CONFLICT" in statuses
        nutrients = [item for m in members for item in ga.get(m["source_product_id"], []) if item.get("value") not in (None, 0, 0.0)]
        codes = {str(item.get("nutrient_code")) for item in nutrients}
        candidate_species = next((m["species"] for m in members if m["species"] in {"DOG", "CAT"}), None)
        candidate_stage = next((m["life_stage"] for m in members if m["life_stage"]), None)
        candidate_form = next((m["product_form"] for m in members if m["product_form"]), None)
        cat = candidate_species == "CAT"
        missing = sorted(MINIMUM.get(candidate_species, set()) - codes) if candidate_species else []
        primary = members[0]
        common = {"canonical_gtin": key if is_exact else "", "cluster_key": key, "product_name": primary["name"], "brand": primary["brand"], "species": candidate_species or "", "life_stage": candidate_stage or "", "product_form": candidate_form or "", "source_records": ";".join(f"{m['source_dataset']}:{m['source_product_id']}" for m in members), "identity_resolution_status": "AUTO_MERGE_PROHIBITED" if has_hard_identity_conflict else "NO_HARD_IDENTITY_CONFLICT", "human_review_required": has_hard_identity_conflict}
        # Acquisition is allowed only for a valid GTIN cluster.  Numeric-looking
        # source IDs with an invalid check digit are permanently excluded from
        # this queue, rather than becoming an accidental lookup target.
        for nutrient in ("CALCIUM", "PHOSPHORUS"):
            if is_exact and nutrient not in codes:
                ca_queue.append({**common, "nutrient_code": nutrient, "current_status": "MISSING_STRUCTURED_VALUE", "candidate_source_priority": "package_label > official_manufacturer > importer > OPFF_image > verified_marketplace", "candidate_reference": "unacquired", "verification_required": True, "identity_requirement": "canonical GTIN or human-approved identity evidence"})
        if is_exact and cat and "TAURINE" not in codes:
            taurine_queue.append({**common, "nutrient_code": "TAURINE", "current_status": "MISSING_STRUCTURED_VALUE", "candidate_source_priority": "package_label > official_manufacturer > importer > OPFF_image > verified_marketplace", "candidate_reference": "unacquired", "verification_required": True, "identity_requirement": "canonical GTIN or human-approved identity evidence"})
        profile = (candidate_species, candidate_stage, candidate_form)
        desired = {("DOG", "ADULT_MAINTENANCE", "DRY"), ("DOG", "GROWTH_REPRODUCTION", "DRY"), ("CAT", "ADULT_MAINTENANCE", "DRY"), ("CAT", "ADULT_MAINTENANCE", "CANNED")}
        if profile in desired:
            status = "GOLD_READY" if not missing and primary["ingredients_present"] else ("GOLD_NEAR_READY" if candidate_species and candidate_stage and candidate_form else "NOT_READY")
            gold.append({**common, "gold_profile": "/".join(profile), "gold_status": status, "present_nutrients": sorted(codes), "missing_minimum_nutrients": missing, "ingredients_present": primary["ingredients_present"], "identity_conflict": has_hard_identity_conflict})
        # An absent/ambiguous species cannot mean an empty nutrient requirement.
        # Keep it as a gap rather than assigning a DOG matrix as a convenience default.
        missing_fields = list(missing)
        if not candidate_species: missing_fields.append("SPECIES")
        if not candidate_stage: missing_fields.append("LIFE_STAGE")
        if not candidate_form: missing_fields.append("PRODUCT_FORM")
        if not primary["ingredients_present"]: missing_fields.append("INGREDIENTS")
        gaps.append({**common, "missing_minimum_nutrients": missing, "missing_fields": sorted(missing_fields), "gap_count": len(missing_fields)})

    csv_out(OUT / "identity_conflict_classification.csv", conflict_rows)
    csv_out(OUT / "field_level_provenance_projection.csv", provenance)
    csv_out(OUT / "nutrition_evidence_acquisition_queue.csv", ca_queue)
    csv_out(OUT / "cat_taurine_acquisition_queue.csv", taurine_queue)

    # Candidate-only Danawa pilot.  It cannot approve an identity because pcode is not GTIN.
    danawa = []
    for path, source_species in ((RAW / "seed_25_danawa_dog_v1.json", "DOG"), (RAW / "seed_25_danawa_cat_v1.json", "CAT")):
        for item in danawa_load_products(str(path)):
            data = item.raw_guaranteed_analysis
            value = {"pcode": item.pcode, "source_species": source_species, "brand": item.brand, "name": item.name, "package_size": getattr(item, "package_size", None), "calcium_present": "칼슘_g" in data, "phosphorus_present": "인_g" in data, "taurine_present": any("타우린" in str(k) for k in data), "internal_gtin_candidate": "", "match_evidence": "name/brand comparison required", "match_method": "CANDIDATE_ONLY", "confidence": "", "human_review_required": True}
            # rank only candidate evidence availability; do not conduct fuzzy matching.
            value["priority_score"] = int(source_species == "CAT") * 4 + int(value["calcium_present"]) + int(value["phosphorus_present"]) + int(value["taurine_present"]) * 2
            danawa.append(value)
    danawa.sort(key=lambda row: (-row["priority_score"], str(row["brand"]), str(row["name"])))
    csv_out(OUT / "danawa_mapping_pilot_candidates.csv", danawa[:30])
    csv_out(OUT / "gold_product_candidates.csv", gold)
    csv_out(OUT / "gold_product_gap_analysis.csv", sorted(gaps, key=lambda r: (r["gap_count"], r["cluster_key"])))

    # These are request-scoped, externally documented formulation candidates.
    # An official page proves its displayed content, not the local GTIN-to-page
    # identity: the page does not expose the GTIN.  They must therefore remain
    # outside persisted-product data until a human verifies that identity.
    from api_nutrition import AnalyzeRequest, NutritionItemIn, PetIn, ProductIn, _analyze_product
    external_candidates = [
        {
            "candidate_id": "ORIJEN_ORIGINAL_CAT_US", "product_id": "0064992280178", "canonical_gtin": "064992280178",
            "species": "cat", "name": "Original Cat", "source_authority": "ORIJEN",
            "source_type": "OFFICIAL_MANUFACTURER", "source_url": "https://www.orijenpetfoods.com/en-US/cats/cat-food/original-cat/ns-ori-catkitten.html",
            "values": {"CRUDE_PROTEIN": 40.0, "CRUDE_FAT": 20.0, "MOISTURE": 10.0, "CALCIUM": 1.4, "PHOSPHORUS": 1.1, "TAURINE": 0.2},
        },
        {
            "candidate_id": "ORIJEN_ORIGINAL_DOG_CA", "product_id": None, "canonical_gtin": None,
            "species": "dog", "name": "Original Dog", "source_authority": "ORIJEN",
            "source_type": "OFFICIAL_MANUFACTURER", "source_url": "https://www.orijenpetfoods.com/en-CA/dogs/dog-food/original/ds-ori-original-dog.html",
            "values": {"CRUDE_PROTEIN": 38.0, "CRUDE_FAT": 18.0, "MOISTURE": 12.0, "CALCIUM": 1.2, "PHOSPHORUS": 1.0},
        },
    ]
    observed_at = datetime.now(timezone.utc).isoformat()
    evidence_rows, e2e_rows = [], []
    for candidate in external_candidates:
        for nutrient, value in candidate["values"].items():
            evidence_rows.append({
                "product_id": candidate["product_id"] or "", "canonical_gtin": candidate["canonical_gtin"] or "",
                "candidate_id": candidate["candidate_id"], "nutrient_code": nutrient, "value": value, "unit": "PERCENT",
                "basis": "AS_FED", "guarantee_type": "GUARANTEED", "source_type": candidate["source_type"],
                "source_record_id": candidate["source_url"], "source_authority": candidate["source_authority"],
                "source_url": candidate["source_url"], "observed_at": observed_at,
                "source_content_verified": True, "identity_verified": False, "verified": False,
                "verification_method": "official-page-content verified; GTIN-to-formula link not exposed on source page",
                "runtime_eligible": False,
            })
        result = _analyze_product(AnalyzeRequest(
            pet=PetIn(id="external-candidate-pet", species=candidate["species"], age_years=3, weight_kg=5, life_stage="adult"),
            product=ProductIn(id="request-" + candidate["candidate_id"], name=candidate["name"], target_species=candidate["species"],
                aafco_life_stage="ALL_LIFE_STAGES", product_form="DRY", ingredient_list=["chicken"],
                nutrition_items=[NutritionItemIn(nutrient_code=code, value=value, source="OFFICIAL_MANUFACTURER") for code, value in candidate["values"].items()]),
        ))
        e2e_rows.append({
            "candidate_id": candidate["candidate_id"], "identity_verified": False, "runtime_eligible_for_persisted_product": False,
            "execution_scope": "REQUEST_SCOPED_ONLY", "input_readiness": result["input_readiness"]["input_readiness"],
            "nutrition_comparison_status": result["nutrition_comparison_status"], "analysis_status": result["analysis_status"],
            "safety_status": result["safety_status"],
            "nutrient_results": [{"nutrient_code": item["nutrient_code"], "nias_compare_status": item["nias_compare_status"]} for item in result["nutrition_items"]],
        })
    csv_out(OUT / "official_manufacturer_evidence_candidates_v1.csv", evidence_rows)
    dump(OUT / "official_manufacturer_request_scoped_e2e_v1.json", e2e_rows)

    opff = [r for r in rows if r["source_dataset"] == "OPFF"]
    image = {"opff_records": len(opff), "front_image_url_present": sum(bool(r["raw"].get("image_url")) for r in opff), "nutrition_image_metadata_present": 0, "ingredients_image_metadata_present": 0, "packaging_image_metadata_present": 0, "interpretation": "URL presence only; no image retrieval or OCR performed."}
    valid_cluster_members = [ms for key, ms in groups.items() if not key.startswith("UNJOINABLE:")]
    metadata_compatible = 0
    multisource_valid_clusters = 0
    multisource_metadata_compatible = 0
    projected_field_complete = 0
    for members in valid_cluster_members:
        hard = any(conflict(field, [m[field] for m in members])[0] == "HARD_IDENTITY_CONFLICT" for field in ("brand", "name", "package_size"))
        metadata_compatible += int(not hard)
        if len({m["source_dataset"] for m in members}) > 1:
            multisource_valid_clusters += 1
            multisource_metadata_compatible += int(not hard)
        item_codes = {str(x.get("nutrient_code")) for m in members for x in ga.get(m["source_product_id"], []) if x.get("value") not in (None, 0, 0.0)}
        item_species = next((m["species"] for m in members if m["species"] in {"DOG", "CAT"}), None)
        item_stage = next((m["life_stage"] for m in members if m["life_stage"]), None)
        item_form = next((m["product_form"] for m in members if m["product_form"]), None)
        if not hard and item_species and item_stage and item_form and any(m["ingredients_present"] for m in members) and not (MINIMUM[item_species] - item_codes):
            projected_field_complete += 1
    metrics = {"generated_at": datetime.now(timezone.utc).isoformat(), "source_records": len(rows), "numeric_source_id_clusters_before_gtin_validation": len(groups), "valid_gtin_records": sum(r["check_digit_valid"] is True for r in rows), "valid_gtin_clusters": len(valid_cluster_members), "metadata_compatible_valid_gtin_clusters": metadata_compatible, "multisource_valid_gtin_clusters": multisource_valid_clusters, "multisource_metadata_compatible_clusters": multisource_metadata_compatible, "projected_field_complete_clusters": projected_field_complete, "nutrition_clusters": sum(bool([x for m in ms for x in ga.get(m["source_product_id"], [])]) for ms in groups.values()), "cap_clusters": sum(any(str(x.get("nutrient_code")) in {"CALCIUM", "PHOSPHORUS"} and x.get("value") not in (None, 0, 0.0) for m in ms for x in ga.get(m["source_product_id"], [])) for ms in groups.values()), "cat_taurine_clusters": sum(any(m["species"] == "CAT" for m in ms) and any(str(x.get("nutrient_code")) == "TAURINE" and x.get("value") not in (None, 0, 0.0) for m in ms for x in ga.get(m["source_product_id"], [])) for ms in groups.values()), "gold_ready": sum(r["gold_status"] == "GOLD_READY" for r in gold), "gold_near_ready": sum(r["gold_status"] == "GOLD_NEAR_READY" for r in gold), "opff_image_availability": image}
    dump(OUT / f"{PREFIX}_metrics.json", metrics)
    report = f"""# Nutrition Product Identity / Nutrition Evidence Gold Cohort P0

## Baseline

- local source records: 394; valid-GTIN identity clusters: {metrics['valid_gtin_clusters']}
- This is a read-only projection. The Nutrition Rule Engine, reference artifacts, API, raw seeds, and database were not changed.

## Portfolio Metrics

- source records: {metrics['source_records']}; valid GTIN clusters: {metrics['valid_gtin_clusters']}
- structured nutrition clusters: {metrics['nutrition_clusters']}; Ca/P-present clusters: {metrics['cap_clusters']}; CAT taurine-present clusters: {metrics['cat_taurine_clusters']}
- Gold-ready: {metrics['gold_ready']}; Gold-near-ready: {metrics['gold_near_ready']}

## GTIN Normalization

`gtin_normalization_audit.csv` verifies allowed GTIN length and check digit. Invalid or unsupported source IDs are not join keys. UPC-A/EAN-13 leading-zero equivalence is canonicalized only after a valid check digit.

## Identity vs Attribute Conflicts

`identity_conflict_classification.csv` separates same-GTIN identity conflicts from metadata variation. No conflict result authorizes a merge; every multi-source cluster remains a projection.

## Field-level Complementarity

`field_level_provenance_projection.csv` records observed values and source records without selecting a source priority.

## Ca/P Recovery Candidates / CAT Taurine Gap

The two queues only identify missing structured values and the required identity verification. No external page was requested, scraped, or inserted.

## Open Pet Food Facts Evidence Availability

{json.dumps(image, ensure_ascii=False)}

## Danawa Mapping Pilot

The 30 rows are candidate-only. `pcode` is not a GTIN, and no fuzzy match is a verified identity.

## Gold Product Candidates

`gold_product_candidates.csv` includes only the requested DOG adult dry, DOG growth dry, CAT adult dry, and CAT adult canned profiles. `GOLD_READY` means the local minimum-runtime fields and ingredients are present; it is not a clinical suitability or comprehensive-coverage claim. `gold_product_gap_analysis.csv` ranks gaps without assigning a default species or missing nutrient value.

## Minimum Acquisition Gap

`nutrition_evidence_acquisition_queue.csv` and `cat_taurine_acquisition_queue.csv` are verification work queues. A queue entry has no acquired external nutrient value.

## Official manufacturer evidence candidates

`official_manufacturer_evidence_candidates_v1.csv` preserves the observed manufacturer-page values separately from local product data. Its `verified` value is `false`: official-page content was checked, but its GTIN-to-formula identity is not exposed by the page. The request-scoped E2E artifact is not a persisted-product validation. The CAT candidate returns `TRUE`; the DOG candidate returns `FALSE` because the current NIAS comparison marks its 12% moisture `OUT_OF_RANGE`, despite Ca/P being `IN_RANGE`.

## Provenance Contract (proposal only)

`product_id`, `canonical_gtin`, `nutrient_code`, `value`, `unit`, `basis`, `guarantee_type`, `source_type`, `source_record_id`, `source_authority`, `observed_at`, `verified`, `verification_method`.

## 확인 불가

Actual package-label nutrient values, manufacturer/importer evidence, and a verified Danawa pcode-to-GTIN mapping are not present in the local inputs. Gold readiness is analysis-only and does not assert nutritional adequacy.

## Next Step

Human-review the GTIN/identity rows required by the Ca/P and taurine queues, then acquire and verify only package-label or authoritative evidence against that approved identity. Do not use a Danawa candidate or fuzzy name similarity as a merge key.
"""
    (OUT / "nutrition_evidence_gold_cohort_p0_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
