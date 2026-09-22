"""Run P1 Gold human-evidence intake without promoting raw source metadata.

The script is an orchestrator only. It never collects web data, changes a raw
seed, or manufactures an approved record: a missing human intake deterministically
produces a zero-row operational artifact.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SCRIPTS), str(SCRIPTS / "nutrition")]

from build_nutrition_evidence_gold_cohort_p0 import gtin_check, records  # noqa: E402
from gold_evidence_gate import evaluate_runtime_eligibility  # noqa: E402
from gold_evidence_intake import (  # noqa: E402
    INTAKE_FIELDS,
    build_operational_evidence_artifact,
    read_human_gold_evidence_intake,
    validate_human_gold_evidence_intake,
    write_operational_evidence_artifact,
)


ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "data" / "eval"
HUMAN_INTAKE_FILENAME = "human_gold_evidence_intake_v1.csv"
OPERATIONAL_CSV_FILENAME = "verified_evidence_gold_v1.csv"
OPERATIONAL_JSON_FILENAME = "verified_evidence_gold_v1.json"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    rows = list(rows)
    names = list(fields) if fields is not None else sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_list(raw: str) -> list[str]:
    try:
        value = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


def _source_product_ids(source_records: object) -> list[str]:
    """Parse existing ``SOURCE:product_id`` locators; never fuzzy-match names."""
    ids: list[str] = []
    for entry in str(source_records or "").split(";"):
        if ":" not in entry:
            continue
        _, product_id = entry.split(":", 1)
        product_id = product_id.strip()
        if product_id:
            ids.append(product_id)
    return sorted(set(ids))


def prepare_candidates(
    candidate_rows: Iterable[dict[str, str]],
    all_records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand P0 GTIN-cluster candidates into P1 local identity contexts.

    Fields from the raw projection remain context only. The validator must still
    receive an authoritative human identity source and duplicate the identity
    fields across the nutrition evidence rows.
    """
    by_gtin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in all_records:
        canonical_gtin = item.get("canonical_gtin")
        if canonical_gtin:
            by_gtin[str(canonical_gtin)].append(item)

    prepared: list[dict[str, Any]] = []
    for item in candidate_rows:
        canonical_gtin = str(item.get("canonical_gtin") or "")
        members = by_gtin.get(canonical_gtin, [])
        sizes = sorted({
            str(member["raw"].get("quantity"))
            for member in members
            if member.get("raw") and member["raw"].get("quantity")
        })
        sources = [entry for entry in str(item.get("source_records") or "").split(";") if entry]
        hard = str(item.get("identity_conflict")) == "True"
        local_consistent = len(sources) >= 2 and not hard
        identity_state = (
            "REVIEW_REQUIRED" if hard
            else "LOCAL_IDENTITY_CONSISTENT" if local_consistent
            else "IDENTITY_PARTIAL"
        )
        missing_nutrients = as_list(str(item.get("missing_minimum_nutrients") or ""))
        species = str(item.get("species") or "").upper()
        prepared.append({
            **item,
            "candidate_id": f"GOLD_{species}_{canonical_gtin}",
            # This exact local-membership list is a lookup constraint, not a
            # claim that raw identity metadata is authoritative.
            "allowed_product_ids": _source_product_ids(item.get("source_records")),
            "package_size_local": "; ".join(sizes),
            "source_count": len(sources),
            "gtin_valid": gtin_check(canonical_gtin),
            "identity_state": identity_state,
            "evidence_state": "CANDIDATE",
            "identity_conflict": "HARD_IDENTITY_CONFLICT" if hard else "NO_HARD_IDENTITY_CONFLICT",
            "missing_runtime_nutrients": "; ".join(missing_nutrients),
            "missing_runtime_nutrient_count": len(missing_nutrients),
            # No product-name/detail inference is added here. A later human
            # intake must provide any DOG growth detail authoritatively.
            "life_stage_detail": "",
            "verified": False,
            "runtime_eligible": False,
        })
    return prepared


def _shortlist(prepared: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple[Any, ...]:
        return (
            item["identity_conflict"] == "HARD_IDENTITY_CONFLICT",
            not bool(item.get("species")),
            not bool(item.get("product_name") and item.get("brand")),
            not bool(item.get("package_size_local")),
            not bool(item.get("product_form")),
            not bool(item.get("life_stage")),
            int(item["missing_runtime_nutrient_count"]),
            -int(item["source_count"]),
            str(item.get("canonical_gtin") or ""),
        )

    result: list[dict[str, Any]] = []
    for species in ("CAT", "DOG"):
        result.extend(sorted((item for item in prepared if item.get("species") == species), key=key)[:3])
    return result


def _baseline_gate_input(candidate: dict[str, Any]) -> dict[str, Any]:
    """Represent a candidate before human evidence without fabricating facts."""
    return {
        "candidate_id": candidate["candidate_id"],
        "canonical_gtin": candidate["canonical_gtin"],
        "gtin_valid": candidate["gtin_valid"],
        "identity_state": candidate["identity_state"],
        "identity_conflict": candidate["identity_conflict"],
        "species": candidate.get("species"),
        "life_stage": candidate.get("life_stage"),
        "life_stage_detail": candidate.get("life_stage_detail"),
        "life_stage_evidence_state": "UNVERIFIED_LOCAL_METADATA",
        "life_stage_detail_evidence_state": "UNVERIFIED_LOCAL_METADATA",
        "product_form": candidate.get("product_form"),
        "product_form_evidence_state": "UNVERIFIED_LOCAL_METADATA",
    }


def _gate_rows(
    shortlist: Iterable[dict[str, Any]],
    operational_candidates: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {str(record.get("candidate_id")): record for record in operational_candidates}
    rows: list[dict[str, Any]] = []
    for item in shortlist:
        operational = by_id.get(str(item["candidate_id"]))
        if operational is not None:
            result = {
                "runtime_eligible": bool(operational["runtime_eligible"]),
                "reason_codes": list(operational["reason_codes"]),
                "missing_nutrients": list(operational["missing_nutrients"]),
                "evidence_row_count": int(operational["evidence_row_count"]),
                "evidence_state": "HUMAN_INTAKE_VALIDATED",
            }
        else:
            result = {
                **evaluate_runtime_eligibility(_baseline_gate_input(item), []),
                "evidence_row_count": 0,
                "evidence_state": "CANDIDATE",
            }
        rows.append({
            "candidate_id": item["candidate_id"],
            "canonical_gtin": item["canonical_gtin"],
            "identity_state": item["identity_state"],
            **result,
        })
    return rows


def _human_queue(shortlist: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for item in shortlist:
        identity_needs = [
            "NEED_PACKAGE_IMAGE_WITH_GTIN_AND_PRODUCT_NAME",
            "NEED_EXACT_VARIANT_SIZE_CONFIRMATION",
            "NEED_FORMULA_MARKET_CONFIRMATION",
        ]
        nutrient_needs = [f"NEED_{code}_GUARANTEE" for code in as_list(str(item.get("missing_minimum_nutrients") or ""))]
        queue.append({
            "candidate_id": item["candidate_id"],
            "canonical_gtin": item["canonical_gtin"],
            "product_name": item.get("product_name"),
            "species": item.get("species"),
            "current_state": item["identity_state"],
            "missing_identity_evidence": "; ".join(identity_needs),
            "missing_nutrient_evidence": "; ".join(nutrient_needs),
            "required_document_or_image": (
                "manufacturer package/document with GTIN, product name, variant/size and market; "
                "plus matching official guaranteed analysis"
            ),
            "next_human_action": (
                "capture immutable document URL or package image reference; do not transcribe "
                "retailer values as manufacturer evidence"
            ),
        })
    return queue


def _pet_life_stage(candidate: dict[str, Any]) -> str:
    """Choose an explicit deterministic test-pet stage from approved evidence."""
    stage = candidate.get("life_stage")
    if stage == "GROWTH_REPRODUCTION":
        return "kitten" if candidate.get("species") == "CAT" else "puppy"
    return "adult"


def _real_persisted_product_e2e(
    operational_candidates: Iterable[dict[str, Any]],
    operational_json_path: Path,
) -> dict[str, Any]:
    """Run only candidates already approved by the strict evidence gate.

    The deterministic pet is an execution harness, not a substituted product
    evidence fixture. This function is never reached for the current zero-row
    human intake.
    """
    eligible = [record for record in operational_candidates if record.get("runtime_eligible") is True]
    if not eligible:
        return {
            "version": "v1",
            "executed": False,
            "execution_scope": "REAL_PERSISTED_PRODUCT_GOLD_E2E",
            "execution_rule": "Run only when strict human validation and runtime eligibility are both true.",
            "reason": "No human-validated runtime-eligible persisted Gold evidence exists.",
            "eligible_candidates": [],
            "candidate_results": [],
        }

    # Import lazily so generation of a legitimate zero-row artifact does not
    # require the HTTP/API dependency stack.
    from api_nutrition import AnalyzeRequest, PetIn, ProductIn, _analyze_product
    from product_input_adapter import load_product_input

    results: list[dict[str, Any]] = []
    for record in eligible:
        candidate = dict(record["runtime_candidate"])
        product_id = str(candidate["product_id"])
        species = str(candidate["species"]).lower()
        try:
            loaded = load_product_input(product_id, operational_evidence_path=operational_json_path)
            product = ProductIn(**loaded["product"])
            pet = PetIn(
                id=f"gold-persisted-e2e-{candidate['candidate_id']}",
                species=species,
                age_years=0.5 if _pet_life_stage(candidate) in {"puppy", "kitten"} else 3.0,
                weight_kg=5.0,
                allergies=[],
                allergy_profile_status="KNOWN_NONE",
                life_stage=_pet_life_stage(candidate),
                life_stage_detail=(
                    candidate.get("life_stage_detail")
                    if candidate.get("life_stage") == "GROWTH_REPRODUCTION" else None
                ),
            )
            result = _analyze_product(AnalyzeRequest(pet=pet, product=product))
            results.append({
                "candidate_id": candidate["candidate_id"],
                "product_id": product_id,
                "runtime_eligible": True,
                "analysis_executed": True,
                "analysis_engine": result.get("analysis_engine"),
                "input_readiness": result.get("input_readiness", {}).get("input_readiness"),
                "nutrition_comparison_status": result.get("nutrition_comparison_status"),
                "safety_status": result.get("safety_status"),
                "analysis_status": result.get("analysis_status"),
                "input_provenance": loaded.get("provenance"),
            })
        except Exception as exc:  # A bad source never becomes a success.
            results.append({
                "candidate_id": candidate["candidate_id"],
                "product_id": product_id,
                "runtime_eligible": True,
                "analysis_executed": False,
                "execution_error_type": type(exc).__name__,
                "analysis_status": "INSUFFICIENT_DATA",
            })
    return {
        "version": "v1",
        "executed": True,
        "execution_scope": "REAL_PERSISTED_PRODUCT_GOLD_E2E",
        "execution_rule": "Run only when strict human validation and runtime eligibility are both true.",
        "eligible_candidates": [record["candidate_id"] for record in eligible],
        "candidate_results": results,
    }


def run(*, eval_dir: Path = EVAL, intake_path: Path | None = None) -> dict[str, Any]:
    """Execute P1 deterministically and return the summary written to disk."""
    required = [
        eval_dir / "gold_product_candidates.csv",
        eval_dir / "gold_product_gap_analysis.csv",
        eval_dir / "blocked_report_gold_v1.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing P0 artifacts: " + ", ".join(missing))

    candidates = read(eval_dir / "gold_product_candidates.csv")
    # Keep this P0 output as a required prerequisite even though it is not
    # mutated: it makes missing-coverage reporting fail rather than disappear.
    read(eval_dir / "gold_product_gap_analysis.csv")
    blocked = read(eval_dir / "blocked_report_gold_v1.csv")
    prepared = prepare_candidates(candidates, records())
    shortlist = _shortlist(prepared)

    write(eval_dir / "gold_fallback_candidate_shortlist_v1.csv", shortlist)
    write(eval_dir / "gold_human_evidence_review_queue_v1.csv", _human_queue(shortlist))
    write(eval_dir / "gold_evidence_intake_manifest_template_v1.csv", [], INTAKE_FIELDS)

    actual_intake_path = intake_path if intake_path is not None else eval_dir / HUMAN_INTAKE_FILENAME
    intake = read_human_gold_evidence_intake(actual_intake_path)
    validation = validate_human_gold_evidence_intake(intake, shortlist)
    operational = build_operational_evidence_artifact(validation)
    operational_csv_path = eval_dir / OPERATIONAL_CSV_FILENAME
    operational_json_path = eval_dir / OPERATIONAL_JSON_FILENAME
    write_operational_evidence_artifact(
        operational,
        csv_path=operational_csv_path,
        json_path=operational_json_path,
    )

    gate_rows = _gate_rows(shortlist, operational["operational_candidates"])
    cat_verified = sum(
        item["runtime_eligible"]
        for item in operational["operational_candidates"]
        if item["runtime_candidate"].get("species") == "CAT"
    )
    dog_verified = sum(
        item["runtime_eligible"]
        for item in operational["operational_candidates"]
        if item["runtime_candidate"].get("species") == "DOG"
    )
    gate_result = {
        "version": "v1",
        "scope": "fallback shortlist, human intake validation, and runtime gate",
        "existing_blocked_preserved": [
            {
                "candidate_id": row.get("candidate_id"),
                "status": row.get("candidate_status"),
                "blocked_reason": row.get("blocked_reason"),
            }
            for row in blocked
        ],
        "human_intake": {
            "source_path": intake.source_path,
            "input_missing": validation["input_missing"],
            "rows": validation["human_intake_rows"],
            "accepted": validation["accepted"],
            "rejected": validation["rejected"],
            "rejection_reason_counts": validation["rejection_reason_counts"],
        },
        "shortlist_gate_results": gate_rows,
        "verified_cat_candidates": cat_verified,
        "verified_dog_candidates": dog_verified,
        "runtime_eligible_count": operational["runtime_eligible_count"],
    }
    (eval_dir / "gold_evidence_gate_result_v1.json").write_text(
        json.dumps(gate_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    e2e = _real_persisted_product_e2e(operational["operational_candidates"], operational_json_path)
    (eval_dir / "gold_persisted_product_e2e_v1.json").write_text(
        json.dumps(e2e, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    report = f"""# Nutrition Gold Fallback Candidate & Evidence Intake Gate P1

## 범위

이 문서는 P0에서 BLOCKED된 후보를 자동으로 VERIFIED로 변경하지 않는다. raw source metadata와 사람이 검증한 operational evidence는 별도 계층이다. `LOCAL_IDENTITY_CONSISTENT`는 repository 내부 source consistency만 뜻하며 manufacturer formulation identity verified가 아니다.

## 이번 실행 실측

- GOLD_NEAR_READY 후보: {len(candidates)}
- CAT shortlist: {sum(item['species'] == 'CAT' for item in shortlist)}
- DOG shortlist: {sum(item['species'] == 'DOG' for item in shortlist)}
- human intake rows: {validation['human_intake_rows']}
- accepted evidence rows: {validation['accepted']}
- rejected evidence rows: {validation['rejected']}
- VERIFIED CAT candidates: {cat_verified}
- VERIFIED DOG candidates: {dog_verified}
- runtime eligible candidates: {operational['runtime_eligible_count']}
- real persisted-product Gold E2E: executed={str(e2e['executed']).lower()}

## Intake와 Operational Artifact

`{HUMAN_INTAKE_FILENAME}`가 없거나 비어 있으면 입력은 0건이며, `{OPERATIONAL_CSV_FILENAME}` 및 `{OPERATIONAL_JSON_FILENAME}`은 정상적인 0-row artifact가 된다. 이 경우 raw PRODUCT_LABEL / seed_13 데이터가 verified evidence로 승격되지 않는다.

검증은 valid canonical GTIN, exact local product-ID membership, GTIN이 보이는 identity evidence, internally consistent product/variant/market/formula binding, authoritative life-stage/product-form provenance, guaranteed nutrient provenance, numeric percent/basis, required nutrient completeness를 요구한다. 제품명 키워드나 raw metadata는 life stage/form evidence가 될 수 없다.

## E2E 구분

테스트의 synthetic row는 fixture-based plumbing integration일 뿐 real persisted-product Gold E2E가 아니다. `{e2e['execution_scope']}`는 strict human validation과 runtime eligibility가 모두 true인 실제 artifact에만 실행한다.
"""
    (eval_dir / "gold_fallback_evidence_closure_p1_report.md").write_text(report, encoding="utf-8")

    return {
        "inventory": len(candidates),
        "shortlist": len(shortlist),
        "human_intake_rows": validation["human_intake_rows"],
        "accepted": validation["accepted"],
        "rejected": validation["rejected"],
        "verified_cat": cat_verified,
        "verified_dog": dog_verified,
        "runtime_eligible": operational["runtime_eligible_count"],
        "e2e_executed": e2e["executed"],
    }


def main() -> None:
    print(json.dumps(run(), ensure_ascii=False))


if __name__ == "__main__":
    main()
