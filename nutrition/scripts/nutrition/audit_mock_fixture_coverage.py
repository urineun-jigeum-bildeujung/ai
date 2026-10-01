"""Read-only audit for the BE Mock integration product master.

Usage inside the Nutrition service environment:
    python scripts/nutrition/audit_mock_fixture_coverage.py --expected-total 286

No INSERT/UPDATE/DDL is performed.  Optional ``--output`` writes only a local
JSON report; it does not write to Service DB or Git.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import service_repository
from mock_integration_fixture import coverage_record, is_mock_sku


def _golden_set(records: list[dict], limit: int = 16) -> list[dict]:
    selected: list[dict] = []
    seen: set[tuple] = set()
    for record in records:
        source = record["service_source"]
        bucket = (
            source.get("category_code"),
            tuple(source.get("target_species") or []),
            source.get("subcategory_code"),
            source.get("target_age_group"),
            record["fixture_status"],
        )
        if bucket not in seen:
            selected.append(record)
            seen.add(bucket)
        if len(selected) >= limit:
            return selected
    for record in records:
        if record not in selected:
            selected.append(record)
        if len(selected) >= limit:
            break
    return selected


def build_report(expected_total: int = 286) -> dict:
    products = service_repository.list_active_products()
    rows = []
    for source in products:
        coverage = coverage_record(source)
        rows.append({**coverage, "service_source": {
            key: source.get(key) for key in (
                "id", "sku", "product_name", "category_code", "subcategory_code",
                "target_age_group", "target_species",
            )
        }})

    fixture_counts = Counter(row["fixture_status"] for row in rows)
    food_rows = [row for row in rows if row["service_source"].get("category_code") == "FOOD"]
    possible = sum(bool(row["nutrition_comparison_possible"]) for row in rows)
    food_unknown = sum(not row["nutrition_comparison_possible"] for row in food_rows)
    not_applicable = len(rows) - len(food_rows)
    mock_count = sum(is_mock_sku(row["service_sku"]) for row in rows)

    report = {
        "artifact_version": "mock_fixture_coverage_v1",
        "scope": "SERVICE_DB_MOCK_INTEGRATION_MASTER",
        "expected_total": expected_total,
        "total": len(rows),
        "all_products_are_mock_sku": mock_count == len(rows),
        "mock_sku_count": mock_count,
        "fixture_coverage": {
            "FIXTURE_READY": fixture_counts["FIXTURE_READY"],
            "FIXTURE_PARTIAL": fixture_counts["FIXTURE_PARTIAL"],
            "FIXTURE_UNAVAILABLE": fixture_counts["FIXTURE_UNAVAILABLE"],
            "NOT_MOCK_PRODUCT": fixture_counts["NOT_MOCK_PRODUCT"],
        },
        "nutrition_comparison_structural": {
            "possible": possible,
            "unknown_expected_food": food_unknown,
            "not_applicable_non_food": not_applicable,
        },
        "acceptance": {
            "expected_total_match": len(rows) == expected_total,
            "all_processed": sum(fixture_counts.values()) == len(rows),
            "all_mock_namespace": mock_count == len(rows),
        },
        "golden_set": _golden_set(rows),
        "products": rows,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-total", type=int, default=286)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.expected_total)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    accepted = all(report["acceptance"].values())
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
