"""Read-only audit for the BE Mock integration product master.

Usage inside the Nutrition service environment:
    python scripts/nutrition/audit_mock_fixture_coverage.py --expected-mock-total 286 --expected-target-total 322

The legacy MOCK-* count remains explicit, while expected_target_total covers
both MOCK-* SKUs and exact Service ID/SKU registrations from
mock_service_identity_v1.json.

No INSERT/UPDATE/DDL is performed. Optional --output writes only a local
JSON report; it does not write to Service DB or Git.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import service_repository
from mock_integration_fixture import coverage_record, is_mock_sku, is_mock_source


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


def _coverage_counts(records: list[dict]) -> Counter:
    return Counter(record["fixture_status"] for record in records)


def _processed_count(counts: Counter) -> int:
    return sum(
        counts[status]
        for status in ("FIXTURE_READY", "FIXTURE_PARTIAL", "FIXTURE_UNAVAILABLE")
    )


def build_report(expected_mock_total: int = 286, expected_target_total: int = 322) -> dict:
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

    mock_rows = [row for row in rows if is_mock_sku(row["service_sku"])]
    target_rows = [row for row in rows if is_mock_source(row["service_source"])]
    registered_rows = [row for row in target_rows if not is_mock_sku(row["service_sku"])]

    mock_counts = _coverage_counts(mock_rows)
    target_counts = _coverage_counts(target_rows)
    food_rows = [row for row in rows if row["service_source"].get("category_code") == "FOOD"]
    possible = sum(bool(row["nutrition_comparison_possible"]) for row in rows)
    food_unknown = sum(not row["nutrition_comparison_possible"] for row in food_rows)
    not_applicable = len(rows) - len(food_rows)

    mock_count = len(mock_rows)
    target_count = len(target_rows)
    registered_count = len(registered_rows)

    report = {
        "artifact_version": "mock_fixture_coverage_v2",
        "scope": "SERVICE_DB_MOCK_INTEGRATION_MASTER",
        "expected_mock_total": expected_mock_total,
        "expected_target_total": expected_target_total,
        "active_product_total": len(rows),
        "total": len(rows),
        "all_products_are_mock_sku": mock_count == len(rows),
        "all_products_are_mock_targets": target_count == len(rows),
        "mock_sku_count": mock_count,
        "mock_product_total": mock_count,
        "registered_mock_product_total": registered_count,
        "mock_target_total": target_count,
        "non_mock_product_total": len(rows) - mock_count,
        "non_target_product_total": len(rows) - target_count,
        "fixture_coverage": {
            "FIXTURE_READY": target_counts["FIXTURE_READY"],
            "FIXTURE_PARTIAL": target_counts["FIXTURE_PARTIAL"],
            "FIXTURE_UNAVAILABLE": target_counts["FIXTURE_UNAVAILABLE"],
        },
        "mock_sku_fixture_coverage": {
            "FIXTURE_READY": mock_counts["FIXTURE_READY"],
            "FIXTURE_PARTIAL": mock_counts["FIXTURE_PARTIAL"],
            "FIXTURE_UNAVAILABLE": mock_counts["FIXTURE_UNAVAILABLE"],
        },
        "nutrition_comparison_structural": {
            "possible": possible,
            "unknown_expected_food": food_unknown,
            "not_applicable_non_food": not_applicable,
        },
        "acceptance": {
            "expected_mock_total_match": mock_count == expected_mock_total,
            "expected_target_total_match": target_count == expected_target_total,
            "all_mock_processed": _processed_count(mock_counts) == mock_count,
            "all_targets_processed": _processed_count(target_counts) == target_count,
        },
        "observations": {
            "all_mock_namespace": mock_count == len(rows),
            "all_active_products_targeted": target_count == len(rows),
        },
        "golden_set": _golden_set(target_rows),
        "products": rows,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expected-mock-total", "--expected-total", dest="expected_mock_total",
        type=int, default=286,
        help="expected active MOCK-* product count (legacy alias: --expected-total)",
    )
    parser.add_argument(
        "--expected-target-total",
        type=int,
        default=322,
        help="expected total synthetic Mock targets including registered Service ID/SKU pairs",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.expected_mock_total, args.expected_target_total)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    accepted = all(
        report["acceptance"][key]
        for key in (
            "expected_mock_total_match",
            "expected_target_total_match",
            "all_mock_processed",
            "all_targets_processed",
        )
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
