"""UCI 주문 상품 행이 재구매용 구매 사건으로 정확히 집계되는지 검증합니다.

제외 행은 사건에 들어오지 않고, 동일 주문·상품의 여러 행은 하나가 되며,
중복 처리 전후 수량·금액이 감사 가능한 형태로 함께 남는지 확인합니다.
"""

from __future__ import annotations

import pandas as pd

from scripts.preprocessing.events import (
    build_uci_purchase_events,
    validate_uci_purchase_events,
)
from scripts.preprocessing.uci import classify_uci_rows
from tests.test_uci_preprocessing import make_uci_fixture


def test_purchase_event_excludes_quarantine_and_preserves_duplicate_views() -> None:
    """격리 행을 제외하고 중복 전후 수량을 한 사건에 보존하는지 확인합니다."""
    classified = classify_uci_rows(make_uci_fixture()).rows

    events = build_uci_purchase_events(classified).set_index(
        ["user_id", "order_id", "product_id"]
    )

    assert len(events) == 3
    assert ("u3", "o4", "POST") not in events.index
    duplicated_event = events.loc[("u5", "o6", "p6")]
    assert duplicated_event["quantity_raw"] == 4
    assert duplicated_event["quantity_deduplicated"] == 2
    assert duplicated_event["quantity_difference"] == 2
    assert bool(duplicated_event["had_suspected_duplicate"])


def test_purchase_event_validation_preserves_source_line_contribution() -> None:
    """모든 사용 가능 원본 행이 정확히 한 구매 사건에 기여하는지 확인합니다."""
    classified = classify_uci_rows(make_uci_fixture()).rows
    events = build_uci_purchase_events(classified)

    summary = validate_uci_purchase_events(classified, events)

    assert summary["accepted_source_line_count"] == 4
    assert summary["purchase_event_count"] == 3
    assert summary["quantity_difference_quantiles_on_affected_events"]["p50"] == 2
    assert all(summary["invariants"].values())


def test_purchase_event_validation_handles_no_duplicate_events() -> None:
    """중복 영향 사건이 없어도 분위수 보고서에 결측값을 만들지 않는지 확인합니다."""
    source = make_uci_fixture().loc[lambda frame: ~frame["source_row_id"].eq("r7")]
    classified = classify_uci_rows(source).rows
    events = build_uci_purchase_events(classified)

    summary = validate_uci_purchase_events(classified, events)

    assert summary["duplicate_affected_event_count"] == 0
    assert set(
        summary["quantity_difference_quantiles_on_affected_events"].values()
    ) == {0.0}


def test_purchase_event_preserves_timestamp_variation() -> None:
    """같은 주문·상품의 시각 범위를 보존하고 사건을 분리하지 않는지 확인합니다."""
    source = make_uci_fixture()
    source.loc[source["source_row_id"].eq("r7"), "ordered_at"] = pd.Timestamp(
        "2026-01-01 00:00:01"
    )
    classified = classify_uci_rows(source).rows

    events = build_uci_purchase_events(classified).set_index(
        ["user_id", "order_id", "product_id"]
    )

    event = events.loc[("u5", "o6", "p6")]
    assert bool(event["has_timestamp_variation"])
    assert event["event_duration_seconds"] == 1
