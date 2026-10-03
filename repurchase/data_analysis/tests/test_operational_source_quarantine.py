"""감사와 모델 입력이 공유하는 원천 격리 경계를 검증합니다."""

from __future__ import annotations

import pandas as pd

from scripts.modeling.operational_source_quarantine import (
    quarantine_unrestorable_orders,
)


def test_quarantine_preserves_valid_order_and_unpaid_pending() -> None:
    orders = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5],
            "order_status": ["PAID", "PAID", "CANCELLED", "CONFIRMED", "PENDING"],
            "paid_at": ["2026-09-29T00:00:00Z"] * 2
            + [None, "2026-09-29T00:00:00Z", None],
        }
    )
    items = pd.DataFrame(
        {"order_item_id": [11, 22, 33, 44, 55], "order_id": [1, 2, 3, 4, 5]}
    )
    histories = pd.DataFrame(
        {
            "history_id": [10, 40],
            "order_id": [1, 4],
            "to_status": ["PAID", "DELIVERED"],
            "changed_at": ["2026-09-29T00:00:00Z"] * 2,
        }
    )
    claims = pd.DataFrame({"claim_id": [100, 300], "order_id": [1, 3]})
    claim_items = pd.DataFrame({"claim_item_id": [101, 301], "claim_id": [100, 300]})

    result = quarantine_unrestorable_orders(
        orders, items, histories, claims, claim_items
    )

    assert result.orders["order_id"].tolist() == [1, 5]
    assert result.order_items["order_item_id"].tolist() == [11, 55]
    assert result.status_histories["history_id"].tolist() == [10]
    assert result.claims["claim_id"].tolist() == [100]
    assert result.claim_items["claim_item_id"].tolist() == [101]
    assert result.missing_history_order_count == 2
    assert result.missing_history_paid_order_count == 1
    assert result.missing_history_order_item_count == 2
    assert result.status_mismatch_order_count == 1
    assert result.status_mismatch_order_item_count == 1
    assert orders["order_id"].tolist() == [1, 2, 3, 4, 5]


def test_latest_history_uses_timestamp_then_id() -> None:
    orders = pd.DataFrame(
        {
            "order_id": [1],
            "order_status": ["CONFIRMED"],
            "paid_at": ["2026-09-29T00:00:00Z"],
        }
    )
    histories = pd.DataFrame(
        {
            "history_id": [3, 2, 1],
            "order_id": [1, 1, 1],
            "to_status": ["CONFIRMED", "DELIVERED", "PAID"],
            "changed_at": [
                "2026-09-29T02:00:00Z",
                "2026-09-29T02:00:00Z",
                "2026-09-29T01:00:00Z",
            ],
        }
    )
    items = pd.DataFrame({"order_item_id": [11], "order_id": [1]})
    claims = pd.DataFrame(columns=["claim_id", "order_id"])
    claim_items = pd.DataFrame(columns=["claim_item_id", "claim_id"])

    result = quarantine_unrestorable_orders(
        orders, items, histories, claims, claim_items
    )

    assert result.status_mismatch_order_count == 0
    assert result.orders["order_id"].tolist() == [1]
