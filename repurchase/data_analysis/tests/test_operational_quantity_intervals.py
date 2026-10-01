"""완료 클레임의 시각에만 주문상품 잔여 수량을 차감하는지 검사합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.operational_orders import OperationalOrderError
from scripts.modeling.operational_quantity_intervals import (
    build_order_item_quantity_intervals,
)


def _sources() -> tuple[pd.DataFrame, ...]:
    orders = pd.DataFrame(
        {
            "order_id": ["o1"],
            "paid_at": ["2026-01-01T00:00:00Z"],
        }
    )
    items = pd.DataFrame(
        {
            "order_item_id": ["i1", "i2"],
            "order_id": ["o1", "o1"],
            "quantity": [5, 2],
            "cancelled_quantity": [2, 0],
            "returned_quantity": [3, 0],
        }
    )
    claims = pd.DataFrame(
        {
            "claim_id": ["c1", "c2", "c3", "c4"],
            "order_id": ["o1"] * 4,
            "claim_type": ["CANCEL", "RETURN", "RETURN", "RETURN"],
            "claim_status": ["COMPLETED", "COMPLETED", "COMPLETED", "REQUESTED"],
            "completed_at": [
                "2026-01-05T00:00:00Z",
                "2026-01-05T00:00:00Z",
                "2026-01-10T00:00:00Z",
                None,
            ],
        }
    )
    claim_items = pd.DataFrame(
        {
            "claim_item_id": ["ci1", "ci2", "ci3", "ci4"],
            "claim_id": ["c1", "c2", "c3", "c4"],
            "order_item_id": ["i1"] * 4,
            "quantity": [2, 1, 2, 1],
        }
    )
    return orders, items, claims, claim_items


def _remaining_at(intervals: pd.DataFrame, item_id: str, when: str) -> int:
    cutoff = pd.Timestamp(when)
    active = intervals.loc[
        intervals["order_item_id"].eq(item_id)
        & intervals["valid_from"].le(cutoff)
        & (intervals["valid_until"].isna() | intervals["valid_until"].gt(cutoff))
    ]
    assert len(active) == 1
    return int(active.iloc[0]["remaining_quantity"])


def test_partial_and_same_time_claims_create_one_new_interval() -> None:
    sources = _sources()
    originals = tuple(frame.copy(deep=True) for frame in sources)

    intervals = build_order_item_quantity_intervals(*sources)

    assert intervals.loc[intervals["order_item_id"].eq("i1")].shape[0] == 3
    assert _remaining_at(intervals, "i1", "2026-01-04T23:59:59Z") == 5
    assert _remaining_at(intervals, "i1", "2026-01-05T00:00:00Z") == 2
    assert _remaining_at(intervals, "i1", "2026-01-10T00:00:00Z") == 0
    assert _remaining_at(intervals, "i2", "2026-01-11T00:00:00Z") == 2
    assert not bool(
        intervals.loc[intervals["order_item_id"].eq("i1")].iloc[-1]["is_valid_quantity"]
    )
    for source, original in zip(sources, originals, strict=True):
        pd.testing.assert_frame_equal(source, original)


def test_completed_claim_at_payment_time_has_no_zero_length_interval() -> None:
    orders, items, claims, claim_items = _sources()
    claims.loc[0, "completed_at"] = "2026-01-01T00:00:00Z"

    intervals = build_order_item_quantity_intervals(orders, items, claims, claim_items)

    assert _remaining_at(intervals, "i1", "2026-01-01T00:00:00Z") == 3
    assert intervals.loc[intervals["order_item_id"].eq("i1")].shape[0] == 3


def test_final_quantity_mismatch_is_rejected() -> None:
    orders, items, claims, claim_items = _sources()
    items.loc[0, "returned_quantity"] = 2
    with pytest.raises(OperationalOrderError, match="이력이 다릅니다"):
        build_order_item_quantity_intervals(orders, items, claims, claim_items)


def test_completed_claim_for_unpaid_order_is_rejected() -> None:
    orders, items, claims, claim_items = _sources()
    orders.loc[0, "paid_at"] = None
    with pytest.raises(OperationalOrderError, match="미결제 주문상품"):
        build_order_item_quantity_intervals(orders, items, claims, claim_items)


def test_claim_completed_before_payment_is_rejected() -> None:
    orders, items, claims, claim_items = _sources()
    claims.loc[0, "completed_at"] = "2025-12-31T00:00:00Z"
    with pytest.raises(OperationalOrderError, match="결제 시각보다 빠릅니다"):
        build_order_item_quantity_intervals(orders, items, claims, claim_items)


def test_completed_claims_cannot_exceed_original_quantity() -> None:
    orders, items, claims, claim_items = _sources()
    items.loc[0, "quantity"] = 4
    with pytest.raises(OperationalOrderError, match="초과합니다"):
        build_order_item_quantity_intervals(orders, items, claims, claim_items)


def test_empty_claim_exports_keep_purchase_quantity() -> None:
    orders, items, claims, claim_items = _sources()
    items[["cancelled_quantity", "returned_quantity"]] = 0
    claims = claims.iloc[0:0]
    claim_items = claim_items.iloc[0:0]

    intervals = build_order_item_quantity_intervals(orders, items, claims, claim_items)

    assert intervals.set_index("order_item_id")["remaining_quantity"].to_dict() == {
        "i1": 5,
        "i2": 2,
    }
    assert intervals["valid_until"].isna().all()


def test_nonempty_claim_quantity_still_requires_integer() -> None:
    orders, items, claims, claim_items = _sources()
    claim_items["quantity"] = claim_items["quantity"].astype(str)

    with pytest.raises(OperationalOrderError, match="quantity"):
        build_order_item_quantity_intervals(orders, items, claims, claim_items)
