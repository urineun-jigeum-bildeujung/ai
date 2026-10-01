"""주문 상태와 상품 잔여 수량의 반열린 교집합을 검증합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.operational_orders import OperationalOrderError
from scripts.modeling.operational_validity_intervals import (
    build_valid_purchase_item_intervals,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    status = pd.DataFrame(
        {
            "order_id": ["o1"] * 4,
            "valid_from": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-03T00:00:00Z",
                    "2026-01-07T00:00:00Z",
                    "2026-01-10T00:00:00Z",
                ]
            ),
            "valid_until": pd.to_datetime(
                [
                    "2026-01-03T00:00:00Z",
                    "2026-01-07T00:00:00Z",
                    "2026-01-10T00:00:00Z",
                    None,
                ]
            ),
            "is_valid_status": [True, True, False, True],
        }
    )
    quantity = pd.DataFrame(
        {
            "order_item_id": ["i1", "i1"],
            "order_id": ["o1", "o1"],
            "valid_from": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-05T00:00:00Z",
                ]
            ),
            "valid_until": pd.to_datetime(["2026-01-05T00:00:00Z", None]),
            "remaining_quantity": [3, 1],
            "is_valid_quantity": [True, True],
        }
    )
    return status, quantity


def _remaining_at(intervals: pd.DataFrame, when: str) -> list[int]:
    cutoff = pd.Timestamp(when)
    active = intervals.loc[
        intervals["valid_from"].le(cutoff)
        & (intervals["valid_until"].isna() | intervals["valid_until"].gt(cutoff))
    ]
    return active["remaining_quantity"].tolist()


def test_adjacent_valid_statuses_merge_but_quantity_change_and_gap_remain() -> None:
    status, quantity = _inputs()
    original_status, original_quantity = (
        status.copy(deep=True),
        quantity.copy(deep=True),
    )

    result = build_valid_purchase_item_intervals(status, quantity)

    assert len(result) == 3
    assert _remaining_at(result, "2026-01-03T00:00:00Z") == [3]
    assert _remaining_at(result, "2026-01-05T00:00:00Z") == [1]
    assert _remaining_at(result, "2026-01-07T00:00:00Z") == []
    assert _remaining_at(result, "2026-01-10T00:00:00Z") == [1]
    assert pd.isna(result.iloc[-1]["valid_until"])
    pd.testing.assert_frame_equal(status, original_status)
    pd.testing.assert_frame_equal(quantity, original_quantity)


def test_missing_order_history_is_rejected() -> None:
    status, quantity = _inputs()
    quantity["order_id"] = "unknown"
    with pytest.raises(OperationalOrderError, match="상태 이력"):
        build_valid_purchase_item_intervals(status, quantity)


def test_quantity_flag_must_match_remaining_quantity() -> None:
    status, quantity = _inputs()
    quantity.loc[0, "is_valid_quantity"] = False
    with pytest.raises(OperationalOrderError, match="유효 표시"):
        build_valid_purchase_item_intervals(status, quantity)


def test_overlapping_status_intervals_are_rejected() -> None:
    status, quantity = _inputs()
    status.loc[0, "valid_until"] = pd.Timestamp("2026-01-04T00:00:00Z")
    with pytest.raises(OperationalOrderError, match="주문 상태.*겹칩니다"):
        build_valid_purchase_item_intervals(status, quantity)


def test_interval_after_infinite_end_is_rejected() -> None:
    status, quantity = _inputs()
    quantity.loc[0, "valid_until"] = pd.NaT
    with pytest.raises(OperationalOrderError, match="상품 수량.*겹칩니다"):
        build_valid_purchase_item_intervals(status, quantity)


def test_textual_false_status_is_not_accepted_as_true() -> None:
    status, quantity = _inputs()
    status["is_valid_status"] = status["is_valid_status"].map(str)
    with pytest.raises(OperationalOrderError, match="boolean"):
        build_valid_purchase_item_intervals(status, quantity)
