"""같은 주문의 여러 SKU를 사건 하나로 합치고 사용자 주문은 보존합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.operational_event_intervals import (
    build_operational_event_intervals,
)
from scripts.modeling.operational_orders import OperationalOrderError


def test_two_skus_share_one_pet_group_event_and_keep_quantity_boundaries() -> None:
    orders = pd.DataFrame(
        {
            "order_id": ["o1"],
            "user_id": ["u1"],
            "ordered_at": ["2026-01-01T00:00:00Z"],
            "paid_at": ["2026-01-01T00:00:00Z"],
        }
    )
    items = pd.DataFrame(
        {
            "order_item_id": ["i1", "i2", "i3"],
            "order_id": ["o1"] * 3,
            "pet_id": ["pet1", "pet1", None],
            "product_group_id_snapshot": ["g1", "g1", "g2"],
            "category_code_snapshot": ["FOOD", "FOOD", "TOY"],
            "is_replenishable_snapshot": [True, True, False],
        }
    )
    pets = pd.DataFrame({"pet_id": ["pet1"], "birth_date": ["2025-12-01"]})
    valid_items = pd.DataFrame(
        {
            "order_item_id": ["i1", "i1", "i2", "i3"],
            "order_id": ["o1"] * 4,
            "valid_from": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-05T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                ]
            ),
            "valid_until": pd.to_datetime(
                [
                    "2026-01-05T00:00:00Z",
                    None,
                    "2026-01-10T00:00:00Z",
                    None,
                ]
            ),
            "remaining_quantity": [3, 1, 2, 1],
        }
    )

    events = build_operational_event_intervals(valid_items, orders, items, pets)

    target = events.pet_targets
    assert target["order_id"].tolist() == ["o1", "o1", "o1"]
    assert target["net_unit_count"].tolist() == [5, 3, 1]
    assert target["source_order_item_count"].tolist() == [2, 2, 1]
    assert target["valid_from"].tolist() == [
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-05T00:00:00Z"),
        pd.Timestamp("2026-01-10T00:00:00Z"),
    ]
    assert events.user_orders["order_id"].nunique() == 1
    assert events.user_orders.iloc[0]["net_unit_count"] == 6


def test_invalid_pet_birth_does_not_remove_user_order() -> None:
    orders = pd.DataFrame(
        {
            "order_id": ["o1"],
            "user_id": ["u1"],
            "ordered_at": ["2026-01-01T00:00:00Z"],
            "paid_at": ["2026-01-01T00:00:00Z"],
        }
    )
    items = pd.DataFrame(
        {
            "order_item_id": ["i1"],
            "order_id": ["o1"],
            "pet_id": ["pet1"],
            "product_group_id_snapshot": ["g1"],
            "category_code_snapshot": ["FOOD"],
            "is_replenishable_snapshot": [True],
        }
    )
    pets = pd.DataFrame({"pet_id": ["pet1"], "birth_date": ["2026-01-10"]})
    valid_items = pd.DataFrame(
        {
            "order_item_id": ["i1"],
            "order_id": ["o1"],
            "valid_from": [pd.Timestamp("2026-01-01T00:00:00Z")],
            "valid_until": [pd.NaT],
            "remaining_quantity": [1],
        }
    )

    events = build_operational_event_intervals(valid_items, orders, items, pets)

    assert len(events.user_orders) == 1
    assert events.pet_targets.empty


def test_same_time_item_replacement_keeps_one_continuous_event() -> None:
    orders = pd.DataFrame(
        {
            "order_id": ["o1"],
            "user_id": ["u1"],
            "ordered_at": ["2026-01-01T00:00:00Z"],
            "paid_at": ["2026-01-01T00:00:00Z"],
        }
    )
    items = pd.DataFrame(
        {
            "order_item_id": ["i1", "i2"],
            "order_id": ["o1", "o1"],
            "pet_id": ["pet1", "pet1"],
            "product_group_id_snapshot": ["g1", "g1"],
            "category_code_snapshot": ["FOOD", "FOOD"],
            "is_replenishable_snapshot": [True, True],
        }
    )
    pets = pd.DataFrame({"pet_id": ["pet1"], "birth_date": ["2025-12-01"]})
    intervals = pd.DataFrame(
        {
            "order_item_id": ["i1", "i2"],
            "order_id": ["o1", "o1"],
            "valid_from": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-05T00:00:00Z",
                ]
            ),
            "valid_until": pd.to_datetime(["2026-01-05T00:00:00Z", None]),
            "remaining_quantity": [2, 2],
        }
    )

    result = build_operational_event_intervals(intervals, orders, items, pets)

    assert len(result.pet_targets) == 1
    assert result.pet_targets.iloc[0]["net_unit_count"] == 2
    assert result.pet_targets.iloc[0]["source_order_item_count"] == 1
    assert pd.isna(result.pet_targets.iloc[0]["valid_until"])


def test_all_items_inactive_gap_does_not_create_purchase_event() -> None:
    orders = pd.DataFrame(
        {
            "order_id": ["o1"],
            "user_id": ["u1"],
            "ordered_at": ["2026-01-01T00:00:00Z"],
            "paid_at": ["2026-01-01T00:00:00Z"],
        }
    )
    items = pd.DataFrame(
        {
            "order_item_id": ["i1"],
            "order_id": ["o1"],
            "pet_id": ["pet1"],
            "product_group_id_snapshot": ["g1"],
            "category_code_snapshot": ["FOOD"],
            "is_replenishable_snapshot": [True],
        }
    )
    pets = pd.DataFrame({"pet_id": ["pet1"], "birth_date": ["2025-12-01"]})
    intervals = pd.DataFrame(
        {
            "order_item_id": ["i1", "i1"],
            "order_id": ["o1", "o1"],
            "valid_from": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-10T00:00:00Z",
                ]
            ),
            "valid_until": pd.to_datetime(["2026-01-05T00:00:00Z", None]),
            "remaining_quantity": [1, 1],
        }
    )

    result = build_operational_event_intervals(intervals, orders, items, pets)

    assert len(result.pet_targets) == 2
    assert result.pet_targets["valid_from"].tolist() == [
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-10T00:00:00Z"),
    ]
    assert result.pet_targets.iloc[0]["valid_until"] == pd.Timestamp(
        "2026-01-05T00:00:00Z"
    )


def test_overlapping_item_intervals_are_rejected_before_aggregation() -> None:
    orders = pd.DataFrame(
        {
            "order_id": ["o1"],
            "user_id": ["u1"],
            "ordered_at": ["2026-01-01T00:00:00Z"],
            "paid_at": ["2026-01-01T00:00:00Z"],
        }
    )
    items = pd.DataFrame(
        {
            "order_item_id": ["i1"],
            "order_id": ["o1"],
            "pet_id": ["pet1"],
            "product_group_id_snapshot": ["g1"],
            "category_code_snapshot": ["FOOD"],
            "is_replenishable_snapshot": [True],
        }
    )
    pets = pd.DataFrame({"pet_id": ["pet1"], "birth_date": ["2025-12-01"]})
    overlapping = pd.DataFrame(
        {
            "order_item_id": ["i1", "i1"],
            "order_id": ["o1", "o1"],
            "valid_from": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-04T00:00:00Z",
                ]
            ),
            "valid_until": pd.to_datetime(["2026-01-05T00:00:00Z", None]),
            "remaining_quantity": [1, 1],
        }
    )

    with pytest.raises(OperationalOrderError, match="주문상품.*겹칩니다"):
        build_operational_event_intervals(overlapping, orders, items, pets)


def test_mixed_replenishable_flags_in_one_target_are_rejected() -> None:
    orders = pd.DataFrame(
        {
            "order_id": ["o1"],
            "user_id": ["u1"],
            "ordered_at": ["2026-01-01T00:00:00Z"],
            "paid_at": ["2026-01-01T00:00:00Z"],
        }
    )
    items = pd.DataFrame(
        {
            "order_item_id": ["i1", "i2"],
            "order_id": ["o1", "o1"],
            "pet_id": ["pet1", "pet1"],
            "product_group_id_snapshot": ["g1", "g1"],
            "category_code_snapshot": ["FOOD", "FOOD"],
            "is_replenishable_snapshot": [True, False],
        }
    )
    pets = pd.DataFrame({"pet_id": ["pet1"], "birth_date": ["2025-12-01"]})
    intervals = pd.DataFrame(
        {
            "order_item_id": ["i1", "i2"],
            "order_id": ["o1", "o1"],
            "valid_from": pd.to_datetime(["2026-01-01T00:00:00Z"] * 2),
            "valid_until": [pd.NaT, pd.NaT],
            "remaining_quantity": [1, 1],
        }
    )

    with pytest.raises(OperationalOrderError, match="반복 소비 여부"):
        build_operational_event_intervals(intervals, orders, items, pets)
