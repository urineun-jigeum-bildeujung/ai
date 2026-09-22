"""운영 주문 상품 행이 상품군 구매 사건으로 안전하게 묶이는지 검증합니다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.modeling.operational_orders import OperationalOrderError
from scripts.modeling.purchase_events import (
    EVENT_COLUMNS,
    build_product_group_purchase_events,
)


def _orders() -> pd.DataFrame:
    """같은 사용자의 두 결제 완료 주문을 준비합니다."""
    return pd.DataFrame(
        {
            "order_id": ["o1", "o2"],
            "user_id": ["u1", "u1"],
            "paid_at": ["2026-01-01T09:00:00+09:00", "2026-02-01T00:00:00Z"],
            "order_status": ["PAID", "PAID"],
            "purchase_type": ["ONE_TIME", "SUBSCRIPTION"],
        }
    )


def _items() -> pd.DataFrame:
    """같은 상품군의 여러 SKU와 반려동물 미지정 구매를 준비합니다."""
    return pd.DataFrame(
        {
            "order_item_id": ["i1", "i2", "i3", "i4", "i5"],
            "order_id": ["o1", "o1", "o1", "o1", "o2"],
            "product_id": ["p1", "p2", "p1", "p3", "p2"],
            "product_group_id_snapshot": ["g1", "g1", "g1", "g2", "g1"],
            "category_code_snapshot": ["FOOD", "FOOD", "FOOD", "TREAT", "FOOD"],
            "is_replenishable_snapshot": [True, True, True, False, True],
            "pet_id": ["pet1", "pet1", None, None, "pet1"],
            "quantity": [2, 1, 1, 1, 1],
            "item_status": ["PAID"] * 5,
            "cancelled_quantity": [0] * 5,
            "returned_quantity": [0] * 5,
        }
    )


def test_same_order_group_pet_lines_become_one_event() -> None:
    """SKU가 달라도 동일 주문·상품군·반려동물은 재구매 사건 한 번입니다."""
    orders = _orders()
    items = _items()
    original_orders = orders.copy(deep=True)
    original_items = items.copy(deep=True)

    result = build_product_group_purchase_events(orders, items)
    selected = result.loc[
        result["order_id"].eq("o1")
        & result["target_id"].eq("g1")
        & result["pet_id"].eq("pet1")
    ].iloc[0]

    assert len(result) == 4
    assert selected["target_scope"] == "PRODUCT_GROUP"
    assert selected["source_order_item_count"] == 2
    assert selected["net_unit_count"] == 3
    assert selected["paid_at"] == pd.Timestamp("2026-01-01T00:00:00Z")
    pd.testing.assert_frame_equal(orders, original_orders)
    pd.testing.assert_frame_equal(items, original_items)


def test_unknown_pet_and_nonreplenishable_item_are_preserved() -> None:
    """미지정 반려동물과 비반복 상품을 조용히 버리거나 합치지 않습니다."""
    result = build_product_group_purchase_events(_orders(), _items())
    order_events = result.loc[result["order_id"].eq("o1")]

    assert len(order_events.loc[order_events["target_id"].eq("g1")]) == 2
    unknown_pet = order_events.loc[
        order_events["target_id"].eq("g1") & order_events["pet_id"].isna()
    ]
    assert len(unknown_pet) == 1
    assert unknown_pet.iloc[0]["net_unit_count"] == 1
    treat = order_events.loc[order_events["target_id"].eq("g2")].iloc[0]
    assert not bool(treat["is_replenishable_snapshot"])


def test_different_pets_are_not_merged() -> None:
    """같은 주문·상품군이어도 사용 대상 반려동물이 다르면 사건이 둘입니다."""
    items = _items()
    items.loc[2, "pet_id"] = "pet2"

    result = build_product_group_purchase_events(_orders(), items)
    group_events = result.loc[
        result["order_id"].eq("o1") & result["target_id"].eq("g1")
    ]

    assert set(group_events["pet_id"]) == {"pet1", "pet2"}


def test_partial_return_and_cancelled_line_use_only_remaining_units() -> None:
    """부분 반품 수량만 남기고 전액 취소 상품 행은 사건에서 제외합니다."""
    items = _items()
    items.loc[0, "quantity"] = 3
    items.loc[0, "returned_quantity"] = 1
    items.loc[0, "item_status"] = "PARTIAL_RETURN"
    items.loc[1, "cancelled_quantity"] = 1
    items.loc[1, "item_status"] = "CANCELLED"

    result = build_product_group_purchase_events(_orders(), items)
    selected = result.loc[
        result["order_id"].eq("o1")
        & result["target_id"].eq("g1")
        & result["pet_id"].eq("pet1")
    ].iloc[0]

    assert selected["net_unit_count"] == 2
    assert selected["source_order_item_count"] == 1


@pytest.mark.parametrize(
    "column", ["category_code_snapshot", "is_replenishable_snapshot"]
)
def test_conflicting_group_snapshots_fail(column: str) -> None:
    """한 상품군 사건에 상충하는 상품 속성이 들어오면 명확히 거절합니다."""
    items = _items()
    items.loc[1, column] = "OTHER" if column == "category_code_snapshot" else False

    with pytest.raises(OperationalOrderError, match="서로 다릅니다"):
        build_product_group_purchase_events(_orders(), items)


def test_net_unit_sum_uses_python_integers() -> None:
    """여러 행의 남은 수량 합이 int64를 넘어도 음수로 돌아가지 않습니다."""
    items = _items()
    maximum = np.iinfo(np.int64).max
    items.loc[0, "quantity"] = maximum
    items.loc[1, "quantity"] = maximum

    result = build_product_group_purchase_events(_orders(), items)
    selected = result.loc[
        result["order_id"].eq("o1")
        & result["target_id"].eq("g1")
        & result["pet_id"].eq("pet1")
    ].iloc[0]

    assert selected["net_unit_count"] == 2 * maximum


def test_no_valid_purchase_returns_empty_event_contract() -> None:
    """유효한 결제 주문이 없으면 동일한 출력 열을 가진 빈 결과를 반환합니다."""
    orders = _orders()
    orders["order_status"] = "PENDING"

    result = build_product_group_purchase_events(orders, _items())

    assert result.empty
    assert tuple(result.columns) == EVENT_COLUMNS
