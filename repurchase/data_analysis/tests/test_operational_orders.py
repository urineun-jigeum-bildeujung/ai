"""백엔드 주문 원천을 유효 구매 상품 행으로 바꾸는 경계를 검증합니다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.modeling.operational_orders import (
    OperationalOrderError,
    build_valid_order_items,
)


def _orders() -> pd.DataFrame:
    """결제·부분환불·취소 상태와 시간대가 다른 시각을 만듭니다."""
    return pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3"],
            "user_id": ["u1", "u1", "u2"],
            "paid_at": [
                "2026-01-01T09:00:00+09:00",
                "2026-01-11T00:00:00Z",
                None,
            ],
            "order_status": ["PAID", "PARTIAL_REFUND", "CANCELLED"],
            "purchase_type": ["ONE_TIME", "SUBSCRIPTION", "ONE_TIME"],
        }
    )


def _items() -> pd.DataFrame:
    """반복·비반복 상품과 부분반품·전액취소 상품을 함께 만듭니다."""
    return pd.DataFrame(
        {
            "order_item_id": ["i1", "i2", "i3", "i4", "i5"],
            "order_id": ["o1", "o1", "o2", "o2", "o3"],
            "product_id": ["p1", "p2", "p1", "p3", "p4"],
            "product_group_id_snapshot": ["g1", "g2", "g1", "g3", "g4"],
            "category_code_snapshot": ["FOOD", "TREAT", "FOOD", "FOOD", "FOOD"],
            "is_replenishable_snapshot": [True, False, True, True, True],
            "pet_id": ["pet1", None, "pet1", "pet1", None],
            "quantity": [1, 1, 3, 1, 1],
            "item_status": ["PAID", "PAID", "PARTIAL_RETURN", "CANCELLED", "PAID"],
            "cancelled_quantity": [0, 0, 0, 1, 0],
            "returned_quantity": [0, 0, 1, 0, 0],
        }
    )


def test_valid_items_preserve_non_replenishable_and_scope_keys() -> None:
    """대상 여부를 뒤에서 판단할 수 있게 모든 유효 구매 키를 보존합니다."""
    orders = _orders()
    items = _items()
    original_orders = orders.copy(deep=True)
    original_items = items.copy(deep=True)

    result = build_valid_order_items(orders, items).set_index("order_item_id")

    assert set(result.index) == {"i1", "i2", "i3"}
    assert not bool(result.loc["i2", "is_replenishable_snapshot"])
    assert pd.isna(result.loc["i2", "pet_id"])
    assert result.loc["i3", "net_quantity"] == 2
    assert result.loc["i3", "purchase_type"] == "SUBSCRIPTION"
    assert result.loc["i1", "paid_at"] == pd.Timestamp("2026-01-01T00:00:00Z")
    assert result.loc["i3", "product_group_id_snapshot"] == "g1"
    pd.testing.assert_frame_equal(orders, original_orders)
    pd.testing.assert_frame_equal(items, original_items)


def test_same_group_lines_are_not_merged_without_scope_decision() -> None:
    """동일 주문의 상품군 묶기는 다음 단계의 명시적 결정으로 남깁니다."""
    items = _items()
    items.loc[1, "product_group_id_snapshot"] = "g1"

    result = build_valid_order_items(_orders(), items)

    assert set(result.loc[result["order_id"].eq("o1"), "order_item_id"]) == {
        "i1",
        "i2",
    }


@pytest.mark.parametrize("status", ["PREPARING", "SHIPPING", "DELIVERED", "CONFIRMED"])
def test_fulfilled_order_preserves_paid_purchase(status: str) -> None:
    """배송 상태가 바뀌어도 결제 시각·유효 수량·구매 행은 변하지 않습니다."""
    orders = _orders()
    expected = build_valid_order_items(orders, _items())
    orders.loc[0, "order_status"] = status

    pd.testing.assert_frame_equal(build_valid_order_items(orders, _items()), expected)


@pytest.mark.parametrize("status", ["PREPARING", "SHIPPING", "DELIVERED", "CONFIRMED"])
def test_fulfilled_order_requires_paid_timestamp(status: str) -> None:
    """배송·확정 상태라도 결제 시각이 없으면 원천 오류로 거절합니다."""
    orders = _orders()
    orders.loc[0, "order_status"] = status
    orders.loc[0, "paid_at"] = None

    with pytest.raises(OperationalOrderError, match="시간대"):
        build_valid_order_items(orders, _items())


@pytest.mark.parametrize("status", ["PENDING", "CANCELLED", "REFUNDED"])
def test_non_purchase_order_is_excluded(status: str) -> None:
    """대상 상태 확장이 미결제·취소·전액환불 주문까지 포함하지 않습니다."""
    orders = _orders()
    orders.loc[0, "order_status"] = status

    result = build_valid_order_items(orders, _items())

    assert result["order_item_id"].tolist() == ["i3"]


@pytest.mark.parametrize(
    ("table", "column", "value", "message"),
    [
        ("orders", "order_status", "UNKNOWN", "order_status"),
        ("orders", "purchase_type", None, "purchase_type"),
        ("orders", "paid_at", "2026-01-01", "시간대"),
        ("items", "item_status", "UNKNOWN", "item_status"),
        ("items", "quantity", 1.5, "정수"),
        ("items", "cancelled_quantity", -1, "정수"),
        ("items", "is_replenishable_snapshot", None, "불리언"),
        ("items", "product_group_id_snapshot", None, "빈 식별키"),
    ],
)
def test_invalid_source_contract_fails(
    table: str, column: str, value: object, message: str
) -> None:
    """누락·형식 오류가 조용한 표본 제외로 바뀌지 않게 합니다."""
    orders = _orders()
    items = _items()
    target = orders if table == "orders" else items
    if column == "quantity":
        target[column] = target[column].astype("float64")
    if column == "is_replenishable_snapshot":
        target[column] = target[column].astype("object")
    target.loc[0, column] = value

    with pytest.raises(OperationalOrderError, match=message):
        build_valid_order_items(orders, items)


def test_invalid_foreign_key_and_duplicate_ids_fail() -> None:
    """잘못된 조인이나 동일 주문 상품의 중복 집계를 허용하지 않습니다."""
    orders = _orders()
    items = _items()
    items.loc[0, "order_id"] = "missing"
    with pytest.raises(OperationalOrderError, match="존재하지 않는"):
        build_valid_order_items(orders, items)

    items = _items()
    items.loc[1, "order_item_id"] = "i1"
    with pytest.raises(OperationalOrderError, match="중복"):
        build_valid_order_items(orders, items)


def test_terminal_item_with_remaining_quantity_fails() -> None:
    """종료 상태인데 수량이 남은 모순을 유효 구매로 세지 않습니다."""
    items = _items()
    items.loc[3, "cancelled_quantity"] = 0

    with pytest.raises(OperationalOrderError, match="유효 수량"):
        build_valid_order_items(_orders(), items)


def test_quantity_arithmetic_does_not_overflow_int64() -> None:
    """고정 폭 정수 연산의 오버플로가 잘못된 구매 판정으로 이어지지 않습니다."""
    items = _items()
    maximum = np.iinfo(np.int64).max
    items.loc[0, "quantity"] = 1
    items.loc[0, "cancelled_quantity"] = maximum
    items.loc[0, "returned_quantity"] = maximum

    with pytest.raises(OperationalOrderError, match="초과"):
        build_valid_order_items(_orders(), items)


def test_paid_order_without_timezone_fails_even_when_no_item_remains() -> None:
    """수량이 모두 취소됐어도 결제 주문의 시각 계약을 검사합니다."""
    orders = _orders()
    orders.loc[1, "paid_at"] = None
    items = _items()
    items.loc[2, "returned_quantity"] = 3
    items.loc[2, "item_status"] = "RETURNED"

    with pytest.raises(OperationalOrderError, match="시간대"):
        build_valid_order_items(orders, items)
