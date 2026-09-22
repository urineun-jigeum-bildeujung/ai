"""클라우드 원천 고정 예제가 구매 사건 계약과 일치하는지 검증합니다."""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from scripts.modeling.operational_orders import (
    OperationalOrderError,
    build_valid_order_items,
)
from scripts.modeling.purchase_events import build_product_group_purchase_events


def _as_rows(payload: dict[str, object], name: str) -> pd.DataFrame:
    """JSON 배열을 DataFrame으로 바꾸되 계약 예제의 잘못된 구조는 거절합니다."""
    value = payload.get(name)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise AssertionError(f"고정 예제의 {name}은 객체 배열이어야 합니다.")
    return pd.DataFrame(cast(list[dict[str, object]], value))


def _normalize_contract_values(rows: pd.DataFrame) -> list[dict[str, object]]:
    """pandas 전용 시각·결측 표현을 언어 중립적인 JSON 값으로 바꿉니다."""
    normalized = rows.copy()
    if "paid_at" in normalized:
        normalized["paid_at"] = normalized["paid_at"].map(
            lambda value: value.isoformat().replace("+00:00", "Z")
        )
    normalized = normalized.astype(object).where(normalized.notna(), None)
    return normalized.to_dict(orient="records")


def test_fixed_source_example_produces_expected_valid_items(
    cloud_source_contract: dict[str, object],
) -> None:
    """결제·부분반품만 남고 전액 취소 행과 취소 주문은 제외됩니다."""
    orders = _as_rows(cloud_source_contract, "orders")
    order_items = _as_rows(cloud_source_contract, "order_items")

    result = build_valid_order_items(orders, order_items)

    assert _normalize_contract_values(result) == cast(
        list[dict[str, object]],
        cloud_source_contract["expected_valid_order_items"],
    )


def test_fixed_source_example_produces_expected_purchase_events(
    cloud_source_contract: dict[str, object],
) -> None:
    """동일 주문·상품군·반려동물 행만 합치고 미지정 반려동물은 보존합니다."""
    orders = _as_rows(cloud_source_contract, "orders")
    order_items = _as_rows(cloud_source_contract, "order_items")

    result = build_product_group_purchase_events(orders, order_items)

    assert _normalize_contract_values(result) == cast(
        list[dict[str, object]],
        cloud_source_contract["expected_purchase_events"],
    )


def test_fixed_source_example_rejects_missing_required_column(
    cloud_source_contract: dict[str, object],
) -> None:
    """필수 필드 누락을 조용히 기본값으로 채우지 않습니다."""
    orders = _as_rows(cloud_source_contract, "orders").drop(columns="purchase_type")
    order_items = _as_rows(cloud_source_contract, "order_items")

    with pytest.raises(OperationalOrderError, match="필수 열"):
        build_valid_order_items(orders, order_items)


def test_fixed_source_example_rejects_duplicate_order_item(
    cloud_source_contract: dict[str, object],
) -> None:
    """중복 원천 행이 구매량과 구매 사건을 이중 집계하지 못하게 합니다."""
    orders = _as_rows(cloud_source_contract, "orders")
    order_items = _as_rows(cloud_source_contract, "order_items")
    order_items.loc[1, "order_item_id"] = order_items.loc[0, "order_item_id"]

    with pytest.raises(OperationalOrderError, match="중복"):
        build_valid_order_items(orders, order_items)
