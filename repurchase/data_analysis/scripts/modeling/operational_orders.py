"""백엔드 주문 원천을 수량이 남은 구매 상품 행으로 정규화합니다.

현재 주문 상태의 스냅샷만 다룹니다. 상품군·반려동물별 구매 사건 병합이나
모델 추론은 하지 않으며, 그 결정에 필요한 원천 키를 결과에 보존합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype


class OperationalOrderError(ValueError):
    """백엔드 주문 원천이 재구매 입력 계약을 위반할 때 발생합니다."""


ORDER_COLUMNS = (
    "order_id",
    "user_id",
    "paid_at",
    "order_status",
    "purchase_type",
)
ITEM_COLUMNS = (
    "order_item_id",
    "order_id",
    "product_id",
    "product_group_id_snapshot",
    "category_code_snapshot",
    "is_replenishable_snapshot",
    "pet_id",
    "quantity",
    "item_status",
    "cancelled_quantity",
    "returned_quantity",
)
VALID_ORDER_STATUSES = frozenset({"PAID", "PARTIAL_REFUND"})
ALL_ORDER_STATUSES = VALID_ORDER_STATUSES | {
    "PENDING",
    "CANCELLED",
    "REFUNDED",
}
VALID_ITEM_STATUSES = frozenset({"PAID", "PARTIAL_RETURN"})
ALL_ITEM_STATUSES = VALID_ITEM_STATUSES | {"CANCELLED", "RETURNED"}


def _require_columns(rows: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    """필수 열이 없거나 열 이름이 중복된 원천 테이블을 거절합니다."""
    if not rows.columns.is_unique:
        raise OperationalOrderError(f"{name}에 중복된 열 이름이 있습니다.")
    missing = set(columns) - set(rows.columns)
    if missing:
        raise OperationalOrderError(f"{name} 필수 열이 누락됐습니다: {sorted(missing)}")


def _require_keys(rows: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    """조인·상품군 판정에 필요한 식별키가 비어 있지 않은지 확인합니다."""
    for column in columns:
        values = rows[column]
        if (
            values.isna().any()
            or values.map(
                lambda value: isinstance(value, str) and not value.strip()
            ).any()
        ):
            raise OperationalOrderError(f"{name}.{column}에 빈 식별키가 있습니다.")


def _require_nonnegative_integer(rows: pd.DataFrame, column: str) -> None:
    """수량이 소수·불리언·음수·결측으로 넘어오지 않았는지 검사합니다."""
    values = rows[column]
    if (
        is_bool_dtype(values.dtype)
        or not is_integer_dtype(values.dtype)
        or values.isna().any()
        or values.lt(0).any()
    ):
        raise OperationalOrderError(f"order_items.{column}은 0 이상의 정수여야 합니다.")


def _paid_at_utc(value: object) -> pd.Timestamp:
    """결제 시각의 명시된 시간대만 UTC로 정규화합니다."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise OperationalOrderError(
            "결제 완료 주문의 paid_at이 잘못됐습니다."
        ) from error
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise OperationalOrderError("결제 완료 주문의 paid_at에는 시간대가 필요합니다.")
    return timestamp.tz_convert("UTC")


def build_valid_order_items(
    orders: pd.DataFrame, order_items: pd.DataFrame
) -> pd.DataFrame:
    """결제·부분환불 주문에서 실제 수량이 남은 상품 행을 보존합니다.

    비반복 상품도 사용자 전체 주문 이력 계산에 필요하므로 여기서는 제외하지 않습니다.
    반환값은 주문 상품 행이며, 곧바로 SKU/상품군 재구매 사건은 아닙니다.
    """
    _require_columns(orders, ORDER_COLUMNS, "orders")
    _require_columns(order_items, ITEM_COLUMNS, "order_items")
    _require_keys(orders, ("order_id", "user_id"), "orders")
    _require_keys(
        order_items,
        (
            "order_item_id",
            "order_id",
            "product_id",
            "product_group_id_snapshot",
            "category_code_snapshot",
        ),
        "order_items",
    )
    if orders["order_id"].duplicated().any():
        raise OperationalOrderError("orders.order_id가 중복됐습니다.")
    if order_items["order_item_id"].duplicated().any():
        raise OperationalOrderError("order_items.order_item_id가 중복됐습니다.")
    if not orders["order_status"].isin(ALL_ORDER_STATUSES).all():
        raise OperationalOrderError("지원하지 않는 order_status가 있습니다.")
    if not orders["purchase_type"].isin({"ONE_TIME", "SUBSCRIPTION"}).all():
        raise OperationalOrderError("지원하지 않는 purchase_type이 있습니다.")
    if not order_items["item_status"].isin(ALL_ITEM_STATUSES).all():
        raise OperationalOrderError("지원하지 않는 item_status가 있습니다.")
    if (
        not order_items["is_replenishable_snapshot"]
        .map(lambda value: isinstance(value, (bool, np.bool_)))
        .all()
    ):
        raise OperationalOrderError("is_replenishable_snapshot은 불리언이어야 합니다.")
    for column in ("quantity", "cancelled_quantity", "returned_quantity"):
        _require_nonnegative_integer(order_items, column)
    if order_items["quantity"].le(0).any():
        raise OperationalOrderError("order_items.quantity는 1 이상이어야 합니다.")

    # pandas의 고정 폭 정수 연산은 매우 큰 수량에서 오버플로할 수 있습니다.
    # 유효 수량은 Python 정수로 계산해 잘못된 양수 판정을 막습니다.
    net_quantity = pd.Series(
        [
            int(quantity) - int(cancelled) - int(returned)
            for quantity, cancelled, returned in zip(
                order_items["quantity"],
                order_items["cancelled_quantity"],
                order_items["returned_quantity"],
                strict=True,
            )
        ],
        index=order_items.index,
        dtype="object",
    )
    if net_quantity.lt(0).any():
        raise OperationalOrderError("취소·반품 수량의 합이 구매 수량을 초과했습니다.")
    if (
        order_items["item_status"].isin({"CANCELLED", "RETURNED"}) & net_quantity.gt(0)
    ).any():
        raise OperationalOrderError("종료된 주문 상품에 유효 수량이 남아 있습니다.")

    if not order_items["order_id"].isin(orders["order_id"]).all():
        raise OperationalOrderError(
            "orders에 존재하지 않는 order_items.order_id가 있습니다."
        )
    items = order_items.loc[:, list(ITEM_COLUMNS)].copy().reset_index(drop=True)
    items["net_quantity"] = net_quantity.to_numpy(dtype="object")
    paid_orders = orders.loc[:, list(ORDER_COLUMNS)].copy()
    paid_mask = paid_orders["order_status"].isin(VALID_ORDER_STATUSES)
    paid_orders.loc[paid_mask, "paid_at"] = paid_orders.loc[paid_mask, "paid_at"].map(
        _paid_at_utc
    )
    joined = items.merge(
        paid_orders,
        on="order_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    eligible = joined.loc[
        joined["order_status"].isin(VALID_ORDER_STATUSES)
        & joined["item_status"].isin(VALID_ITEM_STATUSES)
        & joined["net_quantity"].gt(0)
    ].copy()
    return eligible.loc[
        :,
        [
            "order_item_id",
            "user_id",
            "order_id",
            "product_id",
            "product_group_id_snapshot",
            "category_code_snapshot",
            "is_replenishable_snapshot",
            "pet_id",
            "paid_at",
            "purchase_type",
            "net_quantity",
        ],
    ].reset_index(drop=True)
