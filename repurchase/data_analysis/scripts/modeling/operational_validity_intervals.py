"""주문 상태와 상품 잔여 수량이 동시에 유효한 구매 시간 구간을 만듭니다."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd
from pandas.api.types import is_bool_dtype

from .operational_asof import _utc
from .operational_orders import (
    OperationalOrderError,
    _require_columns,
    _require_keys,
    _require_nonnegative_integer,
)


def _end(value: object) -> pd.Timestamp | None:
    return None if pd.isna(value) else _utc(value, column="valid_until")


def _intersection_end(
    first: pd.Timestamp | None, second: pd.Timestamp | None
) -> pd.Timestamp | None:
    if first is None:
        return second
    if second is None:
        return first
    return min(first, second)


def _require_nonoverlapping(
    groups: dict[object, list[tuple[pd.Timestamp, pd.Timestamp | None]]],
    *,
    name: str,
) -> None:
    """한 주문/상품에서 둘 이상의 구간이 같은 시점을 차지하지 않게 합니다."""
    for spans in groups.values():
        previous_end: pd.Timestamp | None = None
        for index, (start, end) in enumerate(sorted(spans, key=lambda span: span[0])):
            if index and (previous_end is None or start < previous_end):
                raise OperationalOrderError(f"{name}의 시간 구간이 겹칩니다.")
            previous_end = end


def build_valid_purchase_item_intervals(
    status_intervals: pd.DataFrame,
    quantity_intervals: pd.DataFrame,
) -> pd.DataFrame:
    """양쪽 조건을 만족하는 `[valid_from, valid_until)`만 반환합니다.

    같은 상품의 인접 구간은 남은 수량까지 같을 때만 합칩니다. 따라서 주문 상태
    이름만 바뀐 경계는 제거해도, 부분 반품으로 수량이 달라진 시각은 보존합니다.
    """
    _require_columns(
        status_intervals,
        ("order_id", "valid_from", "valid_until", "is_valid_status"),
        "status_intervals",
    )
    _require_columns(
        quantity_intervals,
        (
            "order_item_id",
            "order_id",
            "valid_from",
            "valid_until",
            "remaining_quantity",
            "is_valid_quantity",
        ),
        "quantity_intervals",
    )
    _require_keys(status_intervals, ("order_id",), "status_intervals")
    _require_keys(
        quantity_intervals, ("order_item_id", "order_id"), "quantity_intervals"
    )
    for frame, column in (
        (status_intervals, "is_valid_status"),
        (quantity_intervals, "is_valid_quantity"),
    ):
        if not is_bool_dtype(frame[column].dtype) or frame[column].isna().any():
            raise OperationalOrderError(f"{column}는 결측 없는 boolean이어야 합니다.")
    _require_nonnegative_integer(quantity_intervals, "remaining_quantity")
    status_orders = set(status_intervals["order_id"])
    if not quantity_intervals["order_id"].isin(status_orders).all():
        raise OperationalOrderError("결제된 주문상품에 상태 이력이 없습니다.")

    valid_status: dict[object, list[tuple[pd.Timestamp, pd.Timestamp | None]]] = (
        defaultdict(list)
    )
    status_boundaries: dict[object, list[tuple[pd.Timestamp, pd.Timestamp | None]]] = (
        defaultdict(list)
    )
    for order_id, raw_start, raw_end, is_valid in status_intervals[
        ["order_id", "valid_from", "valid_until", "is_valid_status"]
    ].itertuples(index=False, name=None):
        start, end = _utc(raw_start, column="valid_from"), _end(raw_end)
        if end is not None and end <= start:
            raise OperationalOrderError(
                "주문 상태 구간의 종료 시각이 시작보다 늦지 않습니다."
            )
        status_boundaries[order_id].append((start, end))
        if is_valid:
            valid_status[order_id].append((start, end))
    _require_nonoverlapping(status_boundaries, name="주문 상태")

    by_item: dict[object, list[dict[str, object]]] = defaultdict(list)
    quantity_boundaries: dict[
        object, list[tuple[pd.Timestamp, pd.Timestamp | None]]
    ] = defaultdict(list)
    for (
        item_id,
        order_id,
        raw_start,
        raw_end,
        remaining,
        is_valid,
    ) in quantity_intervals[
        [
            "order_item_id",
            "order_id",
            "valid_from",
            "valid_until",
            "remaining_quantity",
            "is_valid_quantity",
        ]
    ].itertuples(index=False, name=None):
        start, end = _utc(raw_start, column="valid_from"), _end(raw_end)
        if end is not None and end <= start:
            raise OperationalOrderError(
                "상품 수량 구간의 종료 시각이 시작보다 늦지 않습니다."
            )
        quantity_boundaries[item_id].append((start, end))
        if is_valid != (int(remaining) > 0):
            raise OperationalOrderError(
                "상품 수량 구간의 유효 표시와 잔여 수량이 다릅니다."
            )
        if not is_valid:
            continue
        for status_start, status_end in valid_status[order_id]:
            overlap_start = max(start, status_start)
            overlap_end = _intersection_end(end, status_end)
            if overlap_end is not None and overlap_start >= overlap_end:
                continue
            by_item[item_id].append(
                {
                    "order_item_id": item_id,
                    "order_id": order_id,
                    "valid_from": overlap_start,
                    "valid_until": overlap_end,
                    "remaining_quantity": int(remaining),
                }
            )
    _require_nonoverlapping(quantity_boundaries, name="상품 수량")

    result = []
    for item_rows in by_item.values():
        item_rows.sort(key=lambda row: row["valid_from"])
        for row in item_rows:
            if (
                result
                and result[-1]["order_item_id"] == row["order_item_id"]
                and result[-1]["valid_until"] == row["valid_from"]
                and result[-1]["remaining_quantity"] == row["remaining_quantity"]
            ):
                result[-1]["valid_until"] = row["valid_until"]
            else:
                result.append(row)
    return pd.DataFrame(
        result,
        columns=(
            "order_item_id",
            "order_id",
            "valid_from",
            "valid_until",
            "remaining_quantity",
        ),
    )
