"""유효 주문상품 구간을 사용자 주문·반려동물 상품군 사건으로 묶습니다."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_bool_dtype

from .operational_asof import _utc
from .operational_orders import (
    OperationalOrderError,
    _require_columns,
    _require_keys,
    _require_nonnegative_integer,
)
from .operational_validity_intervals import _end, _require_nonoverlapping
from .pet_history import select_pet_history_items


@dataclass(frozen=True)
class OperationalEventIntervals:
    """전체 주문 수용 사건과 반려동물·상품군용 사건을 분리합니다."""

    user_orders: pd.DataFrame
    pet_targets: pd.DataFrame


def _sweep_group(rows: pd.DataFrame) -> list[dict[str, object]]:
    """동일 그룹의 상품 시작·종료를 한 시각에 반영합니다."""
    changes: dict[pd.Timestamp, list[int]] = defaultdict(lambda: [0, 0])
    for start, end, quantity in rows[
        ["valid_from", "valid_until", "remaining_quantity"]
    ].itertuples(index=False, name=None):
        changes[start][0] += int(quantity)
        changes[start][1] += 1
        if pd.notna(end):
            changes[end][0] -= int(quantity)
            changes[end][1] -= 1

    moments = sorted(changes)
    segments: list[dict[str, object]] = []
    running_quantity = running_items = 0
    for index, start in enumerate(moments):
        quantity_delta, item_delta = changes[start]
        running_quantity += quantity_delta
        running_items += item_delta
        if running_quantity < 0 or running_items < 0:
            raise OperationalOrderError(
                "구매 사건 구간의 누적 수량·상품 수가 음수입니다."
            )
        end = moments[index + 1] if index + 1 < len(moments) else pd.NaT
        if running_quantity == 0:
            if running_items != 0:
                raise OperationalOrderError("구매 사건 수량과 상품 수가 서로 다릅니다.")
            continue
        if (
            segments
            and segments[-1]["valid_until"] == start
            and (
                segments[-1]["net_unit_count"] == running_quantity
                and segments[-1]["source_order_item_count"] == running_items
            )
        ):
            segments[-1]["valid_until"] = end
        else:
            segments.append(
                {
                    "valid_from": start,
                    "valid_until": end,
                    "net_unit_count": running_quantity,
                    "source_order_item_count": running_items,
                }
            )
    return segments


def _aggregate(
    rows: pd.DataFrame, *, keys: tuple[str, ...], columns: tuple[str, ...]
) -> pd.DataFrame:
    result: list[dict[str, object]] = []
    for group_key, group in rows.groupby(list(keys), sort=False, observed=True):
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        metadata = dict(zip(keys, values, strict=True))
        for segment in _sweep_group(group):
            result.append({**metadata, **segment})
    return pd.DataFrame(result, columns=columns)


def build_operational_event_intervals(
    valid_item_intervals: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    pets: pd.DataFrame,
) -> OperationalEventIntervals:
    """전체 유효 주문과 반복소비 반려동물·상품군 사건을 별도로 반환합니다.

    `pet_id`가 없거나 생일이 주문일보다 늦은 상품도 전체 사용자 주문에는 남깁니다.
    최종 주문상품 상태는 과거 시점에 소급 적용하지 않습니다.
    """
    _require_columns(
        valid_item_intervals,
        (
            "order_item_id",
            "order_id",
            "valid_from",
            "valid_until",
            "remaining_quantity",
        ),
        "valid_item_intervals",
    )
    _require_columns(orders, ("order_id", "user_id", "ordered_at", "paid_at"), "orders")
    _require_columns(
        order_items,
        (
            "order_item_id",
            "order_id",
            "pet_id",
            "product_group_id_snapshot",
            "category_code_snapshot",
            "is_replenishable_snapshot",
        ),
        "order_items",
    )
    for frame, keys, name in (
        (orders, ("order_id", "user_id"), "orders"),
        (order_items, ("order_item_id", "order_id"), "order_items"),
        (valid_item_intervals, ("order_item_id", "order_id"), "valid_item_intervals"),
    ):
        _require_keys(frame, keys, name)
    if (
        orders["order_id"].duplicated().any()
        or order_items["order_item_id"].duplicated().any()
    ):
        raise OperationalOrderError("주문 또는 주문상품 ID가 중복됐습니다.")
    if not is_bool_dtype(order_items["is_replenishable_snapshot"].dtype) or (
        order_items["is_replenishable_snapshot"].isna().any()
    ):
        raise OperationalOrderError("반복 소비 여부는 결측 없는 boolean이어야 합니다.")
    _require_nonnegative_integer(valid_item_intervals, "remaining_quantity")
    if valid_item_intervals["remaining_quantity"].le(0).any():
        raise OperationalOrderError("유효 구매 구간의 잔여 수량은 양수여야 합니다.")
    intervals = valid_item_intervals.copy()
    intervals["valid_from"] = intervals["valid_from"].map(
        lambda value: _utc(value, column="valid_from")
    )
    intervals["valid_until"] = intervals["valid_until"].map(
        lambda value: pd.NaT if pd.isna(value) else _utc(value, column="valid_until")
    )
    if intervals.groupby("order_item_id")["order_id"].nunique().gt(1).any():
        raise OperationalOrderError("동일 주문상품 ID가 여러 주문에 연결됐습니다.")
    item_spans: dict[object, list[tuple[pd.Timestamp, pd.Timestamp | None]]] = (
        defaultdict(list)
    )
    for item_id, start, raw_end in intervals[
        ["order_item_id", "valid_from", "valid_until"]
    ].itertuples(index=False, name=None):
        end = _end(raw_end)
        if end is not None and end <= start:
            raise OperationalOrderError(
                "유효 구매 구간의 종료 시각이 시작보다 늦지 않습니다."
            )
        item_spans[item_id].append((start, end))
    _require_nonoverlapping(item_spans, name="주문상품")

    joined = intervals.merge(
        order_items[
            [
                "order_item_id",
                "order_id",
                "pet_id",
                "product_group_id_snapshot",
                "category_code_snapshot",
                "is_replenishable_snapshot",
            ]
        ],
        on=["order_item_id", "order_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if joined["_merge"].ne("both").any():
        raise OperationalOrderError(
            "유효 구매 구간에 연결되지 않는 주문상품이 있습니다."
        )
    joined = joined.drop(columns="_merge").merge(
        orders[["order_id", "user_id", "ordered_at", "paid_at"]],
        on="order_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if joined["_merge"].ne("both").any():
        raise OperationalOrderError("유효 구매 구간에 연결되지 않는 주문이 있습니다.")
    joined = joined.drop(columns="_merge")
    user_columns = (
        "user_id",
        "order_id",
        "valid_from",
        "valid_until",
        "net_unit_count",
        "source_order_item_count",
    )
    user_orders = _aggregate(joined, keys=("user_id", "order_id"), columns=user_columns)

    pet_items = select_pet_history_items(
        joined.drop(columns=["ordered_at", "paid_at"]), orders, pets
    )
    target_keys = ("user_id", "pet_id", "product_group_id_snapshot", "order_id")
    if pet_items[list(target_keys)].isna().any(axis=None):
        raise OperationalOrderError("반려동물 구매 사건에 대상 키 결측값이 있습니다.")
    category_counts = pet_items.groupby(list(target_keys), sort=False, observed=True)[
        "category_code_snapshot"
    ].nunique(dropna=False)
    replenishable_counts = pet_items.groupby(
        list(target_keys), sort=False, observed=True
    )["is_replenishable_snapshot"].nunique(dropna=False)
    if (
        category_counts.gt(1).any()
        or replenishable_counts.gt(1).any()
        or pet_items["category_code_snapshot"].isna().any()
    ):
        raise OperationalOrderError(
            "같은 구매 사건의 카테고리 또는 반복 소비 여부가 서로 다릅니다."
        )
    pet_items = pet_items.loc[pet_items["is_replenishable_snapshot"]].copy()
    pet_columns = (
        *target_keys,
        "valid_from",
        "valid_until",
        "net_unit_count",
        "source_order_item_count",
    )
    pet_targets = _aggregate(pet_items, keys=target_keys, columns=pet_columns)
    return OperationalEventIntervals(user_orders=user_orders, pet_targets=pet_targets)
