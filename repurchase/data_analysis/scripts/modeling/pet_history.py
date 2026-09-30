"""목데이터의 반려동물 연결을 과거 구매 시점에 맞게 선별합니다."""

from __future__ import annotations

import pandas as pd

from .operational_orders import OperationalOrderError, _require_columns, _require_keys

SEOUL_TIMEZONE = "Asia/Seoul"


def select_pet_history_items(
    valid_items: pd.DataFrame, orders: pd.DataFrame, pets: pd.DataFrame
) -> pd.DataFrame:
    """반려동물별 이력에 쓸 구매 행만 반환하고 원본 주문은 보존합니다.

    생일이 주문일보다 *늦은* 연결만 제외합니다. 같은 날은 유효합니다.
    pet_id 미지정 행은 반려동물별 이력이 아니므로 이 결과에 넣지 않습니다.
    사용자 전체 주문 수는 이 결과가 아닌 원래 주문에서 계산해야 합니다.
    """
    _require_columns(
        valid_items, ("order_item_id", "order_id", "pet_id"), "valid_items"
    )
    _require_columns(orders, ("order_id", "ordered_at"), "orders")
    _require_columns(pets, ("pet_id", "birth_date"), "pets")
    _require_keys(orders, ("order_id",), "orders")
    _require_keys(pets, ("pet_id",), "pets")
    if orders["order_id"].duplicated().any() or pets["pet_id"].duplicated().any():
        raise OperationalOrderError("주문 또는 반려동물 식별키가 중복됐습니다.")

    pet_items = valid_items.loc[valid_items["pet_id"].notna()].copy()
    if pet_items.empty:
        return pet_items.reset_index(drop=True)
    if not pet_items["order_id"].isin(orders["order_id"]).all():
        raise OperationalOrderError("반려동물 구매 행의 주문이 없습니다.")
    if not pet_items["pet_id"].isin(pets["pet_id"]).all():
        raise OperationalOrderError("반려동물 구매 행의 pet_id가 없습니다.")

    joined = pet_items.merge(
        orders[["order_id", "ordered_at"]], on="order_id", validate="many_to_one"
    ).merge(pets[["pet_id", "birth_date"]], on="pet_id", validate="many_to_one")
    if joined["ordered_at"].isna().any() or joined["birth_date"].isna().any():
        raise OperationalOrderError(
            "반려동물 구매 행의 주문 시각·생일이 비어 있습니다."
        )
    try:
        order_timestamps = joined["ordered_at"].map(pd.Timestamp)
        birth_date = pd.to_datetime(joined["birth_date"], errors="raise")
    except (TypeError, ValueError) as error:
        raise OperationalOrderError(
            "주문 시각 또는 반려동물 생일이 잘못됐습니다."
        ) from error
    if any(value.tzinfo is None for value in order_timestamps):
        raise OperationalOrderError("ordered_at에는 시간대가 필요합니다.")
    if birth_date.dt.tz is not None:
        raise OperationalOrderError("birth_date는 시간대 없는 날짜여야 합니다.")

    # 생일은 날짜만 있으므로 주문 시각을 한국 달력 날짜로 바꾼 뒤 비교합니다.
    order_dates = order_timestamps.map(
        lambda value: value.tz_convert(SEOUL_TIMEZONE).date()
    )
    eligible = birth_date.dt.date.le(order_dates)
    return joined.loc[eligible, list(valid_items.columns)].reset_index(drop=True)
