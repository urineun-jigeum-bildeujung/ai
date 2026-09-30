"""운영 원천을 전체 구매와 반려동물별 구매 이력으로 분리합니다."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .operational_orders import build_valid_order_items
from .pet_history import select_pet_history_items
from .purchase_events import build_product_group_purchase_events_from_valid_items


@dataclass(frozen=True)
class OperationalPurchaseInputs:
    """생일 불일치가 전체 구매 이력을 손상하지 않았는지 함께 보존합니다."""

    valid_items: pd.DataFrame
    all_purchase_events: pd.DataFrame
    pet_history_items: pd.DataFrame
    pet_purchase_events: pd.DataFrame
    excluded_late_birth_item_count: int


def prepare_operational_purchase_inputs(
    orders: pd.DataFrame, order_items: pd.DataFrame, pets: pd.DataFrame
) -> OperationalPurchaseInputs:
    """원천 검증 후 잘못된 pet 연결만 반려동물 이력에서 제외합니다.

    사용자 주문 수는 all_purchase_events의 고유 주문으로 계산할 수 있습니다.
    pet_purchase_events만 이용하면 반려동물 미지정·생일 불일치 구매가 빠집니다.
    """
    valid_items = build_valid_order_items(orders, order_items)
    pet_items = select_pet_history_items(valid_items, orders, pets)
    specified_pet_count = int(valid_items["pet_id"].notna().sum())
    return OperationalPurchaseInputs(
        valid_items=valid_items,
        all_purchase_events=build_product_group_purchase_events_from_valid_items(
            valid_items
        ),
        pet_history_items=pet_items,
        pet_purchase_events=build_product_group_purchase_events_from_valid_items(
            pet_items
        ),
        excluded_late_birth_item_count=specified_pet_count - len(pet_items),
    )
