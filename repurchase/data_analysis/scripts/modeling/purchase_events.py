"""유효 주문 상품 행을 상품군·반려동물 범위의 구매 사건으로 묶습니다.

상품군 사건은 아직 UCI의 SKU 기준 모델 입력이 아닙니다. 범위와 출처를
보존하는 중간 데이터이며, 모델 연결은 별도 검증 후에만 진행합니다.
"""

from __future__ import annotations

import pandas as pd

from .operational_orders import OperationalOrderError, build_valid_order_items

EVENT_COLUMNS = (
    "user_id",
    "order_id",
    "target_scope",
    "target_id",
    "pet_id",
    "paid_at",
    "category_code_snapshot",
    "is_replenishable_snapshot",
    "purchase_type",
    "net_unit_count",
    "source_order_item_count",
)
GROUP_COLUMNS = ("user_id", "order_id", "product_group_id_snapshot", "pet_id")


def build_product_group_purchase_events(
    orders: pd.DataFrame, order_items: pd.DataFrame
) -> pd.DataFrame:
    """같은 주문·상품군·반려동물의 유효 상품 행을 한 사건으로 합칩니다.

    반려동물이 지정되지 않은 행은 지정된 행에 임의로 귀속하지 않습니다.
    비반복 상품도 사용자 전체 주문 이력에 필요하므로 결과에 남깁니다.
    """
    valid_items = build_valid_order_items(orders, order_items)
    events: list[dict[str, object]] = []

    # 결측 pet_id도 독립 그룹으로 유지해 다른 반려동물의 구매와 섞지 않습니다.
    for (user_id, order_id, group_id, pet_id), group in valid_items.groupby(
        list(GROUP_COLUMNS), dropna=False, sort=False
    ):
        if (
            group["category_code_snapshot"].nunique(dropna=False) != 1
            or group["is_replenishable_snapshot"].nunique(dropna=False) != 1
        ):
            raise OperationalOrderError(
                "한 주문·상품군·반려동물의 카테고리 또는 반복 소비 여부가 서로 다릅니다."
            )

        first = group.iloc[0]
        events.append(
            {
                "user_id": user_id,
                "order_id": order_id,
                "target_scope": "PRODUCT_GROUP",
                "target_id": group_id,
                "pet_id": None if pd.isna(pet_id) else pet_id,
                "paid_at": first["paid_at"],
                "category_code_snapshot": first["category_code_snapshot"],
                "is_replenishable_snapshot": bool(first["is_replenishable_snapshot"]),
                "purchase_type": first["purchase_type"],
                # Python 정수로 더해 수량 합산 시 고정 폭 정수 오버플로를 피합니다.
                "net_unit_count": sum(int(value) for value in group["net_quantity"]),
                "source_order_item_count": len(group),
            }
        )

    return pd.DataFrame(events, columns=list(EVENT_COLUMNS))
