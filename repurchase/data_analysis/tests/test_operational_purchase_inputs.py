"""반려동물 연결 제외가 전체 구매·사용자 주문 수를 바꾸지 않는지 검증합니다."""

from __future__ import annotations

import pandas as pd

from scripts.modeling.operational_purchase_inputs import (
    prepare_operational_purchase_inputs,
)


def test_late_birth_changes_only_pet_scoped_events() -> None:
    orders = pd.DataFrame(
        {
            "order_id": ["o1", "o2"],
            "user_id": ["u1", "u1"],
            "ordered_at": ["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"],
            "paid_at": ["2026-01-01T00:01:00Z", "2026-02-01T00:01:00Z"],
            "order_status": ["PAID", "PAID"],
            "purchase_type": ["ONE_TIME", "ONE_TIME"],
        }
    )
    items = pd.DataFrame(
        {
            "order_item_id": ["i1", "i2"],
            "order_id": ["o1", "o2"],
            "product_id": ["p1", "p1"],
            "product_group_id_snapshot": ["g1", "g1"],
            "category_code_snapshot": ["FOOD", "FOOD"],
            "is_replenishable_snapshot": [True, True],
            "pet_id": ["pet1", "pet1"],
            "quantity": [1, 1],
            "item_status": ["PAID", "PAID"],
            "cancelled_quantity": [0, 0],
            "returned_quantity": [0, 0],
        }
    )
    pets = pd.DataFrame({"pet_id": ["pet1"], "birth_date": ["2026-01-15"]})

    result = prepare_operational_purchase_inputs(orders, items, pets)

    assert result.excluded_late_birth_item_count == 1
    assert result.valid_items["order_item_id"].tolist() == ["i1", "i2"]
    assert result.pet_history_items["order_item_id"].tolist() == ["i2"]
    assert result.all_purchase_events["order_id"].tolist() == ["o1", "o2"]
    assert result.pet_purchase_events["order_id"].tolist() == ["o2"]
    assert (
        result.all_purchase_events.groupby("user_id")["order_id"].nunique()["u1"] == 2
    )
