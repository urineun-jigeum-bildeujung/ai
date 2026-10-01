"""운영 CSV 검수 명령이 실제 BE 컬럼 이름을 올바르게 읽는지 검증합니다."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.audit_operational_pet_history import audit_sources


def test_audit_source_csv_preserves_orders(tmp_path: Path) -> None:
    orders_path = tmp_path / "orders.csv"
    items_path = tmp_path / "items.csv"
    pets_path = tmp_path / "pets.csv"
    pd.DataFrame(
        {
            "id": [1, 2],
            "member_id": [10, 10],
            "ordered_at": ["2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"],
            "paid_at": ["2026-01-01T00:01:00Z", "2026-02-01T00:01:00Z"],
            "order_status": ["PAID", "PAID"],
            "purchase_type": ["ONE_TIME", "ONE_TIME"],
        }
    ).to_csv(orders_path, index=False)
    pd.DataFrame(
        {
            "id": [101, 102],
            "order_id": [1, 2],
            "product_id": [5, 5],
            "product_group_id_snapshot": [7, 7],
            "category_code_snapshot": ["FOOD", "FOOD"],
            "is_replenishable_snapshot": ["t", "t"],
            "pet_id": [20, 20],
            "quantity": [1, 1],
            "item_status": ["PAID", "PAID"],
            "cancelled_quantity": [0, 0],
            "returned_quantity": [0, 0],
        }
    ).to_csv(items_path, index=False)
    pd.DataFrame({"pet_id": [20], "birth_date": ["2026-01-15"]}).to_csv(
        pets_path, index=False
    )

    result = audit_sources(orders_path, items_path, pets_path)

    assert result == {
        "source_order_count": 2,
        "source_order_item_count": 2,
        "valid_order_item_count": 2,
        "valid_purchase_order_count": 2,
        "pet_history_item_count": 1,
        "excluded_late_birth_item_count": 1,
    }
