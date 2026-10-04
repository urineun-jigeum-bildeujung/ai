"""로컬 변경 입력에 미래 주문을 복사하지 않도록 후보 선택을 검증합니다."""

import pandas as pd

from scripts.verify_shadow_source_delta import _candidate


def test_candidate_ignores_future_order_even_with_visible_target() -> None:
    orders = pd.DataFrame(
        [
            ("3", "42", "CONFIRMED", "2026-01-04T00:00:00Z", "2026-01-04T01:00:00Z"),
            ("1", "42", "CONFIRMED", "2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"),
            ("2", "42", "CONFIRMED", "2026-01-02T00:00:00Z", "2026-01-02T01:00:00Z"),
        ],
        columns=["order_id", "user_id", "order_status", "ordered_at", "paid_at"],
    )
    items = pd.DataFrame(
        [
            ("30", "3", "7", "PAID", "9", True),
            ("10", "1", "7", "PAID", "9", True),
            ("20", "2", "7", "PAID", "9", True),
        ],
        columns=[
            "order_item_id",
            "order_id",
            "pet_id",
            "item_status",
            "product_group_id_snapshot",
            "is_replenishable_snapshot",
        ],
    )
    baseline = pd.DataFrame([{"user_id": "42", "pet_id": 7, "target_id": "9"}])

    order, item = _candidate(
        {"orders": orders, "order_items": items},
        baseline,
        as_of=pd.Timestamp("2026-01-03T00:00:00Z"),
    )

    assert order["order_id"] == "1"
    assert item["order_item_id"] == "10"
