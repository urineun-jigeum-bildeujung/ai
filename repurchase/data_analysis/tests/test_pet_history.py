"""반려동물 연결 오류만 개인 이력에서 제외되는지 검증합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.operational_orders import OperationalOrderError
from scripts.modeling.pet_history import select_pet_history_items


def _sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    items = pd.DataFrame(
        {
            "order_item_id": ["i1", "i2", "i3", "i4"],
            "order_id": ["o1", "o1", "o2", "o2"],
            "pet_id": ["early", "late", "same_day", None],
            "net_quantity": [1, 1, 1, 1],
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": ["o1", "o2"],
            "ordered_at": ["2026-01-01T15:30:00Z", "2026-01-03T00:30:00+09:00"],
        }
    )
    pets = pd.DataFrame(
        {
            "pet_id": ["early", "late", "same_day"],
            "birth_date": ["2026-01-01", "2026-01-03", "2026-01-03"],
        }
    )
    return items, orders, pets


def test_later_birth_excludes_only_pet_history_line() -> None:
    items, orders, pets = _sources()
    original = items.copy(deep=True)

    result = select_pet_history_items(items, orders, pets)

    assert result["order_item_id"].tolist() == ["i1", "i3"]
    assert result["order_id"].nunique() == 2
    assert len(items) == 4  # 원래 주문상품과 사용자 주문 수의 원천은 그대로입니다.
    pd.testing.assert_frame_equal(items, original)


def test_missing_birth_of_referenced_pet_fails_clearly() -> None:
    items, orders, pets = _sources()
    pets.loc[pets["pet_id"].eq("early"), "birth_date"] = None

    with pytest.raises(OperationalOrderError, match="비어 있습니다"):
        select_pet_history_items(items, orders, pets)


def test_unknown_pet_and_timezone_free_order_fail_clearly() -> None:
    items, orders, pets = _sources()
    with pytest.raises(OperationalOrderError, match="pet_id가 없습니다"):
        select_pet_history_items(items, orders, pets.iloc[1:])

    orders.loc[0, "ordered_at"] = "2026-01-01 15:30:00"
    with pytest.raises(OperationalOrderError, match="시간대"):
        select_pet_history_items(items, orders, pets)
