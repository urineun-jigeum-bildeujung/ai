"""라벨은 관측 종료까지 확인된 유효 구매만 정답으로 사용합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.features import MINIMAL_MODEL_FEATURE_COLUMNS
from scripts.modeling.operational_event_intervals import OperationalEventIntervals
from scripts.modeling.operational_label_rebuild import (
    SERVICE_LABEL_COLUMNS,
    rebuild_service_labels_from_event_intervals,
)
from scripts.modeling.operational_orders import OperationalOrderError


def _events() -> tuple[OperationalEventIntervals, pd.DataFrame]:
    paid = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-11T00:00:00Z",
            "2026-01-31T00:00:00Z",
        ]
    )
    orders = pd.DataFrame(
        {
            "user_id": ["u1"] * 3,
            "order_id": ["o1", "o2", "o3"],
            "paid_at": paid,
        }
    )
    user = pd.DataFrame(
        {
            "user_id": ["u1"] * 3,
            "order_id": orders["order_id"],
            "valid_from": paid,
            "valid_until": pd.to_datetime([None, "2026-01-20T00:00:00Z", None]),
        }
    )
    pet = pd.DataFrame(
        {
            "user_id": ["u1"] * 3,
            "pet_id": ["p1"] * 3,
            "product_group_id_snapshot": ["g1"] * 3,
            "order_id": orders["order_id"],
            "valid_from": paid,
            "valid_until": user["valid_until"],
        }
    )
    return OperationalEventIntervals(user_orders=user, pet_targets=pet), orders


def test_label_cut_includes_later_refunded_purchase_only_before_refund() -> None:
    events, orders = _events()

    early = rebuild_service_labels_from_event_intervals(
        events, orders, observation_end_at=pd.Timestamp("2026-01-15T00:00:00Z")
    ).set_index("order_id")
    late = rebuild_service_labels_from_event_intervals(
        events, orders, observation_end_at=pd.Timestamp("2026-02-01T00:00:00Z")
    ).set_index("order_id")

    assert set(early.index) == {"o1", "o2"}
    assert early.loc["o1", "next_order_id"] == "o2"
    assert early.loc["o1", "duration_days"] == 10.0
    assert bool(early.loc["o2", "is_right_censored"])
    assert early.loc["o2", "duration_days"] == 4.0

    assert set(late.index) == {"o1", "o3"}
    assert late.loc["o1", "next_order_id"] == "o3"
    assert late.loc["o1", "duration_days"] == 30.0
    assert bool(late.loc["o3", "is_right_censored"])


def test_label_cut_at_purchase_has_zero_day_right_censor() -> None:
    events, orders = _events()

    rows = rebuild_service_labels_from_event_intervals(
        events, orders, observation_end_at=pd.Timestamp("2026-01-01T00:00:00Z")
    )

    assert len(rows) == 1
    assert rows.loc[0, "duration_days"] == 0.0
    assert bool(rows.loc[0, "is_right_censored"])
    assert not bool(rows.loc[0, "event_observed"])
    assert tuple(rows.columns) == SERVICE_LABEL_COLUMNS
    assert set(MINIMAL_MODEL_FEATURE_COLUMNS).isdisjoint(rows.columns)


def test_label_rebuild_rejects_pet_event_without_user_order() -> None:
    events, orders = _events()
    missing_user = OperationalEventIntervals(
        user_orders=events.user_orders.loc[events.user_orders["order_id"].ne("o1")],
        pet_targets=events.pet_targets,
    )

    with pytest.raises(OperationalOrderError, match="대응하는 사용자 주문"):
        rebuild_service_labels_from_event_intervals(
            missing_user,
            orders,
            observation_end_at=pd.Timestamp("2026-02-01T00:00:00Z"),
        )
