"""서비스 학습·검증 컷이 미래 결과를 참조하지 않는지 확인합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.operational_event_intervals import OperationalEventIntervals
from scripts.modeling.operational_orders import OperationalOrderError
from scripts.modeling.operational_temporal_split import (
    build_service_train_validation_split,
)


def _events() -> tuple[OperationalEventIntervals, pd.DataFrame]:
    paid = pd.to_datetime(
        ["2026-01-01T00:00:00Z", "2026-01-11T00:00:00Z", "2026-01-31T00:00:00Z"]
    )
    orders = pd.DataFrame(
        {"user_id": ["u1"] * 3, "order_id": ["o1", "o2", "o3"], "paid_at": paid}
    )
    user = pd.DataFrame(
        {
            "user_id": ["u1"] * 3,
            "order_id": orders["order_id"],
            "valid_from": paid,
            "valid_until": pd.to_datetime([None, "2026-01-20T00:00:00Z", None]),
        }
    )
    pet = user.copy()
    pet["pet_id"] = "p1"
    pet["product_group_id_snapshot"] = "g1"
    return OperationalEventIntervals(user_orders=user, pet_targets=pet), orders


def test_split_uses_independent_cut_for_refund_and_censoring() -> None:
    events, orders = _events()

    split = build_service_train_validation_split(
        events,
        orders,
        train_end_at=pd.Timestamp("2026-01-15T00:00:00Z"),
        validation_end_at=pd.Timestamp("2026-02-01T00:00:00Z"),
    )

    train = split.train.set_index("order_id")
    validation = split.validation.set_index("order_id")
    assert set(train.index) == {"o1", "o2"}
    assert train.loc["o1", "next_order_id"] == "o2"
    assert bool(train.loc["o2", "is_right_censored"])
    assert train.loc["o2", "duration_days"] == 4.0
    # o2의 이후 환불은 train 당시 알려지지 않았으므로 train을 바꾸지 않습니다.
    assert set(validation.index) == {"o3"}
    assert validation.loc["o3", "history_median_days"] == 30.0
    assert bool(validation.loc["o3", "is_right_censored"])
    assert validation.loc["o3", "duration_days"] == 1.0


def test_split_assigns_exact_cut_anchor_only_to_earlier_period() -> None:
    events, orders = _events()

    split = build_service_train_validation_split(
        events,
        orders,
        train_end_at=pd.Timestamp("2026-01-11T00:00:00Z"),
        validation_end_at=pd.Timestamp("2026-01-31T00:00:00Z"),
    )

    assert set(split.train["order_id"]) == {"o1", "o2"}
    assert set(split.validation["order_id"]) == {"o3"}
    assert split.train["split"].eq("train").all()
    assert split.validation["split"].eq("validation").all()
    boundary = split.validation.set_index("order_id").loc["o3"]
    assert bool(boundary["is_right_censored"])
    assert boundary["duration_days"] == 0.0


@pytest.mark.parametrize(
    ("train_end", "validation_end"),
    [
        ("2026-01-15", "2026-02-01"),
        ("2026-02-01T00:00:00Z", "2026-01-15T00:00:00Z"),
        ("2026-01-15T00:00:00Z", "2026-01-15T00:00:00Z"),
    ],
)
def test_split_rejects_invalid_cuts(train_end: str, validation_end: str) -> None:
    events, orders = _events()
    with pytest.raises(OperationalOrderError, match="종료 시각"):
        build_service_train_validation_split(
            events,
            orders,
            train_end_at=pd.Timestamp(train_end),
            validation_end_at=pd.Timestamp(validation_end),
        )
