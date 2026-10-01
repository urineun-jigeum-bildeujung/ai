"""전체 사용자 과거 주문 수는 앵커 시각의 유효 주문으로 다시 셉니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.operational_feature_recount import (
    recount_target_interval_features_as_of,
    recount_user_prior_orders_as_of,
)
from scripts.modeling.operational_orders import OperationalOrderError


def test_recount_excludes_same_time_and_future_orders_but_keeps_all_products() -> None:
    samples = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "anchor_at": [
                "2026-01-05T00:00:00Z",
                "2026-01-10T00:00:00Z",
                "2026-01-10T00:00:00Z",
            ],
        }
    )
    orders = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1", "u2", "u3"],
            "order_id": ["o1", "o2", "o3", "o4", "o5", "pending"],
            "paid_at": [
                "2026-01-01T00:00:00Z",
                "2026-01-05T00:00:00Z",
                "2026-01-09T00:00:00Z",
                "2026-01-12T00:00:00Z",
                "2026-01-01T00:00:00Z",
                None,
            ],
        }
    )
    # o1의 두 구간은 수량 변경 전후이며 한 주문으로만 셉니다.
    # o3은 다른 상품·반려동물 미지정 주문이라고 가정해도 전체 주문 수에 남습니다.
    intervals = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1", "u1"],
            "order_id": ["o1", "o1", "o2", "o3", "o4"],
            "valid_from": [
                "2026-01-01T00:00:00Z",
                "2026-01-03T00:00:00Z",
                "2026-01-05T00:00:00Z",
                "2026-01-09T00:00:00Z",
                "2026-01-12T00:00:00Z",
            ],
            "valid_until": ["2026-01-03T00:00:00Z", None, None, None, None],
        }
    )

    result = recount_user_prior_orders_as_of(samples, intervals, orders)

    assert result["as_of_user_prior_order_count"].tolist() == [1, 3, 0]
    assert result["sample_row"].tolist() == [0, 1, 2]
    assert "sample_row" not in samples


def test_recount_respects_end_boundary_and_later_reactivation() -> None:
    samples = pd.DataFrame(
        {
            "user_id": ["u1"] * 3,
            "anchor_at": [
                "2026-01-04T00:00:00Z",
                "2026-01-07T00:00:00Z",
                "2026-01-10T00:00:00Z",
            ],
        }
    )
    orders = pd.DataFrame(
        {
            "user_id": ["u1"],
            "order_id": ["o1"],
            "paid_at": ["2026-01-01T00:00:00Z"],
        }
    )
    intervals = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "order_id": ["o1", "o1"],
            "valid_from": ["2026-01-01T00:00:00Z", "2026-01-10T00:00:00Z"],
            "valid_until": ["2026-01-07T00:00:00Z", None],
        }
    )

    result = recount_user_prior_orders_as_of(samples, intervals, orders)

    assert result["as_of_user_prior_order_count"].tolist() == [1, 0, 1]


def test_recount_rejects_invalid_interval_and_unknown_order() -> None:
    samples = pd.DataFrame(
        {
            "user_id": ["u1"],
            "anchor_at": ["2026-01-03T00:00:00Z"],
        }
    )
    orders = pd.DataFrame(
        {
            "user_id": ["u1"],
            "order_id": ["o1"],
            "paid_at": ["2026-01-01T00:00:00Z"],
        }
    )
    interval = pd.DataFrame(
        {
            "user_id": ["u1"],
            "order_id": ["o1"],
            "valid_from": ["2026-01-03T00:00:00Z"],
            "valid_until": ["2026-01-03T00:00:00Z"],
        }
    )
    with pytest.raises(OperationalOrderError, match="종료 시각"):
        recount_user_prior_orders_as_of(samples, interval, orders)

    interval["valid_until"] = None
    interval["order_id"] = "missing"
    with pytest.raises(OperationalOrderError, match="연결되지 않는 주문"):
        recount_user_prior_orders_as_of(samples, interval, orders)


def test_recount_rejects_naive_anchor_and_interval_before_payment() -> None:
    samples = pd.DataFrame(
        {
            "user_id": ["u1"],
            "anchor_at": ["2026-01-03T00:00:00Z"],
        }
    )
    orders = pd.DataFrame(
        {
            "user_id": ["u1"],
            "order_id": ["o1"],
            "paid_at": ["2026-01-02T00:00:00Z"],
        }
    )
    intervals = pd.DataFrame(
        {
            "user_id": ["u1"],
            "order_id": ["o1"],
            "valid_from": ["2026-01-01T00:00:00Z"],
            "valid_until": [None],
        }
    )
    with pytest.raises(OperationalOrderError, match="결제 시각보다 먼저"):
        recount_user_prior_orders_as_of(samples, intervals, orders)

    intervals["valid_from"] = "2026-01-02T00:00:00Z"
    samples["anchor_at"] = "2026-01-03"
    with pytest.raises(OperationalOrderError, match="시간대"):
        recount_user_prior_orders_as_of(samples, intervals, orders)


def test_target_history_uses_only_events_active_at_each_anchor() -> None:
    orders = pd.DataFrame(
        {
            "user_id": ["u1"] * 3,
            "order_id": ["o1", "o2", "o3"],
            "paid_at": [
                "2026-01-01T00:00:00Z",
                "2026-01-10T00:00:00Z",
                "2026-01-20T00:00:00Z",
            ],
        }
    )
    intervals = pd.DataFrame(
        {
            "user_id": ["u1"] * 3,
            "pet_id": ["p1"] * 3,
            "product_group_id_snapshot": ["g1"] * 3,
            "order_id": ["o1", "o2", "o3"],
            "valid_from": orders["paid_at"],
            "valid_until": ["2026-01-15T00:00:00Z", None, None],
        }
    )
    samples = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "pet_id": ["p1", "p1"],
            "target_id": ["g1", "g1"],
            "order_id": ["o2", "o3"],
            "anchor_at": [orders.loc[1, "paid_at"], orders.loc[2, "paid_at"]],
        }
    )

    result = recount_target_interval_features_as_of(samples, intervals, orders)

    assert result["as_of_history_interval_count"].tolist() == [1, 1]
    assert result["as_of_history_median_days"].tolist() == [9.0, 10.0]
    assert result["as_of_history_mad_days"].isna().all()
    assert result["as_of_history_relative_mad"].isna().all()


def test_target_history_median_and_mad_match_training_definition() -> None:
    dates = [
        "2026-01-01T00:00:00Z",
        "2026-01-11T00:00:00Z",
        "2026-01-31T00:00:00Z",
        "2026-03-02T00:00:00Z",
    ]
    orders = pd.DataFrame(
        {
            "user_id": ["u1"] * 4,
            "order_id": ["o1", "o2", "o3", "o4"],
            "paid_at": dates,
        }
    )
    intervals = pd.DataFrame(
        {
            "user_id": ["u1"] * 4,
            "pet_id": ["p1"] * 4,
            "product_group_id_snapshot": ["g1"] * 4,
            "order_id": orders["order_id"],
            "valid_from": dates,
            "valid_until": [None] * 4,
        }
    )
    samples = pd.DataFrame(
        {
            "user_id": ["u1"],
            "pet_id": ["p1"],
            "target_id": ["g1"],
            "order_id": ["o4"],
            "anchor_at": [dates[-1]],
        }
    )

    result = recount_target_interval_features_as_of(samples, intervals, orders)

    assert result.loc[0, "as_of_history_interval_count"] == 3
    assert result.loc[0, "as_of_history_median_days"] == 20.0
    assert result.loc[0, "as_of_history_mad_days"] == 10.0
    assert result.loc[0, "as_of_history_relative_mad"] == 0.5


def test_target_history_rejects_missing_current_event() -> None:
    orders = pd.DataFrame(
        {
            "user_id": ["u1"],
            "order_id": ["o1"],
            "paid_at": ["2026-01-01T00:00:00Z"],
        }
    )
    intervals = pd.DataFrame(
        {
            "user_id": ["u1"],
            "pet_id": ["p1"],
            "product_group_id_snapshot": ["g1"],
            "order_id": ["o1"],
            "valid_from": ["2026-01-01T00:00:00Z"],
            "valid_until": ["2026-01-05T00:00:00Z"],
        }
    )
    samples = pd.DataFrame(
        {
            "user_id": ["u1"],
            "pet_id": ["p1"],
            "target_id": ["g1"],
            "order_id": ["o1"],
            "anchor_at": ["2026-01-06T00:00:00Z"],
        }
    )

    with pytest.raises(OperationalOrderError, match="현재 대상 구매 사건"):
        recount_target_interval_features_as_of(samples, intervals, orders)


def test_target_history_empty_samples_return_empty_schema() -> None:
    samples = pd.DataFrame(
        columns=[
            "user_id",
            "pet_id",
            "target_id",
            "order_id",
            "anchor_at",
        ]
    )
    intervals = pd.DataFrame(
        columns=[
            "user_id",
            "pet_id",
            "product_group_id_snapshot",
            "order_id",
            "valid_from",
            "valid_until",
        ]
    )
    orders = pd.DataFrame(columns=["user_id", "order_id", "paid_at"])

    result = recount_target_interval_features_as_of(samples, intervals, orders)

    assert result.empty
    assert "as_of_history_interval_count" in result.columns


def test_target_history_reactivation_does_not_duplicate_order() -> None:
    orders = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "order_id": ["o1", "o2"],
            "paid_at": ["2026-01-01T00:00:00Z", "2026-01-20T00:00:00Z"],
        }
    )
    intervals = pd.DataFrame(
        {
            "user_id": ["u1"] * 3,
            "pet_id": ["p1"] * 3,
            "product_group_id_snapshot": ["g1"] * 3,
            "order_id": ["o1", "o1", "o2"],
            "valid_from": [
                "2026-01-01T00:00:00Z",
                "2026-01-10T00:00:00Z",
                "2026-01-20T00:00:00Z",
            ],
            "valid_until": ["2026-01-05T00:00:00Z", None, None],
        }
    )
    samples = pd.DataFrame(
        {
            "user_id": ["u1"],
            "pet_id": ["p1"],
            "target_id": ["g1"],
            "order_id": ["o2"],
            "anchor_at": ["2026-01-20T00:00:00Z"],
        }
    )

    result = recount_target_interval_features_as_of(samples, intervals, orders)

    assert result.loc[0, "as_of_history_interval_count"] == 1
    assert result.loc[0, "as_of_history_median_days"] == 19.0


def test_target_history_rejects_simultaneous_different_orders() -> None:
    paid_at = "2026-01-01T00:00:00Z"
    orders = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "order_id": ["o1", "o2"],
            "paid_at": [paid_at, paid_at],
        }
    )
    intervals = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "pet_id": ["p1", "p1"],
            "product_group_id_snapshot": ["g1", "g1"],
            "order_id": ["o1", "o2"],
            "valid_from": [paid_at, paid_at],
            "valid_until": [None, None],
        }
    )
    samples = pd.DataFrame(
        {
            "user_id": ["u1"],
            "pet_id": ["p1"],
            "target_id": ["g1"],
            "order_id": ["o2"],
            "anchor_at": [paid_at],
        }
    )

    with pytest.raises(OperationalOrderError, match="구매 순서"):
        recount_target_interval_features_as_of(samples, intervals, orders)


def test_target_history_rejects_timezone_free_anchor() -> None:
    paid_at = "2026-01-01T00:00:00Z"
    orders = pd.DataFrame(
        {
            "user_id": ["u1"],
            "order_id": ["o1"],
            "paid_at": [paid_at],
        }
    )
    intervals = pd.DataFrame(
        {
            "user_id": ["u1"],
            "pet_id": ["p1"],
            "product_group_id_snapshot": ["g1"],
            "order_id": ["o1"],
            "valid_from": [paid_at],
            "valid_until": [None],
        }
    )
    samples = pd.DataFrame(
        {
            "user_id": ["u1"],
            "pet_id": ["p1"],
            "target_id": ["g1"],
            "order_id": ["o1"],
            "anchor_at": ["2026-01-02"],
        }
    )

    with pytest.raises(OperationalOrderError, match="시간대"):
        recount_target_interval_features_as_of(samples, intervals, orders)
