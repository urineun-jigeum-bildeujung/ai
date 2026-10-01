"""서비스 상품군·반려동물 라벨과 과거 피처의 시간 경계를 검증합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.operational_purchase_inputs import OperationalPurchaseInputs
from scripts.modeling.service_samples import (
    ServiceSampleError,
    build_current_service_features,
    build_service_repurchase_samples,
)

END = pd.Timestamp("2026-03-12T00:00:00Z")


def _prepared() -> OperationalPurchaseInputs:
    events = pd.DataFrame(
        {
            "user_id": ["u1"] * 5,
            "order_id": ["o1", "o2", "o3", "o4", "o5"],
            "target_scope": ["PRODUCT_GROUP"] * 5,
            "target_id": ["g1"] * 5,
            "pet_id": ["pet1", "pet2", "pet1", "pet1", "pet1"],
            "paid_at": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-06T00:00:00Z",
                    "2026-01-11T00:00:00Z",
                    "2026-01-31T00:00:00Z",
                    "2026-03-02T00:00:00Z",
                ]
            ),
            "is_replenishable_snapshot": [True] * 5,
        }
    )
    return OperationalPurchaseInputs(
        valid_items=pd.DataFrame(),
        all_purchase_events=events.copy(),
        pet_history_items=pd.DataFrame(),
        pet_purchase_events=events.copy(),
        excluded_late_birth_item_count=0,
    )


def test_labels_and_features_use_same_pet_group_history_only() -> None:
    prepared = _prepared()
    original = prepared.pet_purchase_events.copy(deep=True)

    rows = build_service_repurchase_samples(prepared, observation_end_at=END)
    by_order = rows.set_index("order_id")

    assert by_order.loc["o4", "history_interval_count"] == 2
    assert by_order.loc["o4", "history_median_days"] == 15.0
    assert by_order.loc["o4", "history_relative_mad"] == pytest.approx(1 / 3)
    assert by_order.loc["o4", "user_prior_order_count"] == 3
    assert by_order.loc["o4", "next_order_id"] == "o5"
    assert by_order.loc["o4", "duration_days"] == 30.0
    assert by_order.loc["o2", "history_interval_count"] == 0
    assert pd.isna(by_order.loc["o2", "history_median_days"])
    assert bool(by_order.loc["o5", "is_right_censored"])
    assert by_order.loc["o5", "duration_days"] == 10.0
    assert by_order.loc["o5", "history_interval_count"] == 3
    pd.testing.assert_frame_equal(prepared.pet_purchase_events, original)


def test_user_order_count_includes_purchase_not_in_pet_history() -> None:
    prepared = _prepared()
    excluded_pet_event = prepared.pet_purchase_events.loc[
        prepared.pet_purchase_events["order_id"].ne("o2")
    ].copy()
    prepared = OperationalPurchaseInputs(
        valid_items=prepared.valid_items,
        all_purchase_events=prepared.all_purchase_events,
        pet_history_items=prepared.pet_history_items,
        pet_purchase_events=excluded_pet_event,
        excluded_late_birth_item_count=1,
    )

    rows = build_service_repurchase_samples(prepared, observation_end_at=END)
    by_order = rows.set_index("order_id")

    assert "o2" not in by_order.index
    assert by_order.loc["o3", "user_prior_order_count"] == 2
    assert by_order.loc["o3", "history_interval_count"] == 1


def test_same_time_orders_do_not_gain_prior_count() -> None:
    prepared = _prepared()
    prepared.all_purchase_events.loc[
        prepared.all_purchase_events["order_id"].eq("o2"), "paid_at"
    ] = pd.Timestamp("2026-01-11T00:00:00Z")
    prepared.pet_purchase_events.loc[
        prepared.pet_purchase_events["order_id"].eq("o2"), "paid_at"
    ] = pd.Timestamp("2026-01-11T00:00:00Z")

    rows = build_service_repurchase_samples(prepared, observation_end_at=END)

    assert rows.set_index("order_id").loc["o3", "user_prior_order_count"] == 1


def test_future_event_and_timezone_free_cut_are_rejected() -> None:
    prepared = _prepared()
    with pytest.raises(ServiceSampleError, match="관측 종료 시각 이후"):
        build_service_repurchase_samples(
            prepared, observation_end_at=pd.Timestamp("2026-02-01T00:00:00Z")
        )
    with pytest.raises(ServiceSampleError, match="시간대"):
        build_service_repurchase_samples(
            prepared, observation_end_at=pd.Timestamp("2026-03-12")
        )


def test_current_features_match_training_row_and_ignore_future() -> None:
    prepared = _prepared()
    cut = pd.Timestamp("2026-02-05T00:00:00Z")
    current = build_current_service_features(prepared, as_of_timestamp=cut)
    training = build_service_repurchase_samples(prepared, observation_end_at=END)
    selected = current.loc[current["pet_id"].eq("pet1")].iloc[0]
    expected = training.loc[training["order_id"].eq("o4")].iloc[0]

    assert selected["order_id"] == "o4"
    assert selected["elapsed_days"] == 5.0
    for column in (
        "history_interval_count",
        "history_median_days",
        "history_relative_mad",
        "user_prior_order_count",
    ):
        assert selected[column] == pytest.approx(expected[column])
    assert "duration_days" not in current
    assert "next_same_target_at" not in current

    # 기준 시각 뒤 주문을 추가해도 앞선 시점의 운영 피처는 그대로입니다.
    future = prepared.pet_purchase_events.iloc[[-1]].copy()
    future["order_id"] = "o6"
    future["paid_at"] = pd.Timestamp("2026-04-01T00:00:00Z")
    extended = OperationalPurchaseInputs(
        valid_items=prepared.valid_items,
        all_purchase_events=pd.concat(
            [prepared.all_purchase_events, future], ignore_index=True
        ),
        pet_history_items=prepared.pet_history_items,
        pet_purchase_events=pd.concat(
            [prepared.pet_purchase_events, future], ignore_index=True
        ),
        excluded_late_birth_item_count=0,
    )
    pd.testing.assert_frame_equal(
        build_current_service_features(extended, as_of_timestamp=cut), current
    )


def test_product_group_snapshot_change_starts_separate_history() -> None:
    prepared = _prepared()
    for frame in (prepared.all_purchase_events, prepared.pet_purchase_events):
        frame.loc[frame["order_id"].eq("o5"), "target_id"] = "g2"

    rows = build_service_repurchase_samples(prepared, observation_end_at=END)
    by_order = rows.set_index("order_id")

    assert by_order.loc["o5", "history_interval_count"] == 0
    assert bool(by_order.loc["o4", "is_right_censored"])
    assert by_order.loc["o5", "user_prior_order_count"] == 4


def test_zero_day_right_censor_is_not_false_negative() -> None:
    prepared = _prepared()
    cut = pd.Timestamp("2026-01-01T00:00:00Z")
    known_all = prepared.all_purchase_events.iloc[[0]].copy()
    known_pet = prepared.pet_purchase_events.iloc[[0]].copy()
    prepared = OperationalPurchaseInputs(
        valid_items=prepared.valid_items,
        all_purchase_events=known_all,
        pet_history_items=prepared.pet_history_items,
        pet_purchase_events=known_pet,
        excluded_late_birth_item_count=0,
    )

    row = build_service_repurchase_samples(prepared, observation_end_at=cut).iloc[0]

    assert row["duration_days"] == 0.0
    assert not bool(row["event_observed"])
    assert bool(row["is_right_censored"])
    assert not bool(row["has_zero_day_followup"])
