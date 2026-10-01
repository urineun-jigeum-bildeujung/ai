"""라벨 컷과 앵커 당시 피처가 같은 표본에 결합되는지 확인합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.operational_current_features import (
    build_temporal_service_current_features,
)
from scripts.modeling.operational_event_intervals import OperationalEventIntervals
from scripts.modeling.operational_orders import OperationalOrderError
from scripts.modeling.operational_training_samples import (
    TEMPORAL_SERVICE_FEATURE_GENERATION_VERSION,
    _attach_recounted,
    build_temporal_service_training_samples,
)
from scripts.modeling.service_samples import SERVICE_FEATURE_GENERATION_VERSION


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


def test_training_samples_keep_refunded_purchase_in_earlier_anchor_history() -> None:
    events, orders = _events()

    early = build_temporal_service_training_samples(
        events, orders, observation_end_at=pd.Timestamp("2026-01-15T00:00:00Z")
    ).set_index("order_id")
    late = build_temporal_service_training_samples(
        events, orders, observation_end_at=pd.Timestamp("2026-02-01T00:00:00Z")
    ).set_index("order_id")

    assert set(early.index) == {"o1", "o2"}
    assert early.loc["o2", "history_interval_count"] == 1
    assert early.loc["o2", "history_median_days"] == 10.0
    assert early.loc["o2", "user_prior_order_count"] == 1
    assert bool(early.loc["o2", "is_right_censored"])
    assert set(late.index) == {"o1", "o3"}
    assert late.loc["o3", "history_interval_count"] == 1
    assert late.loc["o3", "history_median_days"] == 30.0
    assert late.loc["o3", "user_prior_order_count"] == 1
    assert late.loc["o1", "next_order_id"] == "o3"
    assert (
        late["feature_generation_version"]
        .eq(TEMPORAL_SERVICE_FEATURE_GENERATION_VERSION)
        .all()
    )
    assert (
        TEMPORAL_SERVICE_FEATURE_GENERATION_VERSION
        != SERVICE_FEATURE_GENERATION_VERSION
    )


def test_current_features_share_temporal_training_version_and_cut() -> None:
    events, orders = _events()

    training_at_purchase = build_temporal_service_training_samples(
        events, orders, observation_end_at=pd.Timestamp("2026-01-11T00:00:00Z")
    ).set_index("order_id")
    serving_at_purchase = build_temporal_service_current_features(
        events, orders, as_of_timestamp=pd.Timestamp("2026-01-11T00:00:00Z")
    ).set_index("order_id")
    before_refund = build_temporal_service_current_features(
        events, orders, as_of_timestamp=pd.Timestamp("2026-01-15T00:00:00Z")
    )
    after_refund = build_temporal_service_current_features(
        events, orders, as_of_timestamp=pd.Timestamp("2026-02-01T00:00:00Z")
    )

    assert before_refund.loc[0, "order_id"] == "o2"
    assert before_refund.loc[0, "history_interval_count"] == 1
    assert before_refund.loc[0, "history_median_days"] == 10.0
    assert after_refund.loc[0, "order_id"] == "o3"
    assert after_refund.loc[0, "history_interval_count"] == 1
    assert after_refund.loc[0, "history_median_days"] == 30.0
    for column in (
        "history_interval_count",
        "history_median_days",
        "history_relative_mad",
        "user_prior_order_count",
    ):
        actual = serving_at_purchase.loc["o2", column]
        expected = training_at_purchase.loc["o2", column]
        assert (pd.isna(actual) and pd.isna(expected)) or actual == expected
    assert (
        before_refund["feature_generation_version"]
        .eq(TEMPORAL_SERVICE_FEATURE_GENERATION_VERSION)
        .all()
    )
    assert (
        after_refund["feature_generation_version"]
        .eq(TEMPORAL_SERVICE_FEATURE_GENERATION_VERSION)
        .all()
    )


@pytest.mark.parametrize("sample_rows", [[0, 0], [0], [0, 2]])
def test_recounted_features_require_exact_one_to_one_sample_rows(
    sample_rows: list[int],
) -> None:
    labels = pd.DataFrame({"order_id": ["o1", "o2"]})
    recount = pd.DataFrame(
        {"sample_row": sample_rows, "value": range(len(sample_rows))}
    )

    with pytest.raises(OperationalOrderError, match="표본 행 대응"):
        _attach_recounted(labels, recount, ("value",))


def test_recounted_features_follow_positions_not_original_index() -> None:
    labels = pd.DataFrame({"order_id": ["o2", "o1"]}, index=[20, 10])
    recount = pd.DataFrame({"sample_row": [1, 0], "value": [100, 200]})

    result = _attach_recounted(labels, recount, ("value",))

    assert result["order_id"].tolist() == ["o2", "o1"]
    assert result["value"].tolist() == [200, 100]
