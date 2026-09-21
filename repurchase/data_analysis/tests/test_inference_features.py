"""운영용 피처가 학습 때와 같은 과거 구매만 사용하는지 검증합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.inference_features import (
    InferenceFeatureError,
    build_current_features_from_valid_purchases,
)
from scripts.modeling.samples import build_historical_interval_features
from scripts.preprocessing.labels import build_same_product_repurchase_labels

AS_OF = pd.Timestamp("2026-03-12T00:00:00Z")


def make_valid_purchases() -> pd.DataFrame:
    """동일 상품의 3개 과거 간격과 다른 상품 주문을 함께 만듭니다."""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1", "u1", "u2"],
            "order_id": ["o1", "o2", "o3", "o4", "o5", "x1"],
            "product_id": ["p1", "p2", "p1", "p1", "p1", "p1"],
            "paid_at": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-06T00:00:00Z",
                    "2026-01-11T00:00:00Z",
                    "2026-01-31T00:00:00Z",
                    "2026-03-02T00:00:00Z",
                    "2026-01-05T00:00:00Z",
                ]
            ),
        }
    )


def test_current_features_use_known_intervals_and_all_prior_orders() -> None:
    """최신 구매의 다음 간격은 빼고 다른 상품 주문은 과거 주문 수에 셉니다."""
    result = build_current_features_from_valid_purchases(
        make_valid_purchases(), as_of_timestamp=AS_OF
    ).set_index(["user_id", "product_id"])

    assert len(result) == 3
    p1 = result.loc[("u1", "p1")]
    assert p1["order_id"] == "o5"
    assert p1["anchor_at"] == pd.Timestamp("2026-03-02T00:00:00Z")
    assert p1["as_of_timestamp"] == AS_OF
    assert p1["elapsed_days"] == 10.0
    assert p1["history_interval_count"] == 3
    assert p1["history_median_days"] == 20.0
    assert p1["history_relative_mad"] == 0.5
    assert p1["user_prior_order_count"] == 4
    assert result.loc[("u1", "p2"), "user_prior_order_count"] == 1
    assert result.loc[("u2", "p1"), "history_interval_count"] == 0
    assert pd.isna(result.loc[("u2", "p1"), "history_median_days"])


def test_future_purchase_cannot_change_current_features() -> None:
    """기준 시각 뒤의 구매·간격은 현재 예측 입력에 들어오지 않습니다."""
    events = make_valid_purchases()
    future = pd.DataFrame(
        {
            "user_id": ["u1"],
            "order_id": ["o6"],
            "product_id": ["p1"],
            "paid_at": [pd.Timestamp("2026-04-01T00:00:00Z")],
        }
    )
    with_future = pd.concat([events, future], ignore_index=True)

    expected = build_current_features_from_valid_purchases(
        events, as_of_timestamp=AS_OF
    )
    actual = build_current_features_from_valid_purchases(
        with_future, as_of_timestamp=AS_OF
    )

    pd.testing.assert_frame_equal(actual, expected)


def test_current_features_match_last_training_feature_row() -> None:
    """운영 마지막 행과 동일 시각 학습 행의 네 피처가 일치합니다."""
    events = make_valid_purchases()
    training_events = events.rename(columns={"paid_at": "ordered_at"})
    labels = build_same_product_repurchase_labels(training_events, AS_OF)
    historical = build_historical_interval_features(labels)
    training_last = historical.loc[historical["order_id"].eq("o5")].iloc[0]

    current = build_current_features_from_valid_purchases(events, as_of_timestamp=AS_OF)
    inference_last = current.loc[current["order_id"].eq("o5")].iloc[0]

    for column in (
        "history_interval_count",
        "history_median_days",
        "history_relative_mad",
        "user_prior_order_count",
    ):
        assert inference_last[column] == training_last[column]
    assert "duration_days" not in current.columns
    assert "event_observed" not in current.columns


def test_input_is_not_modified() -> None:
    """피처를 추가해도 전달받은 원본 구매 사건은 그대로 둡니다."""
    events = make_valid_purchases()
    original = events.copy(deep=True)

    build_current_features_from_valid_purchases(events, as_of_timestamp=AS_OF)

    pd.testing.assert_frame_equal(events, original)


def test_timezone_offsets_represent_same_instant() -> None:
    """서로 다른 오프셋은 UTC로 정규화한 뒤 과거 여부를 판단합니다."""
    events = pd.DataFrame(
        {
            "user_id": ["u1"],
            "order_id": ["o1"],
            "product_id": ["p1"],
            "paid_at": ["2026-03-12T09:00:00+09:00"],
        }
    )

    result = build_current_features_from_valid_purchases(events, as_of_timestamp=AS_OF)

    assert result.loc[0, "anchor_at"] == AS_OF
    assert result.loc[0, "elapsed_days"] == 0.0


def test_other_product_in_same_order_is_not_an_earlier_order() -> None:
    """한 주문에 여러 상품이 있어도 이전 주문 수는 중복되지 않습니다."""
    events = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "order_id": ["o1", "o1", "o2"],
            "product_id": ["p1", "p2", "p1"],
            "paid_at": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                    "2026-01-11T00:00:00Z",
                ]
            ),
        }
    )

    result = build_current_features_from_valid_purchases(
        events, as_of_timestamp=AS_OF
    ).set_index("product_id")

    assert result.loc["p1", "user_prior_order_count"] == 1
    assert result.loc["p1", "history_interval_count"] == 1
    assert result.loc["p2", "user_prior_order_count"] == 0


@pytest.mark.parametrize("extra_column", ["order_status", "pet_id"])
def test_raw_order_or_pet_fields_are_not_silently_ignored(
    extra_column: str,
) -> None:
    """주문 상태 판정과 반려동물별 구분이 끝나지 않은 입력은 거절합니다."""
    events = make_valid_purchases()
    events[extra_column] = "unknown"

    with pytest.raises(InferenceFeatureError, match="지원하지 않는 열"):
        build_current_features_from_valid_purchases(events, as_of_timestamp=AS_OF)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("user_id", None, "결측값"),
        ("order_id", " ", "빈 문자열"),
        ("paid_at", None, "시간대"),
        ("paid_at", "2026-01-01", "시간대"),
    ],
)
def test_missing_or_ambiguous_event_fields_fail(
    column: str, value: object, message: str
) -> None:
    """식별키 또는 결제 시각이 모호하면 추측하지 않고 거절합니다."""
    events = make_valid_purchases()
    if column == "paid_at":
        # 시간대형 열에 문자열을 바로 대입하면 pandas가 먼저 UTC로 변환할 수 있습니다.
        events["paid_at"] = events["paid_at"].astype("object")
    events.loc[0, column] = value

    with pytest.raises(InferenceFeatureError, match=message):
        build_current_features_from_valid_purchases(events, as_of_timestamp=AS_OF)


def test_duplicate_event_key_fails() -> None:
    """한 주문 상품이 두 번 전달되면 과거 구매 횟수를 부풀리지 않습니다."""
    events = make_valid_purchases()
    duplicated = pd.concat([events, events.iloc[[0]]], ignore_index=True)

    with pytest.raises(InferenceFeatureError, match="중복"):
        build_current_features_from_valid_purchases(duplicated, as_of_timestamp=AS_OF)


def test_same_product_same_instant_fails() -> None:
    """같은 시각의 별도 주문에 임의의 선후 관계를 만들지 않습니다."""
    events = make_valid_purchases()
    events.loc[2, "paid_at"] = events.loc[0, "paid_at"]

    with pytest.raises(InferenceFeatureError, match="동시 결제"):
        build_current_features_from_valid_purchases(events, as_of_timestamp=AS_OF)


def test_one_order_with_different_paid_at_fails() -> None:
    """같은 주문의 상품들이 서로 다른 결제 시각을 갖는 입력을 거절합니다."""
    events = make_valid_purchases()
    events.loc[1, "order_id"] = "o1"

    with pytest.raises(InferenceFeatureError, match="한 주문"):
        build_current_features_from_valid_purchases(events, as_of_timestamp=AS_OF)


def test_as_of_requires_timezone_and_observed_purchase() -> None:
    """기준 시각이 모호하거나 그 전에 구매가 없으면 추론하지 않습니다."""
    events = make_valid_purchases()
    with pytest.raises(InferenceFeatureError, match="시간대"):
        build_current_features_from_valid_purchases(
            events, as_of_timestamp=pd.Timestamp("2026-03-12")
        )
    with pytest.raises(InferenceFeatureError, match="확인된 유효 구매"):
        build_current_features_from_valid_purchases(
            events, as_of_timestamp=pd.Timestamp("2025-01-01T00:00:00Z")
        )
