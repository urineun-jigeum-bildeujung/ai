"""과거 이력 피처와 시간 분할이 미래 정답을 사용하지 않는지 검증합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.samples import (
    RepurchaseSampleBuildError,
    TemporalSplit,
    assign_temporal_splits,
    build_historical_interval_features,
    make_temporal_split,
)


def make_labels() -> pd.DataFrame:
    """과거 간격 세 개와 마지막 우측검열을 가진 작은 라벨 표본을 만듭니다."""
    anchor_at = pd.to_datetime(["2026-01-01", "2026-01-11", "2026-01-31", "2026-03-02"])
    next_at = pd.to_datetime(["2026-01-11", "2026-01-31", "2026-03-02", None])
    return pd.DataFrame(
        {
            "user_id": ["u1"] * 4,
            "order_id": ["o1", "o2", "o3", "o4"],
            "product_id": ["p1"] * 4,
            "anchor_at": anchor_at,
            "next_same_product_at": next_at,
            "duration_days": [10.0, 20.0, 30.0, 10.0],
            "event_observed": [True, True, True, False],
            "is_right_censored": [False, False, False, True],
        }
    )


def test_history_features_use_only_intervals_known_before_anchor() -> None:
    """현재 행의 미래 정답은 제외하고 앞선 관측 간격만 중앙값에 포함합니다."""
    samples = build_historical_interval_features(make_labels())

    assert samples["history_interval_count"].tolist() == [0, 1, 2, 3]
    assert pd.isna(samples.loc[0, "history_median_days"])
    assert samples.loc[1, "history_median_days"] == 10.0
    assert samples.loc[2, "history_median_days"] == 15.0
    assert samples.loc[3, "history_median_days"] == 20.0
    assert pd.isna(samples.loc[0, "history_mad_days"])
    assert pd.isna(samples.loc[1, "history_mad_days"])
    assert samples.loc[2, "history_mad_days"] == 5.0
    assert samples.loc[3, "history_mad_days"] == 10.0
    assert pd.isna(samples.loc[0, "history_relative_mad"])
    assert pd.isna(samples.loc[1, "history_relative_mad"])
    assert samples.loc[2, "history_relative_mad"] == pytest.approx(1 / 3)
    assert samples.loc[3, "history_relative_mad"] == 0.5
    assert pd.isna(samples.loc[3, "target_duration_days"])


def test_temporal_split_preserves_order_and_outcome_maturity() -> None:
    """시간 구간 순서와 각 구간 종료 전 정답 확인 여부를 함께 기록합니다."""
    samples = build_historical_interval_features(make_labels())
    # 경계 시각은 앞 구간에 포함되므로 의도한 경계를 직접 명시해 검증합니다.
    split = TemporalSplit(
        start_at=pd.Timestamp("2026-01-01"),
        train_end_at=pd.Timestamp("2026-01-20"),
        validation_end_at=pd.Timestamp("2026-02-15"),
        end_at=pd.Timestamp("2026-03-02"),
        train_fraction=0.50,
        validation_fraction=0.25,
        test_fraction=0.25,
    )
    assigned = assign_temporal_splits(samples, split)

    assert assigned["split"].tolist() == ["train", "train", "validation", "test"]
    assert bool(assigned.loc[0, "outcome_available_by_split_end"])
    assert not bool(assigned.loc[1, "outcome_available_by_split_end"])
    assert not bool(assigned.loc[2, "outcome_available_by_split_end"])
    assert not bool(assigned.loc[3, "outcome_available_by_split_end"])
    assert split.start_at < split.train_end_at < split.validation_end_at < split.end_at


def test_temporal_split_fraction_boundaries_are_ordered() -> None:
    """비율로 계산한 시간 경계도 Train·Validation·Test 순서를 지킵니다."""
    split = make_temporal_split(
        build_historical_interval_features(make_labels()),
        train_fraction=0.50,
        validation_fraction=0.25,
    )

    assert split.start_at < split.train_end_at < split.validation_end_at < split.end_at


def test_temporal_split_rejects_nonpositive_test_fraction() -> None:
    """Train·Validation 비율의 합이 1 이상이면 Test 부재를 즉시 알립니다."""
    with pytest.raises(RepurchaseSampleBuildError, match="Test 비율"):
        make_temporal_split(
            build_historical_interval_features(make_labels()),
            train_fraction=0.80,
            validation_fraction=0.20,
        )


def test_history_features_reject_invalid_next_purchase_timestamp() -> None:
    """잘못된 날짜 문자열을 정답 미관측 결측값으로 조용히 바꾸지 않습니다."""
    labels = make_labels()
    labels["next_same_product_at"] = labels["next_same_product_at"].astype("object")
    labels.loc[0, "next_same_product_at"] = "invalid-date"

    with pytest.warns(UserWarning):
        with pytest.raises(ValueError):
            build_historical_interval_features(labels)
