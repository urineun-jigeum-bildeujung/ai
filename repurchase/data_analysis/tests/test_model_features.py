"""LightGBM과 XGBoost AFT가 공유할 최소 피처 계약을 검증합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.features import (
    MINIMAL_MODEL_FEATURE_COLUMNS,
    ModelFeatureError,
    select_minimal_model_features,
)


def make_model_rows() -> pd.DataFrame:
    """과거 피처와 사용 금지 정답 열이 함께 있는 작은 표본을 만듭니다."""
    return pd.DataFrame(
        {
            "history_interval_count": [0, 2],
            "history_median_days": [float("nan"), 20.0],
            "history_relative_mad": [float("nan"), 0.25],
            "user_prior_order_count": [0, 4],
            "duration_days": [10.0, 30.0],
            "event_observed": [True, False],
            "next_same_product_at": pd.to_datetime(["2026-01-11", None]),
        },
        index=[10, 20],
    )


def test_select_minimal_model_features_keeps_only_allowed_columns() -> None:
    """정답과 미래 시각은 버리고 허용한 과거 피처만 순서대로 남깁니다."""
    features = select_minimal_model_features(make_model_rows())

    assert tuple(features.columns) == MINIMAL_MODEL_FEATURE_COLUMNS
    assert features.index.tolist() == [10, 20]
    assert "duration_days" not in features.columns
    assert "event_observed" not in features.columns
    assert "next_same_product_at" not in features.columns


def test_select_minimal_model_features_does_not_modify_source() -> None:
    """반환된 입력표를 바꿔도 원본 학습 표본은 변경되지 않습니다."""
    rows = make_model_rows()
    features = select_minimal_model_features(rows)

    features.loc[10, "history_interval_count"] = 99

    assert rows.loc[10, "history_interval_count"] == 0


def test_select_minimal_model_features_preserves_missing_history() -> None:
    """이력 없음과 실제 0일 간격을 구분하도록 계산 불가 값을 결측으로 유지합니다."""
    features = select_minimal_model_features(make_model_rows())

    assert pd.isna(features.loc[10, "history_median_days"])
    assert pd.isna(features.loc[10, "history_relative_mad"])


def test_select_minimal_model_features_rejects_missing_column() -> None:
    """필수 피처가 빠지면 모델이 학습되기 전에 명확한 오류를 냅니다."""
    rows = make_model_rows().drop(columns="history_relative_mad")

    with pytest.raises(ModelFeatureError, match="history_relative_mad"):
        select_minimal_model_features(rows)


@pytest.mark.parametrize(
    ("column", "invalid_value", "message"),
    [
        ("history_interval_count", -1, "0 이상의 정수"),
        ("user_prior_order_count", 1.5, "0 이상의 정수"),
        ("history_median_days", float("inf"), "유한한 숫자"),
        ("history_relative_mad", "unknown", "숫자 또는 결측값"),
    ],
)
def test_select_minimal_model_features_rejects_invalid_values(
    column: str,
    invalid_value: object,
    message: str,
) -> None:
    """의미가 다른 값으로 자동 변환하지 않고 잘못된 피처를 즉시 거절합니다."""
    rows = make_model_rows()
    if isinstance(invalid_value, str):
        rows[column] = rows[column].astype("object")
    elif column == "user_prior_order_count":
        rows[column] = rows[column].astype("float64")
    rows.loc[10, column] = invalid_value

    with pytest.raises(ModelFeatureError, match=message):
        select_minimal_model_features(rows)
