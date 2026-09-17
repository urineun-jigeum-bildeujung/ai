"""LightGBM 학습 입력이 시간 분할과 IPCW 계약을 지키는지 검증합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.lightgbm_baseline import (
    LightGBMBaselineError,
    build_lightgbm_training_data,
)


def make_lightgbm_rows(*, split: str = "train") -> pd.DataFrame:
    """정답 확인 두 행과 조기 검열 한 행을 가진 작은 학습 표본을 만듭니다."""
    return pd.DataFrame(
        {
            "history_interval_count": [0, 1, 2],
            "history_median_days": [float("nan"), 20.0, 25.0],
            "history_relative_mad": [float("nan"), float("nan"), 0.2],
            "user_prior_order_count": [0, 2, 5],
            "split": [split] * 3,
            "ipcw_horizon_days": [30] * 3,
            "ipcw_outcome_known": pd.array(
                [True, False, True],
                dtype="boolean",
            ),
            "ipcw_event_within_horizon": pd.array(
                [True, pd.NA, False],
                dtype="boolean",
            ),
            "ipcw_weight": [1.25, 0.0, 2.0],
        },
        index=[10, 20, 30],
    )


def test_build_lightgbm_training_data_keeps_only_known_outcomes() -> None:
    """30일 결과가 불명인 조기 검열 행을 정답으로 만들지 않습니다."""
    training_data = build_lightgbm_training_data(make_lightgbm_rows())

    assert training_data.horizon_days == 30
    assert training_data.features.index.tolist() == [10, 30]
    assert training_data.target.index.tolist() == [10, 30]
    assert training_data.sample_weight.index.tolist() == [10, 30]
    assert training_data.target.tolist() == [1, 0]
    assert training_data.sample_weight.tolist() == [1.25, 2.0]


def test_build_lightgbm_training_data_preserves_missing_history_features() -> None:
    """정답은 확인됐지만 과거 이력이 없는 행의 결측 피처는 그대로 유지합니다."""
    training_data = build_lightgbm_training_data(make_lightgbm_rows())

    assert pd.isna(training_data.features.loc[10, "history_median_days"])
    assert pd.isna(training_data.features.loc[10, "history_relative_mad"])


def test_build_lightgbm_training_data_rejects_validation_rows() -> None:
    """Validation 정답이 학습 입력으로 섞이면 즉시 거절합니다."""
    with pytest.raises(LightGBMBaselineError, match="Train"):
        build_lightgbm_training_data(make_lightgbm_rows(split="validation"))


def test_build_lightgbm_training_data_rejects_unknown_boolean_contract() -> None:
    """정답 확인 여부에 결측값이 있으면 행을 조용히 제거하지 않습니다."""
    rows = make_lightgbm_rows()
    rows.loc[20, "ipcw_outcome_known"] = pd.NA

    with pytest.raises(LightGBMBaselineError, match="정답 확인 여부"):
        build_lightgbm_training_data(rows)
