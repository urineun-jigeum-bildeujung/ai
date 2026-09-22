"""현재 시점 조건부 확률의 의미와 저장 모델 연결을 검증합니다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from scripts.modeling.artifacts import (
    LoadedModelArtifact,
    ModelArtifactError,
    load_model_artifact,
    save_model_artifact,
)
from scripts.modeling.current_prediction import predict_current_repurchase_probability
from scripts.modeling.xgboost_aft import (
    XGBoostAFTError,
    XGBoostAFTTrainingResult,
    build_xgboost_aft_training_data,
    calculate_xgboost_aft_conditional_probability,
    train_xgboost_aft_model,
)


def _model_with_distribution(distribution: str) -> XGBoostAFTTrainingResult:
    """닫힌 형태의 정답을 비교할 수 있는 분포 설정을 만듭니다."""
    return XGBoostAFTTrainingResult(
        booster=xgb.Booster(),
        feature_columns=(),
        loss_distribution=distribution,
        loss_distribution_scale=1.0,
        num_boost_round=1,
        training_aft_nloglik=(1.0,),
    )


@pytest.mark.parametrize(
    ("distribution", "at_purchase", "ten_days_later"),
    [
        ("normal", 0.5, 2 * (0.7558914042144173 - 0.5)),
        ("logistic", 0.5, 1 / 3),
        ("extreme", 1 - np.exp(-1), 1 - np.exp(-1)),
    ],
)
def test_conditional_probability_respects_distribution_and_elapsed_time(
    distribution: str, at_purchase: float, ten_days_later: float
) -> None:
    """구매 직후의 누적확률과 10일 미구매 후의 조건부 확률을 구분합니다."""
    duration = pd.Series([10.0, 10.0], index=[7, 3])
    elapsed = pd.Series([0.0, 10.0], index=[7, 3])

    result = calculate_xgboost_aft_conditional_probability(
        _model_with_distribution(distribution), duration, elapsed, window_days=10
    )

    assert result.index.equals(duration.index)
    assert result.iloc[0] == pytest.approx(at_purchase, abs=1e-12)
    assert result.iloc[1] == pytest.approx(ten_days_later, abs=1e-10)


def test_conditional_probability_remains_stable_in_tiny_survival_tail() -> None:
    """생존확률이 일반 실수로 0처럼 보여도 로그 비율은 계산됩니다."""
    result = calculate_xgboost_aft_conditional_probability(
        _model_with_distribution("logistic"),
        pd.Series([10.0], index=[10]),
        pd.Series([1e200], index=[10]),
        window_days=10**200,
    )

    assert result.iloc[0] == pytest.approx(0.5, abs=1e-12)


@pytest.mark.parametrize(
    ("duration", "elapsed", "window", "message"),
    [
        ([10.0], [0.0], 0, "미래 예측 기간"),
        ([10.0], [-1.0], 10, "경과 시간"),
        ([0.0], [0.0], 10, "기본 시간 척도"),
        ([10.0], [float("nan")], 10, "경과 시간"),
        ([10.0], [1e300], 1, "구별 가능한"),
        ([10.0], [0.0], 10**400, "유한한 일수"),
    ],
)
def test_invalid_probability_inputs_are_rejected(
    duration: list[float], elapsed: list[float], window: int, message: str
) -> None:
    """의미 없는 기간·수치나 시간 정밀도 손실을 묵시적으로 통과시키지 않습니다."""
    with pytest.raises(XGBoostAFTError, match=message):
        calculate_xgboost_aft_conditional_probability(
            _model_with_distribution("normal"),
            pd.Series(duration),
            pd.Series(elapsed),
            window_days=window,
        )


def test_elapsed_and_duration_must_have_identical_index_order() -> None:
    """동일한 행 집합이어도 순서가 다르면 다른 사용자의 시간을 붙이지 않습니다."""
    with pytest.raises(XGBoostAFTError, match="인덱스·순서"):
        calculate_xgboost_aft_conditional_probability(
            _model_with_distribution("normal"),
            pd.Series([10.0, 20.0], index=[1, 2]),
            pd.Series([2.0, 1.0], index=[2, 1]),
            window_days=10,
        )


def _trained_model() -> XGBoostAFTTrainingResult:
    """저장 모델을 실제로 불러오는 통합 테스트용 작은 AFT 모델을 학습합니다."""
    rows = pd.DataFrame(
        {
            "history_interval_count": [0, 1, 2, 3],
            "history_median_days": [np.nan, 10.0, 20.0, 30.0],
            "history_relative_mad": [np.nan, np.nan, 0.0, 0.2],
            "user_prior_order_count": [0, 1, 2, 3],
            "survival_observed_duration_days": [10.0, 20.0, 30.0, 40.0],
            "survival_event_observed": [True, False, True, False],
            "split": "train",
        }
    )
    return train_xgboost_aft_model(build_xgboost_aft_training_data(rows))


def _purchase_events() -> pd.DataFrame:
    """미래 주문 제외·현재 경과 시간 확인에 사용할 두 구매를 만듭니다."""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "order_id": ["o1", "o2", "o3"],
            "product_id": ["p1", "p1", "p1"],
            "paid_at": [
                "2026-01-01T00:00:00Z",
                "2026-01-11T00:00:00Z",
                "2026-03-01T00:00:00Z",
            ],
        }
    )


def test_loaded_aft_artifact_predicts_current_window_without_future_leakage(
    tmp_path,
) -> None:
    """저장된 모델·현재 피처·조건부 확률을 연결하고 미래 주문을 제외합니다."""
    artifact = load_model_artifact(
        save_model_artifact(_trained_model(), tmp_path / "aft", horizon_days=30)
    )
    purchases = _purchase_events()
    original = purchases.copy(deep=True)
    result = predict_current_repurchase_probability(
        artifact,
        purchases,
        as_of_timestamp=pd.Timestamp("2026-01-21T00:00:00Z"),
        window_days=30,
    )

    assert len(result) == 1
    assert result.loc[0, "order_id"] == "o2"
    assert result.loc[0, "elapsed_days"] == 10
    assert result.loc[0, "window_days"] == 30
    assert result.loc[0, "artifact_id"] == artifact.artifact_id
    assert 0 <= result.loc[0, "conditional_repurchase_probability"] <= 1
    pd.testing.assert_frame_equal(purchases, original)


def test_fixed_horizon_lightgbm_is_not_a_conditional_survival_model() -> None:
    """고정 기간 분류 모델을 현재 시점 조건부 모델처럼 사용하지 않습니다."""
    artifact = LoadedModelArtifact(
        family="lightgbm",
        model=xgb.Booster(),  # 모델 유형 검사가 먼저 실패해야 합니다.
        feature_columns=("history_interval_count",),
        horizon_days=30,
        artifact_id="lightgbm-test",
    )
    with pytest.raises(ModelArtifactError, match="LightGBM"):
        predict_current_repurchase_probability(
            artifact,
            _purchase_events(),
            as_of_timestamp=pd.Timestamp("2026-01-21T00:00:00Z"),
            window_days=30,
        )
