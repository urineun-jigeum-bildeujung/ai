"""유효 구매 사건을 현재 시점 AFT 재구매 확률로 연결합니다.

모델 선택이나 사용자 알림 결정은 하지 않습니다. 마지막 유효 구매 이후에도
재구매가 관측되지 않았다는 조건에서 지정 기간 안의 구매 확률만 계산합니다.
"""

from __future__ import annotations

from numbers import Integral

import pandas as pd

from .artifacts import LoadedModelArtifact, ModelArtifactError
from .features import FEATURE_GENERATION_VERSION
from .inference_features import build_current_features_from_valid_purchases
from .xgboost_aft import (
    XGBoostAFTTrainingResult,
    build_xgboost_aft_prediction_data,
    calculate_xgboost_aft_conditional_probability,
    predict_xgboost_aft_duration,
)


def predict_current_repurchase_probability(
    artifact: LoadedModelArtifact,
    valid_purchases: pd.DataFrame,
    *,
    as_of_timestamp: pd.Timestamp,
    window_days: int,
) -> pd.DataFrame:
    """현재 유효 구매까지의 피처를 만들고 향후 기간 내 조건부 확률을 반환합니다."""
    if artifact.family != "xgboost_aft" or not isinstance(
        artifact.model, XGBoostAFTTrainingResult
    ):
        raise ModelArtifactError(
            "고정 기간 LightGBM 확률은 현재 시점 조건부 확률로 변환할 수 없습니다."
        )
    if artifact.feature_generation_version != FEATURE_GENERATION_VERSION:
        raise ModelArtifactError("모델의 피처 생성 규칙 버전과 현재 코드가 다릅니다.")
    if (
        isinstance(window_days, bool)
        or not isinstance(window_days, Integral)
        or window_days <= 0
    ):
        raise ModelArtifactError("미래 예측 기간은 0보다 큰 정수 일수여야 합니다.")

    # 학습과 같은 피처 생성 규칙으로 마지막 구매를 찾고, 모델의 피처 순서를 따릅니다.
    current = build_current_features_from_valid_purchases(
        valid_purchases, as_of_timestamp=as_of_timestamp
    )
    prediction_data = build_xgboost_aft_prediction_data(
        current, feature_columns=artifact.feature_columns
    )
    duration = predict_xgboost_aft_duration(artifact.model, prediction_data)
    probability = calculate_xgboost_aft_conditional_probability(
        artifact.model,
        duration,
        current["elapsed_days"],
        window_days=window_days,
    )
    result = current.loc[
        :,
        [
            "user_id",
            "product_id",
            "order_id",
            "anchor_at",
            "as_of_timestamp",
            "elapsed_days",
        ],
    ].copy()
    result["window_days"] = window_days
    result["conditional_repurchase_probability"] = probability.to_numpy(copy=True)
    result["artifact_id"] = artifact.artifact_id
    return result
