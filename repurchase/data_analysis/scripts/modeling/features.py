"""재구매 모델이 공통으로 사용할 입력 피처의 최소 계약을 정의합니다.

LightGBM과 XGBoost AFT가 서로 다른 입력 정보를 사용하면 모델 구조의 차이를
공정하게 비교할 수 없습니다. 따라서 두 모델에 전달할 피처를 이 모듈에서
명시적으로 선택하고, 정답과 미래 정보가 입력에 섞이지 않도록 차단합니다.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype


class ModelFeatureError(ValueError):
    """모델 입력 피처가 정의한 계약을 위반할 때 발생합니다."""


# 첫 실험에서는 이미 검증한 과거 구매 이력 피처만 사용합니다.
# 피처를 추가할 때는 Validation 성능과 누수 여부를 각각 다시 검증합니다.
MINIMAL_MODEL_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "history_interval_count",
    "history_median_days",
    "history_relative_mad",
    "user_prior_order_count",
)

# 개수 피처는 과거 이력이 없더라도 0으로 계산할 수 있으므로 결측값을 허용하지 않습니다.
COUNT_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "history_interval_count",
    "user_prior_order_count",
)

# 과거 간격이 없으면 계산할 수 없는 연속형 피처는 결측값을 정보로 보존합니다.
OPTIONAL_CONTINUOUS_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "history_median_days",
    "history_relative_mad",
)


def _validate_count_features(features: pd.DataFrame) -> None:
    """개수 피처가 결측값 없는 0 이상의 정수인지 검사합니다."""
    for column in COUNT_FEATURE_COLUMNS:
        values = features[column]
        if (
            is_bool_dtype(values.dtype)
            or not is_integer_dtype(values.dtype)
            or values.isna().any()
            or values.lt(0).any()
        ):
            raise ModelFeatureError(f"{column}은 결측값 없는 0 이상의 정수여야 합니다.")


def _validate_optional_continuous_features(features: pd.DataFrame) -> None:
    """연속형 피처는 결측을 허용하되 관측값은 0 이상의 유한한 수로 제한합니다."""
    for column in OPTIONAL_CONTINUOUS_FEATURE_COLUMNS:
        values = features[column]
        if is_bool_dtype(values.dtype) or not is_numeric_dtype(values.dtype):
            raise ModelFeatureError(f"{column}은 숫자 또는 결측값이어야 합니다.")

        observed_values = values.dropna().to_numpy(dtype="float64", copy=False)
        if not np.isfinite(observed_values).all() or (observed_values < 0).any():
            raise ModelFeatureError(
                f"{column}의 관측값은 0 이상의 유한한 숫자여야 합니다."
            )


def select_minimal_model_features(rows: pd.DataFrame) -> pd.DataFrame:
    """허용된 과거 이력 열만 선택해 독립적인 모델 입력표를 반환합니다."""
    missing_columns = set(MINIMAL_MODEL_FEATURE_COLUMNS) - set(rows.columns)
    if missing_columns:
        raise ModelFeatureError(f"모델 피처가 누락됐습니다: {sorted(missing_columns)}")

    # 열 허용 목록을 사용하므로 정답·미래 시각 열이 원본에 있어도 선택되지 않습니다.
    features = rows.loc[:, list(MINIMAL_MODEL_FEATURE_COLUMNS)].copy()
    _validate_count_features(features)
    _validate_optional_continuous_features(features)
    return features
