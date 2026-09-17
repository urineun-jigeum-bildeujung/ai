"""LightGBM 고정 기간 재구매 분류 모델의 학습 입력 계약을 정의합니다.

고정 기간 전에 검열된 행은 재구매 여부를 확정할 수 없으므로 학습 정답으로
사용하지 않습니다. 정답을 확인할 수 있는 Train 행에서 피처·정답·IPCW
가중치를 동일한 인덱스로 묶어 이후 모델 학습 단계에 전달합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Final

import pandas as pd
from lightgbm import LGBMClassifier
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .features import MINIMAL_MODEL_FEATURE_COLUMNS, select_minimal_model_features


class LightGBMBaselineError(ValueError):
    """LightGBM 학습 입력이 정의한 계약을 위반할 때 발생합니다."""


def create_lightgbm_classifier() -> LGBMClassifier:
    """튜닝 전 비교 기준으로 사용할 결정적 이진분류 모델을 만듭니다."""
    return LGBMClassifier(
        objective="binary",
        random_state=42,
        n_jobs=1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


LIGHTGBM_TRAINING_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        *MINIMAL_MODEL_FEATURE_COLUMNS,
        "split",
        "ipcw_horizon_days",
        "ipcw_outcome_known",
        "ipcw_event_within_horizon",
        "ipcw_weight",
    }
)


@dataclass(frozen=True)
class LightGBMTrainingData:
    """동일한 행으로 정렬된 LightGBM 피처·정답·가중치를 보관합니다."""

    horizon_days: int
    features: pd.DataFrame
    target: pd.Series
    sample_weight: pd.Series


def _validate_training_rows(rows: pd.DataFrame) -> int:
    """학습 행의 필수 열·Train 범위·고정 평가 기간을 검사합니다."""
    missing_columns = LIGHTGBM_TRAINING_REQUIRED_COLUMNS - set(rows.columns)
    if missing_columns:
        raise LightGBMBaselineError(
            f"LightGBM 학습 필수 컬럼이 누락됐습니다: {sorted(missing_columns)}"
        )
    if rows.empty:
        raise LightGBMBaselineError("LightGBM을 학습할 표본이 없습니다.")
    if rows["split"].isna().any() or not rows["split"].eq("train").all():
        raise LightGBMBaselineError("LightGBM 학습에는 Train 표본만 사용합니다.")

    horizon_values = rows["ipcw_horizon_days"].drop_duplicates()
    if len(horizon_values) != 1:
        raise LightGBMBaselineError("LightGBM 학습 기간은 하나여야 합니다.")
    horizon_value = horizon_values.iloc[0]
    if isinstance(horizon_value, bool) or not isfinite(float(horizon_value)):
        raise LightGBMBaselineError("LightGBM 학습 기간은 양의 정수여야 합니다.")
    if float(horizon_value) <= 0 or not float(horizon_value).is_integer():
        raise LightGBMBaselineError("LightGBM 학습 기간은 양의 정수여야 합니다.")
    return int(horizon_value)


def build_lightgbm_training_data(rows: pd.DataFrame) -> LightGBMTrainingData:
    """정답을 확인할 수 있는 Train 행으로 LightGBM 학습 입력을 만듭니다."""
    horizon_days = _validate_training_rows(rows)

    outcome_known = rows["ipcw_outcome_known"]
    if outcome_known.isna().any() or not is_bool_dtype(outcome_known.dtype):
        raise LightGBMBaselineError(
            "정답 확인 여부에는 결측값 없는 boolean만 사용할 수 있습니다."
        )

    known_rows = rows.loc[outcome_known].copy()
    if known_rows.empty:
        raise LightGBMBaselineError("30일 정답을 확인할 수 있는 Train 표본이 없습니다.")

    target = known_rows["ipcw_event_within_horizon"]
    if target.isna().any() or not is_bool_dtype(target.dtype):
        raise LightGBMBaselineError(
            "기간 내 재구매 정답에는 결측값 없는 boolean만 사용할 수 있습니다."
        )

    sample_weight = known_rows["ipcw_weight"]
    if (
        sample_weight.isna().any()
        or not is_numeric_dtype(sample_weight.dtype)
        or sample_weight.le(0).any()
        or not sample_weight.map(lambda value: isfinite(float(value))).all()
    ):
        raise LightGBMBaselineError(
            "학습 표본의 IPCW 가중치는 0보다 큰 유한한 숫자여야 합니다."
        )

    features = select_minimal_model_features(known_rows)
    encoded_target = target.astype("int8").copy()
    normalized_weight = sample_weight.astype("float64").copy()
    if not (
        features.index.equals(encoded_target.index)
        and features.index.equals(normalized_weight.index)
    ):
        raise LightGBMBaselineError(
            "LightGBM 피처·정답·가중치의 행 인덱스가 일치하지 않습니다."
        )

    return LightGBMTrainingData(
        horizon_days=horizon_days,
        features=features,
        target=encoded_target,
        sample_weight=normalized_weight,
    )


def train_lightgbm_classifier(
    training_data: LightGBMTrainingData,
) -> LGBMClassifier:
    """검증된 피처·정답·IPCW 가중치로 기준 분류 모델을 학습합니다."""
    model = create_lightgbm_classifier()
    model.fit(
        training_data.features,
        training_data.target,
        sample_weight=training_data.sample_weight,
    )
    return model
