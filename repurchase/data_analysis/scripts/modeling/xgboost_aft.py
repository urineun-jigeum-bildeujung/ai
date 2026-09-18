"""XGBoost AFT 학습에 사용할 생존시간 라벨 계약을 정의합니다.

공용 재구매 라벨은 그대로 보존하고, AFT의 양수 시간 조건 때문에 제외되는
표본 수와 실제 학습 대상 행을 하나의 결과 객체로 관리합니다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import erfc, sqrt
from numbers import Integral, Real
from typing import Final

import numpy as np
import pandas as pd
import xgboost as xgb
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

from .features import MINIMAL_MODEL_FEATURE_COLUMNS, select_minimal_model_features


class XGBoostAFTError(ValueError):
    """AFT 학습 입력이나 결과가 정의한 계약을 위반할 때 발생합니다."""


AFT_LABEL_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "survival_observed_duration_days",
        "survival_event_observed",
    }
)

AFT_LOSS_DISTRIBUTIONS: Final[frozenset[str]] = frozenset(
    {
        "normal",
        "logistic",
        "extreme",
    }
)


def _validate_nonnegative_count(*, name: str, value: object) -> None:
    """표본 수가 boolean이 아닌 0 이상의 정수인지 검사합니다."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise XGBoostAFTError(f"{name}은 0 이상의 정수여야 합니다.")


def create_xgboost_aft_parameters(
    *,
    loss_distribution: str = "normal",
    loss_distribution_scale: float = 1.0,
) -> dict[str, object]:
    """CPU에서 재현 가능한 XGBoost AFT 기준 설정을 만듭니다."""
    if (
        not isinstance(loss_distribution, str)
        or loss_distribution not in AFT_LOSS_DISTRIBUTIONS
    ):
        raise XGBoostAFTError(
            "AFT 손실분포는 normal, logistic, extreme 중 하나여야 합니다."
        )
    if (
        isinstance(loss_distribution_scale, bool)
        or not isinstance(loss_distribution_scale, Real)
        or not np.isfinite(loss_distribution_scale)
        or loss_distribution_scale <= 0
    ):
        raise XGBoostAFTError("AFT 손실분포 scale은 0보다 큰 유한한 숫자여야 합니다.")

    return {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": loss_distribution,
        "aft_loss_distribution_scale": float(loss_distribution_scale),
        "tree_method": "hist",
        "device": "cpu",
        "seed": 42,
        "nthread": 1,
        "validate_parameters": True,
    }


@dataclass(frozen=True)
class AFTLabelBounds:
    """AFT 학습 라벨과 원본 대비 제외 통계를 함께 보관합니다."""

    rows: pd.DataFrame
    source_sample_count: int
    excluded_zero_duration_count: int

    def __post_init__(self) -> None:
        """객체 생성 직후 개별 표본 수와 합계 보존 관계를 검사합니다."""
        _validate_nonnegative_count(
            name="원본 표본 수",
            value=self.source_sample_count,
        )
        _validate_nonnegative_count(
            name="0일 제외 건수",
            value=self.excluded_zero_duration_count,
        )
        if self.excluded_zero_duration_count > self.source_sample_count:
            raise XGBoostAFTError("0일 제외 건수는 원본 표본 수보다 클 수 없습니다.")
        if len(self.rows) != self.included_sample_count:
            raise XGBoostAFTError(
                "AFT 라벨 행 수가 원본 표본 수에서 0일 제외 건수를 뺀 값과 "
                "일치하지 않습니다."
            )

    @property
    def included_sample_count(self) -> int:
        """원본 표본 수에서 0일 제외 건수를 뺀 실제 포함 건수를 계산합니다."""
        return self.source_sample_count - self.excluded_zero_duration_count


@dataclass(frozen=True)
class XGBoostAFTTrainingData:
    """동일한 행으로 정렬된 AFT 피처·구간 라벨과 제외 통계를 보관합니다."""

    matrix: xgb.DMatrix
    row_index: pd.Index
    feature_columns: tuple[str, ...]
    source_sample_count: int
    excluded_zero_duration_count: int

    @property
    def included_sample_count(self) -> int:
        """원본 표본 수에서 0일 제외 건수를 뺀 실제 학습 건수를 계산합니다."""
        return self.source_sample_count - self.excluded_zero_duration_count


@dataclass(frozen=True)
class XGBoostAFTTrainingResult:
    """학습된 AFT 모델과 재현·검증에 필요한 실행 정보를 함께 보관합니다."""

    booster: xgb.Booster
    feature_columns: tuple[str, ...]
    loss_distribution: str
    loss_distribution_scale: float
    num_boost_round: int
    training_aft_nloglik: tuple[float, ...]


@dataclass(frozen=True)
class XGBoostAFTPredictionData:
    """학습 피처 계약에 맞춰 만든 예측 행렬과 원본 행 인덱스를 보관합니다."""

    matrix: xgb.DMatrix
    row_index: pd.Index
    feature_columns: tuple[str, ...]


def _validate_xgboost_aft_training_data(
    training_data: XGBoostAFTTrainingData,
) -> None:
    """학습 직전 DMatrix의 행·피처·AFT 구간 라벨 계약을 다시 검사합니다."""
    matrix = training_data.matrix
    if matrix.num_row() != training_data.included_sample_count:
        raise XGBoostAFTError(
            "AFT 학습 행렬의 행 수가 실제 학습 포함 건수와 일치하지 않습니다."
        )
    if len(training_data.row_index) != matrix.num_row():
        raise XGBoostAFTError(
            "AFT 학습 행렬의 행 수가 예측 추적용 원본 인덱스 수와 일치하지 않습니다."
        )
    if matrix.feature_names != list(training_data.feature_columns):
        raise XGBoostAFTError(
            "AFT 학습 행렬의 피처 이름·순서가 기록된 피처 계약과 일치하지 않습니다."
        )

    lower_bound = matrix.get_float_info("label_lower_bound")
    upper_bound = matrix.get_float_info("label_upper_bound")
    if len(lower_bound) != matrix.num_row() or len(upper_bound) != matrix.num_row():
        raise XGBoostAFTError(
            "AFT 학습 행렬의 하한·상한 수가 학습 행 수와 일치하지 않습니다."
        )
    exact_event = np.isclose(lower_bound, upper_bound)
    right_censored = np.isinf(upper_bound) & (upper_bound > 0)
    if (
        not np.isfinite(lower_bound).all()
        or (lower_bound <= 0).any()
        or not (exact_event | right_censored).all()
    ):
        raise XGBoostAFTError(
            "AFT 학습 라벨은 양수의 정확 사건 [t, t] 또는 "
            "우측검열 [t, inf] 구간이어야 합니다."
        )


def build_aft_label_bounds(rows: pd.DataFrame) -> AFTLabelBounds:
    """공용 생존 관측값을 XGBoost AFT의 하한·상한 라벨로 변환합니다."""
    missing_columns = AFT_LABEL_REQUIRED_COLUMNS - set(rows.columns)
    if missing_columns:
        raise XGBoostAFTError(
            f"AFT 라벨 필수 컬럼이 누락됐습니다: {sorted(missing_columns)}"
        )
    if rows.empty:
        raise XGBoostAFTError("AFT 라벨을 생성할 표본이 없습니다.")

    duration = rows["survival_observed_duration_days"]
    if (
        duration.isna().any()
        or is_bool_dtype(duration.dtype)
        or is_complex_dtype(duration.dtype)
        or not is_numeric_dtype(duration.dtype)
        or not np.isfinite(duration.to_numpy(dtype="float64")).all()
        or duration.lt(0).any()
    ):
        raise XGBoostAFTError(
            "AFT 관측 기간에는 결측·무한대·음수가 없는 숫자만 사용할 수 있습니다."
        )

    event_observed = rows["survival_event_observed"]
    if event_observed.isna().any() or not is_bool_dtype(event_observed.dtype):
        raise XGBoostAFTError(
            "AFT 사건 관측 여부에는 결측값 없는 boolean만 사용할 수 있습니다."
        )

    zero_duration = duration.eq(0)
    result = rows.loc[~zero_duration].copy()
    if result.empty:
        raise XGBoostAFTError("0일 표본을 제외한 뒤 AFT에 사용할 표본이 없습니다.")

    result["aft_lower_bound"] = result["survival_observed_duration_days"].astype(
        "float64"
    )
    result["aft_upper_bound"] = result["aft_lower_bound"]
    censored = ~result["survival_event_observed"]
    result.loc[censored, "aft_upper_bound"] = float("inf")

    return AFTLabelBounds(
        rows=result,
        source_sample_count=len(rows),
        excluded_zero_duration_count=int(zero_duration.sum()),
    )


def build_xgboost_aft_training_data(
    rows: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = MINIMAL_MODEL_FEATURE_COLUMNS,
) -> XGBoostAFTTrainingData:
    """Train 행에서 0일을 제외하고 XGBoost AFT 학습 행렬을 만듭니다."""
    if "split" not in rows.columns:
        raise XGBoostAFTError("XGBoost AFT 학습 필수 컬럼이 누락됐습니다: ['split']")
    if rows["split"].isna().any() or not rows["split"].eq("train").all():
        raise XGBoostAFTError("XGBoost AFT 학습에는 Train 표본만 사용합니다.")

    # AFT에서 사용할 행을 먼저 확정한 뒤 같은 행에서 피처를 선택합니다.
    # 이 순서를 지켜야 0일 제외 후 피처와 구간 라벨의 사용자 대응이 어긋나지 않습니다.
    label_bounds = build_aft_label_bounds(rows)
    if not label_bounds.rows.index.is_unique:
        raise XGBoostAFTError("XGBoost AFT 학습 행의 원본 인덱스에 중복이 있습니다.")
    features = select_minimal_model_features(
        label_bounds.rows,
        feature_columns=feature_columns,
    )
    if not features.index.equals(label_bounds.rows.index):
        raise XGBoostAFTError(
            "XGBoost AFT 피처와 구간 라벨의 행 인덱스가 일치하지 않습니다."
        )

    matrix = xgb.DMatrix(
        features,
        feature_names=list(features.columns),
        missing=np.nan,
    )
    matrix.set_float_info(
        "label_lower_bound",
        label_bounds.rows["aft_lower_bound"].to_numpy(
            dtype="float64",
            copy=True,
        ),
    )
    matrix.set_float_info(
        "label_upper_bound",
        label_bounds.rows["aft_upper_bound"].to_numpy(
            dtype="float64",
            copy=True,
        ),
    )

    return XGBoostAFTTrainingData(
        matrix=matrix,
        row_index=features.index.copy(),
        feature_columns=tuple(features.columns),
        source_sample_count=label_bounds.source_sample_count,
        excluded_zero_duration_count=(label_bounds.excluded_zero_duration_count),
    )


def build_xgboost_aft_prediction_data(
    rows: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
) -> XGBoostAFTPredictionData:
    """학습 때 사용한 피처 이름·순서로 예측용 행렬을 만듭니다."""
    if rows.empty:
        raise XGBoostAFTError("XGBoost AFT로 예측할 표본이 없습니다.")
    if not rows.index.is_unique:
        raise XGBoostAFTError("XGBoost AFT 예측 행의 원본 인덱스에 중복이 있습니다.")

    features = select_minimal_model_features(
        rows,
        feature_columns=feature_columns,
    )
    matrix = xgb.DMatrix(
        features,
        feature_names=list(features.columns),
        missing=np.nan,
    )
    return XGBoostAFTPredictionData(
        matrix=matrix,
        row_index=features.index.copy(),
        feature_columns=tuple(features.columns),
    )


def train_xgboost_aft_model(
    training_data: XGBoostAFTTrainingData,
    *,
    loss_distribution: str = "normal",
    loss_distribution_scale: float = 1.0,
    num_boost_round: int = 5,
) -> XGBoostAFTTrainingResult:
    """검증된 AFT 행렬로 CPU 기준 모델을 학습하고 손실 이력을 반환합니다."""
    if (
        isinstance(num_boost_round, bool)
        or not isinstance(num_boost_round, Integral)
        or num_boost_round <= 0
    ):
        raise XGBoostAFTError("부스팅 반복 횟수는 0보다 큰 정수여야 합니다.")
    _validate_xgboost_aft_training_data(training_data)

    parameters = create_xgboost_aft_parameters(
        loss_distribution=loss_distribution,
        loss_distribution_scale=loss_distribution_scale,
    )
    evaluation_history: dict[str, dict[str, list[float]]] = {}
    booster = xgb.train(
        params=parameters,
        dtrain=training_data.matrix,
        num_boost_round=int(num_boost_round),
        evals=[(training_data.matrix, "train")],
        evals_result=evaluation_history,
        verbose_eval=False,
    )
    training_loss = tuple(
        float(value) for value in evaluation_history["train"]["aft-nloglik"]
    )
    if len(training_loss) != num_boost_round or not np.isfinite(training_loss).all():
        raise XGBoostAFTError(
            "AFT 학습 손실 이력이 반복 횟수와 일치하는 유한한 값이 아닙니다."
        )

    return XGBoostAFTTrainingResult(
        booster=booster,
        feature_columns=training_data.feature_columns,
        loss_distribution=loss_distribution,
        loss_distribution_scale=float(loss_distribution_scale),
        num_boost_round=int(num_boost_round),
        training_aft_nloglik=training_loss,
    )


def predict_xgboost_aft_duration(
    training_result: XGBoostAFTTrainingResult,
    prediction_data: XGBoostAFTPredictionData,
) -> pd.Series:
    """XGBoost AFT의 기본 시간 척도 예측값을 원본 행과 연결해 반환합니다."""
    if prediction_data.feature_columns != training_result.feature_columns:
        raise XGBoostAFTError(
            "AFT 예측 피처 이름·순서가 모델의 학습 피처 계약과 일치하지 않습니다."
        )
    if prediction_data.matrix.feature_names != list(training_result.feature_columns):
        raise XGBoostAFTError(
            "AFT 예측 행렬의 실제 피처 이름·순서가 학습 피처 계약과 일치하지 않습니다."
        )
    if len(prediction_data.row_index) != prediction_data.matrix.num_row():
        raise XGBoostAFTError(
            "AFT 예측 행렬의 행 수가 원본 인덱스 수와 일치하지 않습니다."
        )

    raw_predictions = np.asarray(
        training_result.booster.predict(prediction_data.matrix),
        dtype="float64",
    )
    if raw_predictions.shape != (prediction_data.matrix.num_row(),):
        raise XGBoostAFTError(
            "AFT 예측 결과는 입력 행 수와 같은 1차원 배열이어야 합니다."
        )
    if not np.isfinite(raw_predictions).all() or (raw_predictions <= 0).any():
        raise XGBoostAFTError(
            "AFT 기본 시간 척도 예측값은 0보다 큰 유한한 값이어야 합니다."
        )

    return pd.Series(
        raw_predictions,
        index=prediction_data.row_index.copy(),
        name="predicted_duration_days",
        dtype="float64",
    )


def build_xgboost_aft_evaluation_rows(
    rows: pd.DataFrame,
    predictions: pd.Series,
) -> AFTLabelBounds:
    """원본 행 인덱스로 AFT 예측과 생존 관측값을 정렬해 평가 행을 만듭니다."""
    if rows.empty:
        raise XGBoostAFTError("XGBoost AFT를 평가할 표본이 없습니다.")
    if not rows.index.is_unique or not predictions.index.is_unique:
        raise XGBoostAFTError("AFT 평가 원본과 예측의 행 인덱스는 중복될 수 없습니다.")
    if len(rows) != len(predictions) or not (
        rows.index.difference(predictions.index).empty
        and predictions.index.difference(rows.index).empty
    ):
        raise XGBoostAFTError(
            "AFT 평가 원본과 예측의 행 인덱스 집합이 일치하지 않습니다."
        )
    if (
        not is_numeric_dtype(predictions.dtype)
        or not np.isfinite(predictions.to_numpy(dtype="float64", copy=False)).all()
        or predictions.le(0).any()
    ):
        raise XGBoostAFTError("AFT 평가 예측값은 0보다 큰 유한한 숫자여야 합니다.")

    label_bounds = build_aft_label_bounds(rows)
    evaluation_rows = label_bounds.rows.copy()
    evaluation_rows["predicted_duration_days"] = predictions.reindex(
        evaluation_rows.index
    ).astype("float64")
    return AFTLabelBounds(
        rows=evaluation_rows,
        source_sample_count=label_bounds.source_sample_count,
        excluded_zero_duration_count=label_bounds.excluded_zero_duration_count,
    )


def _calculate_aft_distribution_cdf(
    standardized_time: np.ndarray,
    *,
    loss_distribution: str,
) -> np.ndarray:
    """AFT 표준화 시간에 선택한 잡음 분포의 누적확률을 계산합니다."""
    if loss_distribution == "normal":
        return np.fromiter(
            (0.5 * erfc(-float(value) / sqrt(2.0)) for value in standardized_time),
            dtype="float64",
            count=len(standardized_time),
        )
    if loss_distribution == "logistic":
        positive = standardized_time >= 0
        probability = np.empty_like(standardized_time, dtype="float64")
        probability[positive] = 1.0 / (1.0 + np.exp(-standardized_time[positive]))
        exp_value = np.exp(standardized_time[~positive])
        probability[~positive] = exp_value / (1.0 + exp_value)
        return probability
    if loss_distribution == "extreme":
        # z가 매우 크면 exp(z)는 inf가 되지만 CDF의 수학적 극한은 정확히 1입니다.
        with np.errstate(over="ignore"):
            return -np.expm1(-np.exp(standardized_time))
    raise XGBoostAFTError(
        "AFT 손실분포는 normal, logistic, extreme 중 하나여야 합니다."
    )


def calculate_xgboost_aft_event_probability(
    training_result: XGBoostAFTTrainingResult,
    predicted_duration_days: pd.Series,
    *,
    horizon_days: int,
) -> pd.Series:
    """AFT 기본 시간 척도 예측값을 고정 시점 재구매 누적확률로 변환합니다.

    predicted_duration_days에는 XGBoost 기본 예측인 exp(raw margin)을 사용합니다.
    이 값은 모든 손실분포에서 평균이나 중앙 재구매일을 뜻하지는 않습니다.
    """
    if (
        isinstance(horizon_days, bool)
        or not isinstance(horizon_days, Integral)
        or horizon_days <= 0
    ):
        raise XGBoostAFTError("AFT 확률 평가 시점은 0보다 큰 정수 일수여야 합니다.")
    if predicted_duration_days.empty:
        raise XGBoostAFTError("AFT 재구매 확률을 계산할 예측값이 없습니다.")
    if not predicted_duration_days.index.is_unique:
        raise XGBoostAFTError("AFT 재구매 확률 예측의 원본 인덱스가 중복됐습니다.")
    if (
        is_bool_dtype(predicted_duration_days.dtype)
        or is_complex_dtype(predicted_duration_days.dtype)
        or not is_numeric_dtype(predicted_duration_days.dtype)
        or not np.isfinite(
            predicted_duration_days.to_numpy(dtype="float64", copy=False)
        ).all()
        or predicted_duration_days.le(0).any()
    ):
        raise XGBoostAFTError(
            "AFT 기본 시간 척도 예측값은 0보다 큰 유한한 숫자여야 합니다."
        )

    # 수동 생성된 결과 객체도 확률 계산 전에 분포·scale 계약을 다시 확인합니다.
    create_xgboost_aft_parameters(
        loss_distribution=training_result.loss_distribution,
        loss_distribution_scale=training_result.loss_distribution_scale,
    )
    predicted_duration = predicted_duration_days.to_numpy(
        dtype="float64",
        copy=False,
    )
    standardized_time = (
        np.log(float(horizon_days)) - np.log(predicted_duration)
    ) / training_result.loss_distribution_scale
    probability = _calculate_aft_distribution_cdf(
        standardized_time,
        loss_distribution=training_result.loss_distribution,
    )
    if (
        not np.isfinite(probability).all()
        or (probability < 0).any()
        or (probability > 1).any()
    ):
        raise XGBoostAFTError(
            "AFT 재구매 확률은 0부터 1 사이의 유한한 값이어야 합니다."
        )

    return pd.Series(
        probability,
        index=predicted_duration_days.index.copy(),
        name="predicted_event_probability",
        dtype="float64",
    )
