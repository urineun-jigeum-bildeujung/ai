"""고정 기간 안의 재구매 확률을 학습하는 계층형 베이스라인을 제공합니다.

기간 중앙값 모델과 달리 확률 모델은 "30일 안에 재구매할 가능성"처럼
0부터 1 사이의 값을 생성합니다. 이 파일은 확률을 만드는 책임만 담당하고,
Brier Score와 Calibration 같은 평가는 evaluation.py에서 처리합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Final

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


class ProbabilityBaselineError(ValueError):
    """확률 학습 입력이 정의한 데이터 계약을 위반할 때 발생합니다."""


PROBABILITY_FIT_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "split",
        "split_end_at",
        "ipcw_event_within_horizon",
        "ipcw_horizon_days",
        "ipcw_outcome_known",
        "ipcw_weight",
    }
)
HIERARCHICAL_PROBABILITY_FIT_REQUIRED_COLUMNS: Final[frozenset[str]] = (
    PROBABILITY_FIT_REQUIRED_COLUMNS | {"product_id"}
)
PROBABILITY_PREDICT_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset({"product_id"})


@dataclass(frozen=True)
class HierarchicalEventProbabilityModel:
    """학습 구간에서 계산한 고정 기간 재구매 확률과 근거량을 보관합니다."""

    trained_until: pd.Timestamp
    horizon_days: int
    global_event_probability: float
    global_outcome_known_count: int
    global_ipcw_weight_sum: float
    product_event_probabilities: dict[object, float]
    product_outcome_known_counts: dict[object, int]
    product_ipcw_weight_sums: dict[object, float]
    product_smoothing_strength: float | None


def _normalize_probability(value: object, *, field_name: str) -> float:
    """확률 입력을 0부터 1 사이의 유한한 실수로 정규화합니다."""
    if isinstance(value, (bool, str, bytes)):
        raise ProbabilityBaselineError(
            f"{field_name}은 0부터 1 사이의 유한한 숫자여야 합니다."
        )
    try:
        normalized_value = float(value)
    except (TypeError, ValueError) as error:
        raise ProbabilityBaselineError(
            f"{field_name}은 0부터 1 사이의 유한한 숫자여야 합니다."
        ) from error
    if not isfinite(normalized_value) or not 0 <= normalized_value <= 1:
        raise ProbabilityBaselineError(
            f"{field_name}은 0부터 1 사이의 유한한 숫자여야 합니다."
        )
    return normalized_value


def _normalize_positive_weight(value: object, *, field_name: str) -> float:
    """근거량과 수축 강도를 0보다 큰 유한한 실수로 정규화합니다."""
    if isinstance(value, (bool, str, bytes)):
        raise ProbabilityBaselineError(
            f"{field_name}은 0보다 큰 유한한 숫자여야 합니다."
        )
    try:
        normalized_value = float(value)
    except (TypeError, ValueError) as error:
        raise ProbabilityBaselineError(
            f"{field_name}은 0보다 큰 유한한 숫자여야 합니다."
        ) from error
    if not isfinite(normalized_value) or normalized_value <= 0:
        raise ProbabilityBaselineError(
            f"{field_name}은 0보다 큰 유한한 숫자여야 합니다."
        )
    return normalized_value


def blend_event_probability_with_prior(
    *,
    observed_probability: float,
    observed_weight_sum: float,
    prior_probability: float,
    smoothing_strength: float,
) -> float:
    """관측 근거량에 따라 하위 집단 확률과 상위 prior 확률을 혼합합니다."""
    normalized_observed_probability = _normalize_probability(
        observed_probability,
        field_name="관측 사건 확률",
    )
    normalized_observed_weight = _normalize_positive_weight(
        observed_weight_sum,
        field_name="관측 근거량",
    )
    normalized_prior_probability = _normalize_probability(
        prior_probability,
        field_name="상위 prior 확률",
    )
    normalized_strength = _normalize_positive_weight(
        smoothing_strength,
        field_name="확률 수축 강도",
    )

    observed_weight = normalized_observed_weight / (
        normalized_observed_weight + normalized_strength
    )
    prior_weight = 1.0 - observed_weight
    return (
        observed_weight * normalized_observed_probability
        + prior_weight * normalized_prior_probability
    )


def calculate_weighted_event_probability(
    event_indicator: pd.Series,
    sample_weight: pd.Series,
) -> float:
    """사건 여부에 표본 가중치를 적용해 전체 사건 확률을 계산합니다."""
    if len(event_indicator) != len(sample_weight):
        raise ProbabilityBaselineError("사건 여부와 표본 가중치의 개수가 다릅니다.")
    if event_indicator.empty:
        raise ProbabilityBaselineError("사건 확률을 계산할 표본이 없습니다.")
    if not event_indicator.index.equals(sample_weight.index):
        raise ProbabilityBaselineError(
            "사건 여부와 표본 가중치의 행 인덱스가 일치하지 않습니다."
        )
    if event_indicator.isna().any() or not is_bool_dtype(event_indicator.dtype):
        raise ProbabilityBaselineError(
            "사건 여부에는 결측값 없는 boolean만 사용할 수 있습니다."
        )
    if (
        not is_numeric_dtype(sample_weight.dtype)
        or not np.isfinite(sample_weight.to_numpy(dtype="float64", copy=False)).all()
        or sample_weight.le(0).any()
    ):
        raise ProbabilityBaselineError("표본 가중치는 0보다 큰 유한한 숫자여야 합니다.")

    weighted_event_sum = float(
        event_indicator.astype("float64").mul(sample_weight).sum()
    )
    total_weight = float(sample_weight.sum())
    return weighted_event_sum / total_weight


def fit_global_event_probability_baseline(
    weighted_training_rows: pd.DataFrame,
) -> HierarchicalEventProbabilityModel:
    """Train의 정답 확인 표본으로 전체 재구매 확률 기준선을 학습합니다."""
    missing_columns = PROBABILITY_FIT_REQUIRED_COLUMNS - set(
        weighted_training_rows.columns
    )
    if missing_columns:
        raise ProbabilityBaselineError(
            f"확률 베이스라인 학습 필수 컬럼이 누락됐습니다: {sorted(missing_columns)}"
        )
    if weighted_training_rows.empty:
        raise ProbabilityBaselineError("확률 베이스라인을 학습할 표본이 없습니다.")
    if not weighted_training_rows["split"].eq("train").all():
        raise ProbabilityBaselineError("확률 베이스라인 학습에는 Train만 사용합니다.")

    trained_until_values = pd.to_datetime(
        weighted_training_rows["split_end_at"],
        errors="raise",
    ).drop_duplicates()
    if len(trained_until_values) != 1:
        raise ProbabilityBaselineError("Train 종료 시점은 하나여야 합니다.")

    horizon_values = weighted_training_rows["ipcw_horizon_days"].drop_duplicates()
    if len(horizon_values) != 1:
        raise ProbabilityBaselineError("확률 학습의 고정 기간은 하나여야 합니다.")
    horizon_value = horizon_values.iloc[0]
    if (
        isinstance(horizon_value, (bool, np.bool_))
        or not isinstance(horizon_value, (int, np.integer))
        or int(horizon_value) <= 0
    ):
        raise ProbabilityBaselineError("확률 학습의 고정 기간은 양의 정수여야 합니다.")

    outcome_known = weighted_training_rows["ipcw_outcome_known"]
    if outcome_known.isna().any() or not is_bool_dtype(outcome_known.dtype):
        raise ProbabilityBaselineError(
            "정답 확인 여부에는 결측값 없는 boolean만 사용할 수 있습니다."
        )
    known_rows = weighted_training_rows.loc[outcome_known].copy()
    if known_rows.empty:
        raise ProbabilityBaselineError("확률 학습의 정답 확인 표본이 없습니다.")

    event_probability = calculate_weighted_event_probability(
        known_rows["ipcw_event_within_horizon"],
        known_rows["ipcw_weight"],
    )
    return HierarchicalEventProbabilityModel(
        trained_until=pd.Timestamp(trained_until_values.iloc[0]),
        horizon_days=int(horizon_value),
        global_event_probability=event_probability,
        global_outcome_known_count=int(len(known_rows)),
        global_ipcw_weight_sum=float(known_rows["ipcw_weight"].sum()),
        product_event_probabilities={},
        product_outcome_known_counts={},
        product_ipcw_weight_sums={},
        product_smoothing_strength=None,
    )


def fit_hierarchical_event_probability_baseline(
    weighted_training_rows: pd.DataFrame,
    *,
    product_smoothing_strength: float,
) -> HierarchicalEventProbabilityModel:
    """Train의 전체·상품 확률을 학습하고 희소 상품 확률을 수축합니다."""
    missing_columns = HIERARCHICAL_PROBABILITY_FIT_REQUIRED_COLUMNS - set(
        weighted_training_rows.columns
    )
    if missing_columns:
        raise ProbabilityBaselineError(
            f"계층형 확률 학습 필수 컬럼이 누락됐습니다: {sorted(missing_columns)}"
        )
    if weighted_training_rows["product_id"].isna().any():
        raise ProbabilityBaselineError(
            "확률 학습의 상품 ID에는 결측값을 사용할 수 없습니다."
        )
    normalized_strength = _normalize_positive_weight(
        product_smoothing_strength,
        field_name="상품 확률 수축 강도",
    )

    global_model = fit_global_event_probability_baseline(weighted_training_rows)
    known_rows = weighted_training_rows.loc[
        weighted_training_rows["ipcw_outcome_known"]
    ].copy()
    known_rows["weighted_event_mass"] = (
        known_rows["ipcw_event_within_horizon"]
        .astype("float64")
        .mul(known_rows["ipcw_weight"])
    )
    product_summary = known_rows.groupby(
        "product_id",
        observed=True,
        sort=False,
    ).agg(
        outcome_known_count=("product_id", "size"),
        ipcw_weight_sum=("ipcw_weight", "sum"),
        weighted_event_mass=("weighted_event_mass", "sum"),
    )
    product_summary["observed_event_probability"] = product_summary[
        "weighted_event_mass"
    ].div(product_summary["ipcw_weight_sum"])
    observed_weight = product_summary["ipcw_weight_sum"].div(
        product_summary["ipcw_weight_sum"] + normalized_strength
    )
    product_summary["smoothed_event_probability"] = (
        observed_weight * product_summary["observed_event_probability"]
        + (1.0 - observed_weight) * global_model.global_event_probability
    )

    return HierarchicalEventProbabilityModel(
        trained_until=global_model.trained_until,
        horizon_days=global_model.horizon_days,
        global_event_probability=global_model.global_event_probability,
        global_outcome_known_count=global_model.global_outcome_known_count,
        global_ipcw_weight_sum=global_model.global_ipcw_weight_sum,
        product_event_probabilities=product_summary[
            "smoothed_event_probability"
        ].to_dict(),
        product_outcome_known_counts=product_summary["outcome_known_count"]
        .astype("int64")
        .to_dict(),
        product_ipcw_weight_sums=product_summary["ipcw_weight_sum"]
        .astype("float64")
        .to_dict(),
        product_smoothing_strength=normalized_strength,
    )


def predict_hierarchical_event_probability_baseline(
    model: HierarchicalEventProbabilityModel,
    samples: pd.DataFrame,
) -> pd.DataFrame:
    """상품 확률을 적용하고 학습 이력이 없는 상품은 전체 확률로 대체합니다."""
    missing_columns = PROBABILITY_PREDICT_REQUIRED_COLUMNS - set(samples.columns)
    if missing_columns:
        raise ProbabilityBaselineError(
            f"확률 베이스라인 예측 필수 컬럼이 누락됐습니다: {sorted(missing_columns)}"
        )
    if samples.empty:
        raise ProbabilityBaselineError("재구매 확률을 예측할 표본이 없습니다.")
    if samples["product_id"].isna().any():
        raise ProbabilityBaselineError(
            "확률 예측의 상품 ID에는 결측값을 사용할 수 없습니다."
        )

    predictions = samples.copy()
    predictions["predicted_event_probability"] = model.global_event_probability
    predictions["probability_prediction_source"] = "global_history"
    predictions["probability_observation_count"] = model.global_outcome_known_count
    predictions["probability_ipcw_weight_sum"] = model.global_ipcw_weight_sum

    product_probability = predictions["product_id"].map(
        model.product_event_probabilities
    )
    product_count = predictions["product_id"].map(model.product_outcome_known_counts)
    product_weight_sum = predictions["product_id"].map(model.product_ipcw_weight_sums)
    has_product_history = product_probability.notna()
    predictions.loc[
        has_product_history,
        "predicted_event_probability",
    ] = product_probability
    predictions.loc[has_product_history, "probability_prediction_source"] = (
        "product_history"
    )
    predictions.loc[
        has_product_history,
        "probability_observation_count",
    ] = product_count
    predictions.loc[
        has_product_history,
        "probability_ipcw_weight_sum",
    ] = product_weight_sum

    probabilities = predictions["predicted_event_probability"]
    if probabilities.isna().any() or not probabilities.between(0, 1).all():
        raise ProbabilityBaselineError("예측된 재구매 확률은 0부터 1 사이여야 합니다.")
    predictions["probability_observation_count"] = predictions[
        "probability_observation_count"
    ].astype("int64")
    predictions["probability_ipcw_weight_sum"] = predictions[
        "probability_ipcw_weight_sum"
    ].astype("float64")
    return predictions
