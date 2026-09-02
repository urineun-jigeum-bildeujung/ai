"""과거 재구매 간격의 중앙값을 사용하는 계층형 베이스라인을 제공합니다.

복잡한 모델보다 먼저 전체·상품·사용자 상품 이력의 중앙값을 비교하면 이후
모델이 최소한 어떤 규칙보다 좋아야 하는지 명확해집니다. 중앙값은 극단적으로
긴 구매 간격의 영향을 평균보다 적게 받아 긴 꼬리 분포의 첫 기준에 적합합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from operator import index
from typing import Final

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


class MedianBaselineError(ValueError):
    """베이스라인 학습·예측 입력이 정의한 계약을 위반할 때 발생합니다."""


FIT_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "product_id",
    "next_same_product_at",
    "target_duration_days",
    "event_observed",
)

PREDICT_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "product_id",
    "history_median_days",
    "history_interval_count",
)


@dataclass(frozen=True)
class HierarchicalMedianModel:
    """학습 종료 시점까지 확정된 전체·상품별 구매 간격 통계를 보관합니다."""

    trained_until: pd.Timestamp
    global_median_days: float
    global_observation_count: int
    product_median_days: dict[object, float]
    product_observation_counts: dict[object, int]


def _require_columns(rows: pd.DataFrame, required: tuple[str, ...]) -> None:
    """학습이나 예측에 필요한 열이 빠졌다면 조용히 대체하지 않고 중단합니다."""
    missing_columns = set(required) - set(rows.columns)
    if missing_columns:
        raise MedianBaselineError(
            f"중앙값 베이스라인 필수 컬럼이 누락됐습니다: {sorted(missing_columns)}"
        )


def calculate_personal_history_weight(
    history_interval_count: int,
    shrinkage_strength: float,
) -> float:
    """이력 개수와 수축 강도로 개인 중앙값에 부여할 가중치를 계산합니다."""
    # bool은 정수처럼 계산되므로 먼저 차단해 이력 여부와 이력 개수를 구분합니다.
    if isinstance(history_interval_count, bool):
        raise MedianBaselineError("과거 간격 개수는 음수가 아닌 정수여야 합니다.")
    try:
        normalized_count = index(history_interval_count)
    except TypeError as error:
        raise MedianBaselineError(
            "과거 간격 개수는 음수가 아닌 정수여야 합니다."
        ) from error
    if normalized_count < 0:
        raise MedianBaselineError("과거 간격 개수는 음수가 아닌 정수여야 합니다.")

    normalized_strength = _normalize_shrinkage_strength(shrinkage_strength)

    return normalized_count / (normalized_count + normalized_strength)


def _normalize_shrinkage_strength(value: object) -> float:
    """수축 강도를 0보다 큰 유한한 실수로 정규화합니다."""
    if isinstance(value, bool):
        raise MedianBaselineError("수축 강도는 0보다 큰 유한한 숫자여야 합니다.")
    try:
        normalized_value = float(value)
    except (TypeError, ValueError) as error:
        raise MedianBaselineError(
            "수축 강도는 0보다 큰 유한한 숫자여야 합니다."
        ) from error
    if not isfinite(normalized_value) or normalized_value <= 0:
        raise MedianBaselineError("수축 강도는 0보다 큰 유한한 숫자여야 합니다.")
    return normalized_value


def _normalize_nonnegative_days(value: object, *, field_name: str) -> float:
    """기간 입력을 유한한 0 이상 실수로 정규화합니다."""
    if isinstance(value, (bool, str, bytes)):
        raise MedianBaselineError(f"{field_name}은 0 이상의 유한한 숫자여야 합니다.")
    try:
        normalized_value = float(value)
    except (TypeError, ValueError) as error:
        raise MedianBaselineError(
            f"{field_name}은 0 이상의 유한한 숫자여야 합니다."
        ) from error
    if not isfinite(normalized_value) or normalized_value < 0:
        raise MedianBaselineError(f"{field_name}은 0 이상의 유한한 숫자여야 합니다.")
    return normalized_value


def blend_personal_and_prior_medians(
    *,
    personal_median_days: float,
    prior_median_days: float,
    history_interval_count: int,
    shrinkage_strength: float,
) -> float:
    """이력 수에 따른 가중치로 개인 중앙값과 사전 중앙값을 결합합니다."""
    normalized_personal_median = _normalize_nonnegative_days(
        personal_median_days,
        field_name="개인 중앙값",
    )
    normalized_prior_median = _normalize_nonnegative_days(
        prior_median_days,
        field_name="사전 중앙값",
    )
    personal_weight = calculate_personal_history_weight(
        history_interval_count,
        shrinkage_strength,
    )
    prior_weight = 1.0 - personal_weight

    return (
        personal_weight * normalized_personal_median
        + prior_weight * normalized_prior_median
    )


def _validate_personal_history_features(
    predictions: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """개인 이력 개수와 중앙값의 형식·존재 관계를 검증합니다."""
    history_counts = predictions["history_interval_count"]
    if history_counts.isna().any():
        raise MedianBaselineError("과거 간격 개수에 결측값이 있습니다.")
    if is_bool_dtype(history_counts.dtype) or not is_numeric_dtype(
        history_counts.dtype
    ):
        raise MedianBaselineError("과거 간격 개수는 음수가 아닌 정수여야 합니다.")
    normalized_counts = history_counts.astype("float64")
    if not np.isfinite(normalized_counts.to_numpy(copy=False)).all():
        raise MedianBaselineError("과거 간격 개수는 유한한 정수여야 합니다.")
    if normalized_counts.lt(0).any() or normalized_counts.mod(1).ne(0).any():
        raise MedianBaselineError("과거 간격 개수는 음수가 아닌 정수여야 합니다.")

    personal_medians = predictions["history_median_days"]
    has_count = normalized_counts.gt(0)
    has_median = personal_medians.notna()
    if has_count.ne(has_median).any():
        raise MedianBaselineError(
            "과거 간격 개수와 개인 중앙값의 존재 여부가 일치하지 않습니다."
        )
    known_medians = personal_medians.loc[has_median]
    # 배치 전체가 콜드스타트라 중앙값이 하나도 없으면 prior fallback을 허용합니다.
    if not known_medians.empty:
        if is_bool_dtype(known_medians.dtype) or not is_numeric_dtype(
            known_medians.dtype
        ):
            raise MedianBaselineError("개인 중앙값은 0 이상의 숫자여야 합니다.")
        if not np.isfinite(known_medians.to_numpy(dtype="float64", copy=False)).all():
            raise MedianBaselineError("개인 중앙값은 유한한 숫자여야 합니다.")
        if known_medians.lt(0).any():
            raise MedianBaselineError("개인 중앙값은 음수일 수 없습니다.")

    return normalized_counts, has_count


def fit_hierarchical_median_baseline(
    samples: pd.DataFrame,
    *,
    trained_until: pd.Timestamp,
) -> HierarchicalMedianModel:
    """학습 종료 전에 정답이 확정된 관측 표본으로 중앙값 통계를 학습합니다."""
    _require_columns(samples, FIT_REQUIRED_COLUMNS)
    normalized_cutoff = pd.Timestamp(trained_until)
    next_purchase_at = pd.to_datetime(
        samples["next_same_product_at"],
        errors="raise",
    )
    eligible = (
        samples["event_observed"].astype(bool)
        & samples["target_duration_days"].notna()
        & next_purchase_at.notna()
        & next_purchase_at.le(normalized_cutoff)
    )
    training_rows = samples.loc[eligible].copy()
    if training_rows.empty:
        raise MedianBaselineError("학습 종료 전에 확정된 재구매 간격이 없습니다.")
    if training_rows["target_duration_days"].lt(0).any():
        raise MedianBaselineError("음수 구매 간격은 중앙값 학습에 사용할 수 없습니다.")

    product_summary = training_rows.groupby("product_id", observed=True)[
        "target_duration_days"
    ].agg(["median", "count"])
    return HierarchicalMedianModel(
        trained_until=normalized_cutoff,
        global_median_days=float(training_rows["target_duration_days"].median()),
        global_observation_count=int(len(training_rows)),
        product_median_days=product_summary["median"].astype(float).to_dict(),
        product_observation_counts=product_summary["count"].astype(int).to_dict(),
    )


def predict_hierarchical_median_baseline(
    model: HierarchicalMedianModel,
    samples: pd.DataFrame,
) -> pd.DataFrame:
    """개인 이력·상품 이력·전체 이력 순으로 사용할 수 있는 중앙값을 선택합니다."""
    _require_columns(samples, PREDICT_REQUIRED_COLUMNS)
    predictions = samples.copy()

    # 모든 상품이 처음 등장하더라도 예측할 수 있도록 전체 중앙값에서 시작합니다.
    predictions["predicted_duration_days"] = model.global_median_days
    predictions["prediction_source"] = "global_history"
    predictions["prediction_observation_count"] = model.global_observation_count

    product_median = predictions["product_id"].map(model.product_median_days)
    product_count = predictions["product_id"].map(model.product_observation_counts)
    has_product_history = product_median.notna()
    predictions.loc[has_product_history, "predicted_duration_days"] = product_median
    predictions.loc[has_product_history, "prediction_source"] = "product_history"
    predictions.loc[has_product_history, "prediction_observation_count"] = product_count

    # 개인 이력은 현재 anchor 이전 값만 모은 피처이므로 가장 우선해 사용합니다.
    has_user_product_history = predictions["history_median_days"].notna() & predictions[
        "history_interval_count"
    ].gt(0)
    predictions.loc[
        has_user_product_history,
        "predicted_duration_days",
    ] = predictions.loc[has_user_product_history, "history_median_days"]
    predictions.loc[has_user_product_history, "prediction_source"] = (
        "user_product_history"
    )
    predictions.loc[
        has_user_product_history,
        "prediction_observation_count",
    ] = predictions.loc[has_user_product_history, "history_interval_count"]

    if predictions["predicted_duration_days"].isna().any():
        raise MedianBaselineError("중앙값 fallback 이후에도 예측값이 비어 있습니다.")
    if predictions["predicted_duration_days"].lt(0).any():
        raise MedianBaselineError("중앙값 베이스라인이 음수 기간을 예측했습니다.")
    predictions["prediction_observation_count"] = predictions[
        "prediction_observation_count"
    ].astype("int64")
    return predictions


def predict_shrunk_hierarchical_median_baseline(
    model: HierarchicalMedianModel,
    samples: pd.DataFrame,
    *,
    shrinkage_strength: float,
) -> pd.DataFrame:
    """상품 prior와 개인 중앙값을 이력 수에 따라 혼합해 예측합니다."""
    _require_columns(samples, PREDICT_REQUIRED_COLUMNS)
    normalized_strength = _normalize_shrinkage_strength(shrinkage_strength)
    predictions = samples.copy()
    history_counts, has_user_product_history = _validate_personal_history_features(
        predictions
    )

    # 상품 통계가 없는 경우에도 예측할 수 있도록 전체 중앙값을 prior로 둡니다.
    predictions["prior_duration_days"] = model.global_median_days
    predictions["prior_source"] = "global_history"
    predictions["prior_observation_count"] = model.global_observation_count

    product_median = predictions["product_id"].map(model.product_median_days)
    product_count = predictions["product_id"].map(model.product_observation_counts)
    has_product_history = product_median.notna()
    predictions.loc[has_product_history, "prior_duration_days"] = product_median
    predictions.loc[has_product_history, "prior_source"] = "product_history"
    predictions.loc[has_product_history, "prior_observation_count"] = product_count

    # 개인 이력이 없으면 prior 예측을 그대로 사용합니다.
    predictions["predicted_duration_days"] = predictions["prior_duration_days"]
    predictions["personal_history_weight"] = 0.0
    predictions["shrinkage_strength"] = normalized_strength
    predictions["prediction_source"] = predictions["prior_source"]
    predictions["prediction_observation_count"] = predictions["prior_observation_count"]

    if has_user_product_history.any():
        # 행 반복문 대신 Series끼리 연산해 개인 이력 표본을 한 번에 계산합니다.
        personal_weights = history_counts.loc[has_user_product_history].div(
            history_counts.loc[has_user_product_history] + normalized_strength
        )
        prior_weights = 1.0 - personal_weights
        predictions.loc[
            has_user_product_history,
            "personal_history_weight",
        ] = personal_weights
        predictions.loc[
            has_user_product_history,
            "predicted_duration_days",
        ] = (
            personal_weights
            * predictions.loc[has_user_product_history, "history_median_days"]
            + prior_weights
            * predictions.loc[has_user_product_history, "prior_duration_days"]
        )
        predictions.loc[has_user_product_history, "prediction_source"] = (
            "shrunk_user_product_history"
        )
        predictions.loc[
            has_user_product_history,
            "prediction_observation_count",
        ] = history_counts.loc[has_user_product_history]

    if predictions["predicted_duration_days"].isna().any():
        raise MedianBaselineError("수축 중앙값 예측 이후에도 예측값이 비어 있습니다.")
    if predictions["predicted_duration_days"].lt(0).any():
        raise MedianBaselineError("수축 중앙값 모델이 음수 기간을 예측했습니다.")
    predictions["prior_observation_count"] = predictions[
        "prior_observation_count"
    ].astype("int64")
    predictions["prediction_observation_count"] = predictions[
        "prediction_observation_count"
    ].astype("int64")
    return predictions


def predict_global_median_baseline(
    model: HierarchicalMedianModel,
    samples: pd.DataFrame,
) -> pd.DataFrame:
    """모든 표본에 학습 구간 전체 중앙값만 적용해 비교 기준을 만듭니다."""
    predictions = samples.copy()
    predictions["predicted_duration_days"] = model.global_median_days
    predictions["prediction_source"] = "global_history"
    predictions["prediction_observation_count"] = model.global_observation_count
    return predictions
