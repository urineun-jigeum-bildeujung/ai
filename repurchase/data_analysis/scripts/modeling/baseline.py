"""과거 재구매 간격의 중앙값을 사용하는 계층형 베이스라인을 제공합니다.

복잡한 모델보다 먼저 전체·상품·사용자 상품 이력의 중앙값을 비교하면 이후
모델이 최소한 어떤 규칙보다 좋아야 하는지 명확해집니다. 중앙값은 극단적으로
긴 구매 간격의 영향을 평균보다 적게 받아 긴 꼬리 분포의 첫 기준에 적합합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd


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
