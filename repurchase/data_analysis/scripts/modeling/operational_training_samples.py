"""관측 종료 라벨과 구매 당시 피처를 같은 학습 앵커에 결합합니다."""

from __future__ import annotations

import pandas as pd

from .features import (
    MINIMAL_MODEL_FEATURE_COLUMNS,
    TEMPORAL_SERVICE_FEATURE_GENERATION_VERSION,
    select_minimal_model_features,
)
from .operational_event_intervals import OperationalEventIntervals
from .operational_feature_recount import (
    recount_target_interval_features_as_of,
    recount_user_prior_orders_as_of,
)
from .operational_label_rebuild import rebuild_service_labels_from_event_intervals
from .operational_orders import OperationalOrderError

# 기존 import 경로를 유지하되 버전 정의는 공통 피처 계약에서 가져옵니다.


def _attach_recounted(
    labels: pd.DataFrame, recount: pd.DataFrame, columns: tuple[str, ...]
) -> pd.DataFrame:
    """재계산 결과가 라벨의 모든 행에 정확히 한 번씩 대응하는지 확인합니다."""
    expected = set(range(len(labels)))
    if (
        not recount.columns.is_unique
        or "sample_row" not in recount
        or set(columns) - set(recount.columns)
        or recount["sample_row"].duplicated().any()
        or set(recount["sample_row"]) != expected
    ):
        raise OperationalOrderError("재계산 피처의 표본 행 대응이 불완전합니다.")
    ordered = recount.set_index("sample_row").loc[range(len(labels))]
    result = labels.copy()
    for column in columns:
        result[column] = ordered[column].to_numpy()
    return result


def build_temporal_service_training_samples(
    event_intervals: OperationalEventIntervals,
    orders: pd.DataFrame,
    *,
    observation_end_at: pd.Timestamp,
) -> pd.DataFrame:
    """라벨은 종료 컷에서, 피처는 각 구매 앵커 당시의 유효 이력에서 만듭니다.

    최종 스냅샷으로 계산한 피처를 반환하지 않습니다. 원본 입력도 수정하지 않습니다.
    """
    labels = rebuild_service_labels_from_event_intervals(
        event_intervals, orders, observation_end_at=observation_end_at
    )
    target = recount_target_interval_features_as_of(
        labels, event_intervals.pet_targets, orders
    )
    user = recount_user_prior_orders_as_of(labels, event_intervals.user_orders, orders)
    result = _attach_recounted(
        labels,
        target,
        (
            "as_of_history_interval_count",
            "as_of_history_median_days",
            "as_of_history_mad_days",
            "as_of_history_relative_mad",
        ),
    )
    result = _attach_recounted(result, user, ("as_of_user_prior_order_count",))
    result = result.rename(
        columns={
            "as_of_history_interval_count": "history_interval_count",
            "as_of_history_median_days": "history_median_days",
            "as_of_history_mad_days": "history_mad_days",
            "as_of_history_relative_mad": "history_relative_mad",
            "as_of_user_prior_order_count": "user_prior_order_count",
        }
    )
    result.loc[:, list(MINIMAL_MODEL_FEATURE_COLUMNS)] = select_minimal_model_features(
        result
    )
    result["feature_generation_version"] = TEMPORAL_SERVICE_FEATURE_GENERATION_VERSION
    return result
