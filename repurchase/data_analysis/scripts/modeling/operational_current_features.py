"""시점 복원 구매 사건으로 현재 서비스 예측 입력을 만듭니다."""

from __future__ import annotations

import pandas as pd

from .operational_event_intervals import OperationalEventIntervals
from .operational_label_rebuild import prepare_active_service_purchase_inputs
from .operational_training_samples import TEMPORAL_SERVICE_FEATURE_GENERATION_VERSION
from .service_samples import build_current_service_features


def build_temporal_service_current_features(
    event_intervals: OperationalEventIntervals,
    orders: pd.DataFrame,
    *,
    as_of_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """현재 유효 구매만 사용하고 시점 복원 학습 피처 버전을 명시합니다."""
    prepared = prepare_active_service_purchase_inputs(
        event_intervals, orders, as_of_timestamp=as_of_timestamp
    )
    features = build_current_service_features(prepared, as_of_timestamp=as_of_timestamp)
    features["feature_generation_version"] = TEMPORAL_SERVICE_FEATURE_GENERATION_VERSION
    return features
