"""시점 복원 서비스 학습 표본을 기존 AFT 입력 계약으로 변환합니다."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype

from .operational_orders import OperationalOrderError
from .operational_training_samples import TEMPORAL_SERVICE_FEATURE_GENERATION_VERSION


def build_service_aft_training_rows(train: pd.DataFrame) -> pd.DataFrame:
    """서비스 Train의 생존 기간·사건을 AFT 공용 열로 명시적으로 매핑합니다.

    0일 표본은 원본에 남기며, 기존 AFT 라벨 생성기가 학습 시 제외합니다.
    """
    required = {
        "duration_days",
        "event_observed",
        "is_right_censored",
        "feature_generation_version",
        "split",
    }
    missing = required - set(train.columns)
    if missing:
        raise OperationalOrderError(f"서비스 AFT 입력 필수 열 누락: {sorted(missing)}")
    if train.empty or not train.columns.is_unique or not train.index.is_unique:
        raise OperationalOrderError(
            "서비스 AFT Train 행 또는 열 인덱스가 유효하지 않습니다."
        )
    if (
        train["feature_generation_version"]
        .ne(TEMPORAL_SERVICE_FEATURE_GENERATION_VERSION)
        .any()
    ):
        raise OperationalOrderError("시점 복원 피처 버전이 일치하지 않습니다.")
    if train["split"].ne("train").any():
        raise OperationalOrderError(
            "서비스 AFT 학습에는 Train 표본만 사용할 수 있습니다."
        )
    event = train["event_observed"]
    censored = train["is_right_censored"]
    if (
        event.isna().any()
        or censored.isna().any()
        or not is_bool_dtype(event.dtype)
        or not is_bool_dtype(censored.dtype)
        or event.eq(censored).any()
    ):
        raise OperationalOrderError("사건 관측 여부와 우측검열 여부가 모순됩니다.")
    aliases = {"survival_observed_duration_days", "survival_event_observed"}
    if aliases & set(train.columns):
        raise OperationalOrderError("기존 AFT 라벨 열과 서비스 라벨이 중복됩니다.")

    rows = train.copy()
    rows["split"] = "train"
    rows["survival_observed_duration_days"] = rows["duration_days"]
    rows["survival_event_observed"] = rows["event_observed"]
    return rows
