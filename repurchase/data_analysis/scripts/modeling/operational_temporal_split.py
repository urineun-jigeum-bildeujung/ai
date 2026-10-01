"""서비스 표본을 평가 구간별 관측 종료 시각으로 독립 생성합니다."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .operational_event_intervals import OperationalEventIntervals
from .operational_orders import OperationalOrderError
from .operational_training_samples import build_temporal_service_training_samples


@dataclass(frozen=True)
class ServiceTemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame


def _attach_evaluation_contract(
    rows: pd.DataFrame, *, split_name: str, split_end_at: pd.Timestamp
) -> pd.DataFrame:
    """기존 생존·IPCW 평가기에 필요한 종료 컷 열을 서비스 라벨에서 만듭니다."""
    result = rows.copy()
    result["split"] = split_name
    result["split_end_at"] = split_end_at
    result["outcome_available_by_split_end"] = result["event_observed"]
    result["target_duration_days"] = result["duration_days"].where(
        result["event_observed"]
    )
    return result


def build_service_train_validation_split(
    event_intervals: OperationalEventIntervals,
    orders: pd.DataFrame,
    *,
    train_end_at: pd.Timestamp,
    validation_end_at: pd.Timestamp,
) -> ServiceTemporalSplit:
    """Train은 train 컷, Validation은 validation 컷에서만 정답을 확인합니다.

    Validation의 앵커는 train 컷 이후의 신규 구매만 포함합니다. 기존 대상의
    train 컷 시점 예측을 추적하는 평가는 아닙니다. 종료 시각과 같은 앵커는
    해당 구간에 포함되며, 0일 검열 표본의 모델별 처리 정책은 별도로 적용합니다.
    Test는 모델 선택 과정에서 반복 조회하지 않도록 만들지 않습니다.
    """
    train_end = pd.Timestamp(train_end_at)
    validation_end = pd.Timestamp(validation_end_at)
    if (
        train_end.tzinfo is None
        or validation_end.tzinfo is None
        or train_end >= validation_end
    ):
        raise OperationalOrderError(
            "Train/Validation 종료 시각은 시간대가 있고 엄격히 증가해야 합니다."
        )

    train = build_temporal_service_training_samples(
        event_intervals, orders, observation_end_at=train_end
    )
    validation_cut = build_temporal_service_training_samples(
        event_intervals, orders, observation_end_at=validation_end
    )
    validation = validation_cut.loc[validation_cut["anchor_at"].gt(train_end)].copy()
    if train.empty or validation.empty:
        raise OperationalOrderError("Train 또는 Validation 예측 표본이 비어 있습니다.")
    if (
        train["anchor_at"].gt(train_end).any()
        or validation["anchor_at"].gt(validation_end).any()
    ):
        raise OperationalOrderError("시간 분할 범위를 벗어난 앵커가 있습니다.")
    return ServiceTemporalSplit(
        train=_attach_evaluation_contract(
            train, split_name="train", split_end_at=train_end
        ),
        validation=_attach_evaluation_contract(
            validation, split_name="validation", split_end_at=validation_end
        ),
    )
