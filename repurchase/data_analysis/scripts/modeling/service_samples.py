"""서비스 구매 사건으로 반려동물·상품군 재구매 라벨과 과거 피처를 만듭니다.

UCI의 SKU 기준 시퀀스를 재사용하지 않습니다. 사용자 전체 주문 이력과
반려동물별 반복 구매 이력을 분리한 뒤, 각 예측 기준 시각의 과거만 사용합니다.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final

import numpy as np
import pandas as pd

from .features import MINIMAL_MODEL_FEATURE_COLUMNS, select_minimal_model_features
from .operational_purchase_inputs import OperationalPurchaseInputs
from .samples import _median_absolute_deviation


class ServiceSampleError(ValueError):
    """서비스 라벨·피처의 시각 또는 사건 계약이 잘못됐을 때 발생합니다."""


SERVICE_FEATURE_GENERATION_VERSION: Final[int] = 1
TARGET_KEYS: Final[tuple[str, ...]] = ("user_id", "pet_id", "target_id")


def _utc(value: object, *, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ServiceSampleError(f"{name}을 시각으로 읽을 수 없습니다.") from error
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ServiceSampleError(f"{name}에는 시간대가 필요합니다.")
    return timestamp.tz_convert("UTC")


def _user_prior_order_counts(
    targets: pd.DataFrame, all_events: pd.DataFrame
) -> pd.Series:
    """동일 시각의 주문은 제외하고, 이전의 고유 유효 주문만 셉니다."""
    orders = all_events.loc[:, ["user_id", "order_id", "paid_at"]].drop_duplicates()
    if orders.duplicated(subset=["user_id", "order_id"]).any():
        raise ServiceSampleError("한 사용자·주문의 결제 시각이 서로 다릅니다.")
    result = pd.Series(0, index=targets.index, dtype="int64")
    for user_id, target_rows in targets.groupby("user_id", sort=False, observed=True):
        times = (
            orders.loc[orders["user_id"].eq(user_id), "paid_at"]
            .sort_values()
            .to_numpy(dtype="datetime64[ns]")
        )
        anchors = target_rows["anchor_at"].to_numpy(dtype="datetime64[ns]")
        result.loc[target_rows.index] = np.searchsorted(times, anchors, side="left")
    return result


def build_service_repurchase_samples(
    prepared: OperationalPurchaseInputs, *, observation_end_at: pd.Timestamp
) -> pd.DataFrame:
    """현재 구매 이후의 다음 동일 반려동물·상품군 구매를 라벨로 만듭니다.

    관측 종료 시각은 데이터 생성·추출 기준에서 외부 입력으로 받아 고정합니다.
    마지막 구매에 다음 구매가 없으면 미재구매로 단정하지 않고 우측검열합니다.
    """
    observation_end = _utc(observation_end_at, name="observation_end_at")
    all_events = prepared.all_purchase_events.copy()
    rows = prepared.pet_purchase_events.loc[
        prepared.pet_purchase_events["is_replenishable_snapshot"]
    ].copy()
    if rows.empty:
        raise ServiceSampleError("반복 소비 대상의 반려동물 구매 사건이 없습니다.")
    for name, frame in (("전체", all_events), ("반려동물", rows)):
        frame["paid_at"] = frame["paid_at"].map(
            lambda value, name=name: _utc(value, name=f"{name} 구매 시각")
        )
        if frame["paid_at"].gt(observation_end).any():
            raise ServiceSampleError("관측 종료 시각 이후의 구매 사건이 포함됐습니다.")
    if rows[list(TARGET_KEYS) + ["order_id"]].isna().any(axis=None):
        raise ServiceSampleError("재구매 대상 키에 결측값이 있습니다.")
    if rows.duplicated(subset=[*TARGET_KEYS, "order_id"]).any():
        raise ServiceSampleError("동일 주문·반려동물·상품군 사건이 중복됐습니다.")

    rows = rows.rename(columns={"paid_at": "anchor_at"}).sort_values(
        [*TARGET_KEYS, "anchor_at", "order_id"], kind="stable", ignore_index=True
    )
    sequence = rows.groupby(list(TARGET_KEYS), sort=False, observed=True)
    previous_at = sequence["anchor_at"].shift(1)
    rows["next_order_id"] = sequence["order_id"].shift(-1)
    rows["next_same_target_at"] = sequence["anchor_at"].shift(-1)
    if rows["anchor_at"].eq(previous_at).any():
        raise ServiceSampleError(
            "동일 대상의 동시 결제는 구매 순서를 확정할 수 없습니다."
        )

    rows["observation_end_at"] = observation_end
    rows["event_observed"] = rows["next_same_target_at"].notna()
    rows["is_right_censored"] = ~rows["event_observed"]
    end_at = rows["next_same_target_at"].fillna(observation_end)
    rows["duration_days"] = (end_at - rows["anchor_at"]).dt.total_seconds() / 86_400
    rows["has_zero_day_followup"] = rows["event_observed"] & rows["duration_days"].eq(0)

    # 현재 구매 시각에는 직전→현재 간격을 이미 알 수 있지만 다음 간격은 모릅니다.
    known_interval = (rows["anchor_at"] - previous_at).dt.total_seconds() / 86_400
    history = known_interval.groupby(
        [rows[key] for key in TARGET_KEYS], sort=False, observed=True
    )
    rows["history_interval_count"] = sequence.cumcount().astype("int64")
    rows["history_median_days"] = history.transform(
        lambda values: values.expanding(min_periods=1).median()
    )
    rows["history_mad_days"] = history.transform(
        lambda values: values.expanding(min_periods=2).apply(
            _median_absolute_deviation, raw=True
        )
    )
    rows["history_relative_mad"] = rows["history_mad_days"].div(
        rows["history_median_days"].where(rows["history_median_days"].gt(0))
    )
    rows["user_prior_order_count"] = _user_prior_order_counts(rows, all_events)
    rows["feature_generation_version"] = SERVICE_FEATURE_GENERATION_VERSION
    return rows


def build_current_service_features(
    prepared: OperationalPurchaseInputs, *, as_of_timestamp: pd.Timestamp
) -> pd.DataFrame:
    """기준 시각에 알려진 구매만으로 대상별 최신 운영 피처를 만듭니다.

    학습 라벨 생성과 같은 과거 간격 규칙을 호출하지만 정답 열은 반환하지 않습니다.
    """
    as_of = _utc(as_of_timestamp, name="as_of_timestamp")
    all_events = prepared.all_purchase_events.copy()
    pet_events = prepared.pet_purchase_events.copy()
    all_events["paid_at"] = all_events["paid_at"].map(
        lambda value: _utc(value, name="전체 구매 시각")
    )
    pet_events["paid_at"] = pet_events["paid_at"].map(
        lambda value: _utc(value, name="반려동물 구매 시각")
    )
    known = replace(
        prepared,
        all_purchase_events=all_events.loc[all_events["paid_at"].le(as_of)].copy(),
        pet_purchase_events=pet_events.loc[pet_events["paid_at"].le(as_of)].copy(),
    )
    samples = build_service_repurchase_samples(known, observation_end_at=as_of)
    latest = (
        samples.sort_values([*TARGET_KEYS, "anchor_at", "order_id"], kind="stable")
        .drop_duplicates(subset=list(TARGET_KEYS), keep="last")
        .reset_index(drop=True)
    )
    features = select_minimal_model_features(latest)
    result = latest.loc[:, [*TARGET_KEYS, "order_id", "anchor_at"]].copy()
    result["as_of_timestamp"] = as_of
    result["elapsed_days"] = (as_of - result["anchor_at"]).dt.total_seconds() / 86_400
    result["feature_generation_version"] = SERVICE_FEATURE_GENERATION_VERSION
    for column in MINIMAL_MODEL_FEATURE_COLUMNS:
        result[column] = features[column]
    return result
