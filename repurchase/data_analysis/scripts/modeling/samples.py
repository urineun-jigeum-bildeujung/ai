"""재구매 라벨을 과거 이력 피처와 시간 분할이 포함된 표본으로 변환합니다.

현재 구매 이후에 발생한 다음 구매는 예측 정답일 뿐 입력 피처가 될 수 없습니다.
따라서 같은 사용자·상품의 앞선 구매 간격만 누적하고, 시간 구간 종료 전까지
정답이 확인된 표본을 별도로 표시해 미래 정보 누수를 방지합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd


class RepurchaseSampleBuildError(ValueError):
    """학습 표본 입력이나 시간 분할이 정의한 계약을 위반할 때 발생합니다."""


SAMPLE_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "user_id",
    "order_id",
    "product_id",
    "anchor_at",
    "next_same_product_at",
    "duration_days",
    "event_observed",
    "is_right_censored",
)

SPLIT_NAMES: Final[tuple[str, ...]] = ("train", "validation", "test")
# 정렬이나 필터링 후에도 하나의 사용자·주문·상품 표본을 다시 찾는 식별 열입니다.
SAMPLE_ID_COLUMNS: Final[tuple[str, ...]] = (
    "user_id",
    "order_id",
    "product_id",
)


def _median_absolute_deviation(values: np.ndarray) -> float:
    """여러 구매 간격이 중앙값에서 보통 얼마나 벗어나는지 계산합니다."""
    # expanding 배열의 앞부분에는 shift로 만든 NaN이 있으므로 계산에서 제외합니다.
    median = float(np.nanmedian(values))
    return float(np.nanmedian(np.abs(values - median)))


@dataclass(frozen=True)
class TemporalSplit:
    """시간 순서 분할의 시작·종료 경계를 변경 불가능한 값으로 보관합니다."""

    start_at: pd.Timestamp
    train_end_at: pd.Timestamp
    validation_end_at: pd.Timestamp
    end_at: pd.Timestamp
    train_fraction: float
    validation_fraction: float
    test_fraction: float


def _validate_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """과거 이력 피처 생성에 필요한 열과 기본 시간 관계를 검사합니다."""
    missing_columns = set(SAMPLE_REQUIRED_COLUMNS) - set(labels.columns)
    if missing_columns:
        raise RepurchaseSampleBuildError(
            f"학습 표본 필수 컬럼이 누락됐습니다: {sorted(missing_columns)}"
        )
    if labels.empty:
        raise RepurchaseSampleBuildError("학습 표본을 만들 재구매 라벨이 없습니다.")

    rows = labels.copy()
    rows["anchor_at"] = pd.to_datetime(rows["anchor_at"], errors="raise")
    rows["next_same_product_at"] = pd.to_datetime(
        rows["next_same_product_at"],
        errors="raise",
    )
    if rows["anchor_at"].isna().any():
        raise RepurchaseSampleBuildError("예측 기준 시각에 결측값이 있습니다.")
    if rows["duration_days"].lt(0).any():
        raise RepurchaseSampleBuildError("음수 재구매 기간은 사용할 수 없습니다.")
    if rows.duplicated(subset=list(SAMPLE_ID_COLUMNS)).any():
        raise RepurchaseSampleBuildError("중복된 사용자·주문·상품 라벨이 있습니다.")
    return rows


def build_historical_interval_features(labels: pd.DataFrame) -> pd.DataFrame:
    """각 구매 시점 이전에 확정된 동일 사용자·상품 구매 간격만 누적합니다.

    ``history_median_days``는 현재 행의 정답 ``duration_days``를 포함하지 않습니다.
    예를 들어 두 번째 구매 시점에는 첫 번째→두 번째 구매 간격만 알 수 있고,
    두 번째→세 번째 구매 간격은 아직 미래이므로 피처에서 제외합니다.
    """
    rows = _validate_labels(labels).sort_values(
        ["user_id", "product_id", "anchor_at", "order_id"],
        kind="stable",
        ignore_index=True,
    )
    pair_keys = [rows["user_id"], rows["product_id"]]

    # 관측된 현재 정답을 한 행 뒤부터 사용할 수 있도록 먼저 한 칸 이동합니다.
    known_duration = rows["duration_days"].where(rows["event_observed"])
    historical_duration = known_duration.groupby(
        pair_keys,
        observed=True,
        sort=False,
    ).shift(1)
    historical_sequence = historical_duration.groupby(
        pair_keys,
        observed=True,
        sort=False,
    )
    rows["history_median_days"] = historical_sequence.transform(
        lambda values: values.expanding(min_periods=1).median()
    )

    # 간격이 두 개 이상일 때만 중앙값 절대편차로 불규칙성을 계산합니다.
    rows["history_mad_days"] = historical_sequence.transform(
        lambda values: values.expanding(min_periods=2).apply(
            _median_absolute_deviation,
            raw=True,
        )
    )
    # 같은 MAD라도 대표 주기가 다른 상품을 비교할 수 있도록 비율도 남깁니다.
    positive_history_median = rows["history_median_days"].where(
        rows["history_median_days"].gt(0)
    )
    rows["history_relative_mad"] = rows["history_mad_days"].div(positive_history_median)

    # 현재 행을 제외한 과거 관측 간격의 개수를 함께 남겨 fallback 근거로 씁니다.
    observed_count = rows["event_observed"].astype("int64")
    cumulative_count = observed_count.groupby(
        pair_keys,
        observed=True,
        sort=False,
    ).cumsum()
    rows["history_interval_count"] = cumulative_count - observed_count
    rows["target_duration_days"] = rows["duration_days"].where(rows["event_observed"])
    return rows


def make_temporal_split(
    samples: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> TemporalSplit:
    """전체 관측 기간을 시간 길이 기준 Train·Validation·Test로 나눕니다."""
    if train_fraction <= 0 or validation_fraction <= 0:
        raise RepurchaseSampleBuildError("Train과 Validation 비율은 0보다 커야 합니다.")
    # 이진 부동소수점의 0.15000000000000005 같은 표현을 보고서에 남기지 않습니다.
    test_fraction = round(1.0 - train_fraction - validation_fraction, 12)
    if test_fraction <= 0:
        raise RepurchaseSampleBuildError("Test 비율은 0보다 커야 합니다.")
    if "anchor_at" not in samples.columns or samples.empty:
        raise RepurchaseSampleBuildError("시간 분할할 예측 기준 시각이 없습니다.")

    anchor_at = pd.to_datetime(samples["anchor_at"], errors="raise")
    start_at = pd.Timestamp(anchor_at.min())
    end_at = pd.Timestamp(anchor_at.max())
    if start_at >= end_at:
        raise RepurchaseSampleBuildError(
            "시간 분할에는 서로 다른 구매 시각이 필요합니다."
        )

    observation_span = end_at - start_at
    train_end_at = start_at + observation_span * train_fraction
    validation_end_at = train_end_at + observation_span * validation_fraction
    return TemporalSplit(
        start_at=start_at,
        train_end_at=train_end_at,
        validation_end_at=validation_end_at,
        end_at=end_at,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )


def assign_temporal_splits(
    samples: pd.DataFrame,
    split: TemporalSplit,
) -> pd.DataFrame:
    """각 표본에 시간 구간과 구간 종료 전 정답 확인 여부를 부여합니다."""
    rows = samples.copy()
    anchor_at = pd.to_datetime(rows["anchor_at"], errors="raise")
    next_purchase_at = pd.to_datetime(rows["next_same_product_at"], errors="raise")

    rows["split"] = "test"
    rows.loc[anchor_at.le(split.validation_end_at), "split"] = "validation"
    rows.loc[anchor_at.le(split.train_end_at), "split"] = "train"

    split_end_at = rows["split"].map(
        {
            "train": split.train_end_at,
            "validation": split.validation_end_at,
            "test": split.end_at,
        }
    )
    rows["split_end_at"] = pd.to_datetime(split_end_at, errors="raise")
    rows["outcome_available_by_split_end"] = (
        rows["event_observed"].astype(bool)
        & next_purchase_at.notna()
        & next_purchase_at.le(rows["split_end_at"])
    )
    return rows
