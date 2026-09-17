"""고정 기간 안의 재구매 확률을 학습하는 계층형 베이스라인을 제공합니다.

기간 중앙값 모델과 달리 확률 모델은 "30일 안에 재구매할 가능성"처럼
0부터 1 사이의 값을 생성합니다. 이 파일은 확률을 만드는 책임만 담당하고,
Brier Score와 Calibration 같은 평가는 evaluation.py에서 처리합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class HierarchicalEventProbabilityModel:
    """학습 구간에서 계산한 고정 기간 재구매 확률과 근거량을 보관합니다."""

    trained_until: pd.Timestamp
    horizon_days: int
    global_event_probability: float
    global_outcome_known_count: int
    global_ipcw_weight_sum: float


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
    )
