"""고정 기간 재구매 확률 베이스라인의 기초 계산을 검증합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.maturity_analysis import add_split_ipcw_weights
from scripts.modeling.probability_baseline import (
    ProbabilityBaselineError,
    calculate_weighted_event_probability,
    fit_global_event_probability_baseline,
)


def test_calculate_weighted_event_probability_uses_ipcw_mass() -> None:
    """가중 사건 합을 전체 가중치 합으로 나눠 사건 확률을 계산합니다."""
    event_indicator = pd.Series([True, False, True, False], dtype="boolean")
    sample_weight = pd.Series([1.0, 2.0, 1.0, 1.0], dtype="float64")

    result = calculate_weighted_event_probability(
        event_indicator,
        sample_weight,
    )

    assert result == pytest.approx(0.4)


def test_calculate_weighted_event_probability_rejects_misaligned_rows() -> None:
    """pandas의 자동 인덱스 정렬로 서로 다른 행이 곱해지는 것을 차단합니다."""
    event_indicator = pd.Series(
        [True, False],
        index=[10, 20],
        dtype="boolean",
    )
    sample_weight = pd.Series(
        [1.0, 1.0],
        index=[20, 10],
        dtype="float64",
    )

    with pytest.raises(ProbabilityBaselineError, match="행 인덱스"):
        calculate_weighted_event_probability(
            event_indicator,
            sample_weight,
        )


def make_weighted_rows(*, split: str = "train") -> pd.DataFrame:
    """전체 확률 학습에 사용할 작은 고정 IPCW 표본을 만듭니다."""
    rows = pd.DataFrame(
        {
            "split": [split] * 4,
            "anchor_at": pd.to_datetime(
                ["2026-01-18", "2026-01-17", "2026-01-16", "2026-01-15"]
            ),
            "split_end_at": pd.to_datetime(["2026-01-20"] * 4),
            "outcome_available_by_split_end": [True, False, True, False],
            "target_duration_days": [2.0, float("nan"), 4.0, float("nan")],
        }
    )
    return add_split_ipcw_weights(rows, horizon_days=4)


def test_fit_global_event_probability_baseline_keeps_training_evidence() -> None:
    """Train의 가중 사건 확률과 표본 근거를 변경 불가능한 모델에 저장합니다."""
    model = fit_global_event_probability_baseline(make_weighted_rows())

    assert model.trained_until == pd.Timestamp("2026-01-20")
    assert model.horizon_days == 4
    assert model.global_event_probability == pytest.approx(0.625)
    assert model.global_outcome_known_count == 3
    assert model.global_ipcw_weight_sum == pytest.approx(4.0)


def test_fit_global_event_probability_baseline_rejects_validation_leakage() -> None:
    """Validation 정답이 확률 학습 통계에 들어오면 즉시 거절합니다."""
    with pytest.raises(ProbabilityBaselineError, match="Train만"):
        fit_global_event_probability_baseline(make_weighted_rows(split="validation"))
