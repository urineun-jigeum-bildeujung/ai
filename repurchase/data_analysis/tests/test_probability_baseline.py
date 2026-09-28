"""고정 기간 재구매 확률 베이스라인의 기초 계산을 검증합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.maturity_analysis import add_split_ipcw_weights
from scripts.modeling.probability_baseline import (
    ProbabilityBaselineError,
    blend_event_probability_with_prior,
    calculate_weighted_event_probability,
    fit_global_event_probability_baseline,
    fit_hierarchical_event_probability_baseline,
    predict_hierarchical_event_probability_baseline,
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


def test_blend_event_probability_with_prior_shrinks_sparse_evidence() -> None:
    """근거 한 건의 100% 확률을 전체 확률 쪽으로 수축합니다."""
    result = blend_event_probability_with_prior(
        observed_probability=1.0,
        observed_weight_sum=1.0,
        prior_probability=0.4,
        smoothing_strength=4.0,
    )

    assert result == pytest.approx(0.52)


def test_blend_event_probability_with_prior_trusts_larger_evidence() -> None:
    """같은 관측 확률이라도 근거량이 많으면 관측값을 더 크게 반영합니다."""
    result = blend_event_probability_with_prior(
        observed_probability=1.0,
        observed_weight_sum=20.0,
        prior_probability=0.4,
        smoothing_strength=4.0,
    )

    assert result == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("field_name", "arguments"),
    [
        ("관측 사건 확률", {"observed_probability": 1.1}),
        ("관측 근거량", {"observed_weight_sum": 0.0}),
        ("상위 prior 확률", {"prior_probability": float("nan")}),
        ("확률 수축 강도", {"smoothing_strength": -1.0}),
    ],
)
def test_blend_event_probability_with_prior_rejects_invalid_inputs(
    field_name: str,
    arguments: dict[str, float],
) -> None:
    """범위를 벗어난 확률·근거량·강도를 조용히 보정하지 않습니다."""
    valid_arguments = {
        "observed_probability": 0.8,
        "observed_weight_sum": 3.0,
        "prior_probability": 0.4,
        "smoothing_strength": 4.0,
    }
    valid_arguments.update(arguments)

    with pytest.raises(ProbabilityBaselineError, match=field_name):
        blend_event_probability_with_prior(**valid_arguments)


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
            "product_id": ["p1", "p1", "p2", "p2"],
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


def test_fit_hierarchical_event_probability_baseline_smooths_products() -> None:
    """상품별 IPCW 확률을 전체 확률과 근거량에 따라 혼합해 저장합니다."""
    model = fit_hierarchical_event_probability_baseline(
        make_weighted_rows(),
        product_smoothing_strength=4.0,
    )

    assert model.global_event_probability == pytest.approx(0.625)
    assert model.product_event_probabilities == pytest.approx(
        {
            "p1": 0.7,
            "p2": 4 / 7,
        }
    )
    assert model.product_outcome_known_counts == {"p1": 1, "p2": 2}
    assert model.product_ipcw_weight_sums == pytest.approx({"p1": 1.0, "p2": 3.0})
    assert model.product_smoothing_strength == pytest.approx(4.0)


def test_predict_hierarchical_event_probability_uses_product_then_global() -> None:
    """학습 상품에는 상품 확률을, 새 상품에는 전체 확률을 적용합니다."""
    model = fit_hierarchical_event_probability_baseline(
        make_weighted_rows(),
        product_smoothing_strength=4.0,
    )
    samples = pd.DataFrame(
        {
            "product_id": ["p1", "new"],
            "sample_name": ["known_product", "cold_start_product"],
        }
    )

    result = predict_hierarchical_event_probability_baseline(model, samples)

    assert result["predicted_event_probability"].tolist() == pytest.approx([0.7, 0.625])
    assert result["probability_prediction_source"].tolist() == [
        "product_history",
        "global_history",
    ]
    assert result["probability_observation_count"].tolist() == [1, 3]
    assert result["probability_ipcw_weight_sum"].tolist() == pytest.approx([1.0, 4.0])
    assert "predicted_event_probability" not in samples.columns
