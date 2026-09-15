"""Validation 기반 수축 강도 후보 비교 로직을 검증합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.baseline import HierarchicalMedianModel
from scripts.modeling.model_selection import (
    evaluate_ipcw_shrinkage_candidates,
    evaluate_shrinkage_candidates,
)


def make_model() -> HierarchicalMedianModel:
    """상품 prior와 전체 fallback을 가진 작은 고정 모델을 만듭니다."""
    return HierarchicalMedianModel(
        trained_until=pd.Timestamp("2026-01-31"),
        global_median_days=20.0,
        global_observation_count=100,
        product_median_days={"p1": 30.0},
        product_observation_counts={"p1": 50},
    )


def make_validation_samples() -> pd.DataFrame:
    """수축이 강할수록 상품 prior에 가까워지는 Validation 표본을 만듭니다."""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "order_id": ["o1", "o2"],
            "product_id": ["p1", "new"],
            "history_median_days": [120.0, None],
            "history_interval_count": [1, 0],
            "target_duration_days": [30.0, 20.0],
        }
    )


def make_ipcw_validation_samples() -> pd.DataFrame:
    """동일한 검열 조건에서 기존 모델과 수축 후보를 비교할 표본을 만듭니다."""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "order_id": ["o1", "o2", "o3", "o4"],
            "product_id": ["p1", "new", "p1", "new"],
            "history_median_days": [12.0, None, 3.0, None],
            "history_interval_count": [1, 0, 2, 0],
            "split": ["validation"] * 4,
            "anchor_at": pd.to_datetime(
                ["2026-08-18", "2026-08-17", "2026-08-16", "2026-08-15"]
            ),
            "split_end_at": pd.to_datetime(["2026-08-20"] * 4),
            "outcome_available_by_split_end": [True, False, True, False],
            "target_duration_days": [2.0, float("nan"), 4.0, float("nan")],
        }
    )


def test_evaluate_shrinkage_candidates_preserves_candidates_and_metrics() -> None:
    """후보 순서를 보존하고 동일 표본의 성능·평균 개인 가중치를 반환합니다."""
    result = evaluate_shrinkage_candidates(
        make_validation_samples(),
        make_model(),
        shrinkage_strengths=(1.0, 4.0, 8.0),
    )

    assert result["shrinkage_strength"].tolist() == [1.0, 4.0, 8.0]
    assert result["sample_count"].tolist() == [2, 2, 2]
    assert result["personal_sample_count"].tolist() == [1, 1, 1]
    assert result["mean_personal_history_weight"].tolist() == pytest.approx(
        [0.5, 0.2, 1 / 9]
    )
    # 이 표본에서는 prior가 정답과 같아 수축을 강하게 할수록 MAE가 감소합니다.
    assert result["mae_days"].is_monotonic_decreasing
    # 개인화 표본이 한 건뿐이므로 상위 5% 선택 시 그 한 건이 전체 오차를 차지합니다.
    assert result["tail_sample_count"].tolist() == [1, 1, 1]
    assert result["tail_absolute_error_days"].tolist() == pytest.approx(
        [45.0, 18.0, 10.0]
    )
    assert result["tail_mae_days"].tolist() == pytest.approx([45.0, 18.0, 10.0])
    assert result["tail_absolute_error_share"].tolist() == [1.0, 1.0, 1.0]
    assert result["tail_late_prediction_rate"].tolist() == [1.0, 1.0, 1.0]
    assert result["fixed_cohort_reference_mae_days"].tolist() == [90.0] * 3
    assert result["fixed_cohort_candidate_mae_days"].tolist() == pytest.approx(
        [45.0, 18.0, 10.0]
    )
    assert result["fixed_cohort_mae_improvement_days"].tolist() == pytest.approx(
        [45.0, 72.0, 80.0]
    )
    assert result["fixed_cohort_improved_sample_rate"].tolist() == [1.0] * 3


def test_evaluate_ipcw_shrinkage_candidates_uses_same_population_and_reference() -> (
    None
):
    """기존 모델과 모든 수축 후보를 같은 표본·검열 가중치에서 비교합니다."""
    evaluation = evaluate_ipcw_shrinkage_candidates(
        make_ipcw_validation_samples(),
        make_model(),
        shrinkage_strengths=(1.0, 4.0),
        horizon_days=4,
    )
    result = evaluation.comparison

    assert result["model_candidate"].tolist() == [
        "hierarchical_median",
        "shrunk_hierarchical_median",
        "shrunk_hierarchical_median",
    ]
    assert result["shrinkage_strength"].isna().tolist() == [True, False, False]
    assert result["validation_sample_count"].tolist() == [4, 4, 4]
    assert result["outcome_known_count"].nunique() == 1
    assert result.iloc[0][
        "ipcw_weighted_balanced_accuracy_difference_vs_reference"
    ] == pytest.approx(0.0)
    assert result.iloc[0][
        "ipcw_concordance_index_difference_vs_reference"
    ] == pytest.approx(0.0)
    assert evaluation.reference_binary_evaluation["validation_sample_count"] == 4
    assert evaluation.reference_concordance_evaluation["validation_sample_count"] == 4


@pytest.mark.parametrize(
    "strengths",
    [(), (1.0, 1.0)],
)
def test_evaluate_shrinkage_candidates_rejects_unusable_candidate_set(
    strengths: tuple[float, ...],
) -> None:
    """비어 있거나 중복된 후보 집합으로 불필요한 실험을 실행하지 않습니다."""
    with pytest.raises(ValueError, match="후보|중복"):
        evaluate_shrinkage_candidates(
            make_validation_samples(),
            make_model(),
            shrinkage_strengths=strengths,
        )
