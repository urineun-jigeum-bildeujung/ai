"""Validation 기반 수축 강도 후보 비교 로직을 검증합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.baseline import HierarchicalMedianModel
from scripts.modeling.model_selection import (
    evaluate_ipcw_probability_candidates,
    evaluate_ipcw_shrinkage_candidates,
    evaluate_lightgbm_probability_candidate,
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


def make_ipcw_probability_samples(split: str) -> pd.DataFrame:
    """Train 학습과 Validation 평가에 공통으로 사용할 작은 확률 표본을 만듭니다."""
    split_end = "2026-01-20" if split == "train" else "2026-02-20"
    anchor_month = "2026-01" if split == "train" else "2026-02"
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "order_id": [f"{split}-o1", f"{split}-o2", f"{split}-o3", f"{split}-o4"],
            "product_id": ["p1", "p2", "p1", "p2"],
            "split": [split] * 4,
            "anchor_at": pd.to_datetime(
                [
                    f"{anchor_month}-18",
                    f"{anchor_month}-17",
                    f"{anchor_month}-16",
                    f"{anchor_month}-15",
                ]
            ),
            "split_end_at": pd.to_datetime([split_end] * 4),
            "outcome_available_by_split_end": [True, False, True, False],
            "target_duration_days": [2.0, float("nan"), 4.0, float("nan")],
            "history_interval_count": [0, 1, 2, 3],
            "history_median_days": [float("nan"), 2.0, 3.0, 4.0],
            "history_relative_mad": [float("nan"), float("nan"), 0.2, 0.1],
            "user_prior_order_count": [0, 2, 3, 4],
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


def test_evaluate_ipcw_probability_candidates_uses_train_and_shared_validation() -> (
    None
):
    """Train 확률만 학습하고 모든 후보를 같은 Validation·기준선으로 비교합니다."""
    evaluation = evaluate_ipcw_probability_candidates(
        make_ipcw_probability_samples("train"),
        make_ipcw_probability_samples("validation"),
        product_smoothing_strengths=(1.0, 4.0),
        horizon_days=4,
        calibration_bin_count=2,
        bootstrap_product_smoothing_strength=4.0,
        bootstrap_replicates=100,
        bootstrap_random_seed=42,
    )
    result = evaluation.comparison

    assert result["model_candidate"].tolist() == [
        "global_event_probability",
        "hierarchical_event_probability",
        "hierarchical_event_probability",
    ]
    assert result["product_smoothing_strength"].isna().tolist() == [
        True,
        False,
        False,
    ]
    assert result["evaluation_sample_count"].tolist() == [4, 4, 4]
    assert result["outcome_known_count"].nunique() == 1
    assert result["horizon_days"].tolist() == [4, 4, 4]
    assert result.iloc[0]["product_prediction_rate"] == pytest.approx(0.0)
    assert result.iloc[1:]["product_prediction_rate"].tolist() == [1.0, 1.0]
    assert result.iloc[0]["brier_skill_score"] == pytest.approx(0.0)
    assert result["nonempty_calibration_bin_count"].between(1, 2).all()

    calibration = evaluation.calibration
    assert set(calibration["model_candidate"]) == {
        "global_event_probability",
        "hierarchical_event_probability",
    }
    assert calibration.groupby(
        ["model_candidate", "product_smoothing_strength"],
        dropna=False,
    )["sample_count"].sum().tolist() == [3, 3, 3]
    assert calibration.groupby(
        ["model_candidate", "product_smoothing_strength"],
        dropna=False,
    )["ipcw_weight_share"].sum().tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert evaluation.user_bootstrap is not None
    assert evaluation.user_bootstrap.summary["bootstrap_replicates"] == 100
    assert evaluation.user_bootstrap.summary["user_count"] == 3
    assert len(evaluation.user_bootstrap.trials) == 100


def test_evaluate_ipcw_probability_candidates_rejects_unknown_bootstrap_strength() -> (
    None
):
    """비교하지 않은 수축 강도의 Bootstrap을 요청하면 명확히 거절합니다."""
    with pytest.raises(ValueError, match="후보 집합"):
        evaluate_ipcw_probability_candidates(
            make_ipcw_probability_samples("train"),
            make_ipcw_probability_samples("validation"),
            product_smoothing_strengths=(1.0, 4.0),
            horizon_days=4,
            bootstrap_product_smoothing_strength=8.0,
        )


def test_evaluate_lightgbm_probability_candidate_uses_train_and_validation() -> None:
    """Train으로만 학습한 LightGBM을 동일한 Validation IPCW 기준으로 평가합니다."""
    evaluation = evaluate_lightgbm_probability_candidate(
        make_ipcw_probability_samples("train"),
        make_ipcw_probability_samples("validation"),
        horizon_days=4,
        calibration_bin_count=2,
    )
    result = evaluation.comparison.iloc[0]

    assert result["model_candidate"] == "lightgbm_probability"
    assert pd.isna(result["product_smoothing_strength"])
    assert result["evaluation_sample_count"] == 4
    assert result["outcome_known_count"] == 3
    assert result["horizon_days"] == 4
    assert 0 <= result["ipcw_brier_score"] <= 1
    assert 0 <= result["expected_calibration_error"] <= 1

    calibration = evaluation.calibration
    assert calibration["model_candidate"].eq("lightgbm_probability").all()
    assert calibration["sample_count"].sum() == 3
    assert calibration["ipcw_weight_share"].sum() == pytest.approx(1.0)
    assert evaluation.user_bootstrap is None


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
