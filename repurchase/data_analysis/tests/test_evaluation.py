"""재구매 기간·고정 시점 이진 예측의 평가 계산을 검증합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.evaluation import (
    RepurchaseEvaluationError,
    _FenwickCountTree,
    calculate_ipcw_concordance_pair_weight,
    calculate_pair_concordance_credit,
    evaluate_ipcw_binary_predictions,
    evaluate_ipcw_brier_score,
    evaluate_ipcw_concordance_index,
    is_ipcw_concordance_pair_comparable,
)
from scripts.modeling.maturity_analysis import add_validation_ipcw_weights


def test_fenwick_count_tree_adds_and_queries_prefix_counts() -> None:
    """중복 순위를 포함해 요청한 경계 직전까지의 누적 개수를 반환합니다."""
    tree = _FenwickCountTree(size=5)
    for index in [0, 2, 2, 4]:
        tree.add(index)

    assert tree.prefix_count(0) == 0
    assert tree.prefix_count(1) == 1
    assert tree.prefix_count(3) == 3
    assert tree.prefix_count(5) == 4


@pytest.mark.parametrize(
    ("earlier_prediction_days", "later_prediction_days", "expected_credit"),
    [
        (10.0, 20.0, 1.0),
        (20.0, 20.0, 0.5),
        (30.0, 20.0, 0.0),
    ],
)
def test_calculate_pair_concordance_credit_scores_prediction_order(
    earlier_prediction_days: float,
    later_prediction_days: float,
    expected_credit: float,
) -> None:
    """빠른 사건을 더 이르게 예측하면 1점, 동률이면 0.5점을 부여합니다."""
    assert (
        calculate_pair_concordance_credit(
            earlier_prediction_days,
            later_prediction_days,
        )
        == expected_credit
    )


@pytest.mark.parametrize(
    ("censoring_survival_probability", "expected_weight"),
    [
        (1.0, 1.0),
        (0.5, 4.0),
        (0.25, 16.0),
    ],
)
def test_calculate_ipcw_concordance_pair_weight_uses_squared_inverse_probability(
    censoring_survival_probability: float,
    expected_weight: float,
) -> None:
    """관측 가능성이 낮은 사건일수록 비교 쌍에 더 큰 가중치를 부여합니다."""
    assert calculate_ipcw_concordance_pair_weight(
        censoring_survival_probability
    ) == pytest.approx(expected_weight)


@pytest.mark.parametrize(
    "invalid_probability",
    [float("nan"), float("inf"), 0.0, -0.1, 1.1],
)
def test_calculate_ipcw_concordance_pair_weight_rejects_invalid_probability(
    invalid_probability: float,
) -> None:
    """의미 없는 검열 생존확률을 자동 보정하지 않고 명확히 거절합니다."""
    with pytest.raises(RepurchaseEvaluationError, match="0보다 크고 1 이하"):
        calculate_ipcw_concordance_pair_weight(invalid_probability)


def test_evaluate_ipcw_concordance_index_counts_order_ties_and_weights() -> None:
    """한 사건보다 오래 관찰된 표본을 순서 일치·동률·불일치로 나눕니다."""
    rows = pd.DataFrame(
        {
            "predicted_duration_days": [20.0, 30.0, 20.0, 10.0],
            "survival_observed_duration_days": [2.0, 4.0, 5.0, 6.0],
            "survival_event_observed": [True, False, False, False],
            "ipcw_horizon_days": [3, 3, 3, 3],
            "ipcw_censoring_survival_probability": pd.Series(
                [0.5, pd.NA, pd.NA, pd.NA],
                dtype="Float64",
            ),
        }
    )

    result = evaluate_ipcw_concordance_index(rows)

    assert result == {
        "horizon_days": 3,
        "validation_sample_count": 4,
        "event_reference_count": 1,
        "contributing_event_count": 1,
        "comparable_pair_count": 3,
        "concordant_pair_count": 1,
        "tied_pair_count": 1,
        "discordant_pair_count": 1,
        "unweighted_concordance_index": pytest.approx(0.5),
        "ipcw_weighted_comparable_pair_mass": pytest.approx(12.0),
        "ipcw_weighted_concordance_credit": pytest.approx(6.0),
        "ipcw_concordance_index": pytest.approx(0.5),
    }


@pytest.mark.parametrize(
    (
        "event_duration_days",
        "event_observed",
        "comparison_duration_days",
        "expected_comparable",
    ),
    [
        (10.0, True, 20.0, True),
        (10.0, False, 20.0, False),
        (10.0, True, 10.0, False),
        (40.0, True, 50.0, False),
    ],
)
def test_is_ipcw_concordance_pair_comparable_requires_earlier_event(
    event_duration_days: float,
    event_observed: bool,
    comparison_duration_days: float,
    expected_comparable: bool,
) -> None:
    """기준 시점 안의 재구매가 비교 대상보다 먼저 확인된 쌍만 허용합니다."""
    assert (
        is_ipcw_concordance_pair_comparable(
            event_duration_days,
            event_observed,
            comparison_duration_days,
            horizon_days=30,
        )
        is expected_comparable
    )


def test_evaluate_ipcw_binary_predictions_separates_error_types() -> None:
    """정답 불명은 제외하고 가중 거짓양성·거짓음성 질량을 따로 계산합니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation"] * 4,
            "anchor_at": pd.to_datetime(
                ["2026-08-18", "2026-08-17", "2026-08-16", "2026-08-15"]
            ),
            "split_end_at": pd.to_datetime(["2026-08-20"] * 4),
            "outcome_available_by_split_end": [True, False, True, False],
            "target_duration_days": [2.0, float("nan"), 4.0, float("nan")],
            # 4일 기준으로 정답 확인 행에는 정답·거짓음성·거짓양성을 하나씩 만듭니다.
            "predicted_duration_days": [3.0, 2.0, 5.0, 3.0],
        }
    )
    weighted_rows = add_validation_ipcw_weights(rows, horizon_days=4)

    result = evaluate_ipcw_binary_predictions(weighted_rows)

    assert result == {
        "horizon_days": 4,
        "validation_sample_count": 4,
        "outcome_known_count": 3,
        "predicted_event_count": 2,
        "unweighted_binary_error_rate": pytest.approx(2 / 3),
        "ipcw_weight_sum": pytest.approx(4.0),
        "ipcw_weighted_true_positive_mass": pytest.approx(1.0),
        "ipcw_weighted_true_negative_mass": pytest.approx(0.0),
        "ipcw_weighted_error_mass": pytest.approx(3.0),
        "ipcw_weighted_binary_error_rate": pytest.approx(0.75),
        "ipcw_weighted_binary_accuracy": pytest.approx(0.25),
        "ipcw_weighted_false_positive_mass": pytest.approx(1.5),
        "ipcw_weighted_false_negative_mass": pytest.approx(1.5),
        "ipcw_weighted_precision": pytest.approx(0.4),
        "ipcw_weighted_recall": pytest.approx(0.4),
        "ipcw_weighted_specificity": pytest.approx(0.0),
        "ipcw_weighted_balanced_accuracy": pytest.approx(0.2),
        "ipcw_weighted_f1": pytest.approx(0.4),
        "always_no_event_error_rate": pytest.approx(0.625),
        "always_no_event_accuracy": pytest.approx(0.375),
        "accuracy_difference_vs_always_no_event": pytest.approx(-0.125),
    }


def test_ipcw_binary_evaluation_rejects_multiple_horizons() -> None:
    """서로 다른 기준 시점을 한 지표로 섞으면 명확한 오류를 냅니다."""
    rows = pd.DataFrame(
        {
            "predicted_duration_days": [3.0, 5.0],
            "ipcw_event_within_horizon": pd.Series([True, False], dtype="boolean"),
            "ipcw_horizon_days": [4, 30],
            "ipcw_outcome_known": [True, True],
            "ipcw_weight": [1.0, 1.0],
        }
    )

    with pytest.raises(RepurchaseEvaluationError, match="하나의 고정 시점"):
        evaluate_ipcw_binary_predictions(rows)


def test_evaluate_ipcw_brier_score_compares_probability_reference() -> None:
    """확률 제곱 오차를 IPCW로 보정하고 전체 확률 기준선과 비교합니다."""
    rows = pd.DataFrame(
        {
            "predicted_event_probability": [0.8, 0.2, 0.6, 0.3],
            "ipcw_event_within_horizon": pd.Series(
                [True, pd.NA, True, False],
                dtype="boolean",
            ),
            "ipcw_horizon_days": [4, 4, 4, 4],
            "ipcw_outcome_known": [True, False, True, True],
            "ipcw_weight": [1.0, 0.0, 1.5, 1.5],
        }
    )

    result = evaluate_ipcw_brier_score(
        rows,
        reference_probability=0.625,
    )

    assert result == {
        "horizon_days": 4,
        "validation_sample_count": 4,
        "outcome_known_count": 3,
        "ipcw_weight_sum": pytest.approx(4.0),
        "unweighted_brier_score": pytest.approx((0.04 + 0.16 + 0.09) / 3),
        "ipcw_brier_score": pytest.approx(0.10375),
        "reference_probability": pytest.approx(0.625),
        "ipcw_reference_brier_score": pytest.approx(0.234375),
        "brier_skill_score": pytest.approx(1 - 0.10375 / 0.234375),
    }


@pytest.mark.parametrize(
    "invalid_probability",
    [-0.1, 1.1, float("nan")],
)
def test_evaluate_ipcw_brier_score_rejects_invalid_probability(
    invalid_probability: float,
) -> None:
    """확률 범위를 벗어난 예측값은 잘못된 점수로 계산하지 않습니다."""
    rows = pd.DataFrame(
        {
            "predicted_event_probability": [invalid_probability],
            "ipcw_event_within_horizon": pd.Series([True], dtype="boolean"),
            "ipcw_horizon_days": [30],
            "ipcw_outcome_known": [True],
            "ipcw_weight": [1.0],
        }
    )

    with pytest.raises(RepurchaseEvaluationError, match="예측 사건 확률"):
        evaluate_ipcw_brier_score(rows, reference_probability=0.5)
