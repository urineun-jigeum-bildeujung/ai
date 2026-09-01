"""계층형 중앙값 베이스라인의 학습 시점·fallback·평가 결과를 검증합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.baseline import (
    MedianBaselineError,
    fit_hierarchical_median_baseline,
    predict_hierarchical_median_baseline,
)
from scripts.modeling.evaluation import calculate_regression_metrics


def make_training_samples() -> pd.DataFrame:
    """학습 종료 전후 정답을 함께 넣어 시점 필터가 작동하는지 확인합니다."""
    return pd.DataFrame(
        {
            "product_id": ["p1", "p1", "p2", "p1"],
            "next_same_product_at": pd.to_datetime(
                ["2026-01-10", "2026-01-20", "2026-01-25", "2026-02-10"]
            ),
            "target_duration_days": [10.0, 20.0, 40.0, 1000.0],
            "event_observed": [True, True, True, True],
        }
    )


def test_fit_uses_only_outcomes_known_by_training_cutoff() -> None:
    """학습 종료 뒤 확정된 극단값은 전체·상품 중앙값에 들어가지 않습니다."""
    model = fit_hierarchical_median_baseline(
        make_training_samples(),
        trained_until=pd.Timestamp("2026-01-31"),
    )

    assert model.global_observation_count == 3
    assert model.global_median_days == 20.0
    assert model.product_median_days == {"p1": 15.0, "p2": 40.0}
    assert model.product_observation_counts == {"p1": 2, "p2": 1}


def test_prediction_uses_user_product_then_product_then_global_fallback() -> None:
    """가장 개인화된 과거 이력이 있을 때 우선하고 없으면 단계적으로 대체합니다."""
    model = fit_hierarchical_median_baseline(
        make_training_samples(),
        trained_until=pd.Timestamp("2026-01-31"),
    )
    prediction_rows = pd.DataFrame(
        {
            "product_id": ["p1", "p2", "new"],
            "history_median_days": [12.0, None, None],
            "history_interval_count": [2, 0, 0],
        }
    )

    predictions = predict_hierarchical_median_baseline(model, prediction_rows)

    assert predictions["predicted_duration_days"].tolist() == [12.0, 40.0, 20.0]
    assert predictions["prediction_source"].tolist() == [
        "user_product_history",
        "product_history",
        "global_history",
    ]
    assert predictions["prediction_observation_count"].tolist() == [2, 1, 3]


def test_metrics_calculate_absolute_error_and_hit_rates() -> None:
    """일 단위 절대오차와 ±3일·±7일 적중률을 고정된 값으로 확인합니다."""
    rows = pd.DataFrame(
        {
            "target_duration_days": [10.0, 20.0, 30.0],
            "predicted_duration_days": [9.0, 25.0, 40.0],
            "prediction_source": ["a", "a", "b"],
        }
    )

    metrics = calculate_regression_metrics(rows)

    assert metrics["sample_count"] == 3
    assert metrics["mae_days"] == pytest.approx(16 / 3)
    assert metrics["median_absolute_error_days"] == 5.0
    assert metrics["within_3_days_rate"] == pytest.approx(1 / 3)
    assert metrics["within_7_days_rate"] == pytest.approx(2 / 3)


def test_fit_rejects_training_cutoff_without_matured_outcomes() -> None:
    """학습 시점에 정답이 하나도 없다면 임의 기본값 대신 명시적으로 실패합니다."""
    with pytest.raises(MedianBaselineError, match="확정된 재구매 간격"):
        fit_hierarchical_median_baseline(
            make_training_samples(),
            trained_until=pd.Timestamp("2025-12-31"),
        )
