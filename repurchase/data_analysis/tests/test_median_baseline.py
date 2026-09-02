"""계층형 중앙값 베이스라인의 학습 시점·fallback·평가 결과를 검증합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.baseline import (
    MedianBaselineError,
    blend_personal_and_prior_medians,
    calculate_personal_history_weight,
    fit_hierarchical_median_baseline,
    predict_hierarchical_median_baseline,
    predict_shrunk_hierarchical_median_baseline,
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


@pytest.mark.parametrize(
    ("history_interval_count", "expected_weight"),
    [
        (0, 0.0),
        (1, 0.2),
        (4, 0.5),
        (20, 5 / 6),
    ],
)
def test_personal_history_weight_grows_with_history(
    history_interval_count: int,
    expected_weight: float,
) -> None:
    """같은 수축 강도에서 이력이 쌓일수록 개인값 가중치가 커집니다."""
    result = calculate_personal_history_weight(
        history_interval_count,
        shrinkage_strength=4.0,
    )

    assert result == pytest.approx(expected_weight)


@pytest.mark.parametrize("invalid_count", [-1, 1.5, True])
def test_personal_history_weight_rejects_invalid_count(
    invalid_count: object,
) -> None:
    """음수·소수·논리값을 과거 간격 개수로 허용하지 않습니다."""
    with pytest.raises(MedianBaselineError, match="과거 간격 개수"):
        calculate_personal_history_weight(invalid_count, shrinkage_strength=4.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_strength",
    [0.0, -1.0, float("inf"), float("nan"), True, "four"],
)
def test_personal_history_weight_rejects_invalid_strength(
    invalid_strength: object,
) -> None:
    """0·음수·비유한값·논리값·문자열 수축 강도를 거부합니다."""
    with pytest.raises(MedianBaselineError, match="수축 강도"):
        calculate_personal_history_weight(1, invalid_strength)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("history_interval_count", "expected_prediction"),
    [
        (0, 30.0),
        (1, 48.0),
        (4, 75.0),
        (20, 105.0),
    ],
)
def test_blend_moves_from_prior_toward_personal_history(
    history_interval_count: int,
    expected_prediction: float,
) -> None:
    """이력이 쌓이면 예측값이 사전 중앙값에서 개인 중앙값으로 이동합니다."""
    result = blend_personal_and_prior_medians(
        personal_median_days=120.0,
        prior_median_days=30.0,
        history_interval_count=history_interval_count,
        shrinkage_strength=4.0,
    )

    assert result == pytest.approx(expected_prediction)


@pytest.mark.parametrize(
    ("field_name", "personal_median", "prior_median"),
    [
        ("개인 중앙값", -1.0, 30.0),
        ("개인 중앙값", float("nan"), 30.0),
        ("사전 중앙값", 120.0, float("inf")),
        ("사전 중앙값", 120.0, True),
    ],
)
def test_blend_rejects_invalid_duration(
    field_name: str,
    personal_median: object,
    prior_median: object,
) -> None:
    """음수·비유한값·논리값 기간이 혼합 예측에 들어가지 못하게 합니다."""
    with pytest.raises(MedianBaselineError, match=field_name):
        blend_personal_and_prior_medians(
            personal_median_days=personal_median,  # type: ignore[arg-type]
            prior_median_days=prior_median,  # type: ignore[arg-type]
            history_interval_count=1,
            shrinkage_strength=4.0,
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


def test_shrunk_prediction_blends_personal_history_with_available_prior() -> None:
    """개인 이력은 상품 prior와 혼합하고 이력이 없으면 fallback을 유지합니다."""
    model = fit_hierarchical_median_baseline(
        make_training_samples(),
        trained_until=pd.Timestamp("2026-01-31"),
    )
    prediction_rows = pd.DataFrame(
        {
            "product_id": ["p1", "p2", "new"],
            "history_median_days": [120.0, None, None],
            "history_interval_count": [1, 0, 0],
        }
    )

    predictions = predict_shrunk_hierarchical_median_baseline(
        model,
        prediction_rows,
        shrinkage_strength=4.0,
    )

    # p1의 prior 15일과 개인 중앙값 120일을 80:20으로 혼합해 36일을 예측합니다.
    assert predictions["predicted_duration_days"].tolist() == [36.0, 40.0, 20.0]
    assert predictions["personal_history_weight"].tolist() == [0.2, 0.0, 0.0]
    assert predictions["prior_duration_days"].tolist() == [15.0, 40.0, 20.0]
    assert predictions["prior_source"].tolist() == [
        "product_history",
        "product_history",
        "global_history",
    ]
    assert predictions["prediction_source"].tolist() == [
        "shrunk_user_product_history",
        "product_history",
        "global_history",
    ]

    # 벡터 계산 결과가 앞서 검증한 단일 행 계산 함수와 같은지 확인합니다.
    expected = blend_personal_and_prior_medians(
        personal_median_days=120.0,
        prior_median_days=15.0,
        history_interval_count=1,
        shrinkage_strength=4.0,
    )
    assert predictions.loc[0, "predicted_duration_days"] == pytest.approx(expected)


def test_shrunk_prediction_rejects_inconsistent_personal_history() -> None:
    """이력 개수와 개인 중앙값 중 하나만 존재하는 모순된 피처를 거부합니다."""
    model = fit_hierarchical_median_baseline(
        make_training_samples(),
        trained_until=pd.Timestamp("2026-01-31"),
    )
    prediction_rows = pd.DataFrame(
        {
            "product_id": ["p1"],
            "history_median_days": [None],
            "history_interval_count": [1],
        }
    )

    with pytest.raises(MedianBaselineError, match="존재 여부"):
        predict_shrunk_hierarchical_median_baseline(
            model,
            prediction_rows,
            shrinkage_strength=4.0,
        )


def test_shrunk_prediction_supports_all_cold_start_batch() -> None:
    """개인 이력이 전혀 없는 배치도 상품·전체 prior로 예측합니다."""
    model = fit_hierarchical_median_baseline(
        make_training_samples(),
        trained_until=pd.Timestamp("2026-01-31"),
    )
    prediction_rows = pd.DataFrame(
        {
            "product_id": ["p2", "new"],
            "history_median_days": [None, None],
            "history_interval_count": [0, 0],
        }
    )

    predictions = predict_shrunk_hierarchical_median_baseline(
        model,
        prediction_rows,
        shrinkage_strength=4.0,
    )

    assert predictions["predicted_duration_days"].tolist() == [40.0, 20.0]
    assert predictions["personal_history_weight"].tolist() == [0.0, 0.0]
    assert predictions["prior_source"].tolist() == [
        "product_history",
        "global_history",
    ]


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
