"""Validation 표본에서 재구매 예측 모델 후보를 비교합니다.

모델 학습·예측 구현과 하이퍼파라미터 비교 책임을 분리하고, Test를 보지 않은
상태에서 수축 강도별 성능을 동일한 표본과 지표로 평가합니다.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .baseline import (
    HierarchicalMedianModel,
    predict_hierarchical_median_baseline,
    predict_shrunk_hierarchical_median_baseline,
)
from .error_analysis import (
    compare_error_on_fixed_cohort,
    select_largest_error_rows,
    summarize_largest_error_tail,
)
from .evaluation import evaluate_predictions


def evaluate_shrinkage_candidates(
    samples: pd.DataFrame,
    model: HierarchicalMedianModel,
    *,
    shrinkage_strengths: Sequence[float],
    tail_rate: float = 0.05,
) -> pd.DataFrame:
    """동일 Validation 표본에서 수축 강도별 전체·꼬리 성능을 계산합니다."""
    if samples.empty:
        raise ValueError("수축 강도를 평가할 Validation 표본이 없습니다.")
    if not shrinkage_strengths:
        raise ValueError("평가할 수축 강도 후보가 없습니다.")

    results: list[dict[str, float | int]] = []
    evaluated_strengths: set[float] = set()
    reference_predictions = predict_hierarchical_median_baseline(model, samples)
    fixed_tail_cohort = select_largest_error_rows(
        reference_predictions,
        tail_rate=tail_rate,
    )

    # 후보별 모델 실행은 의도된 실험 반복이며, 각 후보 내부의 행 계산은 벡터화합니다.
    for shrinkage_strength in shrinkage_strengths:
        predictions = predict_shrunk_hierarchical_median_baseline(
            model,
            samples,
            shrinkage_strength=shrinkage_strength,
        )
        normalized_strength = float(predictions["shrinkage_strength"].iat[0])
        if normalized_strength in evaluated_strengths:
            raise ValueError(f"중복된 수축 강도 후보입니다: {normalized_strength}")
        evaluated_strengths.add(normalized_strength)

        evaluation = evaluate_predictions(predictions)
        overall = evaluation["overall"]
        personal_rows = predictions.loc[
            predictions["prediction_source"].eq("shrunk_user_product_history")
        ]
        tail_metrics = summarize_largest_error_tail(
            predictions,
            tail_rate=tail_rate,
        )
        tail_sample_count = int(tail_metrics["tail_sample_count"])
        tail_absolute_error_days = float(tail_metrics["tail_absolute_error_days"])
        fixed_cohort_metrics = compare_error_on_fixed_cohort(
            reference_predictions,
            predictions,
            fixed_tail_cohort,
        )
        mean_personal_weight = (
            0.0
            if personal_rows.empty
            else float(personal_rows["personal_history_weight"].mean())
        )
        results.append(
            {
                "shrinkage_strength": normalized_strength,
                "sample_count": int(overall["sample_count"]),
                "personal_sample_count": int(len(personal_rows)),
                "mae_days": float(overall["mae_days"]),
                "median_absolute_error_days": float(
                    overall["median_absolute_error_days"]
                ),
                "within_3_days_rate": float(overall["within_3_days_rate"]),
                "within_7_days_rate": float(overall["within_7_days_rate"]),
                "mean_personal_history_weight": mean_personal_weight,
                "requested_tail_rate": float(tail_metrics["requested_tail_rate"]),
                "actual_tail_sample_rate": float(
                    tail_metrics["actual_tail_sample_rate"]
                ),
                "tail_sample_count": tail_sample_count,
                "tail_absolute_error_days": tail_absolute_error_days,
                "tail_mae_days": tail_absolute_error_days / tail_sample_count,
                "tail_absolute_error_share": float(
                    tail_metrics["absolute_error_share"]
                ),
                "tail_late_prediction_rate": float(
                    tail_metrics["late_prediction_rate"]
                ),
                "fixed_cohort_sample_count": int(
                    fixed_cohort_metrics["cohort_sample_count"]
                ),
                "fixed_cohort_reference_mae_days": float(
                    fixed_cohort_metrics["reference_mae_days"]
                ),
                "fixed_cohort_candidate_mae_days": float(
                    fixed_cohort_metrics["candidate_mae_days"]
                ),
                "fixed_cohort_mae_improvement_days": float(
                    fixed_cohort_metrics["mae_improvement_days"]
                ),
                "fixed_cohort_improved_sample_count": int(
                    fixed_cohort_metrics["improved_sample_count"]
                ),
                "fixed_cohort_improved_sample_rate": float(
                    fixed_cohort_metrics["improved_sample_rate"]
                ),
                "fixed_cohort_worsened_sample_count": int(
                    fixed_cohort_metrics["worsened_sample_count"]
                ),
                "fixed_cohort_worsened_sample_rate": float(
                    fixed_cohort_metrics["worsened_sample_rate"]
                ),
                "fixed_cohort_late_prediction_rate": float(
                    fixed_cohort_metrics["candidate_late_prediction_rate"]
                ),
            }
        )

    # 입력한 후보 순서를 보존해 강도가 커질 때 지표 변화를 그대로 비교합니다.
    return pd.DataFrame(results)
