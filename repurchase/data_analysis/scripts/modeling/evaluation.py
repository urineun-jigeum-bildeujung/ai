"""재구매 기간 예측의 전체·fallback 단계별 오차 지표를 계산합니다."""

from __future__ import annotations

from typing import Final

import pandas as pd


class RepurchaseEvaluationError(ValueError):
    """평가 입력에 정답이나 예측값이 없을 때 발생합니다."""


EVALUATION_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "target_duration_days",
    "predicted_duration_days",
    "prediction_source",
)

HIT_WINDOWS_DAYS: Final[tuple[int, ...]] = (3, 7)


def calculate_regression_metrics(rows: pd.DataFrame) -> dict[str, float | int]:
    """관측된 재구매 기간의 MAE·중앙 절대오차·허용일 내 적중률을 계산합니다."""
    missing_columns = set(EVALUATION_REQUIRED_COLUMNS) - set(rows.columns)
    if missing_columns:
        raise RepurchaseEvaluationError(
            f"재구매 평가 필수 컬럼이 누락됐습니다: {sorted(missing_columns)}"
        )
    evaluation_rows = rows.dropna(
        subset=["target_duration_days", "predicted_duration_days"]
    ).copy()
    if evaluation_rows.empty:
        raise RepurchaseEvaluationError("평가할 관측 재구매 표본이 없습니다.")

    absolute_error = (
        evaluation_rows["target_duration_days"]
        - evaluation_rows["predicted_duration_days"]
    ).abs()
    metrics: dict[str, float | int] = {
        "sample_count": int(len(evaluation_rows)),
        "mae_days": float(absolute_error.mean()),
        "median_absolute_error_days": float(absolute_error.median()),
    }
    for window_days in HIT_WINDOWS_DAYS:
        metrics[f"within_{window_days}_days_rate"] = float(
            absolute_error.le(window_days).mean()
        )
    return metrics


def evaluate_predictions(rows: pd.DataFrame) -> dict[str, object]:
    """전체 성능과 개인·상품·전역 fallback별 성능 및 사용 비율을 반환합니다."""
    overall = calculate_regression_metrics(rows)
    total_count = int(overall["sample_count"])
    by_source: dict[str, object] = {}
    for source, source_rows in rows.groupby(
        "prediction_source",
        observed=True,
        sort=True,
    ):
        source_metrics = calculate_regression_metrics(source_rows)
        source_metrics["sample_rate"] = float(
            int(source_metrics["sample_count"]) / total_count
        )
        by_source[str(source)] = source_metrics
    return {"overall": overall, "by_prediction_source": by_source}
