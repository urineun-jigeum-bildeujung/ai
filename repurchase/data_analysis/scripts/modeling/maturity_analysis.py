"""평가 표본의 관찰 가능 기간과 라벨 성숙도 편향을 분석합니다."""

from __future__ import annotations

from typing import Final

import pandas as pd
from pandas.api.types import is_bool_dtype

from .error_analysis import add_error_columns

# 구매 시점부터 평가 마감일까지 기다릴 수 있었던 기간을 계산할 때 필요합니다.
FOLLOWUP_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "anchor_at",
        "split_end_at",
    }
)

# Validation 라벨 성숙률을 계산할 때 추가로 필요한 구간·정답 확인 열입니다.
MATURITY_SUMMARY_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "split",
        "outcome_available_by_split_end",
    }
)

# 성숙한 Validation 표본의 실제 재구매 간격과 오차를 월별로 비교할 때 필요합니다.
MATURED_PREDICTION_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "split",
        "anchor_at",
        "outcome_available_by_split_end",
        "target_duration_days",
        "predicted_duration_days",
    }
)

# 전체 Validation 성숙도와 성숙 표본의 예측 품질을 월별로 연결할 때 필요합니다.
MATURITY_RESULT_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "anchor_month",
        "matured_sample_count",
    }
)
QUALITY_RESULT_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "anchor_month",
        "evaluable_sample_count",
    }
)


def add_available_followup_days(rows: pd.DataFrame) -> pd.DataFrame:
    """원본을 변경하지 않고 각 표본의 관찰 가능 일수를 추가합니다."""
    missing_columns = FOLLOWUP_REQUIRED_COLUMNS - set(rows.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"관찰 가능 기간 계산에 필요한 열이 없습니다: {missing_text}")

    if rows.empty:
        raise ValueError("관찰 가능 기간을 계산할 표본이 없습니다.")

    result = rows.copy()
    result["anchor_at"] = pd.to_datetime(result["anchor_at"], errors="raise")
    result["split_end_at"] = pd.to_datetime(
        result["split_end_at"],
        errors="raise",
    )

    if result[["anchor_at", "split_end_at"]].isna().any().any():
        raise ValueError(
            "예측 기준 시각과 평가 종료 시각에는 결측값을 사용할 수 없습니다."
        )
    if result["split_end_at"].lt(result["anchor_at"]).any():
        raise ValueError("평가 종료 시각은 예측 기준 시각보다 이를 수 없습니다.")

    result["available_followup_days"] = (
        (result["split_end_at"] - result["anchor_at"]).dt.total_seconds().div(86_400)
    )
    return result


def summarize_validation_label_maturity_by_anchor_month(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """전체 Validation 표본의 라벨 성숙도를 구매 기준 월별로 요약합니다."""
    missing_columns = MATURITY_SUMMARY_REQUIRED_COLUMNS - set(rows.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"라벨 성숙도 요약에 필요한 열이 없습니다: {missing_text}")

    if not rows["split"].eq("validation").all():
        raise ValueError("라벨 성숙도 요약에는 Validation 표본만 사용할 수 있습니다.")

    outcome_available = rows["outcome_available_by_split_end"]
    if outcome_available.isna().any() or not is_bool_dtype(outcome_available.dtype):
        raise ValueError("정답 확인 여부에는 결측값 없는 boolean만 사용할 수 있습니다.")

    analysis_rows = add_available_followup_days(rows)
    analysis_rows["anchor_month"] = (
        analysis_rows["anchor_at"].dt.to_period("M").astype(str)
    )

    summary = (
        analysis_rows.groupby("anchor_month", observed=True, sort=True)
        .agg(
            validation_sample_count=("outcome_available_by_split_end", "size"),
            matured_sample_count=("outcome_available_by_split_end", "sum"),
            median_available_followup_days=("available_followup_days", "median"),
        )
        .reset_index()
    )
    summary["matured_sample_count"] = summary["matured_sample_count"].astype("int64")
    summary["unmatured_sample_count"] = (
        summary["validation_sample_count"] - summary["matured_sample_count"]
    )
    summary["maturity_rate"] = summary["matured_sample_count"].div(
        summary["validation_sample_count"]
    )
    return summary


def summarize_matured_prediction_quality_by_anchor_month(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """성숙한 Validation 표본의 실제 재구매 간격과 예측 오차를 월별로 요약합니다."""
    missing_columns = MATURED_PREDICTION_REQUIRED_COLUMNS - set(rows.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"월별 예측 품질 요약에 필요한 열이 없습니다: {missing_text}")

    if not rows["split"].eq("validation").all():
        raise ValueError(
            "월별 예측 품질 요약에는 Validation 표본만 사용할 수 있습니다."
        )

    outcome_available = rows["outcome_available_by_split_end"]
    if outcome_available.isna().any() or not is_bool_dtype(outcome_available.dtype):
        raise ValueError("정답 확인 여부에는 결측값 없는 boolean만 사용할 수 있습니다.")
    if not outcome_available.all():
        raise ValueError(
            "월별 예측 품질은 정답이 확인된 표본에서만 계산할 수 있습니다."
        )

    error_rows = add_error_columns(rows)
    error_rows["anchor_at"] = pd.to_datetime(error_rows["anchor_at"], errors="raise")
    if error_rows["anchor_at"].isna().any():
        raise ValueError("예측 기준 시각에는 결측값을 사용할 수 없습니다.")

    error_rows["anchor_month"] = error_rows["anchor_at"].dt.to_period("M").astype(str)
    return (
        error_rows.groupby("anchor_month", observed=True, sort=True)
        .agg(
            evaluable_sample_count=("absolute_error_days", "size"),
            mean_target_duration_days=("target_duration_days", "mean"),
            median_target_duration_days=("target_duration_days", "median"),
            mae_days=("absolute_error_days", "mean"),
            median_absolute_error_days=("absolute_error_days", "median"),
        )
        .reset_index()
    )


def merge_validation_maturity_and_quality(
    maturity_summary: pd.DataFrame,
    quality_summary: pd.DataFrame,
) -> pd.DataFrame:
    """성숙도 월을 모두 보존하며 월별 예측 품질을 일대일로 연결합니다."""
    missing_maturity_columns = MATURITY_RESULT_REQUIRED_COLUMNS - set(
        maturity_summary.columns
    )
    if missing_maturity_columns:
        missing_text = ", ".join(sorted(missing_maturity_columns))
        raise ValueError(f"성숙도 결과에 필요한 열이 없습니다: {missing_text}")

    missing_quality_columns = QUALITY_RESULT_REQUIRED_COLUMNS - set(
        quality_summary.columns
    )
    if missing_quality_columns:
        missing_text = ", ".join(sorted(missing_quality_columns))
        raise ValueError(f"예측 품질 결과에 필요한 열이 없습니다: {missing_text}")

    if maturity_summary.empty:
        raise ValueError("병합할 Validation 성숙도 결과가 없습니다.")
    if maturity_summary["anchor_month"].duplicated().any():
        raise ValueError("성숙도 결과에는 구매 기준 월이 중복될 수 없습니다.")
    if quality_summary["anchor_month"].duplicated().any():
        raise ValueError("예측 품질 결과에는 구매 기준 월이 중복될 수 없습니다.")

    unknown_quality_months = set(quality_summary["anchor_month"]) - set(
        maturity_summary["anchor_month"]
    )
    if unknown_quality_months:
        unknown_text = ", ".join(sorted(str(month) for month in unknown_quality_months))
        raise ValueError(f"성숙도 결과에 없는 예측 품질 월이 있습니다: {unknown_text}")

    result = maturity_summary.merge(
        quality_summary,
        on="anchor_month",
        how="left",
        validate="one_to_one",
    )
    result["evaluable_sample_count"] = (
        result["evaluable_sample_count"].fillna(0).astype("int64")
    )
    if not result["evaluable_sample_count"].eq(result["matured_sample_count"]).all():
        raise ValueError("월별 성숙 표본 수와 평가 가능 표본 수가 일치하지 않습니다.")
    return result
