"""재구매 예측의 행 단위 오차를 보존해 성능 악화 원인을 분석합니다."""

from __future__ import annotations

from math import ceil
from typing import Final

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .samples import SAMPLE_ID_COLUMNS

# 절대오차를 계산하려면 실제 관측값과 모델 예측값이 모두 필요합니다.
REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "target_duration_days",
        "predicted_duration_days",
    }
)

# 개인·상품 이력 개수별 오차를 비교할 때 추가로 필요한 열입니다.
HISTORY_SUMMARY_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "prediction_source",
        "history_interval_count",
    }
)

# 과거 간격의 상대적 불규칙성과 예측 오차의 관계를 분석할 때 필요한 열입니다.
VARIABILITY_SUMMARY_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "prediction_source",
        "history_relative_mad",
    }
)

PRODUCT_SUMMARY_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "prediction_source",
        "product_id",
    }
)

TIME_SUMMARY_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "prediction_source",
        "anchor_at",
    }
)

# 개인 이력을 사용한 예측 방식을 한곳에서 관리해 모든 오차 분석이 같은 범위를 봅니다.
USER_PRODUCT_PREDICTION_SOURCES: Final[frozenset[str]] = frozenset(
    {
        "user_product_history",
        "shrunk_user_product_history",
    }
)


def validate_error_analysis_input(rows: pd.DataFrame) -> None:
    """오차 계산에 필요한 필수 열이 모두 있는지 확인합니다."""
    # 필요한 열에서 실제 입력 열을 빼면 누락된 열만 남습니다.
    missing_columns = REQUIRED_COLUMNS - set(rows.columns)

    if missing_columns:
        # 여러 열이 누락되어도 항상 같은 순서로 오류 메시지를 만듭니다.
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"필수 열이 없습니다: {missing_text}")

    if rows.empty:
        raise ValueError("오차를 분석할 예측 표본이 없습니다.")

    for column in sorted(REQUIRED_COLUMNS):
        values = rows[column]

        if values.isna().any():
            raise ValueError(f"{column}에 결측값이 있습니다.")
        # True와 False는 계산상 1과 0이 될 수 있지만 기간 데이터로는 허용하지 않습니다.
        if is_bool_dtype(values.dtype) or not is_numeric_dtype(values.dtype):
            raise ValueError(f"{column}은 숫자형이어야 합니다.")
        if not np.isfinite(values.to_numpy(dtype="float64", copy=False)).all():
            raise ValueError(f"{column}에는 유한한 숫자만 사용할 수 있습니다.")

    # 0일 후속 구매는 실제 데이터에 존재하므로 음수인 정답만 차단합니다.
    if rows["target_duration_days"].lt(0).any():
        raise ValueError("실제 재구매 간격은 음수일 수 없습니다.")


def add_error_columns(rows: pd.DataFrame) -> pd.DataFrame:
    """원본을 변경하지 않고 각 예측 표본에 방향·절대오차 열을 추가합니다."""
    validate_error_analysis_input(rows)

    result = rows.copy()
    # 음수 예측을 삭제하지 않고 모델 실패 여부를 별도 열에 보존합니다.
    result["is_invalid_prediction"] = result["predicted_duration_days"].lt(0)
    # 양수면 실제보다 늦게, 음수면 실제보다 빠르게 예측했다는 뜻입니다.
    result["prediction_error_days"] = (
        result["predicted_duration_days"] - result["target_duration_days"]
    )
    result["absolute_error_days"] = result["prediction_error_days"].abs()
    return result


def _get_user_product_error_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """오차 열을 계산하고 개인·상품 이력 예측 표본만 반환합니다."""
    if "prediction_source" not in rows.columns:
        raise ValueError("개인 이력 분석 필수 열이 없습니다: prediction_source")

    error_rows = add_error_columns(rows)
    user_product_rows = error_rows.loc[
        error_rows["prediction_source"].isin(USER_PRODUCT_PREDICTION_SOURCES)
    ].copy()

    if user_product_rows.empty:
        raise ValueError("분석할 개인·상품 이력 예측 표본이 없습니다.")

    return user_product_rows


def _select_largest_from_user_product_rows(
    user_product_rows: pd.DataFrame,
    tail_rate: float,
) -> pd.DataFrame:
    """오차가 계산된 개인·상품 이력에서 큰 절대오차 표본을 선택합니다."""
    if not 0 < tail_rate <= 1:
        raise ValueError("tail_rate는 0보다 크고 1 이하여야 합니다.")

    # 표본 수에 비율을 곱한 결과를 올림하고, 작은 데이터에서도 1개는 선택합니다.
    tail_count = max(1, ceil(len(user_product_rows) * tail_rate))
    return user_product_rows.nlargest(
        tail_count,
        "absolute_error_days",
    ).copy()


def select_largest_error_rows(
    rows: pd.DataFrame,
    tail_rate: float = 0.01,
) -> pd.DataFrame:
    """절대오차가 큰 상위 비율의 개인·상품 이력 표본을 선택합니다."""
    # 이번 원인 분석의 대상인 개인·상품 이력 예측만 분리합니다.
    user_product_rows = _get_user_product_error_rows(rows)
    # 오차 방향이 아닌 크기를 기준으로 가장 크게 실패한 행부터 선택합니다.
    return _select_largest_from_user_product_rows(
        user_product_rows,
        tail_rate,
    )


def summarize_largest_error_tail(
    rows: pd.DataFrame,
    tail_rate: float = 0.01,
) -> dict[str, float | int]:
    """큰 절대오차 표본의 전체 오차 기여도와 예측 방향을 요약합니다."""
    user_product_rows = _get_user_product_error_rows(rows)
    tail_rows = _select_largest_from_user_product_rows(
        user_product_rows,
        tail_rate,
    )

    total_absolute_error = float(user_product_rows["absolute_error_days"].sum())
    tail_absolute_error = float(tail_rows["absolute_error_days"].sum())
    tail_sample_count = len(tail_rows)
    late_prediction_count = int(tail_rows["prediction_error_days"].gt(0).sum())
    early_prediction_count = int(tail_rows["prediction_error_days"].lt(0).sum())
    exact_prediction_count = int(tail_rows["prediction_error_days"].eq(0).sum())

    return {
        "requested_tail_rate": tail_rate,
        "tail_sample_count": tail_sample_count,
        # 올림으로 선택하므로 실제 표본 비율은 요청한 비율과 조금 다를 수 있습니다.
        "actual_tail_sample_rate": tail_sample_count / len(user_product_rows),
        "tail_absolute_error_days": tail_absolute_error,
        "absolute_error_share": (
            0.0
            if total_absolute_error == 0
            else tail_absolute_error / total_absolute_error
        ),
        "late_prediction_count": late_prediction_count,
        "late_prediction_rate": late_prediction_count / tail_sample_count,
        "early_prediction_count": early_prediction_count,
        "early_prediction_rate": early_prediction_count / tail_sample_count,
        "exact_prediction_count": exact_prediction_count,
        "exact_prediction_rate": exact_prediction_count / tail_sample_count,
    }


def _validate_fixed_cohort_ids(rows: pd.DataFrame, *, label: str) -> None:
    """고정 코호트 비교용 표본 키가 존재하고 행을 하나씩 식별하는지 확인합니다."""
    missing_columns = set(SAMPLE_ID_COLUMNS) - set(rows.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"{label}에 표본 식별 열이 없습니다: {missing_text}")
    if rows.empty:
        raise ValueError(f"{label}에 비교할 표본이 없습니다.")
    if rows.loc[:, list(SAMPLE_ID_COLUMNS)].isna().any().any():
        raise ValueError(f"{label}의 표본 식별 열에 결측값이 있습니다.")
    if rows.duplicated(subset=list(SAMPLE_ID_COLUMNS)).any():
        raise ValueError(f"{label}에 중복된 표본 식별자가 있습니다.")


def _get_fixed_cohort_error_rows(
    rows: pd.DataFrame,
    cohort_keys: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    """표본 키로 예측 결과를 연결해 고정 코호트의 행별 오차를 반환합니다."""
    _validate_fixed_cohort_ids(rows, label=label)
    error_rows = add_error_columns(rows)
    error_columns = [
        *SAMPLE_ID_COLUMNS,
        "target_duration_days",
        "predicted_duration_days",
        "prediction_error_days",
        "absolute_error_days",
    ]
    matched_rows = cohort_keys.merge(
        error_rows.loc[:, error_columns],
        on=list(SAMPLE_ID_COLUMNS),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if matched_rows["_merge"].ne("both").any():
        raise ValueError(f"{label}에서 고정 코호트 표본을 찾을 수 없습니다.")
    return matched_rows.drop(columns="_merge")


def compare_error_on_fixed_cohort(
    reference_rows: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    cohort_rows: pd.DataFrame,
) -> dict[str, float | int]:
    """같은 코호트에서 기준 모델과 후보 모델의 오차 변화를 요약합니다."""
    comparison = build_fixed_cohort_comparison_rows(
        reference_rows,
        candidate_rows,
        cohort_rows,
    )

    return summarize_fixed_cohort_comparison_rows(comparison)


def summarize_fixed_cohort_comparison_rows(
    comparison: pd.DataFrame,
) -> dict[str, float | int]:
    """이미 연결된 고정 코호트 비교 행을 집계해 모델 수준 지표로 요약합니다."""
    required_columns = {
        "absolute_error_days_reference",
        "absolute_error_days_candidate",
        "prediction_error_days_candidate",
        "comparison_outcome",
    }
    missing_columns = required_columns - set(comparison.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"고정 코호트 요약 필수 열이 없습니다: {missing_text}")
    if comparison.empty:
        raise ValueError("요약할 고정 코호트 비교 행이 없습니다.")

    reference_absolute_error = comparison["absolute_error_days_reference"]
    candidate_absolute_error = comparison["absolute_error_days_candidate"]
    improved = comparison["comparison_outcome"].eq("IMPROVED")
    worsened = comparison["comparison_outcome"].eq("WORSENED")
    unchanged = comparison["comparison_outcome"].eq("UNCHANGED")
    candidate_direction_error = comparison["prediction_error_days_candidate"]
    sample_count = len(comparison)
    reference_mae = float(reference_absolute_error.mean())
    candidate_mae = float(candidate_absolute_error.mean())
    improved_count = int(improved.sum())
    worsened_count = int(worsened.sum())
    unchanged_count = int(unchanged.sum())
    candidate_late_count = int(candidate_direction_error.gt(0).sum())
    candidate_early_count = int(candidate_direction_error.lt(0).sum())

    return {
        "cohort_sample_count": sample_count,
        "reference_mae_days": reference_mae,
        "candidate_mae_days": candidate_mae,
        "mae_improvement_days": reference_mae - candidate_mae,
        "improved_sample_count": improved_count,
        "improved_sample_rate": improved_count / sample_count,
        "worsened_sample_count": worsened_count,
        "worsened_sample_rate": worsened_count / sample_count,
        "unchanged_sample_count": unchanged_count,
        "unchanged_sample_rate": unchanged_count / sample_count,
        "candidate_late_prediction_count": candidate_late_count,
        "candidate_late_prediction_rate": candidate_late_count / sample_count,
        "candidate_early_prediction_count": candidate_early_count,
        "candidate_early_prediction_rate": candidate_early_count / sample_count,
    }


def build_fixed_cohort_comparison_rows(
    reference_rows: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    cohort_rows: pd.DataFrame,
) -> pd.DataFrame:
    """고정 코호트의 기준·후보 오차를 표본별로 연결해 변화 상태를 부여합니다."""
    _validate_fixed_cohort_ids(cohort_rows, label="고정 코호트")
    cohort_keys = cohort_rows.loc[:, list(SAMPLE_ID_COLUMNS)].copy()
    reference_errors = _get_fixed_cohort_error_rows(
        reference_rows,
        cohort_keys,
        label="기준 예측",
    )
    candidate_errors = _get_fixed_cohort_error_rows(
        candidate_rows,
        cohort_keys,
        label="후보 예측",
    )
    comparison = reference_errors.merge(
        candidate_errors,
        on=list(SAMPLE_ID_COLUMNS),
        how="inner",
        validate="one_to_one",
        suffixes=("_reference", "_candidate"),
    )
    comparison["absolute_error_improvement_days"] = (
        comparison["absolute_error_days_reference"]
        - comparison["absolute_error_days_candidate"]
    )
    comparison["comparison_outcome"] = "UNCHANGED"
    comparison.loc[
        comparison["absolute_error_improvement_days"].gt(0),
        "comparison_outcome",
    ] = "IMPROVED"
    comparison.loc[
        comparison["absolute_error_improvement_days"].lt(0),
        "comparison_outcome",
    ] = "WORSENED"
    return comparison


def summarize_largest_error_tail_by_history_count(
    rows: pd.DataFrame,
    tail_rate: float = 0.01,
) -> pd.DataFrame:
    """전체와 꼬리 표본의 개인 이력 개수 분포를 비교합니다."""
    missing_columns = HISTORY_SUMMARY_REQUIRED_COLUMNS - set(rows.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"꼬리 이력 분석 필수 열이 없습니다: {missing_text}")

    user_product_rows = _get_user_product_error_rows(rows)
    history_counts = user_product_rows["history_interval_count"]
    if history_counts.isna().any():
        raise ValueError("개인·상품 이력 개수에 결측값이 있습니다.")
    if history_counts.le(0).any():
        raise ValueError("개인·상품 이력 예측에는 1개 이상의 과거 간격이 필요합니다.")

    tail_rows = _select_largest_from_user_product_rows(
        user_product_rows,
        tail_rate,
    )
    overall_counts = (
        user_product_rows.groupby(
            "history_interval_count",
            observed=True,
            sort=True,
        )
        .size()
        .rename("overall_sample_count")
        .reset_index()
    )
    tail_counts = (
        tail_rows.groupby(
            "history_interval_count",
            observed=True,
            sort=True,
        )
        .agg(
            tail_sample_count=("absolute_error_days", "size"),
            tail_absolute_error_days=("absolute_error_days", "sum"),
        )
        .reset_index()
    )
    summary = overall_counts.merge(
        tail_counts,
        on="history_interval_count",
        how="left",
        validate="one_to_one",
    )
    summary["tail_sample_count"] = (
        summary["tail_sample_count"].fillna(0).astype("int64")
    )
    summary["tail_absolute_error_days"] = summary["tail_absolute_error_days"].fillna(
        0.0
    )
    summary["overall_sample_rate"] = summary["overall_sample_count"].div(
        len(user_product_rows)
    )
    summary["tail_sample_rate"] = summary["tail_sample_count"].div(len(tail_rows))
    summary["tail_membership_rate"] = summary["tail_sample_count"].div(
        summary["overall_sample_count"]
    )
    summary["tail_overrepresentation_ratio"] = summary["tail_sample_rate"].div(
        summary["overall_sample_rate"]
    )

    total_tail_absolute_error = float(tail_rows["absolute_error_days"].sum())
    summary["tail_absolute_error_share"] = (
        0.0
        if total_tail_absolute_error == 0
        else summary["tail_absolute_error_days"].div(total_tail_absolute_error)
    )
    return summary


def summarize_user_product_errors_by_history_count(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """개인·상품 이력 예측의 오차를 과거 구매 간격 개수별로 요약합니다."""
    missing_columns = HISTORY_SUMMARY_REQUIRED_COLUMNS - set(rows.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"이력 개수별 분석 필수 열이 없습니다: {missing_text}")

    user_product_rows = _get_user_product_error_rows(rows)
    if user_product_rows["history_interval_count"].isna().any():
        raise ValueError("개인·상품 이력 개수에 결측값이 있습니다.")
    if user_product_rows["history_interval_count"].le(0).any():
        raise ValueError("개인·상품 이력 예측에는 1개 이상의 과거 간격이 필요합니다.")

    summary = (
        user_product_rows.groupby(
            "history_interval_count",
            observed=True,
            sort=True,
        )
        .agg(
            sample_count=("absolute_error_days", "size"),
            mae_days=("absolute_error_days", "mean"),
            median_absolute_error_days=("absolute_error_days", "median"),
            mean_prediction_error_days=("prediction_error_days", "mean"),
        )
        .reset_index()
    )
    return summary


def _spearman_rank_correlation(
    left: pd.Series,
    right: pd.Series,
) -> float | None:
    """두 숫자의 실제 크기 대신 순위가 함께 변하는 정도를 계산합니다."""
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return None

    left_rank = left.rank(method="average")
    right_rank = right.rank(method="average")
    return float(left_rank.corr(right_rank))


def summarize_user_product_error_variability(
    rows: pd.DataFrame,
) -> dict[str, float | int | None]:
    """개인·상품 이력의 상대 MAD와 절대오차 간 순위 관계를 요약합니다."""
    missing_columns = VARIABILITY_SUMMARY_REQUIRED_COLUMNS - set(rows.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"변동성 분석 필수 열이 없습니다: {missing_text}")

    user_product_rows = _get_user_product_error_rows(rows)
    analysis_rows = user_product_rows.loc[
        user_product_rows["history_relative_mad"].notna()
    ].copy()

    if analysis_rows.empty:
        raise ValueError("변동성을 계산할 수 있는 개인·상품 이력 표본이 없습니다.")

    variability = analysis_rows["history_relative_mad"]
    if is_bool_dtype(variability.dtype) or not is_numeric_dtype(variability.dtype):
        raise ValueError("history_relative_mad는 숫자형이어야 합니다.")
    if not np.isfinite(variability.to_numpy(dtype="float64", copy=False)).all():
        raise ValueError("history_relative_mad에는 유한한 숫자만 사용할 수 있습니다.")
    if variability.lt(0).any():
        raise ValueError("history_relative_mad는 음수일 수 없습니다.")

    return {
        "sample_count": int(len(analysis_rows)),
        "median_relative_mad": float(variability.median()),
        "spearman_relative_mad_absolute_error": _spearman_rank_correlation(
            variability,
            analysis_rows["absolute_error_days"],
        ),
    }


def summarize_user_product_errors_by_product(rows: pd.DataFrame) -> pd.DataFrame:
    """개인·상품 이력 예측의 전체 오차 기여도를 상품별로 요약합니다."""
    missing_columns = PRODUCT_SUMMARY_REQUIRED_COLUMNS - set(rows.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"상품별 분석 필수 열이 없습니다: {missing_text}")
    if rows["product_id"].isna().any():
        raise ValueError("상품별 분석에 사용할 product_id에 결측값이 있습니다.")

    user_product_rows = _get_user_product_error_rows(rows)

    total_absolute_error = float(user_product_rows["absolute_error_days"].sum())
    summary = (
        user_product_rows.groupby("product_id", observed=True, sort=True)
        .agg(
            sample_count=("absolute_error_days", "size"),
            total_absolute_error_days=("absolute_error_days", "sum"),
            mae_days=("absolute_error_days", "mean"),
            median_absolute_error_days=("absolute_error_days", "median"),
            mean_prediction_error_days=("prediction_error_days", "mean"),
        )
        .reset_index()
    )
    summary["sample_rate"] = summary["sample_count"].div(len(user_product_rows))
    if total_absolute_error == 0:
        summary["absolute_error_share"] = 0.0
    else:
        summary["absolute_error_share"] = summary["total_absolute_error_days"].div(
            total_absolute_error
        )
    return summary.sort_values(
        ["total_absolute_error_days", "sample_count", "product_id"],
        ascending=[False, False, True],
        kind="stable",
        ignore_index=True,
    )


def summarize_user_product_errors_by_anchor_month(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """개인·상품 이력 예측의 오차를 예측 기준 월별로 요약합니다."""
    missing_columns = TIME_SUMMARY_REQUIRED_COLUMNS - set(rows.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"월별 분석 필수 열이 없습니다: {missing_text}")

    user_product_rows = _get_user_product_error_rows(rows)
    user_product_rows["anchor_at"] = pd.to_datetime(
        user_product_rows["anchor_at"],
        errors="raise",
    )
    if user_product_rows["anchor_at"].isna().any():
        raise ValueError("예측 기준 시각에 결측값이 있습니다.")

    user_product_rows["anchor_month"] = (
        user_product_rows["anchor_at"].dt.to_period("M").astype(str)
    )
    total_absolute_error = float(user_product_rows["absolute_error_days"].sum())
    summary = (
        user_product_rows.groupby("anchor_month", observed=True, sort=True)
        .agg(
            sample_count=("absolute_error_days", "size"),
            total_absolute_error_days=("absolute_error_days", "sum"),
            mae_days=("absolute_error_days", "mean"),
            median_absolute_error_days=("absolute_error_days", "median"),
            mean_prediction_error_days=("prediction_error_days", "mean"),
        )
        .reset_index()
    )
    summary["sample_rate"] = summary["sample_count"].div(len(user_product_rows))
    if total_absolute_error == 0:
        summary["absolute_error_share"] = 0.0
    else:
        summary["absolute_error_share"] = summary["total_absolute_error_days"].div(
            total_absolute_error
        )
    return summary
