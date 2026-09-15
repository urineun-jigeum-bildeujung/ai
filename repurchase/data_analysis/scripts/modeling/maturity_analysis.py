"""평가 표본의 관찰 가능 기간과 라벨 성숙도 편향을 분석합니다."""

from __future__ import annotations

from collections.abc import Sequence
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

# 공통 관찰 기간 후보를 Validation에서 비교할 때 시간 구간 확인에 필요합니다.
COMMON_FOLLOWUP_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset({"split"})

# 공통 관찰 기간 안에 실제 재구매가 발생했는지 판정할 때 필요합니다.
COMMON_FOLLOWUP_EVENT_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "outcome_available_by_split_end",
        "target_duration_days",
    }
)

# Validation 종료 시점에서 생존분석용 관측 시간과 사건 여부를 만들 때 필요합니다.
SURVIVAL_OBSERVATION_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "split",
        "outcome_available_by_split_end",
        "target_duration_days",
    }
)

# 관찰 기간의 짧은 구간부터 긴 구간까지 분포를 고르게 확인합니다.
FOLLOWUP_DISTRIBUTION_QUANTILES: Final[tuple[float, ...]] = (
    0.0,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    1.0,
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


def _validate_common_followup_horizon_days(horizon_days: object) -> None:
    """공통 관찰 기간이 양의 정수 일수인지 검증합니다."""
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int):
        raise ValueError("공통 관찰 기간은 양의 정수 일수여야 합니다.")
    if horizon_days <= 0:
        raise ValueError("공통 관찰 기간은 양의 정수 일수여야 합니다.")


def _validate_common_followup_candidates(
    horizon_days_candidates: Sequence[int],
) -> list[int]:
    """공통 관찰 기간 후보 목록을 검증해 재사용 가능한 리스트로 반환합니다."""
    candidates = list(horizon_days_candidates)
    if not candidates:
        raise ValueError("비교할 공통 관찰 기간 후보가 하나 이상 필요합니다.")
    for horizon_days in candidates:
        _validate_common_followup_horizon_days(horizon_days)
    if len(set(candidates)) != len(candidates):
        raise ValueError("공통 관찰 기간 후보는 중복될 수 없습니다.")
    return candidates


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


def add_validation_survival_observation(rows: pd.DataFrame) -> pd.DataFrame:
    """Validation 종료 시점에서 확인 가능한 사건·검열 시간만 구성합니다.

    분할 종료 전에 재구매가 확인된 표본은 실제 재구매 간격을 사용합니다.
    그 밖의 표본은 미래의 실제 재구매 간격을 보지 않고, 분할 종료일까지
    기다릴 수 있었던 기간만 우측검열 시간으로 사용합니다.
    """
    missing_columns = SURVIVAL_OBSERVATION_REQUIRED_COLUMNS - set(rows.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"생존분석 관측값 구성에 필요한 열이 없습니다: {missing_text}")
    if not rows["split"].eq("validation").all():
        raise ValueError("생존분석 관측값에는 Validation 표본만 사용할 수 있습니다.")

    result = add_available_followup_days(rows)
    event_observed = result["outcome_available_by_split_end"]
    if event_observed.isna().any() or not is_bool_dtype(event_observed.dtype):
        raise ValueError(
            "분할 종료 전 사건 확인 여부에는 결측값 없는 boolean만 필요합니다."
        )
    if result.loc[event_observed, "target_duration_days"].isna().any():
        raise ValueError(
            "분할 종료 전에 확인된 재구매 사건에는 실제 간격이 필요합니다."
        )
    if result["target_duration_days"].dropna().lt(0).any():
        raise ValueError("실제 재구매 간격은 음수일 수 없습니다.")
    if (
        result.loc[event_observed, "target_duration_days"]
        .gt(result.loc[event_observed, "available_followup_days"])
        .any()
    ):
        raise ValueError(
            "분할 종료 전에 확인된 사건은 관찰 가능 기간을 넘을 수 없습니다."
        )

    # 우선 모든 행을 분할 종료 시점의 우측검열로 보고 관찰 가능 기간을 사용합니다.
    result["survival_observed_duration_days"] = result["available_followup_days"]
    # 실제 사건이 확인된 행만 검열 시간을 실제 재구매 간격으로 교체합니다.
    result.loc[event_observed, "survival_observed_duration_days"] = result.loc[
        event_observed,
        "target_duration_days",
    ]
    result["survival_event_observed"] = event_observed
    result["survival_right_censored"] = ~event_observed
    return result


def estimate_validation_censoring_survival_curve(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Validation의 검열 생존확률을 Kaplan-Meier 방식으로 추정합니다.

    재구매 사건이 아니라 관찰 중단을 관심 사건으로 뒤집어 계산합니다.
    결과는 이후 IPCW 분모로 사용할 검열 생존확률의 시점별 근거가 됩니다.
    """
    observed_rows = add_validation_survival_observation(rows)
    timeline = (
        observed_rows.groupby(
            "survival_observed_duration_days",
            observed=True,
            sort=True,
        )
        .agg(
            exit_count=("survival_observed_duration_days", "size"),
            censoring_event_count=("survival_right_censored", "sum"),
            repurchase_event_count=("survival_event_observed", "sum"),
        )
        .reset_index()
    )
    timeline[["exit_count", "censoring_event_count", "repurchase_event_count"]] = (
        timeline[
            ["exit_count", "censoring_event_count", "repurchase_event_count"]
        ].astype("int64")
    )

    # 현재 시점에 도달하기 전에 빠져나간 표본만 누적해 직전 위험집단을 계산합니다.
    exited_before = timeline["exit_count"].cumsum().shift(fill_value=0)
    timeline["at_risk_count"] = len(observed_rows) - exited_before
    timeline["censoring_survival_step"] = 1.0 - timeline["censoring_event_count"].div(
        timeline["at_risk_count"]
    )
    cumulative_survival = timeline["censoring_survival_step"].cumprod()
    # 같은 시점의 검열을 반영하기 전 확률은 사건 표본의 IPCW 분모에 사용합니다.
    timeline["censoring_survival_probability_before"] = cumulative_survival.shift(
        fill_value=1.0
    )
    timeline["censoring_survival_probability"] = cumulative_survival

    return timeline.loc[
        :,
        [
            "survival_observed_duration_days",
            "at_risk_count",
            "censoring_event_count",
            "repurchase_event_count",
            "censoring_survival_step",
            "censoring_survival_probability_before",
            "censoring_survival_probability",
        ],
    ]


def add_validation_ipcw_weights(
    rows: pd.DataFrame,
    *,
    horizon_days: int,
) -> pd.DataFrame:
    """고정 평가 시점의 재구매 여부와 원시 IPCW 가중치를 추가합니다.

    평가 시점 전에 확인된 재구매에는 사건 시점 직전의 검열 생존확률을,
    평가 시점까지 재구매가 없었던 표본에는 해당 시점의 검열 생존확률을
    사용합니다. 평가 시점 전에 검열된 표본은 결과가 불명이므로 가중치 0입니다.
    """
    _validate_common_followup_horizon_days(horizon_days)
    result = add_validation_survival_observation(rows)
    curve = estimate_validation_censoring_survival_curve(rows)

    observed_duration = result["survival_observed_duration_days"]
    event_by_horizon = result["survival_event_observed"] & observed_duration.le(
        horizon_days
    )
    outcome_known = event_by_horizon | observed_duration.ge(horizon_days)
    no_event_by_horizon = outcome_known & ~event_by_horizon
    if not outcome_known.any():
        raise ValueError("고정 평가 시점의 결과를 확인할 수 있는 표본이 없습니다.")

    # 고정 시점 이전에 검열된 행은 정답이 불명이므로 nullable boolean으로 남깁니다.
    result["ipcw_event_within_horizon"] = event_by_horizon.where(
        outcome_known,
        pd.NA,
    ).astype("boolean")
    result["ipcw_horizon_days"] = horizon_days
    result["ipcw_outcome_known"] = outcome_known
    result["ipcw_censoring_survival_probability"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Float64",
    )

    probability_before_by_duration = curve.set_index("survival_observed_duration_days")[
        "censoring_survival_probability_before"
    ]
    result.loc[event_by_horizon, "ipcw_censoring_survival_probability"] = result.loc[
        event_by_horizon,
        "survival_observed_duration_days",
    ].map(probability_before_by_duration)

    curve_at_or_before_horizon = curve.loc[
        curve["survival_observed_duration_days"].le(horizon_days)
    ]
    horizon_probability = (
        1.0
        if curve_at_or_before_horizon.empty
        else float(
            curve_at_or_before_horizon.iloc[-1]["censoring_survival_probability"]
        )
    )
    result.loc[
        no_event_by_horizon,
        "ipcw_censoring_survival_probability",
    ] = horizon_probability

    known_probabilities = result.loc[
        outcome_known,
        "ipcw_censoring_survival_probability",
    ]
    if known_probabilities.isna().any() or known_probabilities.le(0).any():
        raise ValueError(
            "IPCW 가중치에는 결측값이 없고 0보다 큰 검열 생존확률이 필요합니다."
        )

    result["ipcw_weight"] = 0.0
    result.loc[outcome_known, "ipcw_weight"] = 1.0 / known_probabilities.astype(
        "float64"
    )
    return result


def summarize_validation_ipcw_weight_stability(
    rows: pd.DataFrame,
    *,
    horizon_days: int,
) -> pd.DataFrame:
    """고정 평가 시점의 원시 IPCW 가중치 크기와 유효 표본 수를 요약합니다."""
    weighted_rows = add_validation_ipcw_weights(rows, horizon_days=horizon_days)
    known_rows = weighted_rows.loc[weighted_rows["ipcw_outcome_known"]]
    weights = known_rows["ipcw_weight"].astype("float64")

    weight_sum = float(weights.sum())
    squared_weight_sum = float(weights.pow(2).sum())
    effective_sample_size = weight_sum**2 / squared_weight_sum
    validation_sample_count = len(weighted_rows)
    known_sample_count = len(known_rows)
    event_count = int(known_rows["ipcw_event_within_horizon"].sum())
    event_indicator = known_rows["ipcw_event_within_horizon"].astype("float64")
    weighted_event_mass = float(weights.mul(event_indicator).sum())
    weighted_no_event_mass = weight_sum - weighted_event_mass
    unweighted_event_rate = event_count / known_sample_count
    weighted_event_rate = weighted_event_mass / weight_sum

    return pd.DataFrame(
        [
            {
                "horizon_days": horizon_days,
                "validation_sample_count": validation_sample_count,
                "outcome_known_count": known_sample_count,
                "outcome_unknown_count": validation_sample_count - known_sample_count,
                "outcome_known_rate": known_sample_count / validation_sample_count,
                "event_within_horizon_count": event_count,
                "no_event_within_horizon_count": known_sample_count - event_count,
                "unweighted_known_event_rate": unweighted_event_rate,
                "ipcw_weighted_event_mass": weighted_event_mass,
                "ipcw_weighted_no_event_mass": weighted_no_event_mass,
                "ipcw_weighted_event_rate": weighted_event_rate,
                "ipcw_event_rate_difference": weighted_event_rate
                - unweighted_event_rate,
                "ipcw_weight_sum": weight_sum,
                "ipcw_weight_sum_ratio": weight_sum / validation_sample_count,
                "ipcw_weight_mean": float(weights.mean()),
                "ipcw_weight_median": float(weights.median()),
                "ipcw_weight_p90": float(weights.quantile(0.90)),
                "ipcw_weight_p95": float(weights.quantile(0.95)),
                "ipcw_weight_p99": float(weights.quantile(0.99)),
                "ipcw_weight_max": float(weights.max()),
                "ipcw_effective_sample_size": effective_sample_size,
                "ipcw_effective_to_known_sample_rate": effective_sample_size
                / known_sample_count,
                "ipcw_effective_to_validation_sample_rate": effective_sample_size
                / validation_sample_count,
            }
        ]
    )


def summarize_validation_followup_distribution(rows: pd.DataFrame) -> pd.DataFrame:
    """Validation 관찰 가능 기간의 주요 분위수 분포를 요약합니다."""
    missing_columns = COMMON_FOLLOWUP_REQUIRED_COLUMNS - set(rows.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"관찰 기간 분포에 필요한 열이 없습니다: {missing_text}")
    if not rows["split"].eq("validation").all():
        raise ValueError("관찰 기간 분포에는 Validation 표본만 사용할 수 있습니다.")

    analysis_rows = add_available_followup_days(rows)
    quantiles = analysis_rows["available_followup_days"].quantile(
        FOLLOWUP_DISTRIBUTION_QUANTILES
    )
    return pd.DataFrame(
        {
            "quantile": quantiles.index.astype("float64"),
            "available_followup_days": quantiles.to_numpy(dtype="float64"),
        }
    )


def add_validation_common_followup_eligibility(
    rows: pd.DataFrame,
    *,
    horizon_days: int,
) -> pd.DataFrame:
    """Validation 표본이 공통 관찰 기간을 충족하는지 표시합니다."""
    _validate_common_followup_horizon_days(horizon_days)

    missing_columns = COMMON_FOLLOWUP_REQUIRED_COLUMNS - set(rows.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"공통 관찰 기간 판정에 필요한 열이 없습니다: {missing_text}")
    if not rows["split"].eq("validation").all():
        raise ValueError(
            "공통 관찰 기간 판정에는 Validation 표본만 사용할 수 있습니다."
        )

    result = add_available_followup_days(rows)
    result["common_followup_horizon_days"] = horizon_days
    result["common_followup_eligible"] = result["available_followup_days"].ge(
        horizon_days
    )
    return result


def add_validation_event_within_horizon(
    rows: pd.DataFrame,
    *,
    horizon_days: int,
) -> pd.DataFrame:
    """공통 관찰 기간 안의 재구매 여부를 판단 가능한 행에만 표시합니다."""
    missing_columns = COMMON_FOLLOWUP_EVENT_REQUIRED_COLUMNS - set(rows.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"기간 내 재구매 판정에 필요한 열이 없습니다: {missing_text}")

    outcome_available = rows["outcome_available_by_split_end"]
    if outcome_available.isna().any() or not is_bool_dtype(outcome_available.dtype):
        raise ValueError("정답 확인 여부에는 결측값 없는 boolean만 사용할 수 있습니다.")
    if rows.loc[outcome_available, "target_duration_days"].isna().any():
        raise ValueError("확인된 재구매 사건에는 실제 재구매 간격이 필요합니다.")
    if rows["target_duration_days"].dropna().lt(0).any():
        raise ValueError("실제 재구매 간격은 음수일 수 없습니다.")

    marked_rows = add_validation_common_followup_eligibility(
        rows,
        horizon_days=horizon_days,
    )
    event_within_horizon = marked_rows["outcome_available_by_split_end"] & marked_rows[
        "target_duration_days"
    ].le(horizon_days)
    marked_rows["event_within_horizon"] = event_within_horizon.where(
        marked_rows["common_followup_eligible"],
        pd.NA,
    ).astype("boolean")
    return marked_rows


def summarize_validation_common_followup_outcomes(
    rows: pd.DataFrame,
    *,
    horizon_days: int,
) -> pd.DataFrame:
    """공통 관찰 기간의 평가 범위와 기간 내 재구매 결과를 요약합니다."""
    marked_rows = add_validation_event_within_horizon(
        rows,
        horizon_days=horizon_days,
    )
    eligible_rows = marked_rows.loc[marked_rows["common_followup_eligible"]]

    validation_sample_count = len(marked_rows)
    eligible_sample_count = len(eligible_rows)
    ineligible_sample_count = validation_sample_count - eligible_sample_count
    event_rows = eligible_rows.loc[eligible_rows["event_within_horizon"].eq(True)]
    event_count = len(event_rows)
    no_event_count = eligible_sample_count - event_count
    event_rate = (
        event_count / eligible_sample_count if eligible_sample_count > 0 else pd.NA
    )
    event_durations = event_rows["target_duration_days"]
    if event_durations.empty:
        event_duration_mean_days = pd.NA
        event_duration_q25_days = pd.NA
        event_duration_median_days = pd.NA
        event_duration_q75_days = pd.NA
    else:
        event_duration_mean_days = float(event_durations.mean())
        event_duration_q25_days = float(event_durations.quantile(0.25))
        event_duration_median_days = float(event_durations.median())
        event_duration_q75_days = float(event_durations.quantile(0.75))

    summary = pd.DataFrame(
        [
            {
                "horizon_days": horizon_days,
                "validation_sample_count": validation_sample_count,
                "eligible_sample_count": eligible_sample_count,
                "ineligible_sample_count": ineligible_sample_count,
                "eligible_rate": eligible_sample_count / validation_sample_count,
                "event_within_horizon_count": event_count,
                "no_event_within_horizon_count": no_event_count,
                "event_within_horizon_rate": event_rate,
                "event_duration_mean_days": event_duration_mean_days,
                "event_duration_q25_days": event_duration_q25_days,
                "event_duration_median_days": event_duration_median_days,
                "event_duration_q75_days": event_duration_q75_days,
            }
        ]
    )
    nullable_float_columns = [
        "event_within_horizon_rate",
        "event_duration_mean_days",
        "event_duration_q25_days",
        "event_duration_median_days",
        "event_duration_q75_days",
    ]
    summary[nullable_float_columns] = summary[nullable_float_columns].astype("Float64")
    return summary


def compare_validation_common_followup_candidates(
    rows: pd.DataFrame,
    *,
    horizon_days_candidates: Sequence[int],
) -> pd.DataFrame:
    """여러 공통 관찰 기간 후보의 표본 범위와 재구매 결과를 비교합니다."""
    candidates = _validate_common_followup_candidates(horizon_days_candidates)

    summaries = [
        summarize_validation_common_followup_outcomes(
            rows,
            horizon_days=horizon_days,
        )
        for horizon_days in candidates
    ]
    return pd.concat(summaries, ignore_index=True).sort_values(
        "horizon_days", ignore_index=True
    )


def summarize_validation_common_followup_by_anchor_month(
    rows: pd.DataFrame,
    *,
    horizon_days: int,
) -> pd.DataFrame:
    """공통 관찰 기간 적용 전후의 Validation 구매 월 구성을 비교합니다."""
    marked_rows = add_validation_common_followup_eligibility(
        rows,
        horizon_days=horizon_days,
    )
    marked_rows["anchor_month"] = marked_rows["anchor_at"].dt.to_period("M").astype(str)

    summary = (
        marked_rows.groupby("anchor_month", observed=True, sort=True)
        .agg(
            validation_sample_count=("common_followup_eligible", "size"),
            eligible_sample_count=("common_followup_eligible", "sum"),
        )
        .reset_index()
    )
    summary["eligible_sample_count"] = summary["eligible_sample_count"].astype("int64")
    summary["ineligible_sample_count"] = (
        summary["validation_sample_count"] - summary["eligible_sample_count"]
    )
    summary["eligible_rate"] = summary["eligible_sample_count"].div(
        summary["validation_sample_count"]
    )

    validation_total = int(summary["validation_sample_count"].sum())
    eligible_total = int(summary["eligible_sample_count"].sum())
    summary["validation_sample_share"] = summary["validation_sample_count"].div(
        validation_total
    )
    summary["eligible_sample_share"] = summary["eligible_sample_count"].div(
        eligible_total
    )
    summary["eligible_share_shift"] = (
        summary["eligible_sample_share"] - summary["validation_sample_share"]
    )
    summary.insert(0, "horizon_days", horizon_days)

    nullable_float_columns = [
        "eligible_rate",
        "validation_sample_share",
        "eligible_sample_share",
        "eligible_share_shift",
    ]
    summary[nullable_float_columns] = summary[nullable_float_columns].astype("Float64")
    return summary


def compare_validation_common_followup_monthly_composition(
    rows: pd.DataFrame,
    *,
    horizon_days_candidates: Sequence[int],
) -> pd.DataFrame:
    """후보별 공통 관찰 기간 적용 전후의 구매 월 구성을 비교합니다."""
    candidates = _validate_common_followup_candidates(horizon_days_candidates)
    summaries = [
        summarize_validation_common_followup_by_anchor_month(
            rows,
            horizon_days=horizon_days,
        )
        for horizon_days in candidates
    ]
    return pd.concat(summaries, ignore_index=True).sort_values(
        ["horizon_days", "anchor_month"],
        ignore_index=True,
    )


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
