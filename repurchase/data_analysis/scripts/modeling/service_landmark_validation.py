"""서비스 구매 후 경과 시점의 AFT 조건부 평가 위험집단을 구성합니다."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import pandas as pd

from .features import (
    MINIMAL_MODEL_FEATURE_COLUMNS,
    TEMPORAL_SERVICE_FEATURE_GENERATION_VERSION,
)
from .maturity_analysis import add_split_survival_observation
from .operational_orders import OperationalOrderError


@dataclass(frozen=True)
class ServiceLandmarkCohort:
    rows: pd.DataFrame
    source_sample_count: int
    excluded_prior_event_count: int
    excluded_prior_censor_count: int


def build_service_landmark_cohort(
    split_rows: pd.DataFrame,
    *,
    elapsed_days: int,
    split_name: Literal["train", "validation"],
) -> ServiceLandmarkCohort:
    """시점까지 재구매·검열되지 않은 행만 남기고 잔여 기간을 라벨로 둡니다."""
    if (
        isinstance(elapsed_days, bool)
        or not isinstance(elapsed_days, Integral)
        or elapsed_days < 0
    ):
        raise OperationalOrderError("평가 경과 일수는 0 이상의 정수여야 합니다.")
    if split_name not in ("train", "validation"):
        raise OperationalOrderError(
            "시점별 평가에는 Train 또는 Validation만 사용합니다."
        )
    identifiers = ["user_id", "pet_id", "target_id", "order_id"]
    required = {
        *identifiers,
        "anchor_at",
        "split_end_at",
        "split",
        "outcome_available_by_split_end",
        "target_duration_days",
        "feature_generation_version",
        *MINIMAL_MODEL_FEATURE_COLUMNS,
    }
    if (
        split_rows.empty
        or not split_rows.columns.is_unique
        or not split_rows.index.is_unique
    ):
        raise OperationalOrderError(
            "시점별 서비스 표본이 비었거나 행·열 식별자가 중복됐습니다."
        )
    missing = required - set(split_rows.columns)
    if missing:
        raise OperationalOrderError(
            f"시점별 서비스 입력 열이 누락됐습니다: {sorted(missing)}"
        )
    if split_rows[["user_id", "target_id", "order_id"]].isna().any(axis=None):
        raise OperationalOrderError("시점별 서비스 표본의 필수 식별자가 비었습니다.")
    if split_rows.duplicated(subset=identifiers).any():
        raise OperationalOrderError("시점별 서비스 구매 표본 키가 중복됐습니다.")
    if not split_rows["split"].eq(split_name).all():
        raise OperationalOrderError("시점별 서비스 표본의 시간 분할이 다릅니다.")
    if (
        not split_rows["feature_generation_version"]
        .eq(TEMPORAL_SERVICE_FEATURE_GENERATION_VERSION)
        .all()
    ):
        raise OperationalOrderError("시점별 서비스 피처 버전이 다릅니다.")

    observed = add_split_survival_observation(split_rows)
    duration = observed["survival_observed_duration_days"]
    prior_event = observed["survival_event_observed"] & duration.le(elapsed_days)
    prior_censor = observed["survival_right_censored"] & duration.le(elapsed_days)
    at_risk = ~(prior_event | prior_censor)
    columns = [
        *identifiers,
        "anchor_at",
        "split_end_at",
        "split",
        "outcome_available_by_split_end",
        "target_duration_days",
        "feature_generation_version",
        *MINIMAL_MODEL_FEATURE_COLUMNS,
    ]
    cohort = observed.loc[at_risk, columns].copy()
    if cohort.empty:
        raise OperationalOrderError("해당 시점에 관찰 중인 미재구매 표본이 없습니다.")
    cohort["purchase_anchor_at"] = cohort["anchor_at"]
    cohort["anchor_at"] = cohort["anchor_at"] + pd.to_timedelta(elapsed_days, unit="D")
    cohort["target_duration_days"] = (
        cohort["target_duration_days"]
        .where(cohort["outcome_available_by_split_end"], pd.NA)
        .astype("Float64")
    )
    cohort.loc[cohort["outcome_available_by_split_end"], "target_duration_days"] = (
        cohort.loc[
            cohort["outcome_available_by_split_end"], "target_duration_days"
        ].sub(elapsed_days)
    )
    cohort["elapsed_days"] = elapsed_days
    return ServiceLandmarkCohort(
        rows=cohort,
        source_sample_count=len(split_rows),
        excluded_prior_event_count=int(prior_event.sum()),
        excluded_prior_censor_count=int(prior_censor.sum()),
    )
