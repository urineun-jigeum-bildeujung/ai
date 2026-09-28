"""구매 후 고정 경과 시점의 재구매 조건부 확률 평가 표본을 만듭니다.

구매 직후 모델 피처는 그대로 두고, 지정 시점까지 관찰되면서 아직 같은
상품을 재구매하지 않은 사용자·상품만 위험집단에 남깁니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import pandas as pd

from .features import MINIMAL_MODEL_FEATURE_COLUMNS
from .maturity_analysis import add_split_survival_observation


class LandmarkValidationError(ValueError):
    """landmark 위험집단 입력 또는 시간 경계가 올바르지 않을 때 발생합니다."""


@dataclass(frozen=True)
class LandmarkValidationCohort:
    """평가 위험집단과 제외 사유별 표본 수를 함께 기록합니다."""

    rows: pd.DataFrame
    source_sample_count: int
    excluded_prior_event_count: int
    excluded_prior_censor_count: int


def build_split_landmark_cohort(
    split_rows: pd.DataFrame,
    *,
    elapsed_days: int,
    split_name: Literal["train", "validation"],
) -> LandmarkValidationCohort:
    """Train 또는 Validation에서 아직 관찰 중인 미재구매 행만 남깁니다.

    반환 행의 anchor_at은 구매 시각이 아닌 landmark 시각입니다. 이를 통해
    기존 IPCW 함수가 landmark 이후의 사건·검열 기간만 평가하게 합니다.
    """
    if (
        isinstance(elapsed_days, bool)
        or not isinstance(elapsed_days, Integral)
        or elapsed_days < 0
    ):
        raise LandmarkValidationError("경과 시점은 0 이상의 정수 일수여야 합니다.")
    required = {
        "user_id",
        "order_id",
        "product_id",
        "anchor_at",
        "split_end_at",
        "split",
        "outcome_available_by_split_end",
        "target_duration_days",
        *MINIMAL_MODEL_FEATURE_COLUMNS,
    }
    if split_name not in ("train", "validation"):
        raise LandmarkValidationError(
            "landmark 위험집단에는 Train 또는 Validation만 사용합니다."
        )
    if split_rows.empty:
        raise LandmarkValidationError("landmark 원본 표본이 없습니다.")
    if not split_rows.columns.is_unique:
        raise LandmarkValidationError("landmark 원본의 열 이름에 중복이 있습니다.")
    missing = required - set(split_rows.columns)
    if missing:
        raise LandmarkValidationError(
            f"landmark 입력 열이 누락됐습니다: {sorted(missing)}"
        )
    if not split_rows.index.is_unique:
        raise LandmarkValidationError("landmark 원본 행 인덱스가 중복됐습니다.")
    if split_rows[list(("user_id", "order_id", "product_id"))].isna().any(axis=None):
        raise LandmarkValidationError("landmark 표본 식별자에는 결측값이 없습니다.")
    if split_rows.duplicated(subset=["user_id", "order_id", "product_id"]).any():
        raise LandmarkValidationError("landmark 구매 표본 식별자가 중복됐습니다.")

    if split_rows["split"].isna().any() or not split_rows["split"].eq(split_name).all():
        raise LandmarkValidationError(
            f"landmark 원본에는 {split_name} 표본만 사용할 수 있습니다."
        )
    # 사건과 검열을 해당 분할 종료일까지의 관측 정보로만 판단합니다.
    observed = add_split_survival_observation(split_rows)
    duration = observed["survival_observed_duration_days"]
    prior_event = observed["survival_event_observed"] & duration.le(elapsed_days)
    prior_censor = observed["survival_right_censored"] & duration.le(elapsed_days)
    at_risk = ~(prior_event | prior_censor)
    # 원본에 다음 구매 시각이나 전체 관측 종료까지의 정답이 있어도 반환하지 않습니다.
    # 평가에 필요한 사건 여부·잔여 기간과 구매 당시 피처만 선택합니다.
    cohort = observed.loc[
        at_risk,
        [
            "user_id",
            "order_id",
            "product_id",
            "anchor_at",
            "split_end_at",
            "split",
            "outcome_available_by_split_end",
            "target_duration_days",
            *MINIMAL_MODEL_FEATURE_COLUMNS,
        ],
    ].copy()
    if cohort.empty:
        raise LandmarkValidationError("경과 시점에 관찰 중인 미재구매 표본이 없습니다.")

    # 다음 구매의 정답은 입력 피처로 넘기지 않고, 평가를 위해서만 경과 기간을 뺍니다.
    cohort["purchase_anchor_at"] = cohort["anchor_at"]
    cohort["anchor_at"] = cohort["anchor_at"] + pd.to_timedelta(elapsed_days, unit="D")
    # 분할 종료 뒤 발생한 실제 구매 간격은 원본에 있을 수 있으므로 지웁니다.
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

    return LandmarkValidationCohort(
        rows=cohort,
        source_sample_count=len(split_rows),
        excluded_prior_event_count=int(prior_event.sum()),
        excluded_prior_censor_count=int(prior_censor.sum()),
    )


def build_validation_landmark_cohort(
    validation_rows: pd.DataFrame,
    *,
    elapsed_days: int,
) -> LandmarkValidationCohort:
    """Validation 평가에 필요한 시점별 위험집단을 구성합니다."""
    return build_split_landmark_cohort(
        validation_rows, elapsed_days=elapsed_days, split_name="validation"
    )
