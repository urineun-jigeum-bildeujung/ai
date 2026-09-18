"""XGBoost AFT 학습에 사용할 생존시간 라벨 계약을 정의합니다.

공용 재구매 라벨은 그대로 보존하고, AFT의 양수 시간 조건 때문에 제외되는
표본 수와 실제 학습 대상 행을 하나의 결과 객체로 관리합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Final

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


class XGBoostAFTError(ValueError):
    """AFT 학습 입력이나 결과가 정의한 계약을 위반할 때 발생합니다."""


AFT_LABEL_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "survival_observed_duration_days",
        "survival_event_observed",
    }
)


def _validate_nonnegative_count(*, name: str, value: object) -> None:
    """표본 수가 boolean이 아닌 0 이상의 정수인지 검사합니다."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise XGBoostAFTError(f"{name}은 0 이상의 정수여야 합니다.")


@dataclass(frozen=True)
class AFTLabelBounds:
    """AFT 학습 라벨과 원본 대비 제외 통계를 함께 보관합니다."""

    rows: pd.DataFrame
    source_sample_count: int
    excluded_zero_duration_count: int

    def __post_init__(self) -> None:
        """객체 생성 직후 개별 표본 수와 합계 보존 관계를 검사합니다."""
        _validate_nonnegative_count(
            name="원본 표본 수",
            value=self.source_sample_count,
        )
        _validate_nonnegative_count(
            name="0일 제외 건수",
            value=self.excluded_zero_duration_count,
        )
        if self.excluded_zero_duration_count > self.source_sample_count:
            raise XGBoostAFTError("0일 제외 건수는 원본 표본 수보다 클 수 없습니다.")
        if len(self.rows) != self.included_sample_count:
            raise XGBoostAFTError(
                "AFT 라벨 행 수가 원본 표본 수에서 0일 제외 건수를 뺀 값과 "
                "일치하지 않습니다."
            )

    @property
    def included_sample_count(self) -> int:
        """원본 표본 수에서 0일 제외 건수를 뺀 실제 포함 건수를 계산합니다."""
        return self.source_sample_count - self.excluded_zero_duration_count


def build_aft_label_bounds(rows: pd.DataFrame) -> AFTLabelBounds:
    """공용 생존 관측값을 XGBoost AFT의 하한·상한 라벨로 변환합니다."""
    missing_columns = AFT_LABEL_REQUIRED_COLUMNS - set(rows.columns)
    if missing_columns:
        raise XGBoostAFTError(
            f"AFT 라벨 필수 컬럼이 누락됐습니다: {sorted(missing_columns)}"
        )
    if rows.empty:
        raise XGBoostAFTError("AFT 라벨을 생성할 표본이 없습니다.")

    duration = rows["survival_observed_duration_days"]
    if (
        duration.isna().any()
        or is_bool_dtype(duration.dtype)
        or not is_numeric_dtype(duration.dtype)
        or not np.isfinite(duration.to_numpy(dtype="float64")).all()
        or duration.lt(0).any()
    ):
        raise XGBoostAFTError(
            "AFT 관측 기간에는 결측·무한대·음수가 없는 숫자만 사용할 수 있습니다."
        )

    event_observed = rows["survival_event_observed"]
    if event_observed.isna().any() or not is_bool_dtype(event_observed.dtype):
        raise XGBoostAFTError(
            "AFT 사건 관측 여부에는 결측값 없는 boolean만 사용할 수 있습니다."
        )

    zero_duration = duration.eq(0)
    result = rows.loc[~zero_duration].copy()
    if result.empty:
        raise XGBoostAFTError("0일 표본을 제외한 뒤 AFT에 사용할 표본이 없습니다.")

    result["aft_lower_bound"] = result["survival_observed_duration_days"].astype(
        "float64"
    )
    result["aft_upper_bound"] = result["aft_lower_bound"]
    censored = ~result["survival_event_observed"]
    result.loc[censored, "aft_upper_bound"] = float("inf")

    return AFTLabelBounds(
        rows=result,
        source_sample_count=len(rows),
        excluded_zero_duration_count=int(zero_duration.sum()),
    )
