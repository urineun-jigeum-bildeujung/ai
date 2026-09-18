"""XGBoost AFT 라벨 계약과 제외 통계 계산을 검증합니다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.modeling.xgboost_aft import (
    AFTLabelBounds,
    XGBoostAFTError,
    build_aft_label_bounds,
)


def test_aft_label_bounds_calculates_included_sample_count() -> None:
    """포함 건수를 중복 저장하지 않고 원본과 제외 건수로 계산합니다."""
    bounds = AFTLabelBounds(
        rows=pd.DataFrame(index=range(98)),
        source_sample_count=100,
        excluded_zero_duration_count=2,
    )

    assert bounds.included_sample_count == 98


def test_aft_label_bounds_rejects_inconsistent_row_count() -> None:
    """실제 행 수와 계산된 포함 건수가 다르면 조용히 진행하지 않습니다."""
    with pytest.raises(XGBoostAFTError, match="행 수"):
        AFTLabelBounds(
            rows=pd.DataFrame(index=range(97)),
            source_sample_count=100,
            excluded_zero_duration_count=2,
        )


@pytest.mark.parametrize(
    ("source_sample_count", "excluded_zero_duration_count"),
    [
        (-1, 0),
        (1, -1),
        (1.5, 0),
        (1, 0.5),
        (True, 0),
    ],
)
def test_aft_label_bounds_rejects_invalid_sample_count(
    source_sample_count: object,
    excluded_zero_duration_count: object,
) -> None:
    """음수·실수·boolean 표본 수를 정상 통계로 허용하지 않습니다."""
    with pytest.raises(XGBoostAFTError, match="0 이상의 정수"):
        AFTLabelBounds(
            rows=pd.DataFrame(index=[0]),
            source_sample_count=source_sample_count,  # type: ignore[arg-type]
            excluded_zero_duration_count=(  # type: ignore[arg-type]
                excluded_zero_duration_count
            ),
        )


def test_aft_label_bounds_rejects_excluded_count_above_source() -> None:
    """제외 건수가 원본보다 많으면 합계 관계를 계산하기 전에 거절합니다."""
    with pytest.raises(XGBoostAFTError, match="원본 표본 수보다 클 수 없습니다"):
        AFTLabelBounds(
            rows=pd.DataFrame(),
            source_sample_count=1,
            excluded_zero_duration_count=2,
        )


def test_build_aft_label_bounds_encodes_events_and_censoring() -> None:
    """0일 표본을 제외하고 사건과 우측검열을 서로 다른 상한으로 표현합니다."""
    rows = pd.DataFrame(
        {
            "survival_observed_duration_days": [20.0, 30.0, 0.0, 0.0],
            "survival_event_observed": pd.array(
                [True, False, True, False],
                dtype="boolean",
            ),
        },
        index=[10, 20, 30, 40],
    )

    bounds = build_aft_label_bounds(rows)

    assert bounds.source_sample_count == 4
    assert bounds.excluded_zero_duration_count == 2
    assert bounds.included_sample_count == 2
    assert bounds.rows.index.tolist() == [10, 20]
    assert bounds.rows["aft_lower_bound"].tolist() == [20.0, 30.0]
    assert bounds.rows["aft_upper_bound"].iloc[0] == 20.0
    assert np.isinf(bounds.rows["aft_upper_bound"].iloc[1])
    assert "aft_lower_bound" not in rows.columns
    assert "aft_upper_bound" not in rows.columns


@pytest.mark.parametrize(
    "missing_column",
    [
        "survival_observed_duration_days",
        "survival_event_observed",
    ],
)
def test_build_aft_label_bounds_rejects_missing_required_column(
    missing_column: str,
) -> None:
    """시간이나 사건 여부가 없으면 임의로 값을 추정하지 않습니다."""
    rows = pd.DataFrame(
        {
            "survival_observed_duration_days": [20.0],
            "survival_event_observed": pd.array([True], dtype="boolean"),
        }
    ).drop(columns=missing_column)

    with pytest.raises(XGBoostAFTError, match="필수 컬럼"):
        build_aft_label_bounds(rows)


def test_build_aft_label_bounds_rejects_empty_rows() -> None:
    """열만 있고 표본이 없으면 학습 가능한 라벨로 처리하지 않습니다."""
    rows = pd.DataFrame(
        {
            "survival_observed_duration_days": pd.Series(dtype="float64"),
            "survival_event_observed": pd.Series(dtype="boolean"),
        }
    )

    with pytest.raises(XGBoostAFTError, match="표본이 없습니다"):
        build_aft_label_bounds(rows)


@pytest.mark.parametrize(
    "invalid_duration",
    [
        float("nan"),
        float("inf"),
        -1.0,
        True,
        "20",
    ],
)
def test_build_aft_label_bounds_rejects_invalid_duration(
    invalid_duration: object,
) -> None:
    """결측·무한대·음수·boolean·문자열 시간을 조용히 제외하지 않습니다."""
    rows = pd.DataFrame(
        {
            "survival_observed_duration_days": [20.0, invalid_duration],
            "survival_event_observed": pd.array([True, False], dtype="boolean"),
        }
    )

    with pytest.raises(XGBoostAFTError, match="관측 기간"):
        build_aft_label_bounds(rows)


@pytest.mark.parametrize(
    "invalid_events",
    [
        pd.Series([True, pd.NA], dtype="boolean"),
        pd.Series([1, 0], dtype="int64"),
    ],
)
def test_build_aft_label_bounds_rejects_invalid_event_contract(
    invalid_events: pd.Series,
) -> None:
    """사건 여부는 결측 없는 boolean 열만 허용합니다."""
    rows = pd.DataFrame(
        {
            "survival_observed_duration_days": [20.0, 30.0],
            "survival_event_observed": invalid_events,
        }
    )

    with pytest.raises(XGBoostAFTError, match="사건 관측 여부"):
        build_aft_label_bounds(rows)


def test_build_aft_label_bounds_rejects_only_zero_duration_rows() -> None:
    """0일을 모두 제외한 뒤 남는 표본이 없으면 명확히 중단합니다."""
    rows = pd.DataFrame(
        {
            "survival_observed_duration_days": [0.0, 0.0],
            "survival_event_observed": pd.array([True, False], dtype="boolean"),
        }
    )

    with pytest.raises(XGBoostAFTError, match="0일 표본을 제외한 뒤"):
        build_aft_label_bounds(rows)
