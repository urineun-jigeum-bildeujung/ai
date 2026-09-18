"""XGBoost AFT 라벨 계약과 제외 통계 계산을 검증합니다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.modeling.xgboost_aft import (
    AFTLabelBounds,
    XGBoostAFTError,
    build_aft_label_bounds,
    build_xgboost_aft_training_data,
)


def make_aft_training_rows(*, split: str = "train") -> pd.DataFrame:
    """사건·검열·0일 표본을 함께 가진 작은 AFT 학습 데이터를 만듭니다."""
    return pd.DataFrame(
        {
            "history_interval_count": [2, 0, 1],
            "history_median_days": [25.0, float("nan"), 18.0],
            "history_relative_mad": [0.2, float("nan"), float("nan")],
            "user_prior_order_count": [5, 0, 2],
            "survival_observed_duration_days": [20.0, 0.0, 30.0],
            "survival_event_observed": pd.array(
                [True, True, False],
                dtype="boolean",
            ),
            "split": [split] * 3,
        },
        index=[30, 10, 20],
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


def test_build_aft_label_bounds_rejects_complex_duration() -> None:
    """복소수 관측 기간의 허수부를 버리고 실수로 변환하지 않습니다."""
    rows = pd.DataFrame(
        {
            "survival_observed_duration_days": np.array(
                [20.0 + 1.0j],
                dtype="complex128",
            ),
            "survival_event_observed": pd.array([True], dtype="boolean"),
        }
    )

    with pytest.raises(XGBoostAFTError, match="관측 기간"):
        build_aft_label_bounds(rows)


def test_build_xgboost_aft_training_data_aligns_features_and_bounds() -> None:
    """0일 제외 후 동일한 두 행의 피처와 하한·상한이 행렬에 연결됩니다."""
    training_data = build_xgboost_aft_training_data(make_aft_training_rows())

    assert training_data.source_sample_count == 3
    assert training_data.excluded_zero_duration_count == 1
    assert training_data.included_sample_count == 2
    assert training_data.row_index.tolist() == [30, 20]
    assert training_data.feature_columns == (
        "history_interval_count",
        "history_median_days",
        "history_relative_mad",
        "user_prior_order_count",
    )
    assert training_data.matrix.num_row() == 2
    assert training_data.matrix.num_col() == 4
    feature_values = training_data.matrix.get_data().toarray()
    np.testing.assert_allclose(
        feature_values[:, [0, 1, 3]],
        np.array(
            [
                [2.0, 25.0, 5.0],
                [1.0, 18.0, 2.0],
            ]
        ),
    )
    assert training_data.matrix.get_float_info("label_lower_bound").tolist() == [
        20.0,
        30.0,
    ]
    upper_bound = training_data.matrix.get_float_info("label_upper_bound")
    assert upper_bound[0] == 20.0
    assert np.isinf(upper_bound[1])


def test_build_xgboost_aft_training_data_rejects_non_train_rows() -> None:
    """Validation이나 Test 표본이 AFT 학습 입력으로 섞이면 거절합니다."""
    with pytest.raises(XGBoostAFTError, match="Train 표본만"):
        build_xgboost_aft_training_data(make_aft_training_rows(split="validation"))


def test_build_xgboost_aft_training_data_rejects_missing_split() -> None:
    """시간 분할 정보가 없으면 학습 데이터라고 임의로 가정하지 않습니다."""
    rows = make_aft_training_rows().drop(columns="split")

    with pytest.raises(XGBoostAFTError, match="필수 컬럼"):
        build_xgboost_aft_training_data(rows)


def test_build_xgboost_aft_training_data_rejects_duplicate_row_index() -> None:
    """학습 행의 원본 인덱스가 중복되면 예측 추적이 불가능하므로 거절합니다."""
    rows = make_aft_training_rows()
    rows.index = [7, 8, 7]

    with pytest.raises(XGBoostAFTError, match="중복"):
        build_xgboost_aft_training_data(rows)
