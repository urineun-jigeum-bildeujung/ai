"""서비스 시점 복원 표본과 기존 AFT 학습 계약을 연결합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.operational_aft_input import build_service_aft_training_rows
from scripts.modeling.operational_orders import OperationalOrderError
from scripts.modeling.operational_training_samples import (
    TEMPORAL_SERVICE_FEATURE_GENERATION_VERSION,
)
from scripts.modeling.xgboost_aft import build_xgboost_aft_training_data


def _train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "duration_days": [10.0, 0.0, 4.0],
            "event_observed": [True, False, False],
            "is_right_censored": [False, True, True],
            "split": ["train"] * 3,
            "feature_generation_version": [TEMPORAL_SERVICE_FEATURE_GENERATION_VERSION]
            * 3,
            "history_interval_count": [1, 2, 3],
            "history_median_days": [10.0, 12.0, 13.0],
            "history_relative_mad": [0.1, 0.2, 0.3],
            "user_prior_order_count": [1, 2, 3],
        },
        index=[10, 11, 12],
    )


def test_service_rows_keep_zero_duration_until_aft_training_filter() -> None:
    source = _train()

    adapted = build_service_aft_training_rows(source)
    training = build_xgboost_aft_training_data(adapted)

    assert adapted.index.tolist() == [10, 11, 12]
    assert adapted["survival_observed_duration_days"].tolist() == [10.0, 0.0, 4.0]
    assert adapted["survival_event_observed"].tolist() == [True, False, False]
    assert training.row_index.tolist() == [10, 12]
    assert training.source_sample_count == 3
    assert training.excluded_zero_duration_count == 1
    assert source["split"].eq("train").all()


def test_service_aft_rejects_other_feature_version() -> None:
    source = _train()
    source.loc[11, "feature_generation_version"] = 1

    with pytest.raises(OperationalOrderError, match="피처 버전"):
        build_service_aft_training_rows(source)


def test_service_aft_rejects_premapped_labels() -> None:
    source = _train()
    source["survival_event_observed"] = source["event_observed"]

    with pytest.raises(OperationalOrderError, match="중복"):
        build_service_aft_training_rows(source)


def test_service_aft_rejects_validation_rows() -> None:
    source = _train()
    source.loc[11, "split"] = "validation"

    with pytest.raises(OperationalOrderError, match="Train"):
        build_service_aft_training_rows(source)


def test_service_aft_rejects_missing_split_provenance() -> None:
    source = _train().drop(columns="split")

    with pytest.raises(OperationalOrderError, match="필수 열 누락"):
        build_service_aft_training_rows(source)


def test_service_aft_rejects_event_censor_contradiction() -> None:
    source = _train()
    source.loc[10, "is_right_censored"] = True

    with pytest.raises(OperationalOrderError, match="모순"):
        build_service_aft_training_rows(source)
