"""서비스 모델 비교가 동일 평가 행과 검열 가중치를 유지하는지 확인합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.operational_orders import OperationalOrderError
from scripts.modeling.operational_temporal_split import (
    ServiceTemporalSplit,
    _attach_evaluation_contract,
)
from scripts.modeling.service_model_comparison import (
    _evaluate_candidate,
    compare_service_aft_lightgbm,
)


def _weighted_validation() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u3"],
            "survival_observed_duration_days": [5.0, 10.0, 30.0, 35.0],
            "survival_event_observed": [True, True, False, False],
            "ipcw_horizon_days": [30] * 4,
            "ipcw_censoring_survival_probability": [1.0] * 4,
            "ipcw_outcome_known": [True] * 4,
            "ipcw_event_within_horizon": [True, True, False, False],
            "ipcw_weight": [1.0] * 4,
        },
        index=pd.Index([10, 20, 30, 40]),
    )


def test_candidate_reports_same_cohort_for_brier_concordance_calibration() -> None:
    rows = _weighted_validation()
    probability = pd.Series([0.9, 0.8, 0.1, 0.2], index=rows.index)

    summary, calibration = _evaluate_candidate(
        rows,
        probability,
        model_name="candidate",
        reference_probability=0.5,
        calibration_bin_count=10,
    )

    assert summary["validation_sample_count"] == 4
    assert summary["outcome_known_count"] == 4
    assert summary["ipcw_brier_score"] == pytest.approx(0.025)
    assert summary["ipcw_reference_brier_score"] == pytest.approx(0.25)
    assert summary["ipcw_c_index"] == pytest.approx(1.0)
    assert calibration["sample_count"].sum() == 4
    assert calibration["model"].eq("candidate").all()


def test_candidate_rejects_same_labels_in_different_order() -> None:
    rows = _weighted_validation()
    probability = pd.Series([0.8, 0.9, 0.1, 0.2], index=pd.Index([20, 10, 30, 40]))

    with pytest.raises(OperationalOrderError, match="행 순서"):
        _evaluate_candidate(
            rows,
            probability,
            model_name="candidate",
            reference_probability=0.5,
            calibration_bin_count=10,
        )


def _service_rows(*, split_name: str) -> pd.DataFrame:
    train = split_name == "train"
    count = 8 if train else 6
    anchors = pd.date_range(
        "2026-01-01" if train else "2026-03-02",
        periods=count,
        freq="D",
        tz="UTC",
    )
    split_end = pd.Timestamp(
        "2026-03-01T00:00:00Z" if train else "2026-05-01T00:00:00Z"
    )
    event = [True, True, True, True, False, False, False, False][:count]
    duration = [5.0, 10.0, 15.0, 20.0] + [
        float((split_end - anchor).days) for anchor in anchors[4:]
    ]
    rows = pd.DataFrame(
        {
            "user_id": [f"u{i // 2}" for i in range(count)],
            "anchor_at": anchors,
            "duration_days": duration,
            "event_observed": event,
            "is_right_censored": [not value for value in event],
            "feature_generation_version": [2] * count,
            "history_interval_count": list(range(count)),
            "history_median_days": [float(value) for value in range(count)],
            "history_relative_mad": [0.0] * count,
            "user_prior_order_count": list(range(count)),
        },
        index=pd.Index(range(100, 100 + count) if train else range(200, 200 + count)),
    )
    return _attach_evaluation_contract(
        rows, split_name=split_name, split_end_at=split_end
    )


def test_comparison_trains_both_models_and_preserves_validation_count() -> None:
    comparison = compare_service_aft_lightgbm(
        ServiceTemporalSplit(
            train=_service_rows(split_name="train"),
            validation=_service_rows(split_name="validation"),
        ),
        aft_boost_rounds=2,
        bootstrap_replicates=20,
    )

    assert comparison.summary["model"].tolist() == ["xgboost_aft", "lightgbm"]
    assert comparison.summary["validation_sample_count"].tolist() == [6, 6]
    assert comparison.summary["outcome_known_count"].tolist() == [6, 6]
    assert comparison.paired_bootstrap.summary["outcome_known_count"] == 6
    assert comparison.calibration.groupby("model")["sample_count"].sum().to_dict() == {
        "xgboost_aft": 6,
        "lightgbm": 6,
    }
