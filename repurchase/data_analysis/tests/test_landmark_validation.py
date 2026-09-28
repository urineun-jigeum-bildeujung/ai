"""구매 후 조건부 확률의 시점별 평가 위험집단을 검증합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.landmark_validation import (
    LandmarkValidationError,
    build_split_landmark_cohort,
    build_validation_landmark_cohort,
)
from scripts.modeling.maturity_analysis import add_validation_ipcw_weights
from scripts.preprocessing.labels import build_same_product_repurchase_labels
from scripts.run_uci_conditional_validation import (
    evaluate_conditional_landmarks,
    render_conditional_landmarks,
)
from scripts.run_uci_xgboost_aft import prepare_xgboost_aft_experiment


def _validation_rows() -> pd.DataFrame:
    """landmark 전 사건·경계 사건·검열·미래 사건을 포함한 표본을 만듭니다."""
    anchors = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-01-28T00:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
        ]
    )
    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(6)],
            "order_id": [f"o{i}" for i in range(6)],
            "product_id": ["p1"] * 6,
            "anchor_at": anchors,
            "split_end_at": pd.Timestamp("2026-01-31T00:00:00Z"),
            "split": "validation",
            "outcome_available_by_split_end": pd.array(
                [True, True, True, False, False, False], dtype="boolean"
            ),
            "target_duration_days": [5.0, 7.0, 10.0, None, None, 50.0],
            "next_same_product_at": pd.to_datetime(
                [
                    "2026-01-06T00:00:00Z",
                    "2026-01-08T00:00:00Z",
                    "2026-01-11T00:00:00Z",
                    None,
                    None,
                    "2026-02-20T00:00:00Z",
                ]
            ),
            "history_interval_count": [1, 2, 3, 4, 5, 6],
            "history_median_days": [10.0] * 6,
            "history_relative_mad": [0.1] * 6,
            "user_prior_order_count": [1, 2, 3, 4, 5, 6],
        },
        index=[51, 31, 81, 22, 99, 74],
    )


def test_landmark_keeps_only_observed_at_risk_and_hides_future_outcome() -> None:
    """7일까지 미구매·관찰 중인 표본만 남기고 분할 뒤 정답은 가립니다."""
    source = _validation_rows()
    original = source.copy(deep=True)

    result = build_validation_landmark_cohort(source, elapsed_days=7)
    rows = result.rows

    assert result.source_sample_count == 6
    assert result.excluded_prior_event_count == 2
    assert result.excluded_prior_censor_count == 1
    assert rows.index.tolist() == [81, 99, 74]
    assert rows.loc[81, "purchase_anchor_at"] == pd.Timestamp("2026-01-01T00:00:00Z")
    assert rows.loc[81, "anchor_at"] == pd.Timestamp("2026-01-08T00:00:00Z")
    assert rows.loc[81, "target_duration_days"] == 3.0
    assert pd.isna(rows.loc[74, "target_duration_days"])
    assert "next_same_product_at" not in rows.columns
    assert rows.loc[81, "history_interval_count"] == 3
    assert (
        len(rows)
        + result.excluded_prior_event_count
        + result.excluded_prior_censor_count
        == 6
    )
    pd.testing.assert_frame_equal(source, original)


def test_landmark_ipcw_uses_remaining_followup_and_unknown_outcome() -> None:
    """landmark 뒤 10일 사건과 30일까지의 검열을 기존 IPCW 규칙으로 평가합니다."""
    cohort = build_validation_landmark_cohort(_validation_rows(), elapsed_days=7)
    weighted = add_validation_ipcw_weights(cohort.rows, horizon_days=10)

    assert weighted.loc[81, "survival_observed_duration_days"] == 3.0
    assert bool(weighted.loc[81, "ipcw_event_within_horizon"]) is True
    assert weighted.loc[99, "survival_observed_duration_days"] == 23.0
    assert bool(weighted.loc[99, "ipcw_event_within_horizon"]) is False
    assert bool(weighted.loc[74, "ipcw_outcome_known"]) is True


def test_at_purchase_excludes_same_instant_event_and_no_followup() -> None:
    """구매 직후에도 0일 사건과 관찰 시간이 없는 행은 위험집단이 아닙니다."""
    rows = _validation_rows()
    rows.loc[51, "target_duration_days"] = 0.0
    rows.loc[51, "next_same_product_at"] = rows.loc[51, "anchor_at"]
    rows.loc[22, "anchor_at"] = rows.loc[22, "split_end_at"]

    cohort = build_validation_landmark_cohort(rows, elapsed_days=0)

    assert cohort.excluded_prior_event_count == 1
    assert cohort.excluded_prior_censor_count == 1


@pytest.mark.parametrize("elapsed", [-1, True, 1.5])
def test_invalid_elapsed_time_is_rejected(elapsed: object) -> None:
    """의미 없는 경과 기간을 자동으로 반올림하거나 정수로 바꾸지 않습니다."""
    with pytest.raises(LandmarkValidationError, match="경과 시점"):
        build_validation_landmark_cohort(_validation_rows(), elapsed_days=elapsed)


def test_non_validation_split_is_rejected() -> None:
    """봉인한 Test나 Train을 Validation 결과에 섞지 않습니다."""
    rows = _validation_rows()
    rows.loc[99, "split"] = "test"

    with pytest.raises(LandmarkValidationError, match="validation 표본만"):
        build_validation_landmark_cohort(rows, elapsed_days=7)


def test_train_landmark_uses_train_cutoff_without_test_rows() -> None:
    """학습용 기준 확률은 동일한 규칙으로 Train에서만 계산합니다."""
    rows = _validation_rows()
    rows["split"] = "train"

    cohort = build_split_landmark_cohort(rows, elapsed_days=7, split_name="train")

    assert cohort.rows["split"].eq("train").all()
    assert cohort.rows.index.tolist() == [81, 99, 74]


def test_landmark_refuses_test_even_with_explicit_split_name() -> None:
    """실수로 봉인된 Test를 평가에 투입하는 경로를 허용하지 않습니다."""
    rows = _validation_rows()
    rows["split"] = "test"

    with pytest.raises(LandmarkValidationError, match="Train 또는 Validation"):
        build_split_landmark_cohort(rows, elapsed_days=7, split_name="test")


def test_conditional_e2e_uses_selected_model_and_validation_only(
    uci_e2e_purchase_events: pd.DataFrame,
) -> None:
    """작은 구매 사건에서 선택된 AFT·Train 기준선·Validation 평가를 연결합니다."""
    events = uci_e2e_purchase_events
    labels = build_same_product_repurchase_labels(
        events, observation_end_at=pd.Timestamp(events["ordered_at"].max())
    )
    prepared = prepare_xgboost_aft_experiment(labels)
    validation_before = prepared.validation.copy(deep=True)

    report = evaluate_conditional_landmarks(prepared, landmark_days=(0, 7))
    landmarks = report["landmarks"]

    assert report["split"] == "validation"
    assert report["model"] == {
        "distribution": "normal",
        "scale": 1.0,
        "boost_rounds": 20,
    }
    assert [item["elapsed_days"] for item in landmarks] == [0, 7]
    for item in landmarks:
        assert (
            item["at_risk_count"]
            + item["excluded_prior_event_count"]
            + item["excluded_prior_censor_count"]
            == item["source_validation_count"]
        )
        assert (
            item["outcome_unknown_count"] + item["brier"]["outcome_known_count"]
            == item["at_risk_count"]
        )
        assert 0 <= item["brier"]["ipcw_brier_score"] <= 1
        assert 0 <= item["expected_calibration_error"] <= 1
    assert "Test는 평가하지 않았습니다" in render_conditional_landmarks(report)
    pd.testing.assert_frame_equal(prepared.validation, validation_before)
