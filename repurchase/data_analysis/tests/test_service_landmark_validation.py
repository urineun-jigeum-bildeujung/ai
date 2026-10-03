"""서비스 시점별 위험집단이 미래 정답을 피처로 넘기지 않는지 검증합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.operational_orders import OperationalOrderError
from scripts.modeling.service_landmark_validation import (
    build_service_landmark_cohort,
)


def _rows() -> pd.DataFrame:
    anchors = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z",
            "2026-02-21T00:00:00Z",
            "2026-02-22T00:00:00Z",
        ]
    )
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "pet_id": ["p1", None, "p3", "p4"],
            "target_id": ["g1", "g2", "g3", "g4"],
            "order_id": ["o1", "o2", "o3", "o4"],
            "anchor_at": anchors,
            "split_end_at": [pd.Timestamp("2026-03-01T00:00:00Z")] * 4,
            "split": ["validation"] * 4,
            "outcome_available_by_split_end": [True, True, False, False],
            "target_duration_days": pd.array(
                [5.0, 12.0, 100.0, pd.NA], dtype="Float64"
            ),
            "feature_generation_version": [2] * 4,
            "history_interval_count": [0, 1, 2, 3],
            "history_median_days": [pd.NA, 12.0, 15.0, 20.0],
            "history_relative_mad": [pd.NA, 0.1, 0.2, 0.3],
            "user_prior_order_count": [0, 1, 2, 3],
        },
        index=[10, 20, 30, 40],
    )


def test_service_landmark_excludes_prior_events_and_censors() -> None:
    cohort = build_service_landmark_cohort(
        _rows(), elapsed_days=7, split_name="validation"
    )

    assert cohort.source_sample_count == 4
    assert cohort.excluded_prior_event_count == 1
    assert cohort.excluded_prior_censor_count == 1
    assert cohort.rows.index.tolist() == [20, 30]
    assert cohort.rows.loc[20, "target_duration_days"] == pytest.approx(5.0)
    assert pd.isna(cohort.rows.loc[30, "target_duration_days"])
    assert cohort.rows.loc[20, "anchor_at"] == pd.Timestamp("2026-01-09T00:00:00Z")
    assert cohort.rows.loc[20, "purchase_anchor_at"] == pd.Timestamp(
        "2026-01-02T00:00:00Z"
    )
    assert cohort.rows.loc[20, "history_interval_count"] == 1
    assert "survival_observed_duration_days" not in cohort.rows


@pytest.mark.parametrize("elapsed", [True, -1, 1.5])
def test_service_landmark_rejects_invalid_elapsed_days(elapsed: object) -> None:
    with pytest.raises(OperationalOrderError, match="경과 일수"):
        build_service_landmark_cohort(
            _rows(), elapsed_days=elapsed, split_name="validation"
        )


def test_service_landmark_rejects_duplicate_purchase_key() -> None:
    rows = _rows()
    rows.loc[40, ["user_id", "pet_id", "target_id", "order_id"]] = rows.loc[
        30, ["user_id", "pet_id", "target_id", "order_id"]
    ]
    with pytest.raises(OperationalOrderError, match="키가 중복"):
        build_service_landmark_cohort(rows, elapsed_days=7, split_name="validation")
