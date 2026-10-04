"""Validation/Test 모집단 변화의 읽기 전용 집계 계약을 검증합니다."""

import pandas as pd
import pytest

from scripts.diagnose_service_population_shift import (
    decompose_observed_rate_change,
    summarize_population,
)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["1", "1", "2"],
            "target_id": ["a", "b", "a"],
            "history_interval_count": [0, 1, 2],
            "user_prior_order_count": [0, 2, 4],
            "ipcw_outcome_known": [True, True, False],
            "ipcw_event_within_horizon": [False, True, False],
            "ipcw_weight": [1.0, 2.0, 0.0],
        }
    )


def test_summarize_population_separates_all_rows_from_known_outcomes() -> None:
    result = summarize_population(_rows())
    assert result["sample_count"] == 3
    assert result["outcome_known_count"] == 2
    assert result["weighted_observed_rate"] == pytest.approx(2 / 3)
    assert result["user_count"] == 2
    history = result["segments"]["history_interval_count"]
    assert [item["sample_count"] for item in history] == [1, 1, 1]
    assert history[2]["weighted_observed_rate"] is None
    assert sum(item["ipcw_weight_share"] for item in history) == pytest.approx(1)


@pytest.mark.parametrize(
    ("field", "value"),
    [("history_interval_count", -1), ("user_prior_order_count", 1.5)],
)
def test_summarize_population_rejects_invalid_counts(field: str, value: float) -> None:
    rows = _rows()
    rows[field] = rows[field].astype("float64")
    rows.loc[0, field] = value
    with pytest.raises(ValueError, match="0 이상의 정수"):
        summarize_population(rows)


def test_decompose_observed_rate_change_reconciles_total() -> None:
    before = {
        "weighted_observed_rate": 0.2,
        "segments": {
            "history_interval_count": [
                {
                    "bucket": "0",
                    "ipcw_weight_share": 0.5,
                    "weighted_observed_rate": 0.1,
                },
                {
                    "bucket": "1",
                    "ipcw_weight_share": 0.5,
                    "weighted_observed_rate": 0.3,
                },
            ]
        },
    }
    after = {
        "weighted_observed_rate": 0.45,
        "segments": {
            "history_interval_count": [
                {
                    "bucket": "0",
                    "ipcw_weight_share": 0.25,
                    "weighted_observed_rate": 0.3,
                },
                {
                    "bucket": "1",
                    "ipcw_weight_share": 0.75,
                    "weighted_observed_rate": 0.5,
                },
            ]
        },
    }
    result = decompose_observed_rate_change(before, after, "history_interval_count")
    assert result["composition_component"] == pytest.approx(0.05)
    assert result["within_bucket_component"] == pytest.approx(0.2)
    assert result["total_rate_change"] == pytest.approx(0.25)
