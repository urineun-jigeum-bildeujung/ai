"""봉인 Test의 사후 구간 오차 집계가 정답·가중치 경계를 지키는지 확인합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.diagnose_frozen_service_test import summarize_segment_error


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "target_id": ["g1", "g1", "g2"],
            "history_interval_count": [0, 2, 1],
            "user_prior_order_count": [0, 4, 1],
            "ipcw_outcome_known": [True, True, False],
            "ipcw_event_within_horizon": [False, True, False],
            "ipcw_weight": [1.0, 3.0, 0.0],
            "reference_predicted_event_probability": [0.2, 0.4, 0.5],
            "candidate_predicted_event_probability": [0.1, 0.7, 0.6],
        }
    )


def test_segment_diagnostics_use_only_known_rows_and_global_weight() -> None:
    result = summarize_segment_error(_rows())
    history = [row for row in result if row["segment_kind"] == "history_interval_count"]

    assert [row["segment_value"] for row in history] == ["0", "2+"]
    assert sum(row["outcome_known_count"] for row in history) == 2
    assert history[0]["ipcw_weight_share"] == pytest.approx(0.25)
    assert history[1]["weighted_observed_rate"] == pytest.approx(1.0)
    assert history[1]["weighted_aft_probability"] == pytest.approx(0.4)
    assert history[1]["weighted_lightgbm_probability"] == pytest.approx(0.7)
    assert sum(row["global_brier_difference_contribution"] for row in history) == (
        pytest.approx((0.04 + 3 * 0.36 - 0.01 - 3 * 0.09) / 4)
    )


@pytest.mark.parametrize("invalid", [float("nan"), -0.1, 1.1])
def test_segment_diagnostics_reject_invalid_known_probability(invalid: float) -> None:
    rows = _rows()
    rows.loc[1, "reference_predicted_event_probability"] = invalid

    with pytest.raises(ValueError, match="유효하지 않습니다"):
        summarize_segment_error(rows)


def test_segment_diagnostics_require_contract_columns() -> None:
    with pytest.raises(ValueError, match="필요한 평가 열"):
        summarize_segment_error(_rows().drop(columns="ipcw_weight"))
