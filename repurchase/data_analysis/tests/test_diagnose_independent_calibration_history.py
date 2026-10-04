"""History diagnostics must compare candidates on identical rows and weights."""

import pandas as pd
import pytest

from scripts.diagnose_independent_calibration_history import (
    passes_low_history_guardrail,
    summarize_history_groups,
)


def _rows() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return tiny known-outcome groups with nonuniform IPCW weights."""
    weighted = pd.DataFrame(
        {
            "history_interval_count": [0, 0, 2, 3],
            "ipcw_outcome_known": [True, True, True, False],
            "ipcw_weight": [1.0, 3.0, 2.0, 0.0],
            "ipcw_event_within_horizon": [0.0, 1.0, 1.0, 0.0],
        },
        index=[8, 2, 5, 4],
    )
    raw = pd.Series([0.2, 0.4, 0.5, 0.1], index=weighted.index)
    calibrated = pd.Series([0.3, 0.6, 0.7, 0.2], index=weighted.index)
    return weighted, raw, calibrated


def test_history_groups_preserve_weighted_pairing_and_known_counts() -> None:
    """Use the same known rows and IPCW weights for both candidates."""
    rows, raw, calibrated = _rows()
    result = summarize_history_groups(rows, raw, calibrated)
    assert [row["history_interval_count"] for row in result] == ["0", "1-2"]
    assert result[0]["at_risk_count"] == 2
    assert result[0]["outcome_known_count"] == 2
    assert result[0]["weighted_event_rate"] == pytest.approx(0.75)
    assert result[0]["raw_ipcw_brier"] == pytest.approx(0.28)
    assert result[0]["isotonic_ipcw_brier"] == pytest.approx(0.1425)


def test_history_groups_reject_row_reordering() -> None:
    """A shuffled probability vector cannot silently change score pairing."""
    rows, raw, calibrated = _rows()
    with pytest.raises(ValueError, match="행 순서"):
        summarize_history_groups(rows, raw.iloc[::-1], calibrated)


def test_history_groups_reject_missing_history_count() -> None:
    """Missing group labels must not disappear from the diagnostic."""
    rows, raw, calibrated = _rows()
    rows.loc[8, "history_interval_count"] = pd.NA
    with pytest.raises(ValueError, match="누락"):
        summarize_history_groups(rows, raw, calibrated)


def test_low_history_guardrail_requires_both_groups_without_worsening() -> None:
    """Overall improvement cannot hide an absent or worsening low-history group."""
    groups = [
        {
            "history_interval_count": group,
            "raw_ipcw_brier": 0.2,
            "isotonic_ipcw_brier": 0.1,
        }
        for group in ("0", "1-2")
    ]
    assert passes_low_history_guardrail(groups)
    assert not passes_low_history_guardrail(groups[:1])
    assert not passes_low_history_guardrail([groups[0], groups[0], groups[1]])
    groups[0]["isotonic_ipcw_brier"] = 0.21
    assert not passes_low_history_guardrail(groups)
