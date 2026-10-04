"""Post-hoc reporting must not turn synthetic holdout metrics into approval."""

import copy

import pytest

from scripts.summarize_independent_synthetic_final import summarize


def _result() -> dict:
    rows = []
    pairs = []
    for day in (0, 7, 14, 30):
        for candidate, brier in (("raw", 0.2), ("isotonic", 0.1 if day else 0.3)):
            rows.append(
                {
                    "elapsed_days": day,
                    "candidate": candidate,
                    "outcome_known_count": 10,
                    "ipcw_brier_score": brier,
                    "expected_calibration_error": 0.05,
                }
            )
        pairs.append(
            {
                "elapsed_days": day,
                "outcome_known_count": 10,
                "point_brier_improvement": -0.1 if day == 0 else 0.1,
                "bootstrap_lower_95_brier_improvement": -0.2 if day == 0 else 0.01,
                "bootstrap_upper_95_brier_improvement": 0.01 if day == 0 else 0.2,
            }
        )
    subgroups = []
    for group in ("0", "1-2"):
        for candidate, brier in (("raw", 0.2), ("isotonic", 0.3)):
            subgroups.append(
                {
                    "elapsed_days": 0,
                    "feature": "history_interval_count",
                    "group": group,
                    "candidate": candidate,
                    "outcome_known_count": 5,
                    "ipcw_brier_score": brier,
                }
            )
    return {
        "status": "synthetic_final_evaluation_not_real_user_approval",
        "test_evaluated": True,
        "operational_probability_publication_approved": False,
        "freeze_sha256": "freeze",
        "development_result_sha256": "development",
        "summary": rows,
        "paired_bootstrap_summary": pairs,
        "low_history_subgroups": subgroups,
    }


def test_summary_preserves_mixed_outcome_and_no_go() -> None:
    result = summarize(_result())
    assert result["uniform_brier_improvement"] is False
    assert result["operational_probability_publication_approved"] is False
    assert result["landmarks"][0]["point_brier_improvement"] == pytest.approx(-0.1)
    assert len(result["elapsed_zero_low_history"]) == 2


@pytest.mark.parametrize(
    "change", ["approval", "missing_landmark", "mismatched_pair", "missing_subgroup"]
)
def test_summary_rejects_incomplete_or_unsafe_result(change: str) -> None:
    result = copy.deepcopy(_result())
    if change == "approval":
        result["operational_probability_publication_approved"] = True
    elif change == "missing_landmark":
        result["summary"].pop()
    elif change == "mismatched_pair":
        result["paired_bootstrap_summary"][0]["point_brier_improvement"] = 0.1
    else:
        result["low_history_subgroups"].pop()
    with pytest.raises(ValueError):
        summarize(result)
