"""Frozen synthetic evaluation must reject invalid inputs before final labels."""

import pytest

from scripts import evaluate_independent_history_logistic as evaluator


def _landmark(day, low_candidate=0.18):
    return {
        "elapsed_days": day,
        "summary": {
            "raw": {"ipcw_brier_score": 0.2},
            "history_logistic": {"ipcw_brier_score": 0.19},
        },
        "history_groups": [
            {
                "history_interval_count": label,
                "outcome_known_count": 5,
                "raw_ipcw_brier": 0.2,
                "history_logistic_ipcw_brier": low_candidate,
            }
            for label in ("0", "1-2")
        ],
    }


def test_final_screen_requires_all_frozen_landmarks():
    with pytest.raises(ValueError, match="경과일"):
        evaluator._final_point_screen([_landmark(0)])


def test_final_screen_checks_whole_and_both_low_history_groups():
    landmarks = [_landmark(day) for day in evaluator.LANDMARK_DAYS]
    assert evaluator._final_point_screen(landmarks)["passed"] is True
    landmarks[1]["history_groups"][0]["history_logistic_ipcw_brier"] = 0.21
    assert evaluator._final_point_screen(landmarks)["passed"] is False


def test_final_labels_are_not_read_if_frozen_input_fails(monkeypatch, tmp_path):
    def invalid(*args):
        raise ValueError("invalid frozen input")

    monkeypatch.setattr(evaluator, "validate_frozen_inputs", invalid)
    monkeypatch.setattr(
        evaluator,
        "_events",
        lambda *args: pytest.fail("final labels must remain unopened"),
    )
    with pytest.raises(ValueError, match="invalid frozen input"):
        evaluator.evaluate_frozen(
            freeze_path=tmp_path,
            development_result_path=tmp_path,
            development_comparison_path=tmp_path,
            development_dir=tmp_path,
            evaluation_dir=tmp_path,
            baseline_dir=tmp_path,
            baseline_test_result=tmp_path,
        )
