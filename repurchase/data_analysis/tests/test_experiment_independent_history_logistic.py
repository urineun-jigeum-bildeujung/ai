"""Development candidate selection is complete and never reads final labels."""

import json

import pandas as pd
import pytest

from scripts.experiment_independent_history_logistic import (
    _development_screen,
    _group_scores,
    experiment,
)


def _landmark(day, *, low_candidate=0.18):
    return {
        "elapsed_days": day,
        "summary": {
            "raw": {"ipcw_brier_score": 0.2},
            "history_logistic": {"ipcw_brier_score": 0.19},
        },
        "history_groups": [
            {
                "history_interval_count": label,
                "outcome_known_count": 10,
                "raw_ipcw_brier": 0.2,
                "history_logistic_ipcw_brier": low_candidate,
            }
            for label in ("0", "1-2")
        ],
    }


def test_group_scores_use_identical_known_rows_and_weights():
    weighted = pd.DataFrame(
        {
            "history_interval_count": [0, 0, 2, 3],
            "ipcw_outcome_known": [True, True, False, True],
            "ipcw_weight": [1.0, 3.0, 0.0, 2.0],
            "ipcw_event_within_horizon": [0, 1, 0, 1],
            "user_id": [1, 2, 3, 4],
        }
    )
    raw = pd.Series([0.2, 0.4, 0.3, 0.8])
    candidate = pd.Series([0.1, 0.7, 0.5, 0.9])
    scores = _group_scores(weighted, {"raw": raw, "history_logistic": candidate})
    assert scores[0]["outcome_known_count"] == 2
    assert scores[0]["weighted_event_rate"] == pytest.approx(0.75)
    assert scores[0]["raw_ipcw_brier"] == pytest.approx((0.2**2 + 3 * 0.6**2) / 4)
    assert scores[0]["history_logistic_ipcw_brier"] == pytest.approx(
        (0.1**2 + 3 * 0.3**2) / 4
    )


def test_development_screen_checks_whole_and_both_low_groups():
    landmarks = [_landmark(day) for day in (0, 7, 14, 30)]
    assert _development_screen(landmarks)["passed"] is True
    landmarks[1]["history_groups"][0]["history_logistic_ipcw_brier"] = 0.21
    assert _development_screen(landmarks)["passed"] is False


def test_development_screen_rejects_missing_low_history_group():
    landmark = _landmark(0)
    landmark["history_groups"].pop()
    with pytest.raises(ValueError, match="저이력 평가 표본"):
        _development_screen([landmark])


@pytest.mark.parametrize("days", [[0, 14, 30], [7, 0, 14, 30], [0, 7, 14, 30, 60]])
def test_experiment_rejects_incomplete_landmarks_before_model_fit(
    days, monkeypatch, tmp_path
):
    result_path = tmp_path / "development.json"
    result_path.write_text(json.dumps({"landmark_days": days}), encoding="utf-8")
    monkeypatch.setattr(
        "scripts.experiment_independent_history_logistic.validate_development",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "scripts.experiment_independent_history_logistic._fit_and_reproduce_development",
        lambda *args: pytest.fail("모델을 학습하면 안 됩니다."),
    )
    with pytest.raises(ValueError, match="경과 시점"):
        experiment(result_path, tmp_path)
