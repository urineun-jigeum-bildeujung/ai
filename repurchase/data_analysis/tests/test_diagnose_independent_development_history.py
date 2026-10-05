"""Development history diagnostics must not accept final or altered sources."""

import json

import pytest

from scripts import diagnose_independent_development_history as diagnostic
from scripts.run_service_model_comparison import _file_sha256


def _development(tmp_path, monkeypatch):
    directory = tmp_path / "calibration_development"
    directory.mkdir()
    (directory / "evaluation-plan.json").write_text(
        json.dumps({"dataset_role": "calibration_development"}), encoding="utf-8"
    )
    for name in diagnostic.SOURCE_NAMES:
        (directory / f"{name}.csv").write_text("id\n", encoding="utf-8")
    monkeypatch.setattr(diagnostic, "verify_snapshot", lambda _: None)
    monkeypatch.setattr(diagnostic, "model_code_sha256", lambda: {"model.py": "a"})
    result = {
        "test_evaluated": False,
        "source_sha256": {
            name: _file_sha256(directory / f"{name}.csv")
            for name in diagnostic.SOURCE_NAMES
        },
        "code_sha256": {"model.py": "a"},
    }
    return directory, result


def test_only_unchanged_development_sources_are_accepted(tmp_path, monkeypatch):
    directory, result = _development(tmp_path, monkeypatch)
    diagnostic.validate_development(result, directory)

    (directory / "orders.csv").write_text("id\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="원천 파일 지문"):
        diagnostic.validate_development(result, directory)


def test_final_role_and_previously_evaluated_result_are_rejected(tmp_path, monkeypatch):
    directory, result = _development(tmp_path, monkeypatch)
    (directory / "evaluation-plan.json").write_text(
        json.dumps({"dataset_role": "final_evaluation"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="개발용 원천"):
        diagnostic.validate_development(result, directory)

    (directory / "evaluation-plan.json").write_text(
        json.dumps({"dataset_role": "calibration_development"}), encoding="utf-8"
    )
    result["test_evaluated"] = True
    with pytest.raises(ValueError, match="최종 평가"):
        diagnostic.validate_development(result, directory)


def test_model_code_changes_are_rejected(tmp_path, monkeypatch):
    directory, result = _development(tmp_path, monkeypatch)
    result["code_sha256"]["model.py"] = "changed"
    with pytest.raises(ValueError, match="모델 코드"):
        diagnostic.validate_development(result, directory)


@pytest.mark.parametrize("code_sha256", [None, [], "invalid", 42])
def test_invalid_model_code_fingerprint_is_rejected(tmp_path, monkeypatch, code_sha256):
    directory, result = _development(tmp_path, monkeypatch)
    result["code_sha256"] = code_sha256
    with pytest.raises(ValueError, match="코드 지문이 누락됐거나 올바르지"):
        diagnostic.validate_development(result, directory)


def test_missing_model_code_fingerprint_is_rejected(tmp_path, monkeypatch):
    directory, result = _development(tmp_path, monkeypatch)
    del result["code_sha256"]
    with pytest.raises(ValueError, match="코드 지문이 누락됐거나 올바르지"):
        diagnostic.validate_development(result, directory)


def _landmark(day, *, raw=0.2, candidate=0.19):
    return {
        "elapsed_days": day,
        "segments": {
            "outer_validation": [
                {
                    "history_interval_count": group,
                    "outcome_known_count": 10,
                    "raw_ipcw_brier": raw,
                    "isotonic_ipcw_brier": candidate,
                }
                for group in ("0", "1-2")
            ]
        },
    }


def test_low_history_screen_requires_all_landmarks_and_both_groups():
    landmarks = [_landmark(day) for day in (0, 7, 14, 30)]
    screen = diagnostic.summarize_low_history_screen(landmarks, [0, 7, 14, 30])
    assert screen["passed"] is True
    assert len(screen["groups"]) == 8

    landmarks[1]["segments"]["outer_validation"][0]["isotonic_ipcw_brier"] = 0.21
    screen = diagnostic.summarize_low_history_screen(landmarks, [0, 7, 14, 30])
    assert screen["passed"] is False
    assert screen["groups"][2]["point_brier_improvement"] == pytest.approx(-0.01)


def test_low_history_screen_rejects_missing_or_malformed_evidence():
    with pytest.raises(ValueError, match="경과 시점"):
        diagnostic.summarize_low_history_screen([_landmark(0)], [0, 7])
    incomplete = _landmark(0)
    incomplete["segments"]["outer_validation"].pop()
    with pytest.raises(ValueError, match="결과 확인 표본"):
        diagnostic.summarize_low_history_screen([incomplete], [0])
    invalid = _landmark(0, candidate=float("nan"))
    with pytest.raises(ValueError, match="Brier"):
        diagnostic.summarize_low_history_screen([invalid], [0])
