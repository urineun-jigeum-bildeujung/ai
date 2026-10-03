"""시점별 확률 보정 오차 진단의 입력 계약과 가중 요약을 검증합니다."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import summarize_landmark_calibration as runner
from scripts.summarize_landmark_calibration import summarize_landmark_calibration


def _report() -> dict:
    return {
        "test_evaluated": False,
        "conditional_aft_landmarks": {
            "test_evaluated": False,
            "landmark_days": [7],
            "summary": [
                {
                    "elapsed_days": 7,
                    "at_risk_count": 4,
                    "outcome_known_count": 3,
                    "expected_calibration_error": 0.12,
                }
            ],
            "calibration": [
                {
                    "elapsed_days": 7,
                    "calibration_bin_index": 0,
                    "sample_count": 2,
                    "ipcw_weight_share": 0.6,
                    "mean_predicted_probability": 0.1,
                    "observed_event_rate": 0.3,
                    "calibration_gap": -0.2,
                },
                {
                    "elapsed_days": 7,
                    "calibration_bin_index": 1,
                    "sample_count": 1,
                    "ipcw_weight_share": 0.4,
                    "mean_predicted_probability": 0.25,
                    "observed_event_rate": 0.25,
                    "calibration_gap": 0.0,
                },
            ],
        },
    }


def test_weighted_direction_and_support_are_reported() -> None:
    result = summarize_landmark_calibration(_report())

    assert len(result) == 1
    assert result[0]["outcome_known_count"] == 3
    assert result[0]["weighted_mean_predicted_probability"] == pytest.approx(0.16)
    assert result[0]["weighted_observed_event_rate"] == pytest.approx(0.28)
    assert result[0]["weighted_probability_gap"] == pytest.approx(-0.12)
    assert result[0]["underprediction_ipcw_weight_share"] == pytest.approx(0.6)
    assert result[0]["expected_calibration_error"] == pytest.approx(0.12)
    assert result[0]["bins"][1]["sample_count"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.update(test_evaluated=True),
        lambda r: r["conditional_aft_landmarks"]["summary"][0].update(
            outcome_known_count=2
        ),
        lambda r: r["conditional_aft_landmarks"]["summary"][0].update(
            expected_calibration_error=0.02
        ),
        lambda r: r["conditional_aft_landmarks"]["calibration"][1].update(
            ipcw_weight_share=0.2
        ),
        lambda r: r["conditional_aft_landmarks"]["calibration"][1].update(
            calibration_bin_index=0
        ),
        lambda r: r["conditional_aft_landmarks"]["calibration"][1].update(
            observed_event_rate=float("nan")
        ),
        lambda r: r["conditional_aft_landmarks"]["calibration"][1].update(
            calibration_gap=-0.1
        ),
        lambda r: r["conditional_aft_landmarks"].update(landmark_days=[0, 7]),
        lambda r: r["conditional_aft_landmarks"].update(landmark_days=[7, 7]),
    ],
)
def test_inconsistent_or_test_results_are_rejected(mutation) -> None:
    report = deepcopy(_report())
    mutation(report)

    with pytest.raises(ValueError):
        summarize_landmark_calibration(report)


@pytest.mark.parametrize("same_file_kind", ["same_path", "hard_link", "symbolic_link"])
def test_cli_never_overwrites_its_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    same_file_kind: str,
) -> None:
    source = tmp_path / "validation.json"
    original = json.dumps(_report())
    source.write_text(original, encoding="utf-8")
    output = tmp_path / "diagnostics.json"
    if same_file_kind == "same_path":
        output = source
    elif same_file_kind == "hard_link":
        output.hardlink_to(source)
    else:
        output.symlink_to(source)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_landmark_calibration",
            "--input",
            str(source),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        runner.main()

    assert exc.value.code == 2
    assert source.read_text(encoding="utf-8") == original
