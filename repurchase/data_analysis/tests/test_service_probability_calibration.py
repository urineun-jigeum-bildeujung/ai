"""확률 보정 실험이 시간 분할을 넘겨 학습하지 않는지 검사합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import run_service_probability_calibration as runner
from scripts.modeling.operational_orders import OperationalOrderError
from scripts.modeling.operational_temporal_split import ServiceTemporalSplit
from scripts.modeling.service_probability_calibration import _validate_split_boundaries


def _row(anchor: str, end: str, split: str) -> pd.DataFrame:
    """경계 검사용 단일 시점 표본을 만듭니다."""
    return pd.DataFrame(
        {
            "anchor_at": [pd.Timestamp(f"{anchor}T00:00:00Z")],
            "split_end_at": [pd.Timestamp(f"{end}T00:00:00Z")],
            "split": [split],
        }
    )


def _splits() -> tuple[ServiceTemporalSplit, ServiceTemporalSplit]:
    """내부 학습·보정과 외부 Validation 시간 구간을 구성합니다."""
    inner = ServiceTemporalSplit(
        train=_row("2026-01-01", "2026-02-01", "train"),
        validation=_row("2026-02-15", "2026-03-01", "validation"),
    )
    outer = ServiceTemporalSplit(
        train=_row("2026-02-01", "2026-03-01", "train"),
        validation=_row("2026-03-15", "2026-04-01", "validation"),
    )
    return inner, outer


def test_calibration_temporal_boundaries_are_strict() -> None:
    """서로 다른 세 시간 컷의 순서를 확인합니다."""
    inner, outer = _splits()
    assert _validate_split_boundaries(inner, outer) == (
        pd.Timestamp("2026-02-01T00:00:00Z"),
        pd.Timestamp("2026-03-01T00:00:00Z"),
        pd.Timestamp("2026-04-01T00:00:00Z"),
    )


def test_calibration_rejects_outer_validation_leakage() -> None:
    """외부 평가 행이 보정 구간으로 침범하면 거부합니다."""
    inner, outer = _splits()
    outer.validation.loc[0, "anchor_at"] = pd.Timestamp("2026-02-28T00:00:00Z")
    with pytest.raises(OperationalOrderError, match="시간 분할"):
        _validate_split_boundaries(inner, outer)


def test_calibration_rejects_mismatched_inner_outer_train_cut() -> None:
    """내부 보정 종료와 외부 Train 종료가 다르면 거부합니다."""
    inner, outer = _splits()
    outer.train.loc[0, "split_end_at"] = pd.Timestamp("2026-03-02T00:00:00Z")
    with pytest.raises(OperationalOrderError, match="시간 컷"):
        _validate_split_boundaries(inner, outer)


def test_calibration_cli_does_not_overwrite_result(
    tmp_path, monkeypatch, capsys
) -> None:
    """기존 실험 결과 JSON을 덮어쓰지 않습니다."""
    output = tmp_path / "result.json"
    output.write_text("existing", encoding="utf-8")
    args = ["calibration"]
    for name in ("orders", "order-items", "pets", "histories", "claims", "claim-items"):
        args.extend((f"--{name}", "unused.csv"))
    args.extend(
        (
            "--observation-end-at",
            "2026-10-03T00:00:00+09:00",
            "--output",
            str(output),
        )
    )
    monkeypatch.setattr("sys.argv", args)

    assert runner.main() == 1
    assert output.read_text(encoding="utf-8") == "existing"
    assert "FileExistsError" in capsys.readouterr().err


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf"), True])
def test_calibration_rejects_invalid_aft_scale_before_source_read(scale: float) -> None:
    """잘못된 AFT scale은 원천 파일을 열기 전에 거부합니다."""
    with pytest.raises(ValueError, match="AFT scale"):
        runner.run_calibration(
            {},
            observation_end_at=pd.Timestamp("2026-10-03T00:00:00+09:00"),
            landmark_days=(0,),
            aft_loss_distribution_scale=scale,
        )


def test_calibration_cli_passes_frozen_aft_configuration(
    tmp_path, monkeypatch, capsys
) -> None:
    """CLI의 고정 후보 설정이 실험 호출과 결과 파일에 전달됩니다."""
    output = tmp_path / "result.json"
    args = ["calibration"]
    for name in ("orders", "order-items", "pets", "histories", "claims", "claim-items"):
        args.extend((f"--{name}", "unused.csv"))
    args.extend(
        (
            "--observation-end-at",
            "2026-10-03T00:00:00+09:00",
            "--aft-boost-rounds",
            "20",
            "--aft-loss-distribution-scale",
            "2.0",
            "--output",
            str(output),
        )
    )
    monkeypatch.setattr("sys.argv", args)

    def fake_run(paths, **kwargs):
        assert kwargs["aft_boost_rounds"] == 20
        assert kwargs["aft_loss_distribution_scale"] == 2.0
        return {"aft_configuration": kwargs["aft_loss_distribution_scale"]}

    monkeypatch.setattr(runner, "run_calibration", fake_run)
    assert runner.main() == 0
    assert pd.read_json(output, typ="series")["aft_configuration"] == 2.0
    assert "repurchase_calibration_completed" in capsys.readouterr().out
