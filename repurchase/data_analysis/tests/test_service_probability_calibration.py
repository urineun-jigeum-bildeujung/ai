"""확률 보정 실험이 시간 분할을 넘겨 학습하지 않는지 검사합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import run_service_probability_calibration as runner
from scripts.modeling.operational_orders import OperationalOrderError
from scripts.modeling.operational_temporal_split import ServiceTemporalSplit
from scripts.modeling.service_probability_calibration import _validate_split_boundaries


def _row(anchor: str, end: str, split: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "anchor_at": [pd.Timestamp(f"{anchor}T00:00:00Z")],
            "split_end_at": [pd.Timestamp(f"{end}T00:00:00Z")],
            "split": [split],
        }
    )


def _splits() -> tuple[ServiceTemporalSplit, ServiceTemporalSplit]:
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
    inner, outer = _splits()
    assert _validate_split_boundaries(inner, outer) == (
        pd.Timestamp("2026-02-01T00:00:00Z"),
        pd.Timestamp("2026-03-01T00:00:00Z"),
        pd.Timestamp("2026-04-01T00:00:00Z"),
    )


def test_calibration_rejects_outer_validation_leakage() -> None:
    inner, outer = _splits()
    outer.validation.loc[0, "anchor_at"] = pd.Timestamp("2026-02-28T00:00:00Z")
    with pytest.raises(OperationalOrderError, match="시간 분할"):
        _validate_split_boundaries(inner, outer)


def test_calibration_rejects_mismatched_inner_outer_train_cut() -> None:
    inner, outer = _splits()
    outer.train.loc[0, "split_end_at"] = pd.Timestamp("2026-03-02T00:00:00Z")
    with pytest.raises(OperationalOrderError, match="시간 컷"):
        _validate_split_boundaries(inner, outer)


def test_calibration_cli_does_not_overwrite_result(
    tmp_path, monkeypatch, capsys
) -> None:
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
