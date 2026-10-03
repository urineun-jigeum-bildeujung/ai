"""공용 DB 조회 결과가 CSV 파일 없이 동일한 평가 경계로 전달되는지 검증합니다."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_cloud_service_model_comparison as runner
from scripts.modeling.cloud_source_reader import OrderSourceSnapshot, PetSourceSnapshot


def _snapshots() -> tuple[OrderSourceSnapshot, PetSourceSnapshot]:
    empty = pd.DataFrame()
    return (
        OrderSourceSnapshot(
            extracted_at=pd.Timestamp("2026-09-30T00:00:00Z"),
            orders=empty,
            order_items=empty,
            status_histories=empty,
            claims=empty,
            claim_items=empty,
        ),
        PetSourceSnapshot(
            extracted_at=pd.Timestamp("2026-09-30T00:01:00Z"), pets=empty
        ),
    )


def test_compare_snapshots_passes_audited_frames_without_csv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orders, pets = _snapshots()
    cutoff = pd.Timestamp("2026-09-29T06:44:00Z")
    audit_calls: list[tuple[object, ...]] = []

    # 소유 관계 점검이 먼저 성공해야만 평가 함수가 실행됩니다.
    from scripts.audit_cloud_source_reader import SourceAuditSummary

    summary = SourceAuditSummary(
        as_of_timestamp=cutoff.isoformat(),
        order_extracted_at=orders.extracted_at.isoformat(),
        member_extracted_at=pets.extracted_at.isoformat(),
        order_count=1,
        order_item_count=1,
        status_history_count=1,
        claim_count=0,
        claim_item_count=0,
        pet_count=1,
        valid_order_item_count=1,
        all_purchase_event_count=1,
        pet_purchase_event_count=1,
        excluded_late_birth_item_count=0,
    )
    monkeypatch.setattr(
        runner,
        "audit_snapshots",
        lambda order_source, pet_source, *, as_of_timestamp: (
            audit_calls.append((order_source, pet_source, as_of_timestamp)) or summary
        ),
    )
    comparison_calls: list[tuple[object, dict[str, object]]] = []

    def compare(paths: object, **kwargs: object) -> dict[str, object]:
        comparison_calls.append((paths, kwargs))
        return {"summary": []}

    monkeypatch.setattr(runner, "run_comparison", compare)

    assert runner.compare_snapshots(
        orders, pets, observation_end_at=cutoff, bootstrap_replicates=10
    ) == {"summary": []}
    assert audit_calls == [(orders, pets, cutoff)]
    paths, options = comparison_calls[0]
    assert paths is None
    assert options["sources"]["orders"] is orders.orders
    assert options["sources"]["pets"] is pets.pets
    assert options["source_metadata"]["source_snapshots"]["order_count"] == 1
    assert options["bootstrap_replicates"] == 10
    assert options["train_fraction"] == 0.70
    assert options["validation_fraction"] == 0.85
    assert options["aft_round_candidates"] is None
    assert options["aft_scale_candidates"] is None
    assert options["product_group_smoothing_candidates"] is None
    assert options["inner_train_ratio"] == 0.8


def test_compare_snapshots_forwards_selection_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orders, pets = _snapshots()
    cutoff = pd.Timestamp("2026-09-29T06:44:00Z")
    monkeypatch.setattr(runner, "audit_snapshots", lambda *_, **__: object())
    monkeypatch.setattr(runner, "asdict", lambda _: {})
    received: dict[str, object] = {}

    def compare(paths: object, **kwargs: object) -> dict[str, object]:
        received.update(kwargs)
        return {"summary": []}

    monkeypatch.setattr(runner, "run_comparison", compare)
    assert runner.compare_snapshots(
        orders,
        pets,
        observation_end_at=cutoff,
        train_fraction=0.4,
        validation_fraction=0.55,
        aft_scale_candidates=(0.5, 1.0, 2.0),
        product_group_smoothing_candidates=(1.0, 4.0),
        inner_train_ratio=0.75,
    ) == {"summary": []}
    assert received["train_fraction"] == 0.4
    assert received["validation_fraction"] == 0.55
    assert received["aft_scale_candidates"] == (0.5, 1.0, 2.0)
    assert received["product_group_smoothing_candidates"] == (1.0, 4.0)
    assert received["inner_train_ratio"] == 0.75


@pytest.mark.parametrize("cutoff", ["2026-09-29T15:44:00", "NaT", "invalid"])
def test_invalid_cutoff_rejected_before_db_access(
    cutoff: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        runner,
        "read_cloud_snapshots",
        lambda **_: pytest.fail("잘못된 컷에서 DB를 읽었습니다."),
    )
    assert runner.main(["--observation-end-at", cutoff]) == 2
    assert json.loads(capsys.readouterr().err)["event"] == (
        "repurchase_model_comparison_rejected"
    )


def test_invalid_bootstrap_rejected_before_db_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "read_cloud_snapshots",
        lambda **_: pytest.fail("잘못된 반복 수에서 DB를 읽었습니다."),
    )
    assert (
        runner.main(
            [
                "--observation-end-at",
                "2026-09-29T15:44:00+09:00",
                "--bootstrap-replicates",
                "0",
            ]
        )
        == 2
    )


@pytest.mark.parametrize(
    "options",
    [
        ["--train-fraction", "0.85", "--validation-fraction", "0.70"],
        ["--train-fraction", "nan"],
        ["--aft-round-candidates", "5", "5"],
        ["--aft-scale-candidates", "0.5", "0.5"],
        ["--product-group-smoothing-candidates", "1"],
        ["--product-group-smoothing-candidates", "0", "1"],
        ["--aft-round-candidates", "5", "20", "--aft-scale-candidates", "0.5", "1"],
        ["--inner-train-ratio", "1"],
    ],
)
def test_invalid_model_selection_rejected_before_db_access(
    options: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "read_cloud_snapshots",
        lambda **_: pytest.fail("잘못된 모델 설정에서 DB를 읽었습니다."),
    )
    assert (
        runner.main(["--observation-end-at", "2026-09-29T15:44:00+09:00", *options])
        == 2
    )


def test_main_passes_selection_options_to_comparison(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(runner, "read_cloud_snapshots", lambda **_: _snapshots())
    received: dict[str, object] = {}

    def compare(*_: object, **kwargs: object) -> dict[str, object]:
        received.update(kwargs)
        return {"summary": []}

    monkeypatch.setattr(runner, "compare_snapshots", compare)
    assert (
        runner.main(
            [
                "--observation-end-at",
                "2026-09-29T15:44:00+09:00",
                "--train-fraction",
                "0.55",
                "--validation-fraction",
                "0.70",
                "--aft-round-candidates",
                "5",
                "20",
                "--product-group-smoothing-candidates",
                "1",
                "4",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {"summary": []}
    assert received["train_fraction"] == 0.55
    assert received["validation_fraction"] == 0.70
    assert received["aft_round_candidates"] == (5, 20)
    assert received["product_group_smoothing_candidates"] == (1.0, 4.0)


def test_missing_output_directory_rejected_before_db_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        runner,
        "read_cloud_snapshots",
        lambda **_: pytest.fail("결과 경로 오류에서 DB를 읽었습니다."),
    )
    output = tmp_path / "missing" / "result.json"
    assert (
        runner.main(
            [
                "--observation-end-at",
                "2026-09-29T15:44:00+09:00",
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().err)["message"] == (
        "결과 파일의 상위 디렉터리가 없습니다."
    )


def test_write_failure_returns_structured_error_without_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(runner, "read_cloud_snapshots", lambda **_: _snapshots())
    monkeypatch.setattr(runner, "compare_snapshots", lambda *_, **__: {"summary": []})

    def fail_write(*_: object, **__: object) -> None:
        raise PermissionError("private path")

    monkeypatch.setattr(Path, "write_text", fail_write)
    output = tmp_path / "result.json"
    assert (
        runner.main(
            [
                "--observation-end-at",
                "2026-09-29T15:44:00+09:00",
                "--output",
                str(output),
            ]
        )
        == 1
    )
    error_output = capsys.readouterr().err
    assert json.loads(error_output) == {
        "event": "repurchase_model_comparison_failed",
        "error_type": "PermissionError",
    }
    assert "private path" not in error_output
    assert not output.exists()
