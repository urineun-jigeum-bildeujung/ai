"""최종 Test 이전의 후보 고정은 입력 계약을 먼저 검증합니다."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import freeze_service_models as freezer


def test_build_manifest_fixes_evaluation_contract() -> None:
    comparison = {
        key: {"source_sha256": {}, "code_sha256": {}}.get(key, "value")
        for key in (
            "source_sha256",
            "code_sha256",
            "observation_end_at_assumption",
            "train_end_at",
            "validation_end_at",
            "train_fraction",
            "validation_fraction",
            "feature_generation_version",
            "runtime_versions",
            "model_configuration",
            "evaluation_population_policy",
        )
    }
    manifest = freezer.build_pretest_manifest(comparison)

    assert manifest["aft_configuration"] == freezer.FROZEN_AFT
    assert manifest["evaluation"]["horizon_days"] == 30
    assert manifest["evaluation"]["bootstrap_replicates"] == 1000
    assert manifest["schema_version"] == 1


def test_freeze_rejects_validation_failure_before_reading_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = tmp_path / "validation.json"
    result.write_text(json.dumps({"test_evaluated": True}), encoding="utf-8")
    monkeypatch.setattr(
        freezer,
        "_read_sources",
        lambda paths: pytest.fail("거절된 비교 결과로 원천을 읽으면 안 됩니다."),
    )

    with pytest.raises(ValueError, match="Test 평가가 없는"):
        freezer.freeze_models(result, {}, tmp_path / "frozen")

    assert not (tmp_path / "frozen").exists()


def test_freeze_never_overwrites_existing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = tmp_path / "validation.json"
    result.write_text('{"test_evaluated": false}', encoding="utf-8")
    output = tmp_path / "frozen"
    output.mkdir()
    monkeypatch.setattr(freezer, "build_pretest_manifest", lambda comparison: {})
    monkeypatch.setattr(freezer, "validate_manifest", lambda *args: None)

    with pytest.raises(ValueError, match="덮어쓸 수 없습니다"):
        freezer.freeze_models(result, {}, output)
