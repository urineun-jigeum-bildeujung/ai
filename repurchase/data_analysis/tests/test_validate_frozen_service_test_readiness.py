"""동결 검증은 Test를 열지 않고 파일·모델 계약 변조를 거절합니다."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import validate_frozen_service_test_readiness as readiness
from scripts.run_service_model_comparison import _file_sha256


@pytest.fixture
def frozen_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, Path]]:
    """모델을 학습하지 않고 동결 기록의 구조만 재현합니다."""
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    comparison_path = tmp_path / "validation.json"
    comparison_path.write_text('{"test_evaluated": false}', encoding="utf-8")
    manifest = {
        "validation_end_at": "2026-06-01T00:00:00Z",
        "evaluation": {"horizon_days": 30},
        "feature_generation_version": 2,
        "source_sha256": {name: "source" for name in readiness.SOURCE_NAMES},
        "code_sha256": {"model.py": "code"},
    }
    (frozen / "pretest-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.setattr(readiness, "build_pretest_manifest", lambda _: manifest)
    monkeypatch.setattr(readiness, "validate_manifest", lambda *args: None)
    freeze_code = Path(readiness.__file__).with_name("freeze_service_models.py")
    artifacts = {}
    for family in readiness.MODEL_FAMILIES:
        folder = frozen / family
        folder.mkdir()
        model_manifest = folder / "manifest.json"
        model_manifest.write_text(family, encoding="utf-8")
        artifacts[family] = {
            "manifest_sha256": _file_sha256(model_manifest),
            "artifact_id": f"{family}-id",
        }
    record = {
        "schema_version": 1,
        "test_evaluated": False,
        "validation_result_sha256": _file_sha256(comparison_path),
        "freeze_code_sha256": _file_sha256(freeze_code),
        "source_sha256": manifest["source_sha256"],
        "code_sha256": manifest["code_sha256"],
        "training_cutoff_at": manifest["validation_end_at"],
        "training_sample_count": 10,
        "aft_training_sample_count": 10,
        "lightgbm_training_sample_count": 8,
        "artifacts": artifacts,
    }
    (frozen / "freeze-record.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(
        readiness,
        "load_model_artifact",
        lambda folder: SimpleNamespace(
            family=folder.name,
            artifact_id=f"{folder.name}-id",
            horizon_days=30,
            feature_generation_version=2,
        ),
    )
    sources = {name: tmp_path / f"{name}.csv" for name in readiness.SOURCE_NAMES}
    return frozen, comparison_path, sources


def test_matching_frozen_inputs_are_verified(frozen_files) -> None:
    """두 모델 ID와 컷이 같으면 Test 미평가 상태로 통과합니다."""
    frozen, comparison, sources = frozen_files

    result = readiness.validate_frozen_inputs(frozen, comparison, sources)

    assert result["status"] == "frozen_inputs_verified"
    assert result["test_evaluated"] is False
    assert set(result["artifact_ids"]) == set(readiness.MODEL_FAMILIES)


def test_test_evaluated_record_is_rejected_before_model_loading(
    frozen_files, monkeypatch: pytest.MonkeyPatch
) -> None:
    """이미 Test를 평가한 기록은 모델 파일을 열기 전에 거절합니다."""
    frozen, comparison, sources = frozen_files
    path = frozen / "freeze-record.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["test_evaluated"] = True
    path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(
        readiness,
        "load_model_artifact",
        lambda _: pytest.fail("거절된 기록에서 모델을 읽으면 안 됩니다."),
    )

    with pytest.raises(ValueError, match="Test 미평가"):
        readiness.validate_frozen_inputs(frozen, comparison, sources)


def test_changed_model_manifest_is_rejected(frozen_files) -> None:
    """모델 manifest 변경은 저장된 SHA-256과 대조해 거절합니다."""
    frozen, comparison, sources = frozen_files
    (frozen / "lightgbm" / "manifest.json").write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest 해시"):
        readiness.validate_frozen_inputs(frozen, comparison, sources)
