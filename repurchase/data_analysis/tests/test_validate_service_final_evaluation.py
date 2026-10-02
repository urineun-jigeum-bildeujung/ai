"""최종 평가 사전검증은 설정 변조를 거절하고 Test 성능을 계산하지 않습니다."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

from scripts import validate_service_final_evaluation as preflight
from scripts.modeling.operational_training_samples import (
    TEMPORAL_SERVICE_FEATURE_GENERATION_VERSION,
)
from scripts.run_service_model_comparison import _file_sha256


@pytest.fixture
def frozen_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict, dict, dict[str, Path]]:
    # CI의 경량 테스트 환경에는 학습 패키지의 메타데이터가 없을 수 있습니다.
    package_versions = {"pandas": "2.3.3", "xgboost": "3.2.0", "lightgbm": "4.7.0"}
    monkeypatch.setattr(preflight, "version", package_versions.__getitem__)
    sources = {name: tmp_path / f"{name}.csv" for name in preflight.SOURCE_NAMES}
    for name, path in sources.items():
        path.write_text(f"{name}\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "source_sha256": {name: _file_sha256(path) for name, path in sources.items()},
        "observation_end_at_assumption": "2026-09-29T15:44:00+09:00",
        "train_end_at": "2026-04-01T00:00:00+09:00",
        "validation_end_at": "2026-07-01T00:00:00+09:00",
        "train_fraction": 0.7,
        "validation_fraction": 0.85,
        "feature_generation_version": TEMPORAL_SERVICE_FEATURE_GENERATION_VERSION,
        "runtime_versions": {
            "python": sys.version.split()[0],
            **package_versions,
        },
        "code_sha256": {path.name: _file_sha256(path) for path in preflight.CODE_PATHS},
        "aft_configuration": preflight.FROZEN_AFT.copy(),
        "evaluation": {
            "horizon_days": 30,
            "primary_metric": "ipcw_brier_score",
            "bootstrap_unit": "user",
            "bootstrap_replicates": 1000,
            "bootstrap_random_seed": 42,
        },
    }
    comparison = {
        key: copy.deepcopy(manifest[key])
        for key in (
            "source_sha256",
            "observation_end_at_assumption",
            "train_end_at",
            "validation_end_at",
            "train_fraction",
            "validation_fraction",
            "feature_generation_version",
            "runtime_versions",
        )
    }
    comparison["test_evaluated"] = False
    comparison["aft_scale_selection"] = {
        "loss_distribution": "normal",
        "fixed_num_boost_round": 20,
        "selected_scale": 2.0,
    }
    comparison["summary"] = [
        {"model": name, "horizon_days": 30} for name in ("xgboost_aft", "lightgbm")
    ]
    comparison["calibration"] = [
        {"model": name} for name in ("xgboost_aft", "lightgbm")
    ]
    comparison["paired_bootstrap_summary"] = {
        "bootstrap_replicates": 1000,
        "random_seed": 42,
    }
    return manifest, comparison, sources


def test_preflight_accepts_matching_frozen_inputs(frozen_inputs: tuple) -> None:
    manifest, comparison, sources = frozen_inputs

    result = preflight.validate_manifest(manifest, comparison, sources)

    assert result["status"] == "input_contract_verified"
    assert result["source_count"] == 6
    assert result["test_evaluated"] is False


def test_preflight_rejects_changed_source_file(frozen_inputs: tuple) -> None:
    manifest, comparison, sources = frozen_inputs
    sources["orders"].write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="orders.*SHA-256"):
        preflight.validate_manifest(manifest, comparison, sources)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("observation_end_at_assumption", "2026-09-29T15:44:00", "시간대"),
        ("observation_end_at_assumption", "NaT", "유효한 시각"),
        ("observation_end_at_assumption", "not-a-date", "유효한 시각"),
        ("feature_generation_version", 999, "feature_generation_version"),
        ("runtime_versions", {"python": "0.0.0"}, "runtime_versions"),
        ("aft_configuration", {"loss_distribution": "normal"}, "AFT 후보"),
        ("evaluation", {"horizon_days": 7}, "지표 계약"),
    ],
)
def test_preflight_rejects_changed_contract_before_source_access(
    frozen_inputs: tuple,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    manifest, comparison, sources = frozen_inputs
    manifest[field] = value

    def fail_if_source_opened(_: Path) -> str:
        raise AssertionError("설정 오류 뒤에는 원천 파일을 읽으면 안 됩니다.")

    monkeypatch.setattr(preflight, "_file_sha256", fail_if_source_opened)
    with pytest.raises(ValueError, match=message):
        preflight.validate_manifest(manifest, comparison, sources)


def test_preflight_rejects_result_with_test_evaluated(frozen_inputs: tuple) -> None:
    manifest, comparison, sources = frozen_inputs
    comparison["test_evaluated"] = True

    with pytest.raises(ValueError, match="Test 평가가 없는"):
        preflight.validate_manifest(manifest, comparison, sources)
