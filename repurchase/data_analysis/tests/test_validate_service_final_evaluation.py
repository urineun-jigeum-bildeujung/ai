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
    """동일 원천·설정·디스크 코드 지문을 가진 최소 사전검증 입력을 만듭니다."""
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
        "model_configuration": {
            "xgboost_aft": {
                "parameters": preflight.create_xgboost_aft_parameters(
                    loss_distribution_scale=2.0
                ),
                "num_boost_round": 20,
            },
            "lightgbm": {
                "parameters": preflight.create_lightgbm_classifier().get_params()
            },
        },
        "evaluation": {
            "horizon_days": 30,
            "primary_metric": "ipcw_brier_score",
            "bootstrap_unit": "user",
            "bootstrap_replicates": 1000,
            "bootstrap_random_seed": 42,
        },
        "evaluation_population_policy": (
            preflight.service_evaluation_population_policy()
        ),
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
            "code_sha256",
        )
    }
    comparison["test_evaluated"] = False
    comparison["model_configuration"] = copy.deepcopy(manifest["model_configuration"])
    comparison["evaluation_population_policy"] = copy.deepcopy(
        manifest["evaluation_population_policy"]
    )
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
    """고정 설정·파일 지문이 모두 일치하면 입력 계약만 승인합니다."""
    manifest, comparison, sources = frozen_inputs

    result = preflight.validate_manifest(manifest, comparison, sources)

    assert result["status"] == "input_contract_verified"
    assert result["source_count"] == 6
    assert result["test_evaluated"] is False


def test_preflight_rejects_changed_source_file(frozen_inputs: tuple) -> None:
    """원천 CSV의 바이트가 바뀌면 승인한 스냅샷으로 취급하지 않습니다."""
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
        ("model_configuration", {}, "학습 설정"),
        ("evaluation", {"horizon_days": 7}, "지표 계약"),
        ("evaluation_population_policy", {}, "모집단·제외 기준"),
    ],
)
def test_preflight_rejects_changed_contract_before_source_access(
    frozen_inputs: tuple,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    """설정 불일치는 원천 파일 해시 계산보다 먼저 거절합니다."""
    manifest, comparison, sources = frozen_inputs
    manifest[field] = value

    def fail_if_source_opened(_: Path) -> str:
        raise AssertionError("설정 오류 뒤에는 원천 파일을 읽으면 안 됩니다.")

    monkeypatch.setattr(preflight, "_file_sha256", fail_if_source_opened)
    with pytest.raises(ValueError, match=message):
        preflight.validate_manifest(manifest, comparison, sources)


def test_preflight_rejects_result_with_test_evaluated(frozen_inputs: tuple) -> None:
    """이미 Test를 본 결과를 Validation 기준으로 재사용하지 않습니다."""
    manifest, comparison, sources = frozen_inputs
    comparison["test_evaluated"] = True

    with pytest.raises(ValueError, match="Test 평가가 없는"):
        preflight.validate_manifest(manifest, comparison, sources)


def test_preflight_rejects_changed_validation_model_settings(
    frozen_inputs: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation에 기록된 학습 설정이 승인값과 다르면 거절합니다."""
    manifest, comparison, sources = frozen_inputs
    comparison["model_configuration"]["lightgbm"]["parameters"]["n_estimators"] = 200

    def fail_if_source_opened(_: Path) -> str:
        raise AssertionError("설정 오류 뒤에는 원천 파일을 읽으면 안 됩니다.")

    monkeypatch.setattr(preflight, "_file_sha256", fail_if_source_opened)
    with pytest.raises(ValueError, match="학습 설정"):
        preflight.validate_manifest(manifest, comparison, sources)


def test_preflight_rejects_changed_validation_population_policy(
    frozen_inputs: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation의 모집단 정책이 바뀌면 원천 접근 전에 거절합니다."""
    manifest, comparison, sources = frozen_inputs
    comparison["evaluation_population_policy"]["unknown_outcome"] = "drop_row"

    def fail_if_source_opened(_: Path) -> str:
        raise AssertionError("모집단 계약 오류 뒤에는 원천 파일을 읽으면 안 됩니다.")

    monkeypatch.setattr(preflight, "_file_sha256", fail_if_source_opened)
    with pytest.raises(ValueError, match="모집단·제외 기준"):
        preflight.validate_manifest(manifest, comparison, sources)


def test_preflight_rejects_validation_from_different_model_code(
    frozen_inputs: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """현재 코드로 manifest를 만들어도 과거 코드의 Validation 결과는 거절합니다."""
    manifest, comparison, sources = frozen_inputs
    comparison["code_sha256"] = dict(comparison["code_sha256"])
    comparison["code_sha256"]["run_service_model_comparison.py"] = "0" * 64

    def fail_if_source_opened(_: Path) -> str:
        raise AssertionError("코드 지문 불일치 뒤에는 원천 파일을 읽으면 안 됩니다.")

    monkeypatch.setattr(preflight, "_file_sha256", fail_if_source_opened)
    with pytest.raises(ValueError, match="Validation 결과.*모델 코드 SHA-256"):
        preflight.validate_manifest(manifest, comparison, sources)


def test_preflight_rejects_legacy_validation_without_code_hash(
    frozen_inputs: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """기존 결과의 지문 누락을 원천 파일 접근 이전에 거절합니다."""
    manifest, comparison, sources = frozen_inputs
    del comparison["code_sha256"]

    def fail_if_source_opened(_: Path) -> str:
        raise AssertionError("지문 누락 뒤에는 원천 파일을 읽으면 안 됩니다.")

    monkeypatch.setattr(preflight, "_file_sha256", fail_if_source_opened)

    with pytest.raises(ValueError, match="Validation 결과.*모델 코드 SHA-256"):
        preflight.validate_manifest(manifest, comparison, sources)
