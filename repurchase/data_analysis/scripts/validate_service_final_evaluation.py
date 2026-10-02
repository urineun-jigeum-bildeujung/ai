"""봉인된 최종 평가를 열기 전에 Validation 입력·설정을 읽기 전용으로 검증합니다.

이 명령은 원천 파일의 바이트 해시만 읽습니다. 행·라벨을 파싱하거나 모델을
학습·평가하지 않으며, 사전검증용 manifest와 다른 입력은 실행 전에 거절합니다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from scripts.modeling.lightgbm_baseline import create_lightgbm_classifier
from scripts.modeling.operational_training_samples import (
    TEMPORAL_SERVICE_FEATURE_GENERATION_VERSION,
)
from scripts.modeling.xgboost_aft import create_xgboost_aft_parameters
from scripts.run_service_model_comparison import _file_sha256

SOURCE_NAMES = (
    "orders",
    "order_items",
    "pets",
    "histories",
    "claims",
    "claim_items",
)
FROZEN_AFT = {
    "loss_distribution": "normal",
    "num_boost_round": 20,
    "loss_distribution_scale": 2.0,
}
CODE_PATHS = tuple(sorted((Path(__file__).parent / "modeling").glob("*.py"))) + (
    Path(__file__).parent / "run_service_model_comparison.py",
)


def _aware_timestamp(value: object, name: str) -> pd.Timestamp:
    """잘못된 형식과 NaT를 명시적으로 거절합니다."""
    if not isinstance(value, str):
        raise ValueError(f"{name}은 시간대가 있는 시각이어야 합니다.")
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}은 유효한 시각이어야 합니다.") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{name}은 시간대가 있는 유효한 시각이어야 합니다.")
    return timestamp


def validate_manifest(
    manifest: dict[str, object],
    comparison: dict[str, object],
    sources: dict[str, Path],
) -> dict[str, object]:
    """설정 계약을 먼저, 파일 해시는 마지막에 검사합니다."""
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
    ):
        raise ValueError("지원하지 않는 최종 평가 manifest 버전입니다.")
    if comparison.get("test_evaluated") is not False:
        raise ValueError(
            "Test 평가가 없는 Validation 결과만 기준으로 사용할 수 있습니다."
        )
    expected_hashes = manifest.get("source_sha256")
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(
        SOURCE_NAMES
    ):
        raise ValueError("manifest에는 원천 6개 파일의 SHA-256이 필요합니다.")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for value in expected_hashes.values()
    ):
        raise ValueError("원천 SHA-256은 소문자 64자리 16진수여야 합니다.")
    if set(sources) != set(SOURCE_NAMES):
        raise ValueError("원천 파일 6개의 경로가 모두 필요합니다.")
    if comparison.get("source_sha256") != expected_hashes:
        raise ValueError("manifest와 Validation 결과의 원천 해시가 다릅니다.")

    end = manifest.get("observation_end_at_assumption")
    end_at = _aware_timestamp(end, "관측 종료 컷")
    for key in (
        "observation_end_at_assumption",
        "train_end_at",
        "validation_end_at",
        "train_fraction",
        "validation_fraction",
        "feature_generation_version",
        "runtime_versions",
    ):
        if manifest.get(key) != comparison.get(key):
            raise ValueError(f"manifest와 Validation 결과의 {key}가 다릅니다.")
    train_end_at = _aware_timestamp(manifest.get("train_end_at"), "train_end_at")
    validation_end_at = _aware_timestamp(
        manifest.get("validation_end_at"), "validation_end_at"
    )
    if not (train_end_at < validation_end_at < end_at):
        raise ValueError("시간 컷의 순서가 유효하지 않습니다.")
    train_fraction = manifest.get("train_fraction")
    validation_fraction = manifest.get("validation_fraction")
    if (
        isinstance(train_fraction, bool)
        or isinstance(validation_fraction, bool)
        or not isinstance(train_fraction, (int, float))
        or not isinstance(validation_fraction, (int, float))
        or not math.isfinite(train_fraction)
        or not math.isfinite(validation_fraction)
        or not 0 < train_fraction < validation_fraction < 1
    ):
        raise ValueError("시간 분할 비율이 유효하지 않습니다.")
    if manifest.get("feature_generation_version") != (
        TEMPORAL_SERVICE_FEATURE_GENERATION_VERSION
    ):
        raise ValueError("현재 피처 생성 버전이 승인된 버전과 다릅니다.")
    runtime = manifest.get("runtime_versions")
    if (
        not isinstance(runtime, dict)
        or runtime.get("python") != sys.version.split()[0]
        or any(
            runtime.get(name) != version(name)
            for name in ("pandas", "xgboost", "lightgbm")
        )
    ):
        raise ValueError("현재 모델 런타임 버전이 승인된 버전과 다릅니다.")
    selected = comparison.get("aft_scale_selection")
    if (
        manifest.get("aft_configuration") != FROZEN_AFT
        or not isinstance(selected, dict)
        or selected.get("loss_distribution") != FROZEN_AFT["loss_distribution"]
        or selected.get("fixed_num_boost_round") != FROZEN_AFT["num_boost_round"]
        or selected.get("selected_scale") != FROZEN_AFT["loss_distribution_scale"]
    ):
        raise ValueError("AFT 후보 설정이 고정된 최종 평가 후보와 다릅니다.")
    frozen_model_configuration = {
        "xgboost_aft": {
            "parameters": create_xgboost_aft_parameters(
                loss_distribution_scale=FROZEN_AFT["loss_distribution_scale"]
            ),
            "num_boost_round": FROZEN_AFT["num_boost_round"],
        },
        "lightgbm": {"parameters": create_lightgbm_classifier().get_params()},
    }
    if (
        manifest.get("model_configuration") != frozen_model_configuration
        or comparison.get("model_configuration") != frozen_model_configuration
    ):
        raise ValueError("두 모델의 학습 설정이 승인된 후보와 다릅니다.")
    if manifest.get("evaluation") != {
        "horizon_days": 30,
        "primary_metric": "ipcw_brier_score",
        "bootstrap_unit": "user",
        "bootstrap_replicates": 1000,
        "bootstrap_random_seed": 42,
    }:
        raise ValueError("최종 평가 지표 계약이 다릅니다.")
    summary = comparison.get("summary")
    calibration = comparison.get("calibration")
    bootstrap = comparison.get("paired_bootstrap_summary")
    if (
        not isinstance(summary, list)
        or not all(isinstance(row, dict) for row in summary)
        or {row.get("model") for row in summary} != {"xgboost_aft", "lightgbm"}
        or any(row.get("horizon_days") != 30 for row in summary)
        or not isinstance(calibration, list)
        or not all(isinstance(row, dict) for row in calibration)
        or {row.get("model") for row in calibration} != {"xgboost_aft", "lightgbm"}
        or not isinstance(bootstrap, dict)
        or bootstrap.get("bootstrap_replicates") != 1000
        or bootstrap.get("random_seed") != 42
    ):
        raise ValueError("Validation 결과의 후보·지표 계약이 다릅니다.")

    # 모델 코드 변경 역시 별도 검토가 필요하므로 관련 Python 파일 지문을 고정합니다.
    expected_code_hashes = manifest.get("code_sha256")
    code_paths = {path.name: path for path in CODE_PATHS}
    if not isinstance(expected_code_hashes, dict) or set(expected_code_hashes) != set(
        code_paths
    ):
        raise ValueError("모델 코드 SHA-256 목록이 현재 코드와 다릅니다.")
    for name, path in code_paths.items():
        if _file_sha256(path) != expected_code_hashes[name]:
            raise ValueError(f"{name} 모델 코드 SHA-256이 승인값과 다릅니다.")

    # 설정 불일치는 파일을 읽기 전에 거절합니다. 이후에는 바이트 해시만 읽습니다.
    for name in SOURCE_NAMES:
        if _file_sha256(sources[name]) != expected_hashes[name]:
            raise ValueError(f"{name} 원천 파일의 SHA-256이 승인값과 다릅니다.")
    return {
        "status": "input_contract_verified",
        "source_count": len(SOURCE_NAMES),
        "test_evaluated": False,
        "observation_end_at_assumption": end,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-result", type=Path, required=True)
    for name in SOURCE_NAMES:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        comparison = json.loads(args.validation_result.read_text(encoding="utf-8"))
        result = validate_manifest(
            manifest,
            comparison,
            {name: getattr(args, name) for name in SOURCE_NAMES},
        )
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
