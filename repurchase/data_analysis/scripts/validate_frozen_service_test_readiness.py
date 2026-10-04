"""봉인된 Test를 읽기 전에 모델 아티팩트·원천·코드 지문을 검증합니다.

이 명령은 Test 행이나 정답을 만들지 않으며, 통과해도 최종 평가를 실행하지 않습니다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.freeze_service_models import build_pretest_manifest
from scripts.modeling.artifacts import load_model_artifact
from scripts.run_service_model_comparison import _file_sha256
from scripts.validate_service_final_evaluation import SOURCE_NAMES, validate_manifest

MODEL_FAMILIES = ("xgboost_aft", "lightgbm")


def validate_frozen_inputs(
    frozen_dir: Path,
    comparison_path: Path,
    sources: dict[str, Path],
) -> dict[str, object]:
    """동결 기록과 현재 파일을 대조하며 Test 데이터 행은 파싱하지 않습니다."""
    record = json.loads((frozen_dir / "freeze-record.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (frozen_dir / "pretest-manifest.json").read_text(encoding="utf-8")
    )
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if not all(isinstance(value, dict) for value in (record, manifest, comparison)):
        raise ValueError("모델 고정 기록과 Validation 결과는 JSON 객체여야 합니다.")
    if record.get("schema_version") != 1 or record.get("test_evaluated") is not False:
        raise ValueError("Test 미평가 모델 고정 기록이 아닙니다.")
    if manifest != build_pretest_manifest(comparison):
        raise ValueError("저장된 사전검증 manifest가 Validation 계약과 다릅니다.")
    if record.get("validation_result_sha256") != _file_sha256(comparison_path):
        raise ValueError("모델 고정 이후 Validation 결과 파일이 변경됐습니다.")
    if record.get("freeze_code_sha256") != _file_sha256(
        Path(__file__).with_name("freeze_service_models.py")
    ):
        raise ValueError("모델 고정 코드가 기록과 다릅니다.")
    if record.get("source_sha256") != manifest.get("source_sha256") or record.get(
        "code_sha256"
    ) != manifest.get("code_sha256"):
        raise ValueError("모델 고정 기록의 원천·코드 지문이 다릅니다.")
    cutoff = pd.Timestamp(manifest["validation_end_at"])
    if pd.Timestamp(record.get("training_cutoff_at")) != cutoff:
        raise ValueError("모델 학습 컷이 Validation 종료 시각과 다릅니다.")
    for key in (
        "training_sample_count",
        "aft_training_sample_count",
        "lightgbm_training_sample_count",
    ):
        if type(record.get(key)) is not int or record[key] <= 0:
            raise ValueError(f"{key}는 양의 표본 수여야 합니다.")
    if any(
        record[key] > record["training_sample_count"]
        for key in ("aft_training_sample_count", "lightgbm_training_sample_count")
    ):
        raise ValueError("모델 학습 표본 수가 전체 표본 수를 초과합니다.")

    # 설정·코드·원천 해시까지 통과한 뒤에만 모델 파일을 로딩합니다.
    validate_manifest(manifest, comparison, sources)
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(MODEL_FAMILIES):
        raise ValueError("두 모델 후보의 고정 기록이 모두 필요합니다.")
    verified_ids: dict[str, str] = {}
    for family in MODEL_FAMILIES:
        expected = artifacts[family]
        if not isinstance(expected, dict):
            raise ValueError(f"{family} 모델 고정 기록이 올바르지 않습니다.")
        directory = frozen_dir / family
        if expected.get("manifest_sha256") != _file_sha256(directory / "manifest.json"):
            raise ValueError(f"{family} 모델 manifest 해시가 다릅니다.")
        loaded = load_model_artifact(directory)
        if (
            loaded.family != family
            or loaded.artifact_id != expected.get("artifact_id")
            or loaded.horizon_days != manifest["evaluation"]["horizon_days"]
            or loaded.feature_generation_version
            != manifest["feature_generation_version"]
        ):
            raise ValueError(f"{family} 모델 추론 계약이 기록과 다릅니다.")
        verified_ids[family] = loaded.artifact_id
    return {
        "status": "frozen_inputs_verified",
        "test_evaluated": False,
        "source_count": len(SOURCE_NAMES),
        "training_cutoff_at": record["training_cutoff_at"],
        "artifact_ids": verified_ids,
    }


def main() -> None:
    """CLI 경로를 받아 동결 파일만 확인하고 짧은 결과를 출력합니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--validation-result", type=Path, required=True)
    for name in SOURCE_NAMES:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate_frozen_inputs(
            args.frozen_dir,
            args.validation_result,
            {name: getattr(args, name) for name in SOURCE_NAMES},
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
