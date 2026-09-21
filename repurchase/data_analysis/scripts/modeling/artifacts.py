"""재구매 모델 후보의 네이티브 파일과 추론 계약을 함께 저장·검증합니다.

이 모듈은 운영 모델을 선택하거나 API를 제공하지 않습니다. 검증한 후보를 저장한 뒤
다른 프로세스에서 동일한 피처 순서와 평가 기간으로 읽을 수 있는지만 보장합니다.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMClassifier

from .features import MINIMAL_MODEL_FEATURE_COLUMNS, select_minimal_model_features
from .xgboost_aft import (
    XGBoostAFTTrainingResult,
    build_xgboost_aft_prediction_data,
    calculate_xgboost_aft_event_probability,
    create_xgboost_aft_parameters,
    predict_xgboost_aft_duration,
)

ARTIFACT_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
MODEL_FILENAMES = {"xgboost_aft": "model.json", "lightgbm": "model.txt"}


class ModelArtifactError(ValueError):
    """저장된 모델이나 메타데이터가 추론 계약을 위반할 때 발생합니다."""


@dataclass(frozen=True)
class LoadedModelArtifact:
    """검증된 모델과 학습 시점의 피처·확률 기간을 함께 보관합니다."""

    family: Literal["xgboost_aft", "lightgbm"]
    model: XGBoostAFTTrainingResult | lgb.Booster
    feature_columns: tuple[str, ...]
    horizon_days: int
    artifact_id: str


def _sha256(path: Path) -> str:
    """대형 모델 파일도 한 번에 메모리로 읽지 않고 해시를 계산합니다."""
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_id(manifest: dict[str, object]) -> str:
    """모델 파일 해시와 추론 계약을 합쳐 추적 가능한 버전 ID를 만듭니다."""
    canonical = json.dumps(
        {key: value for key, value in manifest.items() if key != "artifact_id"},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _reject_nonstandard_json_number(value: str) -> None:
    """표준 JSON에 없는 NaN·Infinity를 manifest 입력에서 거절합니다."""
    raise ModelArtifactError(f"모델 manifest에 허용되지 않는 숫자가 있습니다: {value}")


def _validate_contract(feature_columns: tuple[str, ...], horizon_days: int) -> None:
    """아티팩트가 현재 모델 입력과 양의 평가 기간을 선언하는지 검사합니다."""
    if (
        isinstance(horizon_days, bool)
        or not isinstance(horizon_days, int)
        or horizon_days <= 0
    ):
        raise ModelArtifactError("평가 기간은 0보다 큰 정수 일수여야 합니다.")
    if (
        not feature_columns
        or len(feature_columns) != len(set(feature_columns))
        or not set(feature_columns).issubset(MINIMAL_MODEL_FEATURE_COLUMNS)
    ):
        raise ModelArtifactError("모델 피처가 현재 허용 목록과 일치하지 않습니다.")


def save_model_artifact(
    model: XGBoostAFTTrainingResult | LGBMClassifier,
    directory: Path,
    *,
    horizon_days: int,
) -> Path:
    """후보 모델과 피처 계약을 새 디렉터리에 저장하고 기존 결과는 덮지 않습니다."""
    if isinstance(model, XGBoostAFTTrainingResult):
        family = "xgboost_aft"
        feature_columns = model.feature_columns
        library_version = xgb.__version__
        # 확률 계산에는 분포와 scale이 필수이므로 모델 파일과 따로 보존합니다.
        aft_metadata: dict[str, object] = {
            "loss_distribution": model.loss_distribution,
            "loss_distribution_scale": model.loss_distribution_scale,
            "num_boost_round": model.num_boost_round,
            "training_aft_nloglik": list(model.training_aft_nloglik),
        }
        if (
            model.booster.feature_names != list(feature_columns)
            or model.num_boost_round != model.booster.num_boosted_rounds()
            or len(model.training_aft_nloglik) != model.num_boost_round
            or not np.isfinite(model.training_aft_nloglik).all()
        ):
            raise ModelArtifactError(
                "AFT 모델과 피처·학습 반복 기록이 일치하지 않습니다."
            )
        try:
            create_xgboost_aft_parameters(
                loss_distribution=model.loss_distribution,
                loss_distribution_scale=model.loss_distribution_scale,
            )
        except ValueError as error:
            raise ModelArtifactError(
                "AFT 손실분포와 scale이 올바르지 않습니다."
            ) from error
    elif isinstance(model, LGBMClassifier):
        family = "lightgbm"
        if not hasattr(model, "booster_") or list(model.classes_) != [0, 1]:
            raise ModelArtifactError("학습된 이진 LightGBM 모델만 저장할 수 있습니다.")
        feature_columns = tuple(model.booster_.feature_name())
        library_version = lgb.__version__
        aft_metadata = {}
        if getattr(model, "repurchase_horizon_days", None) != horizon_days:
            raise ModelArtifactError(
                "LightGBM 학습 기간과 저장할 평가 기간이 다릅니다."
            )
    else:
        raise ModelArtifactError("지원하지 않는 재구매 모델 유형입니다.")

    _validate_contract(feature_columns, horizon_days)
    directory = Path(directory)
    if directory.exists():
        raise ModelArtifactError("기존 모델 아티팩트 디렉터리는 덮어쓸 수 없습니다.")
    if not directory.parent.is_dir():
        raise ModelArtifactError("아티팩트 상위 디렉터리가 존재하지 않습니다.")

    # 모델을 먼저 임시 디렉터리에 완성한 다음 이동합니다. manifest가 없는
    # 반쪽짜리 저장 결과를 정상 아티팩트로 읽는 일을 막습니다.
    with tempfile.TemporaryDirectory(
        prefix=".repurchase-artifact-", dir=directory.parent
    ) as temporary:
        temporary_path = Path(temporary)
        model_path = temporary_path / MODEL_FILENAMES[family]
        if family == "xgboost_aft":
            model.booster.save_model(model_path)
        else:
            model.booster_.save_model(str(model_path))
        manifest = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_family": family,
            "model_filename": model_path.name,
            "model_sha256": _sha256(model_path),
            "library_version": library_version,
            "feature_columns": list(feature_columns),
            "horizon_days": horizon_days,
            **aft_metadata,
        }
        manifest["artifact_id"] = _artifact_id(manifest)
        (temporary_path / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        temporary_path.rename(directory)
    return directory


def load_model_artifact(directory: Path) -> LoadedModelArtifact:
    """해시·버전·피처 순서를 확인한 후 네이티브 모델을 메모리에 읽습니다."""
    directory = Path(directory)
    try:
        manifest = json.loads(
            (directory / MANIFEST_FILENAME).read_text(encoding="utf-8"),
            parse_constant=_reject_nonstandard_json_number,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ModelArtifactError("모델 manifest를 읽을 수 없습니다.") from error
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("artifact_schema_version")) is not int
        or manifest["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION
    ):
        raise ModelArtifactError("지원하지 않는 모델 아티팩트 형식입니다.")
    family = manifest.get("model_family")
    if not isinstance(family, str) or family not in MODEL_FILENAMES:
        raise ModelArtifactError("모델 유형과 파일명이 일치하지 않습니다.")
    expected_keys = {
        "artifact_schema_version",
        "model_family",
        "model_filename",
        "model_sha256",
        "library_version",
        "feature_columns",
        "horizon_days",
        "artifact_id",
    }
    if family == "xgboost_aft":
        expected_keys.update(
            {
                "loss_distribution",
                "loss_distribution_scale",
                "num_boost_round",
                "training_aft_nloglik",
            }
        )
    if (
        set(manifest) != expected_keys
        or manifest.get("model_filename") != MODEL_FILENAMES[family]
    ):
        raise ModelArtifactError("모델 유형과 파일명이 일치하지 않습니다.")
    columns = manifest.get("feature_columns")
    if not isinstance(columns, list) or not all(
        isinstance(column, str) for column in columns
    ):
        raise ModelArtifactError("모델 피처 목록이 올바르지 않습니다.")
    feature_columns = tuple(columns)
    horizon_days = manifest.get("horizon_days")
    _validate_contract(feature_columns, horizon_days)
    if manifest.get("artifact_id") != _artifact_id(manifest):
        raise ModelArtifactError("모델 아티팩트 ID와 저장된 계약이 일치하지 않습니다.")
    library_version = xgb.__version__ if family == "xgboost_aft" else lgb.__version__
    if manifest.get("library_version") != library_version:
        raise ModelArtifactError("모델 저장 버전과 현재 라이브러리 버전이 다릅니다.")

    model_path = directory / MODEL_FILENAMES[family]
    try:
        digest = _sha256(model_path)
    except OSError as error:
        raise ModelArtifactError("모델 파일을 읽을 수 없습니다.") from error
    if digest != manifest.get("model_sha256"):
        raise ModelArtifactError("모델 파일의 SHA-256이 manifest와 다릅니다.")

    try:
        if family == "xgboost_aft":
            loss_distribution = manifest["loss_distribution"]
            loss_distribution_scale = manifest["loss_distribution_scale"]
            num_boost_round = manifest["num_boost_round"]
            training_loss = manifest["training_aft_nloglik"]
            if (
                loss_distribution not in {"normal", "logistic", "extreme"}
                or isinstance(loss_distribution_scale, bool)
                or not isinstance(loss_distribution_scale, (int, float))
                or not np.isfinite(loss_distribution_scale)
                or loss_distribution_scale <= 0
                or isinstance(num_boost_round, bool)
                or not isinstance(num_boost_round, int)
                or num_boost_round <= 0
                or not isinstance(training_loss, list)
                or len(training_loss) != num_boost_round
                or not all(
                    isinstance(value, (int, float)) and np.isfinite(value)
                    for value in training_loss
                )
            ):
                raise ModelArtifactError(
                    "AFT 분포·반복 횟수·손실 이력이 올바르지 않습니다."
                )
            booster = xgb.Booster(model_file=str(model_path))
            if (
                booster.feature_names != list(feature_columns)
                or booster.num_boosted_rounds() != num_boost_round
            ):
                raise ModelArtifactError(
                    "AFT 모델의 피처·반복 횟수가 manifest와 다릅니다."
                )
            model: XGBoostAFTTrainingResult | lgb.Booster = XGBoostAFTTrainingResult(
                booster=booster,
                feature_columns=feature_columns,
                loss_distribution=loss_distribution,
                loss_distribution_scale=float(loss_distribution_scale),
                num_boost_round=num_boost_round,
                training_aft_nloglik=tuple(float(value) for value in training_loss),
            )
        else:
            model = lgb.Booster(model_file=str(model_path))
            if tuple(model.feature_name()) != feature_columns:
                raise ModelArtifactError(
                    "LightGBM 모델의 피처 이름·순서가 manifest와 다릅니다."
                )
    except (
        KeyError,
        TypeError,
        ValueError,
        xgb.core.XGBoostError,
        lgb.basic.LightGBMError,
    ) as error:
        if isinstance(error, ModelArtifactError):
            raise
        raise ModelArtifactError(
            "모델 파일 또는 메타데이터를 해석할 수 없습니다."
        ) from error
    return LoadedModelArtifact(
        family=family,
        model=model,
        feature_columns=feature_columns,
        horizon_days=horizon_days,
        artifact_id=manifest["artifact_id"],
    )


def predict_artifact_probability(
    artifact: LoadedModelArtifact, rows: pd.DataFrame
) -> pd.Series:
    """구매 anchor부터 선언된 기간 내 재구매 누적확률을 반환합니다.

    이 값은 현재 시점부터 앞으로 N일의 조건부 확률이나 다음 구매일이 아닙니다.
    """
    if rows.empty or not rows.index.is_unique:
        raise ModelArtifactError(
            "추론 표본은 비어 있지 않고 행 인덱스가 고유해야 합니다."
        )
    if artifact.family == "xgboost_aft":
        if not isinstance(artifact.model, XGBoostAFTTrainingResult):
            raise ModelArtifactError("AFT 모델 유형이 아티팩트 선언과 다릅니다.")
        prediction_data = build_xgboost_aft_prediction_data(
            rows, feature_columns=artifact.feature_columns
        )
        duration = predict_xgboost_aft_duration(artifact.model, prediction_data)
        return calculate_xgboost_aft_event_probability(
            artifact.model, duration, horizon_days=artifact.horizon_days
        )
    if artifact.family != "lightgbm" or not isinstance(artifact.model, lgb.Booster):
        raise ModelArtifactError("LightGBM 모델 유형이 아티팩트 선언과 다릅니다.")
    features = select_minimal_model_features(
        rows, feature_columns=artifact.feature_columns
    )
    probability = np.asarray(artifact.model.predict(features), dtype="float64")
    if (
        probability.shape != (len(features),)
        or not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
    ):
        raise ModelArtifactError(
            "LightGBM 확률은 행별 0~1 사이의 유한한 값이어야 합니다."
        )
    return pd.Series(
        probability, index=features.index.copy(), name="predicted_event_probability"
    )
