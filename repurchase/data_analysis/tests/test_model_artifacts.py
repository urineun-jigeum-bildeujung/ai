"""재구매 후보 모델의 저장·로드와 추론 입력 계약을 검증합니다."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from lightgbm import LGBMRegressor

from scripts.modeling.artifacts import (
    ModelArtifactError,
    _artifact_id,
    _sha256,
    load_model_artifact,
    predict_artifact_probability,
    save_model_artifact,
)
from scripts.modeling.lightgbm_baseline import (
    LightGBMTrainingData,
    predict_lightgbm_repurchase_probability,
    train_lightgbm_classifier,
)
from scripts.modeling.xgboost_aft import (
    build_xgboost_aft_prediction_data,
    build_xgboost_aft_training_data,
    calculate_xgboost_aft_event_probability,
    predict_xgboost_aft_duration,
    train_xgboost_aft_model,
)


def _feature_rows() -> pd.DataFrame:
    """저장 전후 예측 비교에 사용할 과거 구매 정보만 만듭니다."""
    return pd.DataFrame(
        {
            "history_interval_count": [0, 1, 2, 3],
            "history_median_days": [np.nan, 18.0, 24.0, 31.0],
            "history_relative_mad": [np.nan, np.nan, 0.2, 0.3],
            "user_prior_order_count": [0, 2, 4, 6],
        },
        index=[40, 10, 30, 20],
    )


def _trained_aft_model():
    """우측검열을 포함한 작은 표본으로 실제 AFT 파일을 생성합니다."""
    rows = _feature_rows().copy()
    rows["survival_observed_duration_days"] = [10.0, 20.0, 30.0, 40.0]
    rows["survival_event_observed"] = pd.array(
        [True, False, True, False], dtype="boolean"
    )
    rows["split"] = "train"
    return train_xgboost_aft_model(build_xgboost_aft_training_data(rows))


def _trained_lightgbm_model():
    """두 정답 클래스를 가진 실제 LightGBM 후보를 학습합니다."""
    features = pd.concat([_feature_rows()] * 10, ignore_index=True)
    training_data = LightGBMTrainingData(
        horizon_days=30,
        features=features,
        target=pd.Series([0, 1] * 20, dtype="int8"),
        sample_weight=pd.Series([1.0] * 40, dtype="float64"),
    )
    return train_lightgbm_classifier(training_data)


def test_aft_artifact_roundtrip_preserves_probability(tmp_path) -> None:
    """AFT를 디스크에서 다시 읽어도 피처 순서·행 인덱스·확률이 같습니다."""
    model = _trained_aft_model()
    rows = _feature_rows()
    data = build_xgboost_aft_prediction_data(
        rows, feature_columns=model.feature_columns
    )
    original_duration = predict_xgboost_aft_duration(model, data)
    original = calculate_xgboost_aft_event_probability(
        model, original_duration, horizon_days=30
    )

    directory = save_model_artifact(model, tmp_path / "aft", horizon_days=30)
    artifact = load_model_artifact(directory)
    restored = predict_artifact_probability(artifact, rows)

    assert artifact.family == "xgboost_aft"
    assert artifact.horizon_days == 30
    assert restored.index.equals(rows.index)
    np.testing.assert_allclose(restored.to_numpy(), original.to_numpy())
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert "user_id" not in manifest


def test_lightgbm_artifact_roundtrip_preserves_probability(tmp_path) -> None:
    """LightGBM 네이티브 파일을 다시 읽어도 sklearn 래퍼와 확률이 같습니다."""
    model = _trained_lightgbm_model()
    rows = _feature_rows()
    original = predict_lightgbm_repurchase_probability(model, rows)

    directory = save_model_artifact(model, tmp_path / "lightgbm", horizon_days=30)
    artifact = load_model_artifact(directory)
    restored = predict_artifact_probability(artifact, rows)

    assert artifact.family == "lightgbm"
    assert artifact.horizon_days == 30
    assert restored.index.equals(rows.index)
    np.testing.assert_allclose(restored.to_numpy(), original.to_numpy())


def test_artifact_rejects_modified_model_file(tmp_path) -> None:
    """모델 파일이 바뀌면 유효한 JSON이더라도 해시 불일치로 거절합니다."""
    directory = save_model_artifact(
        _trained_aft_model(), tmp_path / "aft", horizon_days=30
    )
    model_file = directory / "model.json"
    model_file.write_bytes(model_file.read_bytes() + b" ")

    with pytest.raises(ModelArtifactError, match="SHA-256"):
        load_model_artifact(directory)


def test_artifact_rejects_changed_feature_order(tmp_path) -> None:
    """메타데이터의 피처 순서가 모델 내부 순서와 달라지면 거절합니다."""
    directory = save_model_artifact(
        _trained_lightgbm_model(), tmp_path / "lightgbm", horizon_days=30
    )
    manifest_file = directory / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["feature_columns"] = list(reversed(manifest["feature_columns"]))
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="아티팩트 ID"):
        load_model_artifact(directory)

    # ID까지 다시 계산해도 모델 내부의 실제 피처 순서와 대조해 거절합니다.
    manifest["artifact_id"] = _artifact_id(manifest)
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="피처 이름·순서"):
        load_model_artifact(directory)


@pytest.mark.parametrize("horizon_days", [0, -1, True, 30.5])
def test_artifact_rejects_invalid_horizon(tmp_path, horizon_days) -> None:
    """평가 기간을 정의하지 못하는 아티팩트는 만들지 않습니다."""
    with pytest.raises(ModelArtifactError, match="평가 기간"):
        save_model_artifact(
            _trained_aft_model(), tmp_path / "aft", horizon_days=horizon_days
        )


def test_artifact_rejects_existing_destination(tmp_path) -> None:
    """기존 실험 결과를 실수로 덮어쓰지 않습니다."""
    directory = save_model_artifact(
        _trained_aft_model(), tmp_path / "aft", horizon_days=30
    )

    with pytest.raises(ModelArtifactError, match="덮어쓸 수 없습니다"):
        save_model_artifact(_trained_aft_model(), directory, horizon_days=30)


def test_artifact_rejects_missing_feature_at_prediction(tmp_path) -> None:
    """추론 시 학습 때 사용한 피처가 빠지면 임의의 0으로 채우지 않습니다."""
    directory = save_model_artifact(
        _trained_lightgbm_model(), tmp_path / "lightgbm", horizon_days=30
    )
    artifact = load_model_artifact(directory)

    with pytest.raises(ValueError, match="모델 피처가 누락"):
        predict_artifact_probability(
            artifact, _feature_rows().drop(columns="history_relative_mad")
        )


def test_lightgbm_artifact_rejects_different_training_horizon(tmp_path) -> None:
    """30일 정답으로 학습한 모델을 60일 확률이라고 저장하지 않습니다."""
    with pytest.raises(ModelArtifactError, match="학습 기간"):
        save_model_artifact(
            _trained_lightgbm_model(), tmp_path / "lightgbm", horizon_days=60
        )


def test_artifact_rejects_incompatible_library_version(tmp_path) -> None:
    """운영 환경의 모델 라이브러리 버전이 다르면 묵시적으로 읽지 않습니다."""
    directory = save_model_artifact(
        _trained_aft_model(), tmp_path / "aft", horizon_days=30
    )
    manifest_file = directory / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["library_version"] = "0.0.0"
    manifest["artifact_id"] = _artifact_id(manifest)
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="라이브러리 버전"):
        load_model_artifact(directory)


def test_artifact_rejects_invalid_model_family_without_python_error(tmp_path) -> None:
    """잘못된 manifest 값은 사전 조회 오류 대신 계약 오류로 보고합니다."""
    directory = save_model_artifact(
        _trained_aft_model(), tmp_path / "aft", horizon_days=30
    )
    manifest_file = directory / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["model_family"] = ["xgboost_aft"]
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="모델 유형"):
        load_model_artifact(directory)


def test_artifact_rejects_nonstandard_json_number(tmp_path) -> None:
    """Python JSON 파서가 기본 허용하는 NaN도 모델 계약에서는 거절합니다."""
    directory = save_model_artifact(
        _trained_aft_model(), tmp_path / "aft", horizon_days=30
    )
    (directory / "manifest.json").write_text(
        '{"artifact_schema_version": NaN}', encoding="utf-8"
    )

    with pytest.raises(ModelArtifactError, match="허용되지 않는 숫자"):
        load_model_artifact(directory)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("loss_distribution", "logistic", "손실분포"),
        ("loss_distribution_scale", 1.5, "scale"),
    ],
)
def test_aft_artifact_rejects_native_distribution_mismatch(
    tmp_path, field: str, value: object, message: str
) -> None:
    """ID를 다시 계산한 manifest도 네이티브 AFT 분포와 다르면 거절합니다."""
    directory = save_model_artifact(
        _trained_aft_model(), tmp_path / "aft", horizon_days=30
    )
    manifest_file = directory / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest["artifact_id"] = _artifact_id(manifest)
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelArtifactError, match=message):
        load_model_artifact(directory)


def test_aft_artifact_rejects_non_aft_native_objective(tmp_path) -> None:
    """파일 해시까지 갱신해도 회귀 모델을 AFT 모델로 읽지 않습니다."""
    directory = save_model_artifact(
        _trained_aft_model(), tmp_path / "aft", horizon_days=30
    )
    features = _feature_rows()
    matrix = xgb.DMatrix(features, label=[0.1, 0.2, 0.3, 0.4])
    regression = xgb.train(
        {"objective": "reg:squarederror", "nthread": 1},
        matrix,
        num_boost_round=5,
    )
    model_file = directory / "model.json"
    regression.save_model(model_file)
    manifest_file = directory / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["model_sha256"] = _sha256(model_file)
    manifest["artifact_id"] = _artifact_id(manifest)
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="survival:aft"):
        load_model_artifact(directory)


def test_lightgbm_artifact_rejects_regression_native_objective(tmp_path) -> None:
    """0~1로 예측할 수도 있는 회귀 Booster를 이진 확률 모델로 읽지 않습니다."""
    directory = save_model_artifact(
        _trained_lightgbm_model(), tmp_path / "lightgbm", horizon_days=30
    )
    features = pd.concat([_feature_rows()] * 10, ignore_index=True)
    regression = LGBMRegressor(verbosity=-1, n_jobs=1).fit(
        features, [0.1, 0.2, 0.3, 0.4] * 10
    )
    model_file = directory / "model.txt"
    regression.booster_.save_model(str(model_file))
    manifest_file = directory / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest["model_sha256"] = _sha256(model_file)
    manifest["artifact_id"] = _artifact_id(manifest)
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelArtifactError, match="binary"):
        load_model_artifact(directory)
