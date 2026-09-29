"""LightGBM 학습 입력이 시간 분할과 IPCW 계약을 지키는지 검증합니다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier

from scripts.modeling.features import MINIMAL_MODEL_FEATURE_COLUMNS
from scripts.modeling.lightgbm_baseline import (
    LightGBMBaselineError,
    LightGBMTrainingData,
    build_lightgbm_training_data,
    create_lightgbm_classifier,
    predict_lightgbm_repurchase_probability,
    train_lightgbm_classifier,
)


def make_lightgbm_rows(*, split: str = "train") -> pd.DataFrame:
    """정답 확인 두 행과 조기 검열 한 행을 가진 작은 학습 표본을 만듭니다."""
    return pd.DataFrame(
        {
            "history_interval_count": [0, 1, 2],
            "history_median_days": [float("nan"), 20.0, 25.0],
            "history_relative_mad": [float("nan"), float("nan"), 0.2],
            "user_prior_order_count": [0, 2, 5],
            "split": [split] * 3,
            "ipcw_horizon_days": [30] * 3,
            "ipcw_outcome_known": pd.array(
                [True, False, True],
                dtype="boolean",
            ),
            "ipcw_event_within_horizon": pd.array(
                [True, pd.NA, False],
                dtype="boolean",
            ),
            "ipcw_weight": [1.25, 0.0, 2.0],
        },
        index=[10, 20, 30],
    )


def test_create_lightgbm_classifier_uses_reproducible_binary_baseline() -> None:
    """튜닝 전 기준 모델이 이진분류와 재현성 설정을 명시하는지 검증합니다."""
    model = create_lightgbm_classifier()
    parameters = model.get_params()

    assert parameters["objective"] == "binary"
    assert parameters["random_state"] == 42
    assert parameters["n_jobs"] == 1
    assert parameters["deterministic"] is True
    assert parameters["force_col_wise"] is True


def test_train_lightgbm_classifier_forwards_ipcw_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """학습 함수가 피처·정답과 함께 IPCW 가중치를 모델에 전달합니다."""
    training_data = build_lightgbm_training_data(make_lightgbm_rows())
    captured: dict[str, object] = {}

    def capture_fit(
        model: LGBMClassifier,
        features: pd.DataFrame,
        target: pd.Series,
        *,
        sample_weight: pd.Series,
    ) -> LGBMClassifier:
        """실제 학습 대신 fit 호출에 전달된 값을 기록합니다."""
        captured["model"] = model
        captured["features"] = features
        captured["target"] = target
        captured["sample_weight"] = sample_weight
        return model

    monkeypatch.setattr(LGBMClassifier, "fit", capture_fit)

    trained_model = train_lightgbm_classifier(training_data)

    assert trained_model is captured["model"]
    assert captured["features"] is training_data.features
    assert captured["target"] is training_data.target
    assert captured["sample_weight"] is training_data.sample_weight


def test_train_lightgbm_classifier_fits_real_model() -> None:
    """작은 계약 표본으로 실제 LightGBM 학습이 완료되는지 검증합니다."""
    training_data = build_lightgbm_training_data(make_lightgbm_rows())

    trained_model = train_lightgbm_classifier(training_data)

    assert trained_model.classes_.tolist() == [0, 1]
    assert trained_model.n_features_in_ == len(MINIMAL_MODEL_FEATURE_COLUMNS)


@pytest.mark.parametrize("target_value", [0, 1])
def test_train_lightgbm_classifier_rejects_single_target_class(
    target_value: int,
) -> None:
    """생성 함수를 우회해도 한 종류의 정답만 있는 학습을 거절합니다."""
    valid_training_data = build_lightgbm_training_data(make_lightgbm_rows())
    single_class_training_data = LightGBMTrainingData(
        horizon_days=valid_training_data.horizon_days,
        features=valid_training_data.features,
        target=pd.Series(
            [target_value] * len(valid_training_data.target),
            index=valid_training_data.target.index,
            dtype="int8",
        ),
        sample_weight=valid_training_data.sample_weight,
    )

    with pytest.raises(LightGBMBaselineError, match="미발생 0과 발생 1"):
        train_lightgbm_classifier(single_class_training_data)


def test_predict_lightgbm_probability_selects_positive_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """모든 행에서 재구매 정답 1에 해당하는 두 번째 확률만 반환합니다."""
    rows = make_lightgbm_rows()
    model = train_lightgbm_classifier(build_lightgbm_training_data(rows))
    probability_matrix = np.array(
        [
            [0.70, 0.30],
            [0.20, 0.80],
            [0.55, 0.45],
        ]
    )

    def return_probability_matrix(
        fitted_model: LGBMClassifier,
        features: pd.DataFrame,
    ) -> np.ndarray:
        """확률 열 선택을 검증하도록 정해진 2열 행렬을 반환합니다."""
        assert fitted_model is model
        assert features.index.tolist() == [10, 20, 30]
        return probability_matrix

    monkeypatch.setattr(LGBMClassifier, "predict_proba", return_probability_matrix)

    probabilities = predict_lightgbm_repurchase_probability(model, rows)

    assert probabilities.index.tolist() == [10, 20, 30]
    assert probabilities.tolist() == pytest.approx([0.30, 0.80, 0.45])
    assert probabilities.name == "predicted_repurchase_probability"


def test_predict_lightgbm_probability_rejects_wrong_matrix_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """이진분류 확률이 2열이 아니면 잘못된 모델 출력으로 거절합니다."""
    model = train_lightgbm_classifier(
        build_lightgbm_training_data(make_lightgbm_rows())
    )
    monkeypatch.setattr(
        LGBMClassifier,
        "predict_proba",
        lambda _model, _features: np.array([[0.30], [0.80], [0.45]]),
    )

    with pytest.raises(LightGBMBaselineError, match="2열 행렬"):
        predict_lightgbm_repurchase_probability(model, make_lightgbm_rows())


def test_predict_lightgbm_probability_rejects_out_of_range_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """0부터 1 사이를 벗어난 값은 사용자에게 전달할 확률로 허용하지 않습니다."""
    model = train_lightgbm_classifier(
        build_lightgbm_training_data(make_lightgbm_rows())
    )
    monkeypatch.setattr(
        LGBMClassifier,
        "predict_proba",
        lambda _model, _features: np.array([[0.70, 0.30], [-0.10, 1.10], [0.55, 0.45]]),
    )

    with pytest.raises(LightGBMBaselineError, match="0부터 1 사이"):
        predict_lightgbm_repurchase_probability(model, make_lightgbm_rows())


def test_build_lightgbm_training_data_keeps_only_known_outcomes() -> None:
    """30일 결과가 불명인 조기 검열 행을 정답으로 만들지 않습니다."""
    training_data = build_lightgbm_training_data(make_lightgbm_rows())

    assert training_data.horizon_days == 30
    assert training_data.features.index.tolist() == [10, 30]
    assert training_data.target.index.tolist() == [10, 30]
    assert training_data.sample_weight.index.tolist() == [10, 30]
    assert training_data.target.tolist() == [1, 0]
    assert training_data.sample_weight.tolist() == [1.25, 2.0]


def test_subset_model_predicts_with_training_columns_and_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """예측 원본의 열 순서와 추가 열에 관계없이 학습 피처 순서를 재사용합니다."""
    columns = ("user_prior_order_count", "history_interval_count")
    rows = make_lightgbm_rows().drop(
        columns=["history_median_days", "history_relative_mad"]
    )
    training_data = build_lightgbm_training_data(rows, feature_columns=columns)
    model = train_lightgbm_classifier(training_data)
    assert model.feature_name_ == list(columns)
    assert training_data.features.index.tolist() == [10, 30]
    assert training_data.sample_weight.tolist() == [1.25, 2.0]
    original_predict = model.predict_proba

    def check_prediction_features(features: pd.DataFrame) -> np.ndarray:
        """실제 예측 직전에 학습 열 순서와 각 값의 대응을 검사합니다."""
        pd.testing.assert_frame_equal(features, rows.loc[:, list(columns)])
        return original_predict(features)

    monkeypatch.setattr(model, "predict_proba", check_prediction_features)
    probabilities = predict_lightgbm_repurchase_probability(model, rows.iloc[:, ::-1])

    assert probabilities.index.equals(rows.index)
    assert probabilities.between(0, 1).all()


def test_build_lightgbm_training_data_preserves_missing_history_features() -> None:
    """정답은 확인됐지만 과거 이력이 없는 행의 결측 피처는 그대로 유지합니다."""
    training_data = build_lightgbm_training_data(make_lightgbm_rows())

    assert pd.isna(training_data.features.loc[10, "history_median_days"])
    assert pd.isna(training_data.features.loc[10, "history_relative_mad"])


def test_build_lightgbm_training_data_rejects_validation_rows() -> None:
    """Validation 정답이 학습 입력으로 섞이면 즉시 거절합니다."""
    with pytest.raises(LightGBMBaselineError, match="Train"):
        build_lightgbm_training_data(make_lightgbm_rows(split="validation"))


def test_build_lightgbm_training_data_rejects_unknown_boolean_contract() -> None:
    """정답 확인 여부에 결측값이 있으면 행을 조용히 제거하지 않습니다."""
    rows = make_lightgbm_rows()
    rows.loc[20, "ipcw_outcome_known"] = pd.NA

    with pytest.raises(LightGBMBaselineError, match="정답 확인 여부"):
        build_lightgbm_training_data(rows)


@pytest.mark.parametrize("target_value", [False, True])
def test_build_lightgbm_training_data_rejects_single_target_class(
    target_value: bool,
) -> None:
    """형식이 boolean이어도 미발생·발생 중 하나만 있으면 학습하지 않습니다."""
    rows = make_lightgbm_rows()
    rows.loc[rows["ipcw_outcome_known"], "ipcw_event_within_horizon"] = target_value

    with pytest.raises(LightGBMBaselineError, match="미발생 0과 발생 1"):
        build_lightgbm_training_data(rows)
