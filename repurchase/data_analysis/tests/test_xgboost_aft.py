"""XGBoost AFT 라벨 계약과 제외 통계 계산을 검증합니다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.modeling.xgboost_aft import (
    AFTLabelBounds,
    XGBoostAFTError,
    build_aft_label_bounds,
    build_xgboost_aft_evaluation_rows,
    build_xgboost_aft_prediction_data,
    build_xgboost_aft_training_data,
    create_xgboost_aft_parameters,
    predict_xgboost_aft_duration,
    train_xgboost_aft_model,
)


def make_aft_training_rows(*, split: str = "train") -> pd.DataFrame:
    """사건·검열·0일 표본을 함께 가진 작은 AFT 학습 데이터를 만듭니다."""
    return pd.DataFrame(
        {
            "history_interval_count": [2, 0, 1],
            "history_median_days": [25.0, float("nan"), 18.0],
            "history_relative_mad": [0.2, float("nan"), float("nan")],
            "user_prior_order_count": [5, 0, 2],
            "survival_observed_duration_days": [20.0, 0.0, 30.0],
            "survival_event_observed": pd.array(
                [True, True, False],
                dtype="boolean",
            ),
            "split": [split] * 3,
        },
        index=[30, 10, 20],
    )


def test_create_xgboost_aft_parameters_uses_reproducible_cpu_baseline() -> None:
    """튜닝 전 기준 설정이 AFT 목적·평가지표·CPU·고정 seed를 명시합니다."""
    parameters = create_xgboost_aft_parameters()

    assert parameters == {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.0,
        "tree_method": "hist",
        "device": "cpu",
        "seed": 42,
        "nthread": 1,
        "validate_parameters": True,
    }


def test_create_xgboost_aft_parameters_accepts_experiment_candidate() -> None:
    """공통 실행 조건을 유지하면서 손실분포와 scale만 교체할 수 있습니다."""
    parameters = create_xgboost_aft_parameters(
        loss_distribution="logistic",
        loss_distribution_scale=1.5,
    )

    assert parameters["aft_loss_distribution"] == "logistic"
    assert parameters["aft_loss_distribution_scale"] == 1.5
    assert parameters["device"] == "cpu"
    assert parameters["seed"] == 42


def test_create_xgboost_aft_parameters_rejects_unknown_distribution() -> None:
    """분포 이름의 오타를 XGBoost 실행 전 명확한 오류로 거절합니다."""
    with pytest.raises(XGBoostAFTError, match="normal, logistic, extreme"):
        create_xgboost_aft_parameters(loss_distribution="gaussian")


def test_create_xgboost_aft_parameters_rejects_non_string_distribution() -> None:
    """해시할 수 없는 값도 Python 오류가 아닌 AFT 설정 오류로 거절합니다."""
    with pytest.raises(XGBoostAFTError, match="normal, logistic, extreme"):
        create_xgboost_aft_parameters(
            loss_distribution=["normal"],  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_scale",
    [
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        True,
        "1.0",
    ],
)
def test_create_xgboost_aft_parameters_rejects_invalid_scale(
    invalid_scale: object,
) -> None:
    """0·음수·비유한값·boolean·문자열 scale을 학습 설정으로 허용하지 않습니다."""
    with pytest.raises(XGBoostAFTError, match="0보다 큰 유한한 숫자"):
        create_xgboost_aft_parameters(
            loss_distribution_scale=invalid_scale,  # type: ignore[arg-type]
        )


def test_aft_label_bounds_calculates_included_sample_count() -> None:
    """포함 건수를 중복 저장하지 않고 원본과 제외 건수로 계산합니다."""
    bounds = AFTLabelBounds(
        rows=pd.DataFrame(index=range(98)),
        source_sample_count=100,
        excluded_zero_duration_count=2,
    )

    assert bounds.included_sample_count == 98


def test_aft_label_bounds_rejects_inconsistent_row_count() -> None:
    """실제 행 수와 계산된 포함 건수가 다르면 조용히 진행하지 않습니다."""
    with pytest.raises(XGBoostAFTError, match="행 수"):
        AFTLabelBounds(
            rows=pd.DataFrame(index=range(97)),
            source_sample_count=100,
            excluded_zero_duration_count=2,
        )


@pytest.mark.parametrize(
    ("source_sample_count", "excluded_zero_duration_count"),
    [
        (-1, 0),
        (1, -1),
        (1.5, 0),
        (1, 0.5),
        (True, 0),
    ],
)
def test_aft_label_bounds_rejects_invalid_sample_count(
    source_sample_count: object,
    excluded_zero_duration_count: object,
) -> None:
    """음수·실수·boolean 표본 수를 정상 통계로 허용하지 않습니다."""
    with pytest.raises(XGBoostAFTError, match="0 이상의 정수"):
        AFTLabelBounds(
            rows=pd.DataFrame(index=[0]),
            source_sample_count=source_sample_count,  # type: ignore[arg-type]
            excluded_zero_duration_count=(  # type: ignore[arg-type]
                excluded_zero_duration_count
            ),
        )


def test_aft_label_bounds_rejects_excluded_count_above_source() -> None:
    """제외 건수가 원본보다 많으면 합계 관계를 계산하기 전에 거절합니다."""
    with pytest.raises(XGBoostAFTError, match="원본 표본 수보다 클 수 없습니다"):
        AFTLabelBounds(
            rows=pd.DataFrame(),
            source_sample_count=1,
            excluded_zero_duration_count=2,
        )


def test_build_aft_label_bounds_encodes_events_and_censoring() -> None:
    """0일 표본을 제외하고 사건과 우측검열을 서로 다른 상한으로 표현합니다."""
    rows = pd.DataFrame(
        {
            "survival_observed_duration_days": [20.0, 30.0, 0.0, 0.0],
            "survival_event_observed": pd.array(
                [True, False, True, False],
                dtype="boolean",
            ),
        },
        index=[10, 20, 30, 40],
    )

    bounds = build_aft_label_bounds(rows)

    assert bounds.source_sample_count == 4
    assert bounds.excluded_zero_duration_count == 2
    assert bounds.included_sample_count == 2
    assert bounds.rows.index.tolist() == [10, 20]
    assert bounds.rows["aft_lower_bound"].tolist() == [20.0, 30.0]
    assert bounds.rows["aft_upper_bound"].iloc[0] == 20.0
    assert np.isinf(bounds.rows["aft_upper_bound"].iloc[1])
    assert "aft_lower_bound" not in rows.columns
    assert "aft_upper_bound" not in rows.columns


@pytest.mark.parametrize(
    "missing_column",
    [
        "survival_observed_duration_days",
        "survival_event_observed",
    ],
)
def test_build_aft_label_bounds_rejects_missing_required_column(
    missing_column: str,
) -> None:
    """시간이나 사건 여부가 없으면 임의로 값을 추정하지 않습니다."""
    rows = pd.DataFrame(
        {
            "survival_observed_duration_days": [20.0],
            "survival_event_observed": pd.array([True], dtype="boolean"),
        }
    ).drop(columns=missing_column)

    with pytest.raises(XGBoostAFTError, match="필수 컬럼"):
        build_aft_label_bounds(rows)


def test_build_aft_label_bounds_rejects_empty_rows() -> None:
    """열만 있고 표본이 없으면 학습 가능한 라벨로 처리하지 않습니다."""
    rows = pd.DataFrame(
        {
            "survival_observed_duration_days": pd.Series(dtype="float64"),
            "survival_event_observed": pd.Series(dtype="boolean"),
        }
    )

    with pytest.raises(XGBoostAFTError, match="표본이 없습니다"):
        build_aft_label_bounds(rows)


@pytest.mark.parametrize(
    "invalid_duration",
    [
        float("nan"),
        float("inf"),
        -1.0,
        True,
        "20",
    ],
)
def test_build_aft_label_bounds_rejects_invalid_duration(
    invalid_duration: object,
) -> None:
    """결측·무한대·음수·boolean·문자열 시간을 조용히 제외하지 않습니다."""
    rows = pd.DataFrame(
        {
            "survival_observed_duration_days": [20.0, invalid_duration],
            "survival_event_observed": pd.array([True, False], dtype="boolean"),
        }
    )

    with pytest.raises(XGBoostAFTError, match="관측 기간"):
        build_aft_label_bounds(rows)


@pytest.mark.parametrize(
    "invalid_events",
    [
        pd.Series([True, pd.NA], dtype="boolean"),
        pd.Series([1, 0], dtype="int64"),
    ],
)
def test_build_aft_label_bounds_rejects_invalid_event_contract(
    invalid_events: pd.Series,
) -> None:
    """사건 여부는 결측 없는 boolean 열만 허용합니다."""
    rows = pd.DataFrame(
        {
            "survival_observed_duration_days": [20.0, 30.0],
            "survival_event_observed": invalid_events,
        }
    )

    with pytest.raises(XGBoostAFTError, match="사건 관측 여부"):
        build_aft_label_bounds(rows)


def test_build_aft_label_bounds_rejects_only_zero_duration_rows() -> None:
    """0일을 모두 제외한 뒤 남는 표본이 없으면 명확히 중단합니다."""
    rows = pd.DataFrame(
        {
            "survival_observed_duration_days": [0.0, 0.0],
            "survival_event_observed": pd.array([True, False], dtype="boolean"),
        }
    )

    with pytest.raises(XGBoostAFTError, match="0일 표본을 제외한 뒤"):
        build_aft_label_bounds(rows)


def test_build_aft_label_bounds_rejects_complex_duration() -> None:
    """복소수 관측 기간의 허수부를 버리고 실수로 변환하지 않습니다."""
    rows = pd.DataFrame(
        {
            "survival_observed_duration_days": np.array(
                [20.0 + 1.0j],
                dtype="complex128",
            ),
            "survival_event_observed": pd.array([True], dtype="boolean"),
        }
    )

    with pytest.raises(XGBoostAFTError, match="관측 기간"):
        build_aft_label_bounds(rows)


def test_build_xgboost_aft_training_data_aligns_features_and_bounds() -> None:
    """0일 제외 후 동일한 두 행의 피처와 하한·상한이 행렬에 연결됩니다."""
    training_data = build_xgboost_aft_training_data(make_aft_training_rows())

    assert training_data.source_sample_count == 3
    assert training_data.excluded_zero_duration_count == 1
    assert training_data.included_sample_count == 2
    assert training_data.row_index.tolist() == [30, 20]
    assert training_data.feature_columns == (
        "history_interval_count",
        "history_median_days",
        "history_relative_mad",
        "user_prior_order_count",
    )
    assert training_data.matrix.num_row() == 2
    assert training_data.matrix.num_col() == 4
    feature_values = training_data.matrix.get_data().toarray()
    np.testing.assert_allclose(
        feature_values[:, [0, 1, 3]],
        np.array(
            [
                [2.0, 25.0, 5.0],
                [1.0, 18.0, 2.0],
            ]
        ),
    )
    assert training_data.matrix.get_float_info("label_lower_bound").tolist() == [
        20.0,
        30.0,
    ]
    upper_bound = training_data.matrix.get_float_info("label_upper_bound")
    assert upper_bound[0] == 20.0
    assert np.isinf(upper_bound[1])


def test_build_xgboost_aft_training_data_rejects_non_train_rows() -> None:
    """Validation이나 Test 표본이 AFT 학습 입력으로 섞이면 거절합니다."""
    with pytest.raises(XGBoostAFTError, match="Train 표본만"):
        build_xgboost_aft_training_data(make_aft_training_rows(split="validation"))


def test_build_xgboost_aft_training_data_rejects_missing_split() -> None:
    """시간 분할 정보가 없으면 학습 데이터라고 임의로 가정하지 않습니다."""
    rows = make_aft_training_rows().drop(columns="split")

    with pytest.raises(XGBoostAFTError, match="필수 컬럼"):
        build_xgboost_aft_training_data(rows)


def test_build_xgboost_aft_training_data_rejects_duplicate_row_index() -> None:
    """학습 행의 원본 인덱스가 중복되면 예측 추적이 불가능하므로 거절합니다."""
    rows = make_aft_training_rows()
    rows.index = [7, 8, 7]

    with pytest.raises(XGBoostAFTError, match="중복"):
        build_xgboost_aft_training_data(rows)


def test_build_xgboost_aft_prediction_data_reuses_training_feature_order() -> None:
    """예측 행렬은 원본 열 순서가 달라도 학습 피처 계약 순서를 재사용합니다."""
    training_data = build_xgboost_aft_training_data(make_aft_training_rows())
    rows = make_aft_training_rows().loc[[20, 30]].copy()
    rows = rows.loc[:, list(reversed(rows.columns))]

    prediction_data = build_xgboost_aft_prediction_data(
        rows,
        feature_columns=training_data.feature_columns,
    )

    assert prediction_data.row_index.tolist() == [20, 30]
    assert prediction_data.feature_columns == training_data.feature_columns
    assert prediction_data.matrix.feature_names == list(training_data.feature_columns)
    feature_values = prediction_data.matrix.get_data().toarray()
    np.testing.assert_allclose(
        feature_values[:, [0, 1, 3]],
        np.array(
            [
                [1.0, 18.0, 2.0],
                [2.0, 25.0, 5.0],
            ]
        ),
    )


def test_build_xgboost_aft_prediction_data_rejects_empty_rows() -> None:
    """예측할 행이 없으면 빈 결과를 정상 예측으로 반환하지 않습니다."""
    rows = make_aft_training_rows().iloc[0:0]

    with pytest.raises(XGBoostAFTError, match="예측할 표본이 없습니다"):
        build_xgboost_aft_prediction_data(
            rows,
            feature_columns=("history_interval_count",),
        )


def test_build_xgboost_aft_prediction_data_rejects_duplicate_index() -> None:
    """예측을 원본 행에 되돌릴 수 없는 중복 인덱스를 거절합니다."""
    rows = make_aft_training_rows().loc[[20, 30]].copy()
    rows.index = [7, 7]

    with pytest.raises(XGBoostAFTError, match="인덱스에 중복"):
        build_xgboost_aft_prediction_data(
            rows,
            feature_columns=("history_interval_count",),
        )


def test_train_xgboost_aft_model_completes_small_cpu_smoke_run() -> None:
    """작은 계약 표본으로 실제 AFT 학습 경로와 손실 기록을 확인합니다."""
    training_data = build_xgboost_aft_training_data(make_aft_training_rows())

    result = train_xgboost_aft_model(training_data, num_boost_round=5)

    assert result.booster.num_boosted_rounds() == 5
    assert result.feature_columns == training_data.feature_columns
    assert result.loss_distribution == "normal"
    assert result.loss_distribution_scale == 1.0
    assert result.num_boost_round == 5
    assert len(result.training_aft_nloglik) == 5
    assert np.isfinite(result.training_aft_nloglik).all()


def test_train_xgboost_aft_model_records_candidate_conditions() -> None:
    """후보별로 달라지는 분포·scale과 공통 반복 횟수를 결과에 보존합니다."""
    training_data = build_xgboost_aft_training_data(make_aft_training_rows())

    result = train_xgboost_aft_model(
        training_data,
        loss_distribution="logistic",
        loss_distribution_scale=1.5,
        num_boost_round=3,
    )

    assert result.booster.num_boosted_rounds() == 3
    assert result.loss_distribution == "logistic"
    assert result.loss_distribution_scale == 1.5
    assert result.num_boost_round == 3
    assert len(result.training_aft_nloglik) == 3


@pytest.mark.parametrize("loss_distribution", ["normal", "logistic", "extreme"])
def test_train_xgboost_aft_model_supports_each_loss_distribution(
    loss_distribution: str,
) -> None:
    """허용한 세 손실분포가 모두 실제 AFT 학습 경로에서 동작합니다."""
    training_data = build_xgboost_aft_training_data(make_aft_training_rows())

    result = train_xgboost_aft_model(
        training_data,
        loss_distribution=loss_distribution,
        num_boost_round=1,
    )

    assert result.loss_distribution == loss_distribution
    assert len(result.training_aft_nloglik) == 1


def test_train_xgboost_aft_model_rejects_changed_feature_order() -> None:
    """DMatrix 내부 피처 순서가 생성 후 바뀌면 잘못된 의미로 학습하지 않습니다."""
    training_data = build_xgboost_aft_training_data(make_aft_training_rows())
    training_data.matrix.feature_names = list(reversed(training_data.feature_columns))

    with pytest.raises(XGBoostAFTError, match="피처 이름·순서"):
        train_xgboost_aft_model(training_data)


def test_train_xgboost_aft_model_rejects_changed_row_index_count() -> None:
    """예측을 원본 행에 되돌릴 인덱스가 유실되면 학습 전에 중단합니다."""
    training_data = build_xgboost_aft_training_data(make_aft_training_rows())
    object.__setattr__(training_data, "row_index", training_data.row_index[:1])

    with pytest.raises(XGBoostAFTError, match="원본 인덱스 수"):
        train_xgboost_aft_model(training_data)


def test_train_xgboost_aft_model_rejects_changed_zero_day_label() -> None:
    """DMatrix 내부 라벨이 0일로 바뀌면 AFT 양수 시간 계약 위반으로 거절합니다."""
    training_data = build_xgboost_aft_training_data(make_aft_training_rows())
    training_data.matrix.set_float_info("label_lower_bound", [0.0, 30.0])
    training_data.matrix.set_float_info("label_upper_bound", [0.0, float("inf")])

    with pytest.raises(XGBoostAFTError, match="양수의 정확 사건"):
        train_xgboost_aft_model(training_data)


@pytest.mark.parametrize("invalid_rounds", [0, -1, 1.5, True])
def test_train_xgboost_aft_model_rejects_invalid_boost_rounds(
    invalid_rounds: object,
) -> None:
    """0·음수·실수·boolean 반복 횟수를 학습 전에 명확히 거절합니다."""
    training_data = build_xgboost_aft_training_data(make_aft_training_rows())

    with pytest.raises(XGBoostAFTError, match="0보다 큰 정수"):
        train_xgboost_aft_model(
            training_data,
            num_boost_round=invalid_rounds,  # type: ignore[arg-type]
        )


def test_predict_xgboost_aft_duration_restores_original_row_index() -> None:
    """실제 모델 예측값을 입력 순서가 아닌 원본 행 식별자와 연결합니다."""
    training_data = build_xgboost_aft_training_data(make_aft_training_rows())
    training_result = train_xgboost_aft_model(training_data)
    rows = make_aft_training_rows().loc[[20, 30]].copy()
    prediction_data = build_xgboost_aft_prediction_data(
        rows,
        feature_columns=training_result.feature_columns,
    )

    predictions = predict_xgboost_aft_duration(training_result, prediction_data)

    assert predictions.index.tolist() == [20, 30]
    assert predictions.name == "predicted_duration_days"
    assert predictions.dtype == "float64"
    assert len(predictions) == 2
    assert np.isfinite(predictions).all()
    assert predictions.gt(0).all()


def test_predict_xgboost_aft_duration_rejects_different_feature_contract() -> None:
    """예측용 피처 이름이나 순서가 학습 계약과 다르면 추론 전에 거절합니다."""
    training_data = build_xgboost_aft_training_data(make_aft_training_rows())
    training_result = train_xgboost_aft_model(training_data)
    prediction_data = build_xgboost_aft_prediction_data(
        make_aft_training_rows().loc[[20, 30]],
        feature_columns=("user_prior_order_count", "history_interval_count"),
    )

    with pytest.raises(XGBoostAFTError, match="모델의 학습 피처 계약"):
        predict_xgboost_aft_duration(training_result, prediction_data)


@pytest.mark.parametrize(
    "invalid_predictions",
    [
        np.array([[20.0], [30.0]]),
        np.array([20.0]),
        np.array([20.0, float("nan")]),
        np.array([20.0, float("inf")]),
        np.array([20.0, 0.0]),
        np.array([20.0, -1.0]),
    ],
)
def test_predict_xgboost_aft_duration_rejects_invalid_model_output(
    invalid_predictions: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """형태가 다르거나 비유한·비양수인 모델 출력을 서비스 값으로 허용하지 않습니다."""
    training_data = build_xgboost_aft_training_data(make_aft_training_rows())
    training_result = train_xgboost_aft_model(training_data)
    prediction_data = build_xgboost_aft_prediction_data(
        make_aft_training_rows().loc[[20, 30]],
        feature_columns=training_result.feature_columns,
    )
    monkeypatch.setattr(
        training_result.booster,
        "predict",
        lambda _matrix: invalid_predictions,
    )

    with pytest.raises(XGBoostAFTError, match="예측 결과|예상 재구매 소요일"):
        predict_xgboost_aft_duration(training_result, prediction_data)


def test_build_xgboost_aft_evaluation_rows_aligns_by_original_index() -> None:
    """예측 순서가 달라도 원본 행 인덱스로 정답과 정확히 다시 연결합니다."""
    rows = make_aft_training_rows()
    original_rows = rows.copy(deep=True)
    predictions = pd.Series(
        [200.0, 300.0, 100.0],
        index=[20, 30, 10],
        name="predicted_duration_days",
    )

    evaluation = build_xgboost_aft_evaluation_rows(rows, predictions)

    assert evaluation.source_sample_count == 3
    assert evaluation.excluded_zero_duration_count == 1
    assert evaluation.included_sample_count == 2
    assert evaluation.rows.index.tolist() == [30, 20]
    assert evaluation.rows["predicted_duration_days"].tolist() == [300.0, 200.0]
    assert evaluation.rows["survival_observed_duration_days"].tolist() == [20.0, 30.0]
    pd.testing.assert_frame_equal(rows, original_rows)


@pytest.mark.parametrize(
    "predictions",
    [
        pd.Series([30.0, 20.0], index=[30, 20]),
        pd.Series([30.0, 20.0, 10.0], index=[30, 20, 99]),
        pd.Series([30.0, 20.0, 10.0], index=[30, 20, 20]),
    ],
)
def test_build_xgboost_aft_evaluation_rows_rejects_unmatched_index(
    predictions: pd.Series,
) -> None:
    """누락·추가·중복된 예측 행을 내부 결합으로 조용히 버리지 않습니다."""
    with pytest.raises(XGBoostAFTError, match="인덱스"):
        build_xgboost_aft_evaluation_rows(
            make_aft_training_rows(),
            predictions,
        )


@pytest.mark.parametrize(
    "invalid_predictions",
    [
        [30.0, 20.0, float("nan")],
        [30.0, 20.0, float("inf")],
        [30.0, 20.0, 0.0],
        [30.0, 20.0, -1.0],
    ],
)
def test_build_xgboost_aft_evaluation_rows_rejects_invalid_prediction(
    invalid_predictions: list[float],
) -> None:
    """비유한·비양수 예측을 평가 지표 입력으로 허용하지 않습니다."""
    predictions = pd.Series(invalid_predictions, index=[30, 10, 20])

    with pytest.raises(XGBoostAFTError, match="0보다 큰 유한한 숫자"):
        build_xgboost_aft_evaluation_rows(
            make_aft_training_rows(),
            predictions,
        )
