"""Validation 표본에서 재구매 예측 모델 후보를 비교합니다.

모델 학습·예측 구현과 하이퍼파라미터 비교 책임을 분리하고, Test를 보지 않은
상태에서 수축 강도별 성능을 동일한 표본과 지표로 평가합니다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from .baseline import (
    HierarchicalMedianModel,
    predict_hierarchical_median_baseline,
    predict_shrunk_hierarchical_median_baseline,
)
from .error_analysis import (
    compare_error_on_fixed_cohort,
    select_largest_error_rows,
    summarize_largest_error_tail,
)
from .evaluation import (
    IPCWUserBootstrapResult,
    bootstrap_ipcw_brier_difference_by_user,
    bootstrap_ipcw_brier_pair_difference_by_user,
    evaluate_ipcw_binary_predictions,
    evaluate_ipcw_brier_score,
    evaluate_ipcw_concordance_index,
    evaluate_predictions,
    summarize_ipcw_calibration,
)
from .features import MINIMAL_MODEL_FEATURE_COLUMNS
from .lightgbm_baseline import (
    build_lightgbm_training_data,
    predict_lightgbm_repurchase_probability,
    train_lightgbm_classifier,
)
from .maturity_analysis import (
    add_split_ipcw_weights,
    add_validation_ipcw_weights,
    add_validation_survival_observation,
)
from .probability_baseline import (
    fit_global_event_probability_baseline,
    fit_hierarchical_event_probability_baseline,
    predict_hierarchical_event_probability_baseline,
)
from .xgboost_aft import (
    AFTLabelBounds,
    XGBoostAFTTrainingResult,
    build_xgboost_aft_evaluation_rows,
    calculate_xgboost_aft_event_probability,
)

IPCW_CANDIDATE_ID_COLUMNS = ("user_id", "order_id", "product_id")

# 동일한 표본에 정보를 단계적으로 추가하며, C는 기존 네 피처 기준을 재현합니다.
LIGHTGBM_FEATURE_SETS = (
    ("A_counts", ("history_interval_count", "user_prior_order_count")),
    (
        "B_counts_median",
        ("history_interval_count", "history_median_days", "user_prior_order_count"),
    ),
    ("C_counts_median_variability", MINIMAL_MODEL_FEATURE_COLUMNS),
)


@dataclass(frozen=True)
class IPCWShrinkageCandidateEvaluation:
    """후보 비교표와 기존 계층형 모델의 상세 IPCW 결과를 함께 보관합니다."""

    comparison: pd.DataFrame
    reference_binary_evaluation: dict[str, float | int | None]
    reference_concordance_evaluation: dict[str, float | int]


@dataclass(frozen=True)
class IPCWProbabilityCandidateEvaluation:
    """확률 후보 비교표와 후보별 Calibration 구간 상세를 함께 보관합니다."""

    comparison: pd.DataFrame
    calibration: pd.DataFrame
    user_bootstrap: IPCWUserBootstrapResult | None


@dataclass(frozen=True)
class XGBoostAFTProbabilityEvaluation:
    """한 AFT 후보의 Brier 요약과 구간별 Calibration 근거를 함께 보관합니다."""

    summary: dict[str, float | int | None]
    calibration: pd.DataFrame


def _validate_candidate_alignment(
    weighted_samples: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    """후보 비교 전 표본 식별자·개수·순서가 같은지 공통 검증합니다."""
    if not weighted_samples.columns.is_unique or not predictions.columns.is_unique:
        raise ValueError("IPCW 표본과 예측 데이터의 열 이름은 중복될 수 없습니다.")
    missing_columns = set(IPCW_CANDIDATE_ID_COLUMNS) - set(weighted_samples.columns)
    missing_columns |= set(IPCW_CANDIDATE_ID_COLUMNS) - set(predictions.columns)
    if missing_columns:
        raise ValueError(
            f"IPCW 후보 정렬 확인 열이 누락됐습니다: {sorted(missing_columns)}"
        )
    if len(weighted_samples) != len(predictions):
        raise ValueError("IPCW 기준 표본 수와 후보 예측 표본 수가 다릅니다.")
    for rows in (weighted_samples, predictions):
        if rows.loc[:, list(IPCW_CANDIDATE_ID_COLUMNS)].isna().any(axis=None):
            raise ValueError("IPCW 표본 식별자에는 결측값을 사용할 수 없습니다.")
    if weighted_samples.duplicated(subset=list(IPCW_CANDIDATE_ID_COLUMNS)).any():
        raise ValueError("IPCW 기준 표본 식별자가 중복됐습니다.")
    if predictions.duplicated(subset=list(IPCW_CANDIDATE_ID_COLUMNS)).any():
        raise ValueError("후보 예측 표본 식별자가 중복됐습니다.")

    weighted_ids = weighted_samples.loc[:, IPCW_CANDIDATE_ID_COLUMNS].reset_index(
        drop=True
    )
    prediction_ids = predictions.loc[:, IPCW_CANDIDATE_ID_COLUMNS].reset_index(
        drop=True
    )
    if not weighted_ids.equals(prediction_ids):
        raise ValueError("IPCW 기준 표본과 후보 예측 표본의 순서가 다릅니다.")


def _attach_probability_candidate_predictions(
    weighted_samples: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """검증된 동일 표본에 확률 예측값만 연결합니다."""
    _validate_candidate_alignment(weighted_samples, predictions)

    evaluation_rows = weighted_samples.copy()
    evaluation_rows["predicted_event_probability"] = predictions[
        "predicted_event_probability"
    ].to_numpy(copy=True)
    return evaluation_rows


def build_paired_probability_predictions(
    weighted_samples: pd.DataFrame,
    reference_predictions: pd.DataFrame,
    candidate_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """같은 구매 표본의 정답·가중치를 보존하며 기준·후보 확률을 연결합니다.

    인덱스 번호 대신 구매 식별자와 순서를 검증합니다. 확률값 범위와 IPCW
    값의 유효성은 이후 공통 평가 함수에서 검사합니다. 호출부는 두 예측의
    사건 정의·예측 기간·기준 시점이 같음을 보장해야 합니다.
    """
    paired_rows = weighted_samples.copy()
    for role, predictions in (
        ("reference", reference_predictions),
        ("candidate", candidate_predictions),
    ):
        _validate_candidate_alignment(weighted_samples, predictions)
        if "predicted_event_probability" not in predictions.columns:
            raise ValueError(
                f"{role} 예측에 predicted_event_probability 열이 없습니다."
            )
        # 키 순서를 확인했으므로 서로 다른 인덱스 번호에 의한 자동 정렬을 막습니다.
        paired_rows[f"{role}_predicted_event_probability"] = predictions[
            "predicted_event_probability"
        ].to_numpy(copy=True)
    return paired_rows


def build_lightgbm_feature_pair_predictions(
    training_samples: pd.DataFrame,
    validation_samples: pd.DataFrame,
    *,
    reference_feature_columns: Sequence[str],
    candidate_feature_columns: Sequence[str],
    horizon_days: int = 30,
) -> pd.DataFrame:
    """같은 기간·표본으로 두 피처 후보를 한 번씩 학습해 쌍 비교 입력을 만듭니다.

    같은 사건 정의로 생성한 Train·Validation을 전달해야 합니다. 검열 가중치는
    분할마다 한 번 계산해 두 후보가 공유하며, Bootstrap과 지표 계산은 하지
    않습니다. 반환한 예측표를 재사용해 후속 비교마다 재학습하지 않습니다.
    """
    if training_samples.empty or validation_samples.empty:
        raise ValueError("피처 쌍 비교에는 Train과 Validation 표본이 모두 필요합니다.")
    if (
        "split" not in validation_samples
        or not validation_samples["split"].eq("validation").fillna(False).all()
    ):
        raise ValueError("피처 쌍 비교에는 Validation 표본만 사용합니다.")
    # 구매 키의 결측·중복은 비용이 드는 학습 전에 거절합니다.
    _validate_candidate_alignment(training_samples, training_samples)
    _validate_candidate_alignment(validation_samples, validation_samples)
    weighted_training = add_split_ipcw_weights(
        training_samples, horizon_days=horizon_days
    )
    weighted_validation = add_split_ipcw_weights(
        validation_samples, horizon_days=horizon_days
    )
    # 두 후보의 입력을 모두 검증한 뒤 학습하므로 잘못된 두 번째 피처도 먼저 거절합니다.
    training_inputs = [
        build_lightgbm_training_data(weighted_training, feature_columns=columns)
        for columns in (reference_feature_columns, candidate_feature_columns)
    ]
    prediction_tables = []
    for training_data in training_inputs:
        model = train_lightgbm_classifier(training_data)
        probabilities = predict_lightgbm_repurchase_probability(
            model, validation_samples
        )
        predictions = validation_samples.loc[:, IPCW_CANDIDATE_ID_COLUMNS].copy()
        predictions["predicted_event_probability"] = probabilities.to_numpy(copy=True)
        prediction_tables.append(predictions)

    paired_rows = build_paired_probability_predictions(
        weighted_validation, prediction_tables[0], prediction_tables[1]
    )
    # 평가 시점 이후 정보를 섞지 않도록 Train에서 정답을 확인한 상품 행만 셉니다.
    product_training_counts = (
        weighted_training.loc[weighted_training["ipcw_outcome_known"]]
        .groupby("product_id", observed=True, sort=False)
        .size()
    )
    paired_rows["product_train_outcome_count"] = (
        paired_rows["product_id"].map(product_training_counts).fillna(0).astype("int64")
    )
    return paired_rows


def evaluate_lightgbm_probability_candidate(
    training_samples: pd.DataFrame,
    validation_samples: pd.DataFrame,
    *,
    horizon_days: int,
    feature_columns: Sequence[str] = MINIMAL_MODEL_FEATURE_COLUMNS,
    calibration_bin_count: int = 10,
    bootstrap_reference_product_smoothing_strength: float | None = None,
    bootstrap_replicates: int = 1_000,
    bootstrap_random_seed: int = 42,
) -> IPCWProbabilityCandidateEvaluation:
    """Train으로 LightGBM을 학습하고 동일한 Validation IPCW 기준으로 평가합니다."""
    if training_samples.empty or validation_samples.empty:
        raise ValueError("LightGBM 평가에는 Train과 Validation 표본이 모두 필요합니다.")
    if (
        "split" not in validation_samples
        or not validation_samples["split"].eq("validation").fillna(False).all()
    ):
        raise ValueError("LightGBM 후보 비교에는 Validation 표본만 사용합니다.")

    weighted_training = add_split_ipcw_weights(
        training_samples,
        horizon_days=horizon_days,
    )
    weighted_validation = add_split_ipcw_weights(
        validation_samples,
        horizon_days=horizon_days,
    )
    training_data = build_lightgbm_training_data(
        weighted_training, feature_columns=feature_columns
    )
    model = train_lightgbm_classifier(training_data)
    probabilities = predict_lightgbm_repurchase_probability(
        model,
        validation_samples,
    )

    predictions = validation_samples.loc[:, IPCW_CANDIDATE_ID_COLUMNS].copy()
    predictions["predicted_event_probability"] = probabilities.to_numpy(copy=True)
    evaluation_rows = _attach_probability_candidate_predictions(
        weighted_validation,
        predictions,
    )
    global_model = fit_global_event_probability_baseline(weighted_training)
    metrics = evaluate_ipcw_brier_score(
        evaluation_rows,
        reference_probability=global_model.global_event_probability,
    )
    calibration = summarize_ipcw_calibration(
        evaluation_rows,
        bin_count=calibration_bin_count,
    )
    calibration.insert(0, "model_candidate", "lightgbm_probability")
    calibration.insert(
        1,
        "product_smoothing_strength",
        pd.Series([None] * len(calibration), dtype="Float64"),
    )
    expected_calibration_error = float(
        calibration["weighted_absolute_gap_contribution"].sum()
    )
    maximum_calibration_error = float(calibration["absolute_calibration_gap"].max())
    user_bootstrap: IPCWUserBootstrapResult | None = None
    if bootstrap_reference_product_smoothing_strength is not None:
        reference_model = fit_hierarchical_event_probability_baseline(
            weighted_training,
            product_smoothing_strength=(bootstrap_reference_product_smoothing_strength),
        )
        reference_predictions = predict_hierarchical_event_probability_baseline(
            reference_model,
            validation_samples,
        )
        pair_rows = build_paired_probability_predictions(
            weighted_validation,
            reference_predictions,
            predictions,
        )
        user_bootstrap = bootstrap_ipcw_brier_pair_difference_by_user(
            pair_rows,
            bootstrap_replicates=bootstrap_replicates,
            random_seed=bootstrap_random_seed,
        )
    comparison = pd.DataFrame(
        [
            {
                "model_candidate": "lightgbm_probability",
                "feature_columns": list(training_data.features.columns),
                "training_sample_count": len(training_data.features),
                "training_weight_sum": float(training_data.sample_weight.sum()),
                "product_smoothing_strength": None,
                "horizon_days": int(metrics["horizon_days"]),
                "training_global_event_probability": (
                    global_model.global_event_probability
                ),
                "evaluation_sample_count": int(metrics["validation_sample_count"]),
                "outcome_known_count": int(metrics["outcome_known_count"]),
                "product_prediction_rate": None,
                "ipcw_brier_score": float(metrics["ipcw_brier_score"]),
                "ipcw_reference_brier_score": float(
                    metrics["ipcw_reference_brier_score"]
                ),
                "brier_skill_score": metrics["brier_skill_score"],
                "expected_calibration_error": expected_calibration_error,
                "maximum_calibration_error": maximum_calibration_error,
                "nonempty_calibration_bin_count": int(len(calibration)),
            }
        ]
    )
    comparison["product_smoothing_strength"] = pd.Series(
        [float("nan")],
        dtype="float64",
    )
    comparison["product_prediction_rate"] = pd.Series(
        [float("nan")],
        dtype="float64",
    )
    return IPCWProbabilityCandidateEvaluation(
        comparison=comparison,
        calibration=calibration,
        user_bootstrap=user_bootstrap,
    )


def evaluate_lightgbm_feature_sets(
    training_samples: pd.DataFrame,
    validation_samples: pd.DataFrame,
    *,
    horizon_days: int = 30,
    calibration_bin_count: int = 10,
) -> IPCWProbabilityCandidateEvaluation:
    """동일한 표본과 설정에서 횟수·중앙값·불규칙성을 단계적으로 추가합니다."""
    comparisons = []
    calibrations = []
    for name, columns in LIGHTGBM_FEATURE_SETS:
        # 결측 피처가 있는 행도 보존하고, 모델에 전달할 열만 변경합니다.
        evaluation = evaluate_lightgbm_probability_candidate(
            training_samples,
            validation_samples,
            horizon_days=horizon_days,
            calibration_bin_count=calibration_bin_count,
            feature_columns=columns,
        )
        evaluation.comparison.insert(0, "feature_set", name)
        evaluation.calibration.insert(0, "feature_set", name)
        comparisons.append(evaluation.comparison)
        calibrations.append(evaluation.calibration)

    comparison = pd.concat(comparisons, ignore_index=True)
    for column in (
        "training_sample_count",
        "training_weight_sum",
        "evaluation_sample_count",
        "outcome_known_count",
    ):
        if comparison[column].nunique(dropna=False) != 1:
            raise RuntimeError(
                f"피처 비교 후보의 공통 평가 조건이 달라졌습니다: {column}"
            )
    # 양수는 앞 후보보다 Brier 오차가 줄었다는 뜻이며 첫 후보는 비교 대상이 없습니다.
    comparison["brier_improvement_vs_previous"] = -comparison["ipcw_brier_score"].diff()
    return IPCWProbabilityCandidateEvaluation(
        comparison=comparison,
        calibration=pd.concat(calibrations, ignore_index=True),
        user_bootstrap=None,
    )


def evaluate_ipcw_probability_candidates(
    training_samples: pd.DataFrame,
    validation_samples: pd.DataFrame,
    *,
    product_smoothing_strengths: Sequence[float],
    horizon_days: int,
    calibration_bin_count: int = 10,
    bootstrap_product_smoothing_strength: float | None = None,
    bootstrap_replicates: int = 1_000,
    bootstrap_random_seed: int = 42,
) -> IPCWProbabilityCandidateEvaluation:
    """Train으로 확률 후보를 학습하고 Validation 성능과 불확실성을 비교합니다."""
    if training_samples.empty or validation_samples.empty:
        raise ValueError(
            "확률 후보 비교에는 Train과 Validation 표본이 모두 필요합니다."
        )
    if not product_smoothing_strengths:
        raise ValueError("비교할 상품 확률 수축 강도 후보가 없습니다.")

    normalized_strengths = [float(value) for value in product_smoothing_strengths]
    if len(normalized_strengths) != len(set(normalized_strengths)):
        raise ValueError("중복된 상품 확률 수축 강도 후보입니다.")
    normalized_bootstrap_strength = (
        None
        if bootstrap_product_smoothing_strength is None
        else float(bootstrap_product_smoothing_strength)
    )
    if (
        normalized_bootstrap_strength is not None
        and normalized_bootstrap_strength not in normalized_strengths
    ):
        raise ValueError("Bootstrap 대상 상품 확률 수축 강도가 후보 집합에 없습니다.")

    weighted_training = add_split_ipcw_weights(
        training_samples,
        horizon_days=horizon_days,
    )
    weighted_validation = add_split_ipcw_weights(
        validation_samples,
        horizon_days=horizon_days,
    )
    global_model = fit_global_event_probability_baseline(weighted_training)

    candidates = [
        (
            "global_event_probability",
            None,
            global_model,
        )
    ]
    candidates.extend(
        (
            "hierarchical_event_probability",
            smoothing_strength,
            fit_hierarchical_event_probability_baseline(
                weighted_training,
                product_smoothing_strength=smoothing_strength,
            ),
        )
        for smoothing_strength in normalized_strengths
    )

    results: list[dict[str, float | int | str | None]] = []
    calibration_results: list[pd.DataFrame] = []
    user_bootstrap: IPCWUserBootstrapResult | None = None
    for candidate_name, smoothing_strength, model in candidates:
        predictions = predict_hierarchical_event_probability_baseline(
            model,
            validation_samples,
        )
        evaluation_rows = _attach_probability_candidate_predictions(
            weighted_validation,
            predictions,
        )
        metrics = evaluate_ipcw_brier_score(
            evaluation_rows,
            reference_probability=global_model.global_event_probability,
        )
        calibration = summarize_ipcw_calibration(
            evaluation_rows,
            bin_count=calibration_bin_count,
        )
        calibration.insert(0, "model_candidate", candidate_name)
        calibration.insert(
            1,
            "product_smoothing_strength",
            pd.Series(
                [smoothing_strength] * len(calibration),
                index=calibration.index,
                dtype="Float64",
            ),
        )
        calibration_results.append(calibration)
        if (
            normalized_bootstrap_strength is not None
            and smoothing_strength == normalized_bootstrap_strength
        ):
            user_bootstrap = bootstrap_ipcw_brier_difference_by_user(
                evaluation_rows,
                reference_probability=global_model.global_event_probability,
                bootstrap_replicates=bootstrap_replicates,
                random_seed=bootstrap_random_seed,
            )
        expected_calibration_error = float(
            calibration["weighted_absolute_gap_contribution"].sum()
        )
        maximum_calibration_error = float(calibration["absolute_calibration_gap"].max())
        product_prediction_rate = float(
            predictions["probability_prediction_source"].eq("product_history").mean()
        )
        results.append(
            {
                "model_candidate": candidate_name,
                "product_smoothing_strength": smoothing_strength,
                "horizon_days": int(metrics["horizon_days"]),
                "training_global_event_probability": (
                    global_model.global_event_probability
                ),
                "evaluation_sample_count": int(metrics["validation_sample_count"]),
                "outcome_known_count": int(metrics["outcome_known_count"]),
                "product_prediction_rate": product_prediction_rate,
                "ipcw_brier_score": float(metrics["ipcw_brier_score"]),
                "ipcw_reference_brier_score": float(
                    metrics["ipcw_reference_brier_score"]
                ),
                "brier_skill_score": metrics["brier_skill_score"],
                "expected_calibration_error": expected_calibration_error,
                "maximum_calibration_error": maximum_calibration_error,
                "nonempty_calibration_bin_count": int(len(calibration)),
            }
        )

    return IPCWProbabilityCandidateEvaluation(
        comparison=pd.DataFrame(results),
        calibration=pd.concat(calibration_results, ignore_index=True),
        user_bootstrap=user_bootstrap,
    )


def _attach_candidate_predictions(
    weighted_samples: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """같은 표본 순서인지 확인한 뒤 후보 예측값만 IPCW 평가 행에 연결합니다."""
    _validate_candidate_alignment(weighted_samples, predictions)

    evaluation_rows = weighted_samples.copy()
    evaluation_rows["predicted_duration_days"] = predictions[
        "predicted_duration_days"
    ].to_numpy(copy=True)
    return evaluation_rows


def _prepare_xgboost_aft_ipcw_evaluation_rows(
    validation_samples: pd.DataFrame,
    predictions: pd.Series,
    *,
    horizon_days: int,
) -> AFTLabelBounds:
    """0일을 먼저 제외한 동일 Validation 집단에서 IPCW 평가 행을 만듭니다."""
    observed_samples = add_validation_survival_observation(validation_samples)
    aligned_evaluation = build_xgboost_aft_evaluation_rows(
        observed_samples,
        predictions,
    )
    included_index = aligned_evaluation.rows.index
    weighted_samples = add_validation_ipcw_weights(
        validation_samples.loc[included_index],
        horizon_days=horizon_days,
    )
    weighted_samples["predicted_duration_days"] = aligned_evaluation.rows[
        "predicted_duration_days"
    ]
    return AFTLabelBounds(
        rows=weighted_samples,
        source_sample_count=aligned_evaluation.source_sample_count,
        excluded_zero_duration_count=(aligned_evaluation.excluded_zero_duration_count),
    )


def evaluate_xgboost_aft_ipcw_concordance(
    validation_samples: pd.DataFrame,
    predictions: pd.Series,
    *,
    horizon_days: int,
) -> dict[str, float | int]:
    """AFT 평가 집단에서 공통 IPCW C-index와 표본 흐름을 계산합니다."""
    evaluation = _prepare_xgboost_aft_ipcw_evaluation_rows(
        validation_samples,
        predictions,
        horizon_days=horizon_days,
    )
    metrics = evaluate_ipcw_concordance_index(evaluation.rows)
    return {
        "source_validation_sample_count": evaluation.source_sample_count,
        "excluded_zero_duration_count": evaluation.excluded_zero_duration_count,
        "aft_evaluation_sample_count": evaluation.included_sample_count,
        **metrics,
    }


def evaluate_xgboost_aft_ipcw_brier(
    validation_samples: pd.DataFrame,
    training_result: XGBoostAFTTrainingResult,
    predictions: pd.Series,
    *,
    horizon_days: int,
    training_reference_probability: float,
) -> dict[str, float | int | None]:
    """AFT 확률의 IPCW Brier를 같은 시점의 Train 기준 확률과 비교합니다.

    training_reference_probability는 동일 horizon의 Train에서 계산해 전달하며,
    Validation의 실제 결과로 다시 추정하지 않습니다.
    """
    evaluation = _prepare_xgboost_aft_ipcw_probability_rows(
        validation_samples,
        training_result,
        predictions,
        horizon_days=horizon_days,
    )
    metrics = evaluate_ipcw_brier_score(
        evaluation.rows,
        reference_probability=training_reference_probability,
    )
    return {
        "source_validation_sample_count": evaluation.source_sample_count,
        "excluded_zero_duration_count": evaluation.excluded_zero_duration_count,
        "aft_evaluation_sample_count": evaluation.included_sample_count,
        **metrics,
    }


def _prepare_xgboost_aft_ipcw_probability_rows(
    validation_samples: pd.DataFrame,
    training_result: XGBoostAFTTrainingResult,
    predictions: pd.Series,
    *,
    horizon_days: int,
) -> AFTLabelBounds:
    """공통 AFT 평가 집단에 동일 모델의 고정 시점 재구매 확률을 추가합니다."""
    evaluation = _prepare_xgboost_aft_ipcw_evaluation_rows(
        validation_samples,
        predictions,
        horizon_days=horizon_days,
    )
    probability_rows = evaluation.rows.copy()
    probability_rows["predicted_event_probability"] = (
        calculate_xgboost_aft_event_probability(
            training_result,
            probability_rows["predicted_duration_days"],
            horizon_days=horizon_days,
        )
    )
    return AFTLabelBounds(
        rows=probability_rows,
        source_sample_count=evaluation.source_sample_count,
        excluded_zero_duration_count=evaluation.excluded_zero_duration_count,
    )


def evaluate_xgboost_aft_ipcw_probability(
    validation_samples: pd.DataFrame,
    training_result: XGBoostAFTTrainingResult,
    predictions: pd.Series,
    *,
    horizon_days: int,
    training_reference_probability: float,
    calibration_bin_count: int = 10,
) -> XGBoostAFTProbabilityEvaluation:
    """같은 AFT 확률 행에서 IPCW Brier와 Calibration을 함께 계산합니다.

    training_reference_probability는 동일 horizon의 Train에서 계산해 전달하며,
    Validation의 실제 결과로 다시 추정하지 않습니다.
    """
    evaluation = _prepare_xgboost_aft_ipcw_probability_rows(
        validation_samples,
        training_result,
        predictions,
        horizon_days=horizon_days,
    )
    brier_metrics = evaluate_ipcw_brier_score(
        evaluation.rows,
        reference_probability=training_reference_probability,
    )
    calibration = summarize_ipcw_calibration(
        evaluation.rows,
        bin_count=calibration_bin_count,
    )
    summary: dict[str, float | int | None] = {
        "source_validation_sample_count": evaluation.source_sample_count,
        "excluded_zero_duration_count": evaluation.excluded_zero_duration_count,
        "aft_evaluation_sample_count": evaluation.included_sample_count,
        **brier_metrics,
        "requested_calibration_bin_count": calibration_bin_count,
        "expected_calibration_error": float(
            calibration["weighted_absolute_gap_contribution"].sum()
        ),
        "maximum_calibration_error": float(
            calibration["absolute_calibration_gap"].max()
        ),
        "nonempty_calibration_bin_count": int(len(calibration)),
    }
    return XGBoostAFTProbabilityEvaluation(
        summary=summary,
        calibration=calibration,
    )


def evaluate_ipcw_shrinkage_candidates(
    samples: pd.DataFrame,
    model: HierarchicalMedianModel,
    *,
    shrinkage_strengths: Sequence[float],
    horizon_days: int,
) -> IPCWShrinkageCandidateEvaluation:
    """동일한 IPCW 조건에서 기존 계층형 모델과 수축 후보를 직접 비교합니다."""
    if samples.empty:
        raise ValueError("IPCW로 비교할 Validation 표본이 없습니다.")
    if not shrinkage_strengths:
        raise ValueError("IPCW로 평가할 수축 강도 후보가 없습니다.")

    weighted_samples = add_validation_ipcw_weights(
        samples,
        horizon_days=horizon_days,
    )
    candidate_predictions: list[tuple[str, float | None, pd.DataFrame]] = [
        (
            "hierarchical_median",
            None,
            predict_hierarchical_median_baseline(model, samples),
        )
    ]
    evaluated_strengths: set[float] = set()
    for shrinkage_strength in shrinkage_strengths:
        predictions = predict_shrunk_hierarchical_median_baseline(
            model,
            samples,
            shrinkage_strength=shrinkage_strength,
        )
        normalized_strength = float(predictions["shrinkage_strength"].iat[0])
        if normalized_strength in evaluated_strengths:
            raise ValueError(f"중복된 수축 강도 후보입니다: {normalized_strength}")
        evaluated_strengths.add(normalized_strength)
        candidate_predictions.append(
            (
                "shrunk_hierarchical_median",
                normalized_strength,
                predictions,
            )
        )

    results: list[dict[str, float | int | str | None]] = []
    reference_binary_evaluation: dict[str, float | int | None] | None = None
    reference_concordance_evaluation: dict[str, float | int] | None = None
    for candidate_name, shrinkage_strength, predictions in candidate_predictions:
        evaluation_rows = _attach_candidate_predictions(
            weighted_samples,
            predictions,
        )
        binary_evaluation = evaluate_ipcw_binary_predictions(evaluation_rows)
        concordance_evaluation = evaluate_ipcw_concordance_index(evaluation_rows)
        if shrinkage_strength is None:
            reference_binary_evaluation = binary_evaluation
            reference_concordance_evaluation = concordance_evaluation
        results.append(
            {
                "model_candidate": candidate_name,
                "shrinkage_strength": shrinkage_strength,
                "validation_sample_count": int(
                    binary_evaluation["validation_sample_count"]
                ),
                "outcome_known_count": int(binary_evaluation["outcome_known_count"]),
                "ipcw_weighted_binary_accuracy": float(
                    binary_evaluation["ipcw_weighted_binary_accuracy"]
                ),
                "ipcw_weighted_precision": binary_evaluation["ipcw_weighted_precision"],
                "ipcw_weighted_recall": binary_evaluation["ipcw_weighted_recall"],
                "ipcw_weighted_specificity": binary_evaluation[
                    "ipcw_weighted_specificity"
                ],
                "ipcw_weighted_balanced_accuracy": binary_evaluation[
                    "ipcw_weighted_balanced_accuracy"
                ],
                "ipcw_weighted_f1": binary_evaluation["ipcw_weighted_f1"],
                "ipcw_concordance_index": float(
                    concordance_evaluation["ipcw_concordance_index"]
                ),
            }
        )

    result = pd.DataFrame(results)
    reference = result.iloc[0]
    for metric in (
        "ipcw_weighted_binary_accuracy",
        "ipcw_weighted_balanced_accuracy",
        "ipcw_concordance_index",
    ):
        result[f"{metric}_difference_vs_reference"] = result[metric] - reference[metric]
    if reference_binary_evaluation is None or reference_concordance_evaluation is None:
        raise RuntimeError("기존 계층형 모델의 IPCW 기준 결과가 생성되지 않았습니다.")
    return IPCWShrinkageCandidateEvaluation(
        comparison=result,
        reference_binary_evaluation=reference_binary_evaluation,
        reference_concordance_evaluation=reference_concordance_evaluation,
    )


def evaluate_shrinkage_candidates(
    samples: pd.DataFrame,
    model: HierarchicalMedianModel,
    *,
    shrinkage_strengths: Sequence[float],
    tail_rate: float = 0.05,
) -> pd.DataFrame:
    """동일 Validation 표본에서 수축 강도별 전체·꼬리 성능을 계산합니다."""
    if samples.empty:
        raise ValueError("수축 강도를 평가할 Validation 표본이 없습니다.")
    if not shrinkage_strengths:
        raise ValueError("평가할 수축 강도 후보가 없습니다.")

    results: list[dict[str, float | int]] = []
    evaluated_strengths: set[float] = set()
    reference_predictions = predict_hierarchical_median_baseline(model, samples)
    fixed_tail_cohort = select_largest_error_rows(
        reference_predictions,
        tail_rate=tail_rate,
    )

    # 후보별 모델 실행은 의도된 실험 반복이며, 각 후보 내부의 행 계산은 벡터화합니다.
    for shrinkage_strength in shrinkage_strengths:
        predictions = predict_shrunk_hierarchical_median_baseline(
            model,
            samples,
            shrinkage_strength=shrinkage_strength,
        )
        normalized_strength = float(predictions["shrinkage_strength"].iat[0])
        if normalized_strength in evaluated_strengths:
            raise ValueError(f"중복된 수축 강도 후보입니다: {normalized_strength}")
        evaluated_strengths.add(normalized_strength)

        evaluation = evaluate_predictions(predictions)
        overall = evaluation["overall"]
        personal_rows = predictions.loc[
            predictions["prediction_source"].eq("shrunk_user_product_history")
        ]
        tail_metrics = summarize_largest_error_tail(
            predictions,
            tail_rate=tail_rate,
        )
        tail_sample_count = int(tail_metrics["tail_sample_count"])
        tail_absolute_error_days = float(tail_metrics["tail_absolute_error_days"])
        fixed_cohort_metrics = compare_error_on_fixed_cohort(
            reference_predictions,
            predictions,
            fixed_tail_cohort,
        )
        mean_personal_weight = (
            0.0
            if personal_rows.empty
            else float(personal_rows["personal_history_weight"].mean())
        )
        results.append(
            {
                "shrinkage_strength": normalized_strength,
                "sample_count": int(overall["sample_count"]),
                "personal_sample_count": int(len(personal_rows)),
                "mae_days": float(overall["mae_days"]),
                "median_absolute_error_days": float(
                    overall["median_absolute_error_days"]
                ),
                "within_3_days_rate": float(overall["within_3_days_rate"]),
                "within_7_days_rate": float(overall["within_7_days_rate"]),
                "mean_personal_history_weight": mean_personal_weight,
                "requested_tail_rate": float(tail_metrics["requested_tail_rate"]),
                "actual_tail_sample_rate": float(
                    tail_metrics["actual_tail_sample_rate"]
                ),
                "tail_sample_count": tail_sample_count,
                "tail_absolute_error_days": tail_absolute_error_days,
                "tail_mae_days": tail_absolute_error_days / tail_sample_count,
                "tail_absolute_error_share": float(
                    tail_metrics["absolute_error_share"]
                ),
                "tail_late_prediction_rate": float(
                    tail_metrics["late_prediction_rate"]
                ),
                "fixed_cohort_sample_count": int(
                    fixed_cohort_metrics["cohort_sample_count"]
                ),
                "fixed_cohort_reference_mae_days": float(
                    fixed_cohort_metrics["reference_mae_days"]
                ),
                "fixed_cohort_candidate_mae_days": float(
                    fixed_cohort_metrics["candidate_mae_days"]
                ),
                "fixed_cohort_mae_improvement_days": float(
                    fixed_cohort_metrics["mae_improvement_days"]
                ),
                "fixed_cohort_improved_sample_count": int(
                    fixed_cohort_metrics["improved_sample_count"]
                ),
                "fixed_cohort_improved_sample_rate": float(
                    fixed_cohort_metrics["improved_sample_rate"]
                ),
                "fixed_cohort_worsened_sample_count": int(
                    fixed_cohort_metrics["worsened_sample_count"]
                ),
                "fixed_cohort_worsened_sample_rate": float(
                    fixed_cohort_metrics["worsened_sample_rate"]
                ),
                "fixed_cohort_late_prediction_rate": float(
                    fixed_cohort_metrics["candidate_late_prediction_rate"]
                ),
            }
        )

    # 입력한 후보 순서를 보존해 강도가 커질 때 지표 변화를 그대로 비교합니다.
    return pd.DataFrame(results)
