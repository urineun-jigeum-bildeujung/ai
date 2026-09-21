"""UCI Train으로 XGBoost AFT를 학습하고 Validation에서 평가합니다.

공통 전처리·시간 분할을 재사용하되 기존 베이스라인 E2E와 실행 책임을
분리합니다. Test는 모델 선택이 끝나기 전까지 열어보지 않습니다.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from math import isclose, isfinite, log
from numbers import Integral, Real
from typing import Final

import pandas as pd

from .loaders import load_uci_online_retail_ii
from .modeling.baseline import (
    fit_hierarchical_median_baseline,
    predict_hierarchical_median_baseline,
    predict_shrunk_hierarchical_median_baseline,
)
from .modeling.evaluation import (
    IPCWUserBootstrapResult,
    bootstrap_ipcw_brier_pair_difference_by_user,
    calculate_regression_metrics,
    evaluate_ipcw_brier_score,
    evaluate_ipcw_concordance_index,
    summarize_ipcw_calibration,
    summarize_ipcw_probability_pair_by_count_segment,
)
from .modeling.lightgbm_baseline import (
    build_lightgbm_training_data,
    predict_lightgbm_repurchase_probability,
    train_lightgbm_classifier,
)
from .modeling.maturity_analysis import (
    add_split_ipcw_weights,
    add_split_survival_observation,
)
from .modeling.model_selection import (
    IPCW_CANDIDATE_ID_COLUMNS,
    XGBoostAFTProbabilityEvaluation,
    build_paired_probability_predictions,
    evaluate_xgboost_aft_ipcw_concordance,
    evaluate_xgboost_aft_ipcw_probability,
)
from .modeling.probability_baseline import fit_global_event_probability_baseline
from .modeling.rolling_validation import validate_temporal_split_boundaries
from .modeling.samples import (
    TemporalSplit,
    assign_temporal_splits,
    build_historical_interval_features,
    make_temporal_split,
)
from .modeling.xgboost_aft import (
    AFT_LOSS_DISTRIBUTIONS,
    XGBoostAFTNumericalPredictionError,
    XGBoostAFTPredictionData,
    XGBoostAFTTrainingData,
    build_xgboost_aft_prediction_data,
    build_xgboost_aft_training_data,
    predict_xgboost_aft_duration,
    train_xgboost_aft_model,
)
from .paths import REPORT_DIR
from .preprocessing.events import build_uci_purchase_events
from .preprocessing.labels import build_same_product_repurchase_labels
from .preprocessing.uci import classify_uci_rows
from .reporting import dataframe_to_nullable_records, write_text_atomically

HORIZON_DAYS: Final[int] = 30
CALIBRATION_BIN_COUNT: Final[int] = 10
BOOTSTRAP_REPLICATES: Final[int] = 1_000
BOOTSTRAP_RANDOM_SEED: Final[int] = 42
AFT_NUM_BOOST_ROUND: Final[int] = 5
AFT_BOOST_ROUND_CANDIDATES: Final[tuple[int, ...]] = (5, 20, 50, 100)
AFT_LOSS_DISTRIBUTION_CANDIDATES: Final[tuple[str, ...]] = (
    "normal",
    "logistic",
    "extreme",
)
AFT_LOGISTIC_SCALE_CANDIDATES: Final[tuple[float, ...]] = (
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
)
JSON_REPORT_PATH = REPORT_DIR / "uci_xgboost_aft_evaluation.json"
MARKDOWN_REPORT_PATH = REPORT_DIR / "uci_xgboost_aft_evaluation.md"
BOOTSTRAP_TRIALS_REPORT_PATH = REPORT_DIR / "uci_xgboost_aft_bootstrap_trials.json"
ROUND_COMPARISON_JSON_REPORT_PATH = (
    REPORT_DIR / "uci_xgboost_aft_boost_round_comparison.json"
)
ROUND_COMPARISON_MARKDOWN_REPORT_PATH = (
    REPORT_DIR / "uci_xgboost_aft_boost_round_comparison.md"
)
DISTRIBUTION_COMPARISON_JSON_REPORT_PATH = (
    REPORT_DIR / "uci_xgboost_aft_distribution_comparison.json"
)
DISTRIBUTION_COMPARISON_MARKDOWN_REPORT_PATH = (
    REPORT_DIR / "uci_xgboost_aft_distribution_comparison.md"
)
LOGISTIC_SCALE_COMPARISON_JSON_REPORT_PATH = (
    REPORT_DIR / "uci_xgboost_aft_logistic_scale_comparison.json"
)
LOGISTIC_SCALE_COMPARISON_MARKDOWN_REPORT_PATH = (
    REPORT_DIR / "uci_xgboost_aft_logistic_scale_comparison.md"
)
CANDIDATE_PAIR_BOOTSTRAP_JSON_REPORT_PATH = (
    REPORT_DIR / "uci_xgboost_aft_candidate_pair_bootstrap.json"
)
CANDIDATE_PAIR_BOOTSTRAP_MARKDOWN_REPORT_PATH = (
    REPORT_DIR / "uci_xgboost_aft_candidate_pair_bootstrap.md"
)
CANDIDATE_PAIR_BOOTSTRAP_TRIALS_REPORT_PATH = (
    REPORT_DIR / "uci_xgboost_aft_candidate_pair_bootstrap_trials.json"
)
SEGMENT_COMPARISON_JSON_REPORT_PATH = (
    REPORT_DIR / "uci_xgboost_aft_segment_comparison.json"
)
SEGMENT_COMPARISON_MARKDOWN_REPORT_PATH = (
    REPORT_DIR / "uci_xgboost_aft_segment_comparison.md"
)
AFT_SEGMENT_COUNT_COLUMNS: Final[tuple[str, ...]] = (
    "history_interval_count",
    "user_prior_order_count",
)
AFT_TIME_COMPARISON_SHRINKAGE_STRENGTHS: Final[tuple[float, ...]] = (
    1.0,
    2.0,
    4.0,
    8.0,
)
TIME_COMPARISON_JSON_REPORT_PATH = (
    REPORT_DIR / "uci_xgboost_aft_observed_time_comparison.json"
)
TIME_COMPARISON_MARKDOWN_REPORT_PATH = (
    REPORT_DIR / "uci_xgboost_aft_observed_time_comparison.md"
)
LIGHTGBM_COMPARISON_JSON_REPORT_PATH = (
    REPORT_DIR / "uci_xgboost_aft_lightgbm_comparison.json"
)
LIGHTGBM_COMPARISON_MARKDOWN_REPORT_PATH = (
    REPORT_DIR / "uci_xgboost_aft_lightgbm_comparison.md"
)
LIGHTGBM_COMPARISON_TRIALS_REPORT_PATH = (
    REPORT_DIR / "uci_xgboost_aft_lightgbm_bootstrap_trials.json"
)


@dataclass(frozen=True)
class XGBoostAFTExperimentResult:
    """AFT 학습 근거와 Validation 평가 결과를 한 실행 단위로 보관합니다."""

    split: TemporalSplit
    training_summary: dict[str, object]
    validation_predictions: pd.Series
    concordance: dict[str, float | int]
    probability: XGBoostAFTProbabilityEvaluation


@dataclass(frozen=True)
class XGBoostAFTPreparedExperiment:
    """후보 모델들이 공통으로 사용할 시간 분할·학습·평가 데이터를 보관합니다.

    frozen은 필드 교체를 막지만 내부 DataFrame까지 불변으로 만들지는 않으므로,
    후보 평가 함수는 전달받은 공통 데이터를 직접 수정하지 않아야 합니다.
    """

    horizon_days: int
    split: TemporalSplit
    training: pd.DataFrame
    training_data: XGBoostAFTTrainingData
    validation: pd.DataFrame
    prediction_data: XGBoostAFTPredictionData
    training_reference_probability: float
    reference_population_sample_count: int


@dataclass(frozen=True)
class XGBoostAFTLightGBMComparison:
    """동일 모집단 AFT·LightGBM 확률 비교 결과를 함께 보관합니다."""

    comparison: pd.DataFrame
    calibration: pd.DataFrame
    segments: pd.DataFrame
    paired_rows: pd.DataFrame
    user_bootstrap: IPCWUserBootstrapResult


def prepare_xgboost_aft_experiment(
    labels: pd.DataFrame,
    *,
    horizon_days: int = HORIZON_DAYS,
    split: TemporalSplit | None = None,
) -> XGBoostAFTPreparedExperiment:
    """후보마다 반복할 필요가 없는 피처·분할·평가 기준을 한 번만 준비합니다."""
    samples = build_historical_interval_features(labels)
    selected_split = make_temporal_split(samples) if split is None else split
    validate_temporal_split_boundaries(selected_split)
    samples = assign_temporal_splits(samples, selected_split)
    training = samples.loc[samples["split"].eq("train")].copy()
    validation = samples.loc[samples["split"].eq("validation")].copy()

    # AFT 학습은 고정 시점 IPCW와 분리해 생존시간·검열 정보만 사용합니다.
    survival_training = add_split_survival_observation(training)
    training_data = build_xgboost_aft_training_data(survival_training)

    # 모든 후보가 동일한 Validation 행과 피처 이름·순서를 사용하게 고정합니다.
    prediction_data = build_xgboost_aft_prediction_data(
        validation,
        feature_columns=training_data.feature_columns,
    )

    # AFT와 기준선이 같은 Train 모집단을 보도록 0일 제외 후 IPCW를 재계산합니다.
    reference_training = training.loc[training_data.row_index].copy()
    weighted_reference_training = add_split_ipcw_weights(
        reference_training,
        horizon_days=horizon_days,
    )
    global_probability_model = fit_global_event_probability_baseline(
        weighted_reference_training
    )
    return XGBoostAFTPreparedExperiment(
        horizon_days=horizon_days,
        split=selected_split,
        training=training,
        training_data=training_data,
        validation=validation,
        prediction_data=prediction_data,
        training_reference_probability=(
            global_probability_model.global_event_probability
        ),
        reference_population_sample_count=len(weighted_reference_training),
    )


def run_xgboost_aft_experiment(
    labels: pd.DataFrame,
    *,
    horizon_days: int = HORIZON_DAYS,
    calibration_bin_count: int = CALIBRATION_BIN_COUNT,
    bootstrap_replicates: int | None = BOOTSTRAP_REPLICATES,
    bootstrap_random_seed: int = BOOTSTRAP_RANDOM_SEED,
    loss_distribution: str = "normal",
    loss_distribution_scale: float = 1.0,
    num_boost_round: int = AFT_NUM_BOOST_ROUND,
) -> XGBoostAFTExperimentResult:
    """같은 시간 분할에서 AFT를 학습하고 Validation 성능만 계산합니다."""
    prepared = prepare_xgboost_aft_experiment(
        labels,
        horizon_days=horizon_days,
    )
    return evaluate_xgboost_aft_candidate(
        prepared,
        calibration_bin_count=calibration_bin_count,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_random_seed=bootstrap_random_seed,
        loss_distribution=loss_distribution,
        loss_distribution_scale=loss_distribution_scale,
        num_boost_round=num_boost_round,
    )


def evaluate_xgboost_aft_candidate(
    prepared: XGBoostAFTPreparedExperiment,
    *,
    calibration_bin_count: int = CALIBRATION_BIN_COUNT,
    bootstrap_replicates: int | None = BOOTSTRAP_REPLICATES,
    bootstrap_random_seed: int = BOOTSTRAP_RANDOM_SEED,
    loss_distribution: str = "normal",
    loss_distribution_scale: float = 1.0,
    num_boost_round: int = AFT_NUM_BOOST_ROUND,
) -> XGBoostAFTExperimentResult:
    """공통 준비 데이터를 사용해 AFT 후보 하나를 학습하고 평가합니다."""
    training_result = train_xgboost_aft_model(
        prepared.training_data,
        loss_distribution=loss_distribution,
        loss_distribution_scale=loss_distribution_scale,
        num_boost_round=num_boost_round,
    )

    validation_predictions = predict_xgboost_aft_duration(
        training_result,
        prepared.prediction_data,
    )

    concordance = evaluate_xgboost_aft_ipcw_concordance(
        prepared.validation,
        validation_predictions,
        horizon_days=prepared.horizon_days,
    )
    probability = evaluate_xgboost_aft_ipcw_probability(
        prepared.validation,
        training_result,
        validation_predictions,
        horizon_days=prepared.horizon_days,
        training_reference_probability=prepared.training_reference_probability,
        calibration_bin_count=calibration_bin_count,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_random_seed=bootstrap_random_seed,
    )

    training_summary: dict[str, object] = {
        "training_split": "train",
        "evaluation_split": "validation",
        "trained_until": prepared.split.train_end_at,
        "source_sample_count": prepared.training_data.source_sample_count,
        "excluded_zero_duration_count": (
            prepared.training_data.excluded_zero_duration_count
        ),
        "included_sample_count": prepared.training_data.included_sample_count,
        "reference_population_sample_count": (
            prepared.reference_population_sample_count
        ),
        "feature_columns": list(training_result.feature_columns),
        "loss_distribution": training_result.loss_distribution,
        "loss_distribution_scale": training_result.loss_distribution_scale,
        "num_boost_round": training_result.num_boost_round,
        "training_aft_nloglik": list(training_result.training_aft_nloglik),
        "training_reference_probability": prepared.training_reference_probability,
    }
    return XGBoostAFTExperimentResult(
        split=prepared.split,
        training_summary=training_summary,
        validation_predictions=validation_predictions,
        concordance=concordance,
        probability=probability,
    )


def evaluate_xgboost_aft_observed_event_time(
    result: XGBoostAFTExperimentResult,
) -> dict[str, float | int]:
    """실제 재구매가 관측된 AFT 행에서만 시점 오차를 계산합니다."""
    rows = result.probability.rows
    required_columns = {
        "survival_event_observed",
        "survival_observed_duration_days",
        "predicted_duration_days",
    }
    missing_columns = required_columns - set(rows.columns)
    if missing_columns:
        raise ValueError(
            f"AFT 관측 사건 시점 평가 컬럼이 누락됐습니다: {sorted(missing_columns)}"
        )
    event_observed = rows["survival_event_observed"]
    if event_observed.isna().any() or not pd.api.types.is_bool_dtype(
        event_observed.dtype
    ):
        raise ValueError(
            "AFT 관측 사건 여부에는 결측값 없는 boolean만 사용할 수 있습니다."
        )
    observed_rows = rows.loc[event_observed].copy()
    if observed_rows.empty:
        raise ValueError("시점 오차를 계산할 관측 재구매 사건이 없습니다.")
    observed_rows["target_duration_days"] = observed_rows[
        "survival_observed_duration_days"
    ]
    observed_rows["prediction_source"] = "xgboost_aft"
    return calculate_regression_metrics(observed_rows)


def _validate_xgboost_aft_result_matches_prepared(
    prepared: XGBoostAFTPreparedExperiment,
    result: XGBoostAFTExperimentResult,
) -> pd.DataFrame:
    """AFT 결과가 동일한 준비 데이터에서 생성됐는지 계약값으로 검증합니다."""
    if result.split != prepared.split:
        raise ValueError("AFT 결과와 준비 데이터의 시간 분할이 다릅니다.")
    expected_training_summary = {
        "trained_until": prepared.split.train_end_at,
        "source_sample_count": len(prepared.training),
        "included_sample_count": len(prepared.training_data.row_index),
        "feature_columns": list(prepared.training_data.feature_columns),
        "training_reference_probability": prepared.training_reference_probability,
    }
    for field, expected_value in expected_training_summary.items():
        if result.training_summary.get(field) != expected_value:
            raise ValueError(f"AFT 결과와 준비 데이터의 {field} 값이 다릅니다.")
    if int(result.probability.summary["horizon_days"]) != prepared.horizon_days:
        raise ValueError("AFT 결과와 준비 데이터의 horizon_days 값이 다릅니다.")

    aft_rows = result.probability.rows
    if (
        aft_rows["ipcw_horizon_days"].nunique() != 1
        or int(aft_rows["ipcw_horizon_days"].iat[0]) != prepared.horizon_days
    ):
        raise ValueError("AFT 평가 행과 준비 데이터의 horizon_days 값이 다릅니다.")
    expected_validation = add_split_survival_observation(prepared.validation)
    expected_validation = expected_validation.loc[
        expected_validation["survival_observed_duration_days"].ne(0)
    ]
    if not aft_rows.index.equals(expected_validation.index):
        raise ValueError("AFT 결과와 준비 데이터의 0일 제외 Validation 행이 다릅니다.")
    comparison_contract_columns = (
        *IPCW_CANDIDATE_ID_COLUMNS,
        *prepared.training_data.feature_columns,
        "target_duration_days",
        "survival_event_observed",
        "survival_observed_duration_days",
    )
    for column in comparison_contract_columns:
        if not aft_rows[column].equals(expected_validation[column]):
            raise ValueError(f"AFT 결과와 준비 데이터의 {column} 값이 다릅니다.")
    return expected_validation


def compare_xgboost_aft_observed_time_with_median_baselines(
    prepared: XGBoostAFTPreparedExperiment,
    result: XGBoostAFTExperimentResult,
    *,
    shrinkage_strengths: Sequence[float] = (AFT_TIME_COMPARISON_SHRINKAGE_STRENGTHS),
) -> pd.DataFrame:
    """같은 AFT 학습·Validation 표본에서 시점 예측 후보를 비교합니다."""
    strengths = tuple(shrinkage_strengths)
    if not strengths:
        raise ValueError("시점 비교에는 수축 강도 후보가 하나 이상 필요합니다.")
    if len(set(strengths)) != len(strengths):
        raise ValueError("시점 비교 수축 강도 후보에는 중복을 사용할 수 없습니다.")
    _validate_xgboost_aft_result_matches_prepared(prepared, result)

    # AFT가 실제로 사용한 0일 제외 Train 행만 중앙값 후보에도 제공합니다.
    training_rows = prepared.training.loc[prepared.training_data.row_index].copy()
    survival_training = add_split_survival_observation(training_rows)
    aft_censored_training_count = int(
        (~survival_training["survival_event_observed"]).sum()
    )
    median_model = fit_hierarchical_median_baseline(
        training_rows,
        trained_until=prepared.split.train_end_at,
    )
    median_training_sample_count = int(
        survival_training["survival_event_observed"].sum()
    )
    if median_model.global_observation_count != median_training_sample_count:
        raise RuntimeError(
            "중앙값 모델의 실제 학습 사건 수가 공통 생존 관측 정의와 다릅니다."
        )

    aft_rows = result.probability.rows
    observed_index = aft_rows.index[aft_rows["survival_event_observed"]]
    if observed_index.empty:
        raise ValueError("시점 후보를 비교할 관측 재구매 사건이 없습니다.")
    if not observed_index.isin(prepared.validation.index).all():
        raise ValueError("AFT 관측 사건을 공통 Validation 원본에 연결할 수 없습니다.")
    validation_rows = prepared.validation.loc[observed_index].copy()
    if validation_rows["target_duration_days"].isna().any():
        raise ValueError("관측 사건의 실제 재구매 간격이 비어 있습니다.")

    # AFT와 중앙값 후보가 같은 행 순서와 같은 정답을 사용하도록 명시합니다.
    aft_predictions = validation_rows.copy()
    aft_predictions["predicted_duration_days"] = aft_rows.loc[
        observed_index,
        "predicted_duration_days",
    ]
    aft_predictions["prediction_source"] = "xgboost_aft"

    candidates: list[tuple[str, float | None, pd.DataFrame]] = [
        ("xgboost_aft", None, aft_predictions),
        (
            "hierarchical_median",
            None,
            predict_hierarchical_median_baseline(median_model, validation_rows),
        ),
    ]
    for strength in strengths:
        predictions = predict_shrunk_hierarchical_median_baseline(
            median_model,
            validation_rows,
            shrinkage_strength=strength,
        )
        normalized_strength = float(predictions["shrinkage_strength"].iat[0])
        candidates.append(
            ("shrunk_hierarchical_median", normalized_strength, predictions)
        )

    rows: list[dict[str, float | int | str | None]] = []
    for candidate_name, strength, predictions in candidates:
        if not predictions.index.equals(observed_index):
            raise RuntimeError(
                f"{candidate_name} 시점 예측의 Validation 행 순서가 달라졌습니다."
            )
        metrics = calculate_regression_metrics(predictions)
        is_aft = candidate_name == "xgboost_aft"
        rows.append(
            {
                "candidate_name": candidate_name,
                "shrinkage_strength": strength,
                "sample_count": int(metrics["sample_count"]),
                "mae_days": float(metrics["mae_days"]),
                "median_absolute_error_days": float(
                    metrics["median_absolute_error_days"]
                ),
                "within_3_days_rate": float(metrics["within_3_days_rate"]),
                "within_7_days_rate": float(metrics["within_7_days_rate"]),
                "common_training_source_sample_count": len(training_rows),
                "actual_training_sample_count": (
                    len(training_rows) if is_aft else median_training_sample_count
                ),
                "used_censored_training_sample_count": (
                    aft_censored_training_count if is_aft else 0
                ),
            }
        )

    comparison = pd.DataFrame(rows)
    if comparison["sample_count"].nunique() != 1:
        raise RuntimeError("시점 예측 후보별 평가 표본 수가 서로 다릅니다.")
    aft_metrics = comparison.loc[comparison["candidate_name"].eq("xgboost_aft")]
    if len(aft_metrics) != 1:
        raise RuntimeError("시점 비교에는 XGBoost AFT 기준 결과가 하나여야 합니다.")
    aft_metrics = aft_metrics.iloc[0]
    comparison["mae_improvement_vs_aft_days"] = (
        float(aft_metrics["mae_days"]) - comparison["mae_days"]
    )
    comparison["median_ae_improvement_vs_aft_days"] = (
        float(aft_metrics["median_absolute_error_days"])
        - comparison["median_absolute_error_days"]
    )
    comparison["within_7_days_rate_improvement_vs_aft"] = comparison[
        "within_7_days_rate"
    ] - float(aft_metrics["within_7_days_rate"])
    return comparison


def _nullable_float(value: object) -> float | None:
    """pandas 결측값은 JSON의 비표준 NaN 대신 null로 바꿉니다."""
    return None if pd.isna(value) else float(value)


def build_xgboost_aft_observed_time_comparison_report(
    result: XGBoostAFTExperimentResult,
    comparison: pd.DataFrame,
) -> dict[str, object]:
    """같은 관측 사건에서 AFT와 중앙값 후보의 시점 오차를 요약합니다."""
    required_columns = {
        "candidate_name",
        "shrinkage_strength",
        "sample_count",
        "mae_days",
        "median_absolute_error_days",
        "within_3_days_rate",
        "within_7_days_rate",
        "mae_improvement_vs_aft_days",
        "median_ae_improvement_vs_aft_days",
        "within_7_days_rate_improvement_vs_aft",
        "common_training_source_sample_count",
        "actual_training_sample_count",
        "used_censored_training_sample_count",
    }
    missing_columns = required_columns - set(comparison.columns)
    if missing_columns:
        raise ValueError(
            f"AFT 시점 비교 보고서 컬럼이 누락됐습니다: {sorted(missing_columns)}"
        )
    if comparison.empty or comparison["sample_count"].nunique() != 1:
        raise ValueError(
            "AFT 시점 비교 후보는 동일한 비어 있지 않은 표본이어야 합니다."
        )

    best_mae = comparison.loc[comparison["mae_days"].idxmin()]
    best_median_ae = comparison.loc[comparison["median_absolute_error_days"].idxmin()]
    best_within_seven = comparison.loc[comparison["within_7_days_rate"].idxmax()]
    aft_row = comparison.loc[comparison["candidate_name"].eq("xgboost_aft")]
    median_row = comparison.loc[comparison["candidate_name"].eq("hierarchical_median")]
    if len(aft_row) != 1 or len(median_row) != 1:
        raise ValueError("시점 비교에는 AFT와 계층형 중앙값 기준이 하나씩 필요합니다.")
    aft_row = aft_row.iloc[0]
    median_row = median_row.iloc[0]
    return {
        "dataset": "uci_online_retail_ii",
        "experiment_version": "xgboost_aft_observed_time_comparison_v1",
        "evaluation_split": "validation",
        "model": _build_xgboost_aft_candidate_identity(result),
        "evaluation_sample_count": int(comparison["sample_count"].iat[0]),
        "training_population": {
            "common_zero_duration_excluded_source_sample_count": int(
                aft_row["common_training_source_sample_count"]
            ),
            "aft_actual_training_sample_count": int(
                aft_row["actual_training_sample_count"]
            ),
            "aft_used_censored_training_sample_count": int(
                aft_row["used_censored_training_sample_count"]
            ),
            "median_actual_observed_event_training_sample_count": int(
                median_row["actual_training_sample_count"]
            ),
        },
        "candidates": dataframe_to_nullable_records(comparison),
        "best_mae_candidate": {
            "candidate_name": str(best_mae["candidate_name"]),
            "shrinkage_strength": _nullable_float(best_mae["shrinkage_strength"]),
            "mae_days": float(best_mae["mae_days"]),
        },
        "best_median_ae_candidate": {
            "candidate_name": str(best_median_ae["candidate_name"]),
            "shrinkage_strength": _nullable_float(best_median_ae["shrinkage_strength"]),
            "median_absolute_error_days": float(
                best_median_ae["median_absolute_error_days"]
            ),
        },
        "best_within_7_days_candidate": {
            "candidate_name": str(best_within_seven["candidate_name"]),
            "shrinkage_strength": _nullable_float(
                best_within_seven["shrinkage_strength"]
            ),
            "within_7_days_rate": float(best_within_seven["within_7_days_rate"]),
        },
        "scope": (
            "두 모델은 동일한 0일 제외 Train 원본에서 시작합니다. AFT는 관측 "
            "사건과 우측검열 행을 함께 학습하고, 중앙값 후보는 Train 종료 전에 "
            "확정된 관측 사건만 학습하는 알고리즘 차이가 있습니다. "
            "AFT Validation 중 실제 재구매가 관측된 동일 행에서만 시점 오차를 "
            "비교했습니다. 검열 행과 Test는 사용하지 않았습니다. 이 결과는 "
            "30일 이내 확률 성능이 아니라 정확한 구매 시점 예측 역할을 비교합니다."
        ),
    }


def render_xgboost_aft_observed_time_comparison_report(
    report: dict[str, object],
) -> str:
    """동일 표본 시점 예측 비교 결과를 사람이 검토할 Markdown으로 만듭니다."""
    candidate_lines = []
    for row in report["candidates"]:
        strength = row["shrinkage_strength"]
        label = str(row["candidate_name"])
        if strength is not None:
            label = f"{label} (k={float(strength):g})"
        candidate_lines.append(
            f"| {label} | {row['sample_count']:,} | {row['mae_days']:.2f}일 | "
            f"{row['median_absolute_error_days']:.2f}일 | "
            f"{row['within_7_days_rate']:.2%} | "
            f"{row['mae_improvement_vs_aft_days']:+.2f}일 | "
            f"{row['within_7_days_rate_improvement_vs_aft']:+.2%}p |"
        )
    model = report["model"]
    training = report["training_population"]
    lines = [
        "# UCI XGBoost AFT 관측 시점 동일 표본 비교",
        "",
        f"- AFT 설정: `{model['loss_distribution']}`, "
        f"`scale={model['loss_distribution_scale']}`, "
        f"`{model['num_boost_round']} rounds`",
        f"- 공통 관측 사건: `{report['evaluation_sample_count']:,}`건",
        f"- 공통 0일 제외 Train 원본: "
        f"`{training['common_zero_duration_excluded_source_sample_count']:,}`건",
        f"- AFT 실제 학습: `{training['aft_actual_training_sample_count']:,}`건 "
        f"(우측검열 `{training['aft_used_censored_training_sample_count']:,}`건 포함)",
        "- 중앙값 후보 실제 학습: "
        f"`{training['median_actual_observed_event_training_sample_count']:,}`건 "
        "(확정된 관측 사건만)",
        "- AFT 대비 개선량은 양수일수록 비교 후보가 더 좋습니다.",
        "",
        "| 후보 | 표본 | MAE | Median AE | ±7일 | MAE 개선 | ±7일 개선 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *candidate_lines,
        "",
        str(report["scope"]),
        "",
    ]
    return "\n".join(lines)


def compare_xgboost_aft_with_lightgbm_probability(
    prepared: XGBoostAFTPreparedExperiment,
    result: XGBoostAFTExperimentResult,
    *,
    calibration_bin_count: int = CALIBRATION_BIN_COUNT,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_random_seed: int = BOOTSTRAP_RANDOM_SEED,
) -> XGBoostAFTLightGBMComparison:
    """동일 학습 원본·Validation·피처에서 AFT와 LightGBM을 비교합니다."""
    _validate_xgboost_aft_result_matches_prepared(prepared, result)
    common_training = prepared.training.loc[prepared.training_data.row_index].copy()
    weighted_training = add_split_ipcw_weights(
        common_training,
        horizon_days=prepared.horizon_days,
    )
    lightgbm_training = build_lightgbm_training_data(
        weighted_training,
        feature_columns=prepared.training_data.feature_columns,
    )
    lightgbm_model = train_lightgbm_classifier(lightgbm_training)

    aft_rows = result.probability.rows.copy()
    validation_rows = prepared.validation.loc[aft_rows.index].copy()
    lightgbm_probability = predict_lightgbm_repurchase_probability(
        lightgbm_model,
        validation_rows,
    )
    if not lightgbm_probability.index.equals(aft_rows.index):
        raise RuntimeError("LightGBM 확률과 AFT Validation 행 순서가 다릅니다.")

    lightgbm_rows = aft_rows.copy()
    lightgbm_rows["predicted_event_probability"] = lightgbm_probability
    # C-index에는 실제 일수가 아니라 확률과 반대 방향인 순위 점수를 사용합니다.
    lightgbm_concordance_rows = lightgbm_rows.copy()
    lightgbm_concordance_rows["predicted_duration_days"] = lightgbm_probability.rsub(
        1.0
    )

    model_rows = {
        "xgboost_aft": aft_rows,
        "lightgbm_probability": lightgbm_rows,
    }
    training_counts = {
        "xgboost_aft": len(common_training),
        "lightgbm_probability": len(lightgbm_training.features),
    }
    concordance_rows = {
        "xgboost_aft": aft_rows,
        "lightgbm_probability": lightgbm_concordance_rows,
    }
    comparison_rows: list[dict[str, object]] = []
    calibration_tables = []
    for model_name in ("xgboost_aft", "lightgbm_probability"):
        probability_rows = model_rows[model_name]
        brier = evaluate_ipcw_brier_score(
            probability_rows,
            reference_probability=prepared.training_reference_probability,
        )
        calibration = summarize_ipcw_calibration(
            probability_rows,
            bin_count=calibration_bin_count,
        )
        calibration.insert(0, "model_candidate", model_name)
        calibration_tables.append(calibration)
        concordance = evaluate_ipcw_concordance_index(concordance_rows[model_name])
        comparison_rows.append(
            {
                "model_candidate": model_name,
                "feature_columns": list(prepared.training_data.feature_columns),
                "common_training_source_sample_count": len(common_training),
                "actual_training_sample_count": training_counts[model_name],
                "evaluation_sample_count": int(brier["validation_sample_count"]),
                "outcome_known_count": int(brier["outcome_known_count"]),
                "ipcw_brier_score": float(brier["ipcw_brier_score"]),
                "ipcw_reference_brier_score": float(
                    brier["ipcw_reference_brier_score"]
                ),
                "brier_skill_score": brier["brier_skill_score"],
                "ipcw_concordance_index": float(concordance["ipcw_concordance_index"]),
                "expected_calibration_error": float(
                    calibration["weighted_absolute_gap_contribution"].sum()
                ),
                "maximum_calibration_error": float(
                    calibration["absolute_calibration_gap"].max()
                ),
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    fixed_columns = (
        "feature_columns",
        "common_training_source_sample_count",
        "evaluation_sample_count",
        "outcome_known_count",
        "ipcw_reference_brier_score",
    )
    for column in fixed_columns:
        if comparison[column].map(str).nunique() != 1:
            raise RuntimeError(f"AFT·LightGBM 공통 비교 조건이 다릅니다: {column}")

    aft_predictions = aft_rows.loc[:, IPCW_CANDIDATE_ID_COLUMNS].copy()
    aft_predictions["predicted_event_probability"] = aft_rows[
        "predicted_event_probability"
    ]
    lightgbm_predictions = lightgbm_rows.loc[:, IPCW_CANDIDATE_ID_COLUMNS].copy()
    lightgbm_predictions["predicted_event_probability"] = lightgbm_rows[
        "predicted_event_probability"
    ]
    paired_rows = build_paired_probability_predictions(
        aft_rows,
        aft_predictions,
        lightgbm_predictions,
    )
    user_bootstrap = bootstrap_ipcw_brier_pair_difference_by_user(
        paired_rows,
        bootstrap_replicates=bootstrap_replicates,
        random_seed=bootstrap_random_seed,
    )
    segment_tables = [
        summarize_ipcw_probability_pair_by_count_segment(
            paired_rows,
            count_column=count_column,
            calibration_bin_count=calibration_bin_count,
        )
        for count_column in AFT_SEGMENT_COUNT_COLUMNS
    ]
    segments = pd.concat(segment_tables, ignore_index=True)
    segments["validation_lower_brier_model"] = segments["brier_improvement"].map(
        lambda improvement: (
            "insufficient_outcome"
            if pd.isna(improvement)
            else ("lightgbm_probability" if improvement > 0 else "xgboost_aft")
        )
    )
    return XGBoostAFTLightGBMComparison(
        comparison=comparison,
        calibration=pd.concat(calibration_tables, ignore_index=True),
        segments=segments,
        paired_rows=paired_rows,
        user_bootstrap=user_bootstrap,
    )


def build_xgboost_aft_lightgbm_comparison_report(
    result: XGBoostAFTExperimentResult,
    comparison: XGBoostAFTLightGBMComparison,
) -> dict[str, object]:
    """동일 모집단 AFT·LightGBM 비교를 JSON 저장 구조로 변환합니다."""
    bootstrap = comparison.user_bootstrap.summary
    lower = float(bootstrap["bootstrap_lower_95_brier_improvement"])
    upper = float(bootstrap["bootstrap_upper_95_brier_improvement"])
    if lower > 0:
        decision = "LightGBM의 Brier Score가 AFT보다 일관되게 낮았습니다."
    elif upper < 0:
        decision = "AFT의 Brier Score가 LightGBM보다 일관되게 낮았습니다."
    else:
        decision = "95% 구간이 0을 포함해 두 모델의 안정적인 우위를 확정하지 않습니다."
    return {
        "dataset": "uci_online_retail_ii",
        "experiment_version": "xgboost_aft_lightgbm_comparison_v1",
        "evaluation_split": "validation",
        "horizon_days": int(result.probability.summary["horizon_days"]),
        "aft_model": _build_xgboost_aft_candidate_identity(result),
        "brier_improvement_direction": "aft_brier_minus_lightgbm_brier",
        "comparison": dataframe_to_nullable_records(comparison.comparison),
        "calibration": dataframe_to_nullable_records(comparison.calibration),
        "segments": dataframe_to_nullable_records(comparison.segments),
        "user_bootstrap": bootstrap,
        "decision": decision,
        "scope": (
            "AFT가 사용한 0일 제외 Train 원본·Validation·네 피처를 LightGBM과 "
            "공유했습니다. LightGBM은 고정 30일 정답을 확인할 수 있는 Train "
            "행만 실제 학습하며, AFT는 우측검열 행도 사용합니다. LightGBM의 "
            "C-index는 1-확률을 순위 점수로 사용한 결과이며 예상 일수 해석이 "
            "아닙니다. 사용자 Bootstrap은 고정된 두 모델의 Validation 사용자 "
            "구성 불확실성만 측정하며 Test는 사용하지 않았습니다."
        ),
    }


def build_xgboost_aft_lightgbm_bootstrap_trials_report(
    report: dict[str, object],
    comparison: XGBoostAFTLightGBMComparison,
) -> dict[str, object]:
    """AFT·LightGBM 사용자 Bootstrap 반복 원자료를 별도 구조로 만듭니다."""
    return {
        "dataset": report["dataset"],
        "experiment_version": report["experiment_version"],
        "evaluation_split": report["evaluation_split"],
        "brier_improvement_direction": report["brier_improvement_direction"],
        "summary": comparison.user_bootstrap.summary,
        "trials": dataframe_to_nullable_records(comparison.user_bootstrap.trials),
    }


def render_xgboost_aft_lightgbm_comparison_report(
    report: dict[str, object],
) -> str:
    """AFT·LightGBM 동일 모집단 비교를 사람이 검토할 Markdown으로 만듭니다."""
    comparison_lines = [
        (
            f"| {row['model_candidate']} | "
            f"{row['actual_training_sample_count']:,} | "
            f"{row['evaluation_sample_count']:,} | "
            f"{row['ipcw_brier_score']:.6f} | "
            f"{row['ipcw_concordance_index']:.6f} | "
            f"{row['expected_calibration_error']:.6f} | "
            f"{row['maximum_calibration_error']:.6f} |"
        )
        for row in report["comparison"]
    ]
    segment_lines = [
        (
            f"| {row['count_column']} | {row['count_bucket']} | "
            f"{row['sample_count']:,} | {row['user_count']:,} | "
            f"{_format_optional_metric(row['reference_ipcw_brier_score'])} | "
            f"{_format_optional_metric(row['candidate_ipcw_brier_score'])} | "
            f"{_format_optional_signed_metric(row['brier_improvement'])} | "
            f"{_format_optional_metric(row['reference_expected_calibration_error'])} | "
            f"{_format_optional_metric(row['candidate_expected_calibration_error'])} | "
            f"{row['validation_lower_brier_model']} |"
        )
        for row in report["segments"]
    ]
    bootstrap = report["user_bootstrap"]
    model = report["aft_model"]
    lines = [
        "# UCI XGBoost AFT·LightGBM 동일 모집단 비교",
        "",
        f"- AFT 설정: `{model['loss_distribution']}`, "
        f"`scale={model['loss_distribution_scale']}`, "
        f"`{model['num_boost_round']} rounds`",
        f"- 평가 시점: `{report['horizon_days']}`일",
        "- Bootstrap 개선량: `AFT Brier - LightGBM Brier`",
        "",
        "| 모델 | 실제 학습 | 평가 표본 | Brier | C-index | ECE | MCE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *comparison_lines,
        "",
        "## 사용자 단위 paired Bootstrap",
        "",
        f"- 사용자: `{bootstrap['user_count']:,}`명",
        f"- 반복 수 / seed: `{bootstrap['bootstrap_replicates']:,}` / "
        f"`{bootstrap['random_seed']}`",
        f"- 점 개선량: `{bootstrap['point_brier_improvement']:+.6f}`",
        f"- 평균 개선량: `{bootstrap['bootstrap_mean_brier_improvement']:+.6f}`",
        f"- 95% 구간: `[{bootstrap['bootstrap_lower_95_brier_improvement']:+.6f}, "
        f"{bootstrap['bootstrap_upper_95_brier_improvement']:+.6f}]`",
        f"- LightGBM 개선 비율: "
        f"`{bootstrap['bootstrap_positive_improvement_rate']:.2%}`",
        "",
        str(report["decision"]),
        "",
        "## 이력량 구간별 비교",
        "",
        "구간 개선량은 `AFT Brier - LightGBM Brier`이며 양수면 LightGBM, "
        "0 이하면 AFT가 낮습니다. 이 열은 Validation에서 Brier가 낮았던 모델을 "
        "표시할 뿐 운영 라우팅 규칙이 아니며 rolling cutoff 재검증이 필요합니다.",
        "",
        "| 기준 | 구간 | 표본 | 사용자 | AFT Brier | LightGBM Brier | 개선량 | "
        "AFT ECE | LightGBM ECE | Validation 최저 Brier |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        *segment_lines,
        "",
        str(report["scope"]),
        "",
    ]
    return "\n".join(lines)


def summarize_xgboost_aft_probability_by_count_segments(
    result: XGBoostAFTExperimentResult,
    *,
    count_columns: Sequence[str] = AFT_SEGMENT_COUNT_COLUMNS,
    calibration_bin_count: int = CALIBRATION_BIN_COUNT,
) -> pd.DataFrame:
    """이력량 구간별로 Train 상수확률과 AFT의 확률 성능을 비교합니다."""
    columns = tuple(count_columns)
    if not columns:
        raise ValueError("AFT 세그먼트 분석에는 이력량 컬럼이 하나 이상 필요합니다.")
    if len(set(columns)) != len(columns):
        raise ValueError("AFT 세그먼트 분석 컬럼에는 중복을 사용할 수 없습니다.")

    evaluation_rows = result.probability.rows.copy()
    missing_columns = set(columns) - set(evaluation_rows.columns)
    if missing_columns:
        raise ValueError(
            f"AFT 세그먼트 분석 컬럼이 누락됐습니다: {sorted(missing_columns)}"
        )
    evaluation_rows["reference_predicted_event_probability"] = float(
        result.training_summary["training_reference_probability"]
    )
    evaluation_rows["candidate_predicted_event_probability"] = evaluation_rows[
        "predicted_event_probability"
    ]

    summaries = [
        summarize_ipcw_probability_pair_by_count_segment(
            evaluation_rows,
            count_column=count_column,
            calibration_bin_count=calibration_bin_count,
        )
        for count_column in columns
    ]
    result_rows = pd.concat(summaries, ignore_index=True)
    result_rows["validation_fallback_candidate"] = result_rows["brier_improvement"].le(
        0.0
    )
    return result_rows


def _build_xgboost_aft_comparison_metrics(
    result: XGBoostAFTExperimentResult,
) -> dict[str, object]:
    """AFT 후보 결과에서 모든 비교 실험이 공유할 평가 지표를 추출합니다."""
    training_loss = result.training_summary["training_aft_nloglik"]
    if not isinstance(training_loss, list) or not training_loss:
        raise ValueError("후보 모델의 AFT 학습 손실 이력이 비어 있습니다.")

    probability = result.probability.summary
    observed_time = evaluate_xgboost_aft_observed_event_time(result)
    return {
        "final_training_aft_nloglik": float(training_loss[-1]),
        "minimum_training_aft_nloglik": float(min(training_loss)),
        "ipcw_concordance_index": float(result.concordance["ipcw_concordance_index"]),
        "ipcw_brier_score": float(probability["ipcw_brier_score"]),
        "ipcw_reference_brier_score": float(probability["ipcw_reference_brier_score"]),
        "brier_skill_score": probability["brier_skill_score"],
        "expected_calibration_error": float(probability["expected_calibration_error"]),
        "maximum_calibration_error": float(probability["maximum_calibration_error"]),
        "weighted_calibration_gap": float(probability["weighted_calibration_gap"]),
        "observed_event_time_sample_count": observed_time["sample_count"],
        "observed_event_mae_days": observed_time["mae_days"],
        "observed_event_median_absolute_error_days": observed_time[
            "median_absolute_error_days"
        ],
        "observed_event_within_7_days_rate": observed_time["within_7_days_rate"],
        "horizon_days": int(probability["horizon_days"]),
        "validation_sample_count": int(probability["validation_sample_count"]),
        "outcome_known_count": int(probability["outcome_known_count"]),
        "ipcw_weight_sum": float(probability["ipcw_weight_sum"]),
        "reference_probability": float(probability["reference_probability"]),
        "aft_evaluation_sample_count": int(probability["aft_evaluation_sample_count"]),
    }


def compare_xgboost_aft_boosting_rounds(
    prepared: XGBoostAFTPreparedExperiment,
    *,
    round_candidates: Sequence[int] = AFT_BOOST_ROUND_CANDIDATES,
    calibration_bin_count: int = CALIBRATION_BIN_COUNT,
    loss_distribution: str = "normal",
    loss_distribution_scale: float = 1.0,
) -> pd.DataFrame:
    """동일한 공통 데이터에서 부스팅 반복 횟수 후보의 Validation 지표를 비교합니다."""
    candidates = tuple(round_candidates)
    if not candidates:
        raise ValueError("비교할 부스팅 반복 횟수 후보가 하나 이상 필요합니다.")
    if any(
        isinstance(candidate, bool)
        or not isinstance(candidate, Integral)
        or candidate <= 0
        for candidate in candidates
    ):
        raise ValueError("부스팅 반복 횟수 후보는 0보다 큰 정수여야 합니다.")
    if len(set(candidates)) != len(candidates):
        raise ValueError("부스팅 반복 횟수 후보에는 중복된 값을 사용할 수 없습니다.")

    comparison_rows: list[dict[str, float | int | None]] = []
    for round_count in candidates:
        result = evaluate_xgboost_aft_candidate(
            prepared,
            calibration_bin_count=calibration_bin_count,
            bootstrap_replicates=None,
            loss_distribution=loss_distribution,
            loss_distribution_scale=loss_distribution_scale,
            num_boost_round=round_count,
        )
        comparison_rows.append(
            {
                "num_boost_round": int(round_count),
                **_build_xgboost_aft_comparison_metrics(result),
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    for fixed_column in (
        "ipcw_reference_brier_score",
        "aft_evaluation_sample_count",
        "horizon_days",
        "validation_sample_count",
        "outcome_known_count",
        "ipcw_weight_sum",
        "reference_probability",
    ):
        if comparison[fixed_column].nunique(dropna=False) != 1:
            raise RuntimeError(
                f"AFT 반복 횟수 후보의 공통 평가 조건이 달라졌습니다: {fixed_column}"
            )
    return comparison


def compare_xgboost_aft_loss_distributions(
    prepared: XGBoostAFTPreparedExperiment,
    *,
    num_boost_round: int,
    distribution_candidates: Sequence[str] = AFT_LOSS_DISTRIBUTION_CANDIDATES,
    calibration_bin_count: int = CALIBRATION_BIN_COUNT,
    loss_distribution_scale: float = 1.0,
) -> pd.DataFrame:
    """동일한 공통 데이터에서 AFT 손실분포 후보의 Validation 지표를 비교합니다."""
    candidates = tuple(distribution_candidates)
    if not candidates:
        raise ValueError("비교할 AFT 손실분포 후보가 하나 이상 필요합니다.")
    invalid_candidates = [
        candidate
        for candidate in candidates
        if not isinstance(candidate, str) or candidate not in AFT_LOSS_DISTRIBUTIONS
    ]
    if invalid_candidates:
        raise ValueError(
            "지원하지 않는 AFT 손실분포 후보가 있습니다: "
            f"{invalid_candidates}. 허용값: {sorted(AFT_LOSS_DISTRIBUTIONS)}"
        )
    if len(set(candidates)) != len(candidates):
        raise ValueError("AFT 손실분포 후보에는 중복된 값을 사용할 수 없습니다.")

    comparison_rows: list[dict[str, object]] = []
    for distribution in candidates:
        try:
            result = evaluate_xgboost_aft_candidate(
                prepared,
                calibration_bin_count=calibration_bin_count,
                bootstrap_replicates=None,
                loss_distribution=distribution,
                loss_distribution_scale=loss_distribution_scale,
                num_boost_round=num_boost_round,
            )
        except XGBoostAFTNumericalPredictionError as error:
            comparison_rows.append(
                {
                    "loss_distribution": distribution,
                    "loss_distribution_scale": float(loss_distribution_scale),
                    "num_boost_round": int(num_boost_round),
                    "status": "failed",
                    "failure_reason": str(error),
                    "prediction_sample_count": error.sample_count,
                    "invalid_prediction_count": error.invalid_prediction_count,
                    "nan_count": error.nan_count,
                    "positive_infinity_count": error.positive_infinity_count,
                    "negative_infinity_count": error.negative_infinity_count,
                    "nonpositive_finite_count": error.nonpositive_finite_count,
                    "final_training_aft_nloglik": None,
                    "minimum_training_aft_nloglik": None,
                    "ipcw_concordance_index": None,
                    "ipcw_brier_score": None,
                    "ipcw_reference_brier_score": None,
                    "brier_skill_score": None,
                    "expected_calibration_error": None,
                    "maximum_calibration_error": None,
                    "weighted_calibration_gap": None,
                    "observed_event_time_sample_count": None,
                    "observed_event_mae_days": None,
                    "observed_event_median_absolute_error_days": None,
                    "observed_event_within_7_days_rate": None,
                    "horizon_days": prepared.horizon_days,
                    "validation_sample_count": None,
                    "outcome_known_count": None,
                    "ipcw_weight_sum": None,
                    "reference_probability": None,
                    "aft_evaluation_sample_count": None,
                }
            )
            continue
        comparison_rows.append(
            {
                "loss_distribution": distribution,
                "loss_distribution_scale": result.training_summary[
                    "loss_distribution_scale"
                ],
                "num_boost_round": result.training_summary["num_boost_round"],
                "status": "success",
                "failure_reason": None,
                "prediction_sample_count": len(result.validation_predictions),
                "invalid_prediction_count": 0,
                "nan_count": 0,
                "positive_infinity_count": 0,
                "negative_infinity_count": 0,
                "nonpositive_finite_count": 0,
                **_build_xgboost_aft_comparison_metrics(result),
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    successful = comparison.loc[comparison["status"].eq("success")]
    if successful.empty:
        return comparison
    for fixed_column in (
        "loss_distribution_scale",
        "num_boost_round",
        "ipcw_reference_brier_score",
        "aft_evaluation_sample_count",
        "horizon_days",
        "validation_sample_count",
        "outcome_known_count",
        "ipcw_weight_sum",
        "reference_probability",
    ):
        if successful[fixed_column].nunique(dropna=False) != 1:
            raise RuntimeError(
                f"AFT 손실분포 후보의 공통 평가 조건이 달라졌습니다: {fixed_column}"
            )
    return comparison


def compare_xgboost_aft_loss_distribution_scales(
    prepared: XGBoostAFTPreparedExperiment,
    *,
    loss_distribution: str,
    num_boost_round: int,
    scale_candidates: Sequence[float] = AFT_LOGISTIC_SCALE_CANDIDATES,
    calibration_bin_count: int = CALIBRATION_BIN_COUNT,
) -> pd.DataFrame:
    """한 손실분포에서 scale만 바꿔 수치 안정성과 Validation 지표를 비교합니다."""
    candidates = tuple(scale_candidates)
    if not candidates:
        raise ValueError("비교할 AFT scale 후보가 하나 이상 필요합니다.")
    if loss_distribution not in AFT_LOSS_DISTRIBUTIONS:
        raise ValueError(
            "지원하지 않는 AFT 손실분포입니다: "
            f"{loss_distribution}. 허용값: {sorted(AFT_LOSS_DISTRIBUTIONS)}"
        )
    invalid_candidates = [
        candidate
        for candidate in candidates
        if isinstance(candidate, bool)
        or not isinstance(candidate, Real)
        or not isfinite(float(candidate))
        or float(candidate) <= 0.0
    ]
    if invalid_candidates:
        raise ValueError(
            f"AFT scale 후보는 0보다 큰 유한한 실수여야 합니다: {invalid_candidates}"
        )
    normalized_candidates = tuple(float(candidate) for candidate in candidates)
    if len(set(normalized_candidates)) != len(normalized_candidates):
        raise ValueError("AFT scale 후보에는 중복된 값을 사용할 수 없습니다.")

    comparisons = [
        compare_xgboost_aft_loss_distributions(
            prepared,
            num_boost_round=num_boost_round,
            distribution_candidates=(loss_distribution,),
            calibration_bin_count=calibration_bin_count,
            loss_distribution_scale=scale,
        )
        for scale in normalized_candidates
    ]
    comparison_columns = comparisons[0].columns
    frames_without_all_missing_columns = [
        frame.dropna(axis="columns", how="all") for frame in comparisons
    ]
    comparison = pd.concat(
        frames_without_all_missing_columns,
        ignore_index=True,
    ).reindex(columns=comparison_columns)

    for fixed_column in (
        "loss_distribution",
        "num_boost_round",
        "horizon_days",
        "prediction_sample_count",
    ):
        if comparison[fixed_column].nunique(dropna=False) != 1:
            raise RuntimeError(
                f"AFT scale 후보의 공통 평가 조건이 달라졌습니다: {fixed_column}"
            )

    successful = comparison.loc[comparison["status"].eq("success")]
    for fixed_column in (
        "ipcw_reference_brier_score",
        "aft_evaluation_sample_count",
        "validation_sample_count",
        "outcome_known_count",
        "ipcw_weight_sum",
        "reference_probability",
    ):
        if successful[fixed_column].nunique(dropna=False) > 1:
            raise RuntimeError(
                f"AFT scale 후보의 공통 평가 조건이 달라졌습니다: {fixed_column}"
            )
    return comparison


def bootstrap_xgboost_aft_candidate_pair_by_user(
    reference: XGBoostAFTExperimentResult,
    candidate: XGBoostAFTExperimentResult,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    random_seed: int = BOOTSTRAP_RANDOM_SEED,
) -> IPCWUserBootstrapResult:
    """같은 AFT Validation 표본에서 두 후보의 Brier 차이를 사용자별 추정합니다."""
    reference_rows = reference.probability.rows
    candidate_rows = candidate.probability.rows
    reference_horizon = reference.probability.summary["horizon_days"]
    candidate_horizon = candidate.probability.summary["horizon_days"]
    if reference_horizon != candidate_horizon:
        raise ValueError(
            "AFT 후보 쌍의 요약 평가 horizon이 다릅니다: "
            f"기준={reference_horizon}, 후보={candidate_horizon}"
        )
    shared_evaluation_columns = (
        "ipcw_horizon_days",
        "ipcw_outcome_known",
        "ipcw_event_within_horizon",
        "ipcw_weight",
    )
    for column in shared_evaluation_columns:
        if column not in reference_rows or column not in candidate_rows:
            raise ValueError(f"AFT 후보 쌍 평가 필수 컬럼이 누락됐습니다: {column}")
        reference_values = reference_rows[column].reset_index(drop=True)
        candidate_values = candidate_rows[column].reset_index(drop=True)
        if not reference_values.equals(candidate_values):
            raise ValueError(f"AFT 후보 쌍의 정답·검열·IPCW 조건이 다릅니다: {column}")

    prediction_columns = [
        *IPCW_CANDIDATE_ID_COLUMNS,
        "predicted_event_probability",
    ]
    paired_rows = build_paired_probability_predictions(
        reference_rows,
        reference_rows.loc[:, prediction_columns],
        candidate_rows.loc[:, prediction_columns],
    )
    bootstrap = bootstrap_ipcw_brier_pair_difference_by_user(
        paired_rows,
        bootstrap_replicates=bootstrap_replicates,
        random_seed=random_seed,
    )

    expected_scores = {
        "point_reference_brier_score": reference.probability.summary[
            "ipcw_brier_score"
        ],
        "point_candidate_brier_score": candidate.probability.summary[
            "ipcw_brier_score"
        ],
    }
    for metric, expected_value in expected_scores.items():
        actual_value = bootstrap.summary[metric]
        if not isclose(
            float(actual_value),
            float(expected_value),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"AFT 후보 쌍 Bootstrap 점 지표가 개별 평가와 다릅니다: {metric}"
            )
    return bootstrap


def _validate_xgboost_aft_selection_table(
    comparison: pd.DataFrame,
    *,
    candidate_column: str,
) -> None:
    """AFT 후보 선택표의 공통 지표와 후보 식별 컬럼을 검사합니다."""
    required_columns = (
        candidate_column,
        "ipcw_brier_score",
        "ipcw_concordance_index",
        "expected_calibration_error",
        "weighted_calibration_gap",
    )
    if comparison.columns.duplicated().any():
        raise ValueError("AFT 후보 선택 결과에 중복된 컬럼 이름이 있습니다.")
    missing_columns = [
        column for column in required_columns if column not in comparison.columns
    ]
    if missing_columns:
        raise ValueError(f"AFT 후보 선택 필수 컬럼이 누락됐습니다: {missing_columns}")
    if comparison.empty:
        raise ValueError("선택할 AFT 후보 결과가 없습니다.")

    metric_ranges = {
        "ipcw_brier_score": (0.0, 1.0),
        "ipcw_concordance_index": (0.0, 1.0),
        "expected_calibration_error": (0.0, 1.0),
        "weighted_calibration_gap": (-1.0, 1.0),
    }
    for column, (lower_bound, upper_bound) in metric_ranges.items():
        values = comparison[column].tolist()
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not lower_bound <= float(value) <= upper_bound
            for value in values
        ):
            raise ValueError(
                f"AFT 후보 선택 지표 {column}은 "
                f"{lower_bound}부터 {upper_bound} 사이의 유한한 실수여야 합니다."
            )


def _rank_xgboost_aft_candidates(
    comparison: pd.DataFrame,
    *,
    final_tie_breaker_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """공통 Validation 우선순위와 마지막 동률 해소 기준으로 후보를 정렬합니다."""
    ranked = comparison.copy()
    ranked["absolute_weighted_calibration_gap"] = ranked[
        "weighted_calibration_gap"
    ].abs()
    return ranked.sort_values(
        by=[
            "ipcw_brier_score",
            "ipcw_concordance_index",
            "expected_calibration_error",
            "absolute_weighted_calibration_gap",
            *final_tie_breaker_columns,
        ],
        ascending=[True, False, True, True] + [True] * len(final_tie_breaker_columns),
        kind="stable",
    )


def select_xgboost_aft_boosting_round(comparison: pd.DataFrame) -> int:
    """Brier를 우선하고 순위·확률 신뢰도·비용 순으로 반복 횟수를 선택합니다."""
    _validate_xgboost_aft_selection_table(
        comparison,
        candidate_column="num_boost_round",
    )
    round_values = comparison["num_boost_round"].tolist()
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0
        for value in round_values
    ):
        raise ValueError("AFT 후보 반복 횟수는 0보다 큰 정수여야 합니다.")
    if comparison["num_boost_round"].duplicated().any():
        raise ValueError("AFT 후보 선택 결과에 중복된 반복 횟수가 있습니다.")

    ranked = _rank_xgboost_aft_candidates(
        comparison,
        final_tie_breaker_columns=("num_boost_round",),
    )
    return int(ranked.iloc[0]["num_boost_round"])


def select_xgboost_aft_loss_distribution(comparison: pd.DataFrame) -> str:
    """공통 Validation 우선순위와 기준 분포 선호로 AFT 손실분포를 선택합니다."""
    selection_rows = comparison
    if "status" in comparison.columns:
        statuses = comparison["status"]
        if statuses.isna().any() or not statuses.isin({"success", "failed"}).all():
            raise ValueError("AFT 손실분포 후보 상태는 success 또는 failed여야 합니다.")
        selection_rows = comparison.loc[statuses.eq("success")]
    _validate_xgboost_aft_selection_table(
        selection_rows,
        candidate_column="loss_distribution",
    )
    distribution_values = comparison["loss_distribution"].tolist()
    if any(
        not isinstance(value, str) or value not in AFT_LOSS_DISTRIBUTIONS
        for value in distribution_values
    ):
        raise ValueError(
            "AFT 후보 손실분포는 normal, logistic, extreme 중 하나여야 합니다."
        )
    if comparison["loss_distribution"].duplicated().any():
        raise ValueError("AFT 후보 선택 결과에 중복된 손실분포가 있습니다.")

    distribution_preference = {
        distribution: index
        for index, distribution in enumerate(AFT_LOSS_DISTRIBUTION_CANDIDATES)
    }
    candidates_with_preference = selection_rows.copy()
    candidates_with_preference["distribution_preference"] = candidates_with_preference[
        "loss_distribution"
    ].map(distribution_preference)
    ranked = _rank_xgboost_aft_candidates(
        candidates_with_preference,
        final_tie_breaker_columns=("distribution_preference",),
    )
    return str(ranked.iloc[0]["loss_distribution"])


def select_xgboost_aft_loss_distribution_scale(comparison: pd.DataFrame) -> float:
    """성공한 후보 중 Brier를 우선해 한 손실분포의 scale을 선택합니다."""
    if "status" not in comparison.columns:
        raise ValueError("AFT scale 후보 결과에 상태 컬럼이 없습니다.")
    statuses = comparison["status"]
    if statuses.isna().any() or not statuses.isin({"success", "failed"}).all():
        raise ValueError("AFT scale 후보 상태는 success 또는 failed여야 합니다.")

    selection_rows = comparison.loc[statuses.eq("success")]
    _validate_xgboost_aft_selection_table(
        selection_rows,
        candidate_column="loss_distribution_scale",
    )
    scale_values = comparison["loss_distribution_scale"].tolist()
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(float(value))
        or float(value) <= 0.0
        for value in scale_values
    ):
        raise ValueError("AFT 후보 scale은 0보다 큰 유한한 실수여야 합니다.")
    if comparison["loss_distribution_scale"].duplicated().any():
        raise ValueError("AFT 후보 선택 결과에 중복된 scale이 있습니다.")

    candidates_with_preference = selection_rows.copy()
    candidates_with_preference["scale_distance_from_one"] = (
        candidates_with_preference["loss_distribution_scale"]
        .astype(float)
        .map(lambda value: abs(log(value)))
    )
    ranked = _rank_xgboost_aft_candidates(
        candidates_with_preference,
        final_tie_breaker_columns=(
            "scale_distance_from_one",
            "loss_distribution_scale",
        ),
    )
    return float(ranked.iloc[0]["loss_distribution_scale"])


def validate_selected_xgboost_aft_result(
    comparison: pd.DataFrame,
    result: XGBoostAFTExperimentResult,
) -> None:
    """선택 후보의 재학습 결과가 후보 비교 시점의 점 지표와 같은지 확인합니다."""
    selected_num_boost_round = select_xgboost_aft_boosting_round(comparison)
    result_num_boost_round = result.training_summary["num_boost_round"]
    if result_num_boost_round != selected_num_boost_round:
        raise RuntimeError(
            "최종 AFT 평가 모델의 반복 횟수가 Validation 선택 후보와 다릅니다."
        )
    selected_row = comparison.loc[
        comparison["num_boost_round"].eq(selected_num_boost_round)
    ].iloc[0]
    training_loss = result.training_summary["training_aft_nloglik"]
    if not isinstance(training_loss, list) or not training_loss:
        raise RuntimeError("최종 AFT 평가 모델의 학습 손실 이력이 비어 있습니다.")
    probability = result.probability.summary
    actual_metrics = {
        "final_training_aft_nloglik": float(training_loss[-1]),
        "ipcw_concordance_index": float(result.concordance["ipcw_concordance_index"]),
        "ipcw_brier_score": float(probability["ipcw_brier_score"]),
        "ipcw_reference_brier_score": float(probability["ipcw_reference_brier_score"]),
        "expected_calibration_error": float(probability["expected_calibration_error"]),
        "maximum_calibration_error": float(probability["maximum_calibration_error"]),
        "weighted_calibration_gap": float(probability["weighted_calibration_gap"]),
    }
    for metric, actual_value in actual_metrics.items():
        expected_value = float(selected_row[metric])
        if not isclose(actual_value, expected_value, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(
                f"최종 AFT 재학습 결과가 후보 비교 시점과 일치하지 않습니다: {metric}"
            )


def validate_selected_xgboost_aft_distribution_result(
    comparison: pd.DataFrame,
    result: XGBoostAFTExperimentResult,
) -> None:
    """최종 모델의 분포·고정 조건·점 지표가 분포 선택 결과와 같은지 확인합니다."""
    selected_distribution = select_xgboost_aft_loss_distribution(comparison)
    training_summary = result.training_summary
    if training_summary["loss_distribution"] != selected_distribution:
        raise RuntimeError(
            "최종 AFT 평가 모델의 손실분포가 Validation 선택 후보와 다릅니다."
        )

    selected_row = comparison.loc[
        comparison["loss_distribution"].eq(selected_distribution)
    ].iloc[0]
    if training_summary["num_boost_round"] != int(selected_row["num_boost_round"]):
        raise RuntimeError(
            "최종 AFT 평가 모델의 반복 횟수가 손실분포 비교 조건과 다릅니다."
        )
    if not isclose(
        float(training_summary["loss_distribution_scale"]),
        float(selected_row["loss_distribution_scale"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "최종 AFT 평가 모델의 scale이 손실분포 비교 조건과 다릅니다."
        )

    actual_metrics = _build_xgboost_aft_comparison_metrics(result)
    for metric, actual_value in actual_metrics.items():
        if metric not in selected_row:
            continue
        expected_value = selected_row[metric]
        if actual_value is None or expected_value is None:
            if actual_value is not expected_value:
                raise RuntimeError(
                    "최종 AFT 재학습 결과가 손실분포 비교 시점과 일치하지 "
                    f"않습니다: {metric}"
                )
            continue
        if not isclose(
            float(actual_value),
            float(expected_value),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "최종 AFT 재학습 결과가 손실분포 비교 시점과 일치하지 "
                f"않습니다: {metric}"
            )


def validate_selected_xgboost_aft_scale_result(
    comparison: pd.DataFrame,
    result: XGBoostAFTExperimentResult,
) -> None:
    """재학습한 scale 후보의 설정·점 지표가 선택 시점과 같은지 확인합니다."""
    selected_scale = select_xgboost_aft_loss_distribution_scale(comparison)
    selected_row = comparison.loc[
        comparison["loss_distribution_scale"].eq(selected_scale)
    ].iloc[0]
    training = result.training_summary
    if training["loss_distribution"] != selected_row["loss_distribution"]:
        raise RuntimeError("재학습한 AFT scale 후보의 손실분포가 선택 결과와 다릅니다.")
    if training["num_boost_round"] != int(selected_row["num_boost_round"]):
        raise RuntimeError(
            "재학습한 AFT scale 후보의 반복 횟수가 선택 결과와 다릅니다."
        )
    if not isclose(
        float(training["loss_distribution_scale"]),
        selected_scale,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError("재학습한 AFT scale 후보의 scale이 선택 결과와 다릅니다.")

    actual_metrics = _build_xgboost_aft_comparison_metrics(result)
    for metric, actual_value in actual_metrics.items():
        if metric not in selected_row:
            continue
        expected_value = selected_row[metric]
        if actual_value is None or expected_value is None:
            if actual_value is not expected_value:
                raise RuntimeError(
                    f"재학습한 AFT scale 후보가 선택 시점과 일치하지 않습니다: {metric}"
                )
            continue
        if not isclose(
            float(actual_value),
            float(expected_value),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"재학습한 AFT scale 후보가 선택 시점과 일치하지 않습니다: {metric}"
            )


def build_xgboost_aft_round_comparison_report(
    comparison: pd.DataFrame,
    *,
    selected_num_boost_round: int,
) -> dict[str, object]:
    """반복 횟수 후보의 공통 Validation 결과와 선택 규칙을 저장합니다."""
    selected_by_policy = select_xgboost_aft_boosting_round(comparison)
    if selected_num_boost_round != selected_by_policy:
        raise ValueError(
            "전달된 선택 반복 횟수가 사전에 정의한 AFT 후보 선택 규칙과 다릅니다."
        )
    horizon_days = int(comparison["horizon_days"].iloc[0])
    return {
        "dataset": "uci_online_retail_ii",
        "experiment_version": "xgboost_aft_boost_round_comparison_v1",
        "evaluation_split": "validation",
        "horizon_days": horizon_days,
        "selected_num_boost_round": selected_num_boost_round,
        "selection_policy": [
            "lowest_ipcw_brier_score",
            "highest_ipcw_concordance_index_on_exact_brier_tie",
            "lowest_expected_calibration_error_on_exact_tie",
            "lowest_absolute_weighted_calibration_gap_on_exact_tie",
            "lowest_num_boost_round_on_exact_tie",
        ],
        "candidates": dataframe_to_nullable_records(comparison),
        "scope": (
            f"같은 Train·Validation·피처·{horizon_days}일 horizon·IPCW 기준선에서 "
            "반복 횟수만 "
            "변경했습니다. 선택값은 Validation 후보 탐색 결과이며 Test 성능이나 "
            "배포 가능성을 의미하지 않습니다. 후보 비교에서는 Bootstrap을 생략하고 "
            "선택 후보에만 사용자 단위 Bootstrap을 별도로 수행합니다. 같은 "
            "Validation으로 후보를 선택하고 평가했으므로 선택 과정의 불확실성과 "
            "낙관성은 Bootstrap 구간에 포함되지 않습니다."
        ),
    }


def _format_optional_metric(value: object) -> str:
    """정의되지 않은 선택 지표는 실패 대신 N/A로 표시합니다."""
    if value is None:
        return "N/A"
    return f"{float(value):.6f}"


def _format_optional_signed_metric(value: object) -> str:
    """정의되지 않은 변화량은 N/A, 숫자는 부호와 함께 표시합니다."""
    if value is None:
        return "N/A"
    return f"{float(value):+.6f}"


def render_xgboost_aft_round_comparison_report(report: dict[str, object]) -> str:
    """반복 횟수 후보와 선택 근거를 사람이 검토할 Markdown으로 만듭니다."""
    candidates = report["candidates"]
    selected_num_boost_round = report["selected_num_boost_round"]
    candidate_lines = [
        (
            f"| {row['num_boost_round']:,} | "
            f"{row['final_training_aft_nloglik']:.6f} | "
            f"{row['ipcw_brier_score']:.6f} | "
            f"{row['ipcw_reference_brier_score']:.6f} | "
            f"{_format_optional_metric(row['brier_skill_score'])} | "
            f"{row['ipcw_concordance_index']:.6f} | "
            f"{row['expected_calibration_error']:.6f} | "
            f"{row['maximum_calibration_error']:.6f} | "
            f"{row['weighted_calibration_gap']:+.6f} |"
        )
        for row in candidates
    ]
    lines = [
        "# UCI XGBoost AFT 반복 횟수 비교",
        "",
        f"- Validation 선택 후보: `{selected_num_boost_round:,}` rounds",
        "- 1순위는 IPCW Brier Score이며, 나머지 지표는 정확한 동률일 때만 "
        "순서대로 사용합니다.",
        "",
        "| rounds | Train loss | AFT Brier | 기준 Brier | Brier Skill | "
        "C-index | ECE | MCE | 확률 편향 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *candidate_lines,
        "",
        str(report["scope"]),
        "",
    ]
    return "\n".join(lines)


def build_xgboost_aft_distribution_comparison_report(
    comparison: pd.DataFrame,
    *,
    selected_loss_distribution: str,
) -> dict[str, object]:
    """손실분포 후보의 공통 Validation 결과와 선택 규칙을 저장합니다."""
    selected_by_policy = select_xgboost_aft_loss_distribution(comparison)
    if selected_loss_distribution != selected_by_policy:
        raise ValueError(
            "전달된 선택 손실분포가 사전에 정의한 AFT 후보 선택 규칙과 다릅니다."
        )
    return {
        "dataset": "uci_online_retail_ii",
        "experiment_version": "xgboost_aft_distribution_comparison_v1",
        "evaluation_split": "validation",
        "horizon_days": int(comparison["horizon_days"].iloc[0]),
        "num_boost_round": int(comparison["num_boost_round"].iloc[0]),
        "loss_distribution_scale": float(comparison["loss_distribution_scale"].iloc[0]),
        "selected_loss_distribution": selected_loss_distribution,
        "selection_policy": [
            "lowest_ipcw_brier_score",
            "highest_ipcw_concordance_index_on_exact_brier_tie",
            "lowest_expected_calibration_error_on_exact_tie",
            "lowest_absolute_weighted_calibration_gap_on_exact_tie",
            "normal_then_logistic_then_extreme_on_exact_tie",
        ],
        "candidates": dataframe_to_nullable_records(comparison),
        "scope": (
            "같은 Train·Validation·피처·horizon·IPCW 기준선에서 손실분포만 "
            "변경했습니다. 선택값은 Validation 점 지표의 1차 후보 탐색 결과이며 "
            "Test 성능이나 배포 가능성을 의미하지 않습니다. 점 지표 상위 두 "
            "후보는 동일 사용자 재표본의 paired Bootstrap으로 차이의 안정성을 "
            "추가 확인해야 합니다."
        ),
    }


def render_xgboost_aft_distribution_comparison_report(
    report: dict[str, object],
) -> str:
    """손실분포 후보와 선택 근거를 사람이 검토할 Markdown으로 만듭니다."""
    candidates = report["candidates"]
    selected_loss_distribution = report["selected_loss_distribution"]
    candidate_lines = []
    failure_lines = []
    for row in candidates:
        if row.get("status", "success") == "failed":
            candidate_lines.append(
                f"| {row['loss_distribution']} | 실패 | N/A | N/A | N/A | "
                "N/A | N/A | N/A | N/A | N/A |"
            )
            failure_lines.append(
                f"- `{row['loss_distribution']}`: 무효 예측 "
                f"`{row['invalid_prediction_count']}/{row['prediction_sample_count']}`건 "
                f"(NaN {row['nan_count']}, +inf {row['positive_infinity_count']}, "
                f"-inf {row['negative_infinity_count']}, 유한한 0 이하 "
                f"{row['nonpositive_finite_count']})"
            )
            continue
        candidate_lines.append(
            f"| {row['loss_distribution']} | 성공 | "
            f"{row['final_training_aft_nloglik']:.6f} | "
            f"{row['ipcw_brier_score']:.6f} | "
            f"{row['ipcw_reference_brier_score']:.6f} | "
            f"{_format_optional_metric(row['brier_skill_score'])} | "
            f"{row['ipcw_concordance_index']:.6f} | "
            f"{row['expected_calibration_error']:.6f} | "
            f"{row['maximum_calibration_error']:.6f} | "
            f"{row['weighted_calibration_gap']:+.6f} |"
        )
    lines = [
        "# UCI XGBoost AFT 손실분포 비교",
        "",
        f"- Validation 선택 후보: `{selected_loss_distribution}`",
        f"- 고정 반복 횟수 / scale: `{report['num_boost_round']}` / "
        f"`{report['loss_distribution_scale']}`",
        "- 1순위는 IPCW Brier Score이며, 나머지 지표는 정확한 동률일 때만 "
        "순서대로 사용합니다.",
        "",
        "| 분포 | 상태 | Train loss | AFT Brier | 기준 Brier | Brier Skill | "
        "C-index | ECE | MCE | 확률 편향 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *candidate_lines,
        "",
    ]
    if failure_lines:
        lines.extend(["## 수치 실패 진단", "", *failure_lines, ""])
    lines.extend([str(report["scope"]), ""])
    return "\n".join(lines)


def build_xgboost_aft_logistic_scale_comparison_report(
    comparison: pd.DataFrame,
    *,
    selected_loss_distribution_scale: float | None,
) -> dict[str, object]:
    """logistic scale 후보의 수치 안정성과 Validation 결과를 저장합니다."""
    successful = comparison.loc[comparison["status"].eq("success")]
    if successful.empty:
        if selected_loss_distribution_scale is not None:
            raise ValueError(
                "모든 logistic scale이 실패해 선택 scale을 지정할 수 없습니다."
            )
    else:
        selected_by_policy = select_xgboost_aft_loss_distribution_scale(comparison)
        if selected_loss_distribution_scale is None or not isclose(
            selected_loss_distribution_scale,
            selected_by_policy,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "전달된 선택 scale이 사전에 정의한 AFT 후보 선택 규칙과 다릅니다."
            )
    distributions = comparison["loss_distribution"].drop_duplicates().tolist()
    if distributions != ["logistic"]:
        raise ValueError("logistic scale 비교 보고서에는 logistic 후보만 허용합니다.")
    return {
        "dataset": "uci_online_retail_ii",
        "experiment_version": "xgboost_aft_logistic_scale_comparison_v1",
        "evaluation_split": "validation",
        "horizon_days": int(comparison["horizon_days"].iloc[0]),
        "loss_distribution": "logistic",
        "num_boost_round": int(comparison["num_boost_round"].iloc[0]),
        "selected_loss_distribution_scale": selected_loss_distribution_scale,
        "selection_policy": [
            "exclude_numerically_failed_candidates",
            "lowest_ipcw_brier_score",
            "highest_ipcw_concordance_index_on_exact_brier_tie",
            "lowest_expected_calibration_error_on_exact_tie",
            "lowest_absolute_weighted_calibration_gap_on_exact_tie",
            "closest_multiplicative_distance_to_scale_one_on_exact_tie",
            "lowest_scale_on_exact_tie",
        ],
        "candidates": dataframe_to_nullable_records(comparison),
        "scope": (
            "같은 Train·Validation·피처·logistic 분포·반복 횟수·horizon·IPCW "
            "기준선에서 scale만 변경했습니다. 실패 후보는 삭제하거나 보정하지 "
            "않고 무효 예측 유형과 개수를 기록했습니다. 선택 scale은 logistic "
            "분포 내부의 수치 안정성·민감도 진단 결과이며 normal을 포함한 전체 "
            "AFT 후보의 최종 선택을 변경하지 않습니다. Test는 사용하지 않았습니다."
        ),
    }


def render_xgboost_aft_logistic_scale_comparison_report(
    report: dict[str, object],
) -> str:
    """logistic scale별 성공·실패와 지표를 사람이 검토할 Markdown으로 만듭니다."""
    candidate_lines = []
    failure_lines = []
    for row in report["candidates"]:
        scale = row["loss_distribution_scale"]
        if row["status"] == "failed":
            candidate_lines.append(
                f"| {scale} | 실패 | {row['invalid_prediction_count']:,} | "
                "N/A | N/A | N/A | N/A | N/A |"
            )
            failure_lines.append(
                f"- `scale={scale}`: 무효 예측 "
                f"`{row['invalid_prediction_count']:,}/"
                f"{row['prediction_sample_count']:,}`건 "
                f"(NaN {row['nan_count']:,}, +inf "
                f"{row['positive_infinity_count']:,}, -inf "
                f"{row['negative_infinity_count']:,}, 유한한 0 이하 "
                f"{row['nonpositive_finite_count']:,})"
            )
            continue
        candidate_lines.append(
            f"| {scale} | 성공 | 0 | {row['ipcw_brier_score']:.6f} | "
            f"{row['ipcw_concordance_index']:.6f} | "
            f"{row['expected_calibration_error']:.6f} | "
            f"{row['maximum_calibration_error']:.6f} | "
            f"{row['weighted_calibration_gap']:+.6f} |"
        )

    selected_scale = report["selected_loss_distribution_scale"]
    selected_scale_text = "N/A" if selected_scale is None else str(selected_scale)
    lines = [
        "# UCI XGBoost AFT logistic scale 비교",
        "",
        f"- logistic 내부 선택 scale: `{selected_scale_text}`",
        f"- 고정 반복 횟수: `{report['num_boost_round']}`",
        "- 수치 실패 후보를 먼저 제외하고 성공 후보의 IPCW Brier Score를 "
        "1순위로 비교합니다.",
        "",
        "| scale | 상태 | 무효 예측 | AFT Brier | C-index | ECE | MCE | 확률 편향 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *candidate_lines,
        "",
    ]
    if failure_lines:
        lines.extend(["## 수치 실패 진단", "", *failure_lines, ""])
    lines.extend([str(report["scope"]), ""])
    return "\n".join(lines)


def _build_xgboost_aft_candidate_identity(
    result: XGBoostAFTExperimentResult,
) -> dict[str, object]:
    """후보 쌍 보고서에서 모델을 재현할 최소 학습 설정을 추출합니다."""
    training = result.training_summary
    return {
        "loss_distribution": training["loss_distribution"],
        "loss_distribution_scale": training["loss_distribution_scale"],
        "num_boost_round": training["num_boost_round"],
        "feature_columns": list(training["feature_columns"]),
    }


def build_xgboost_aft_candidate_pair_bootstrap_report(
    reference: XGBoostAFTExperimentResult,
    candidate: XGBoostAFTExperimentResult | None,
    bootstrap: IPCWUserBootstrapResult | None,
    *,
    unavailable_reason: str | None = None,
) -> dict[str, object]:
    """기준 AFT와 비교 후보의 paired Bootstrap 판단을 저장합니다."""
    reference_identity = _build_xgboost_aft_candidate_identity(reference)
    common_scope = (
        "고정된 두 모델의 동일 Validation 행·정답·IPCW 가중치에서 사용자를 "
        "함께 복원추출했습니다. Test는 사용하지 않았습니다. 이 구간은 사용자 "
        "구성의 불확실성만 반영하며 모델 재학습과 같은 Validation 후보 선택의 "
        "낙관성은 포함하지 않습니다."
    )
    if candidate is None or bootstrap is None:
        if candidate is not None or bootstrap is not None or not unavailable_reason:
            raise ValueError(
                "후보 쌍 비교 불가 보고서에는 후보·Bootstrap 없이 사유가 필요합니다."
            )
        return {
            "dataset": "uci_online_retail_ii",
            "experiment_version": "xgboost_aft_candidate_pair_bootstrap_v1",
            "evaluation_split": "validation",
            "status": "unavailable",
            "reference": reference_identity,
            "candidate": {
                "loss_distribution": "logistic",
                "loss_distribution_scale": None,
                "num_boost_round": reference_identity["num_boost_round"],
                "feature_columns": reference_identity["feature_columns"],
            },
            "improvement_direction": "reference_brier_minus_candidate_brier",
            "summary": None,
            "decision": unavailable_reason,
            "scope": common_scope,
        }

    summary = bootstrap.summary
    candidate_identity = _build_xgboost_aft_candidate_identity(candidate)
    reference_name = str(reference_identity["loss_distribution"])
    candidate_name = str(candidate_identity["loss_distribution"])
    lower = float(summary["bootstrap_lower_95_brier_improvement"])
    upper = float(summary["bootstrap_upper_95_brier_improvement"])
    if lower > 0.0:
        decision = (
            f"{candidate_name} 후보의 Brier 우위가 사용자 재표본에서도 일관됐습니다."
        )
    elif upper < 0.0:
        decision = (
            f"{reference_name} 기준의 Brier 우위가 사용자 재표본에서도 일관됐습니다."
        )
    else:
        decision = "95% Bootstrap 구간이 0을 포함해 두 후보의 우위를 확정하지 않습니다."
    return {
        "dataset": "uci_online_retail_ii",
        "experiment_version": "xgboost_aft_candidate_pair_bootstrap_v1",
        "evaluation_split": "validation",
        "status": "complete",
        "reference": reference_identity,
        "candidate": candidate_identity,
        "improvement_direction": "reference_brier_minus_candidate_brier",
        "summary": summary,
        "decision": decision,
        "scope": common_scope,
    }


def build_xgboost_aft_candidate_pair_bootstrap_trials_report(
    report: dict[str, object],
    bootstrap: IPCWUserBootstrapResult | None,
) -> dict[str, object]:
    """후보 쌍 Bootstrap 반복 원자료를 판단 요약과 분리해 저장합니다."""
    trials = (
        [] if bootstrap is None else dataframe_to_nullable_records(bootstrap.trials)
    )
    return {
        "dataset": report["dataset"],
        "experiment_version": report["experiment_version"],
        "evaluation_split": report["evaluation_split"],
        "status": report["status"],
        "improvement_direction": report["improvement_direction"],
        "summary": report["summary"],
        "trials": trials,
    }


def render_xgboost_aft_candidate_pair_bootstrap_report(
    report: dict[str, object],
) -> str:
    """두 AFT 후보의 사용자 paired Bootstrap 결과를 Markdown으로 만듭니다."""
    reference = report["reference"]
    candidate = report["candidate"]
    lines = [
        "# UCI XGBoost AFT 후보 쌍 사용자 Bootstrap",
        "",
        f"- 기준: `{reference['loss_distribution']}`, "
        f"`scale={reference['loss_distribution_scale']}`",
        f"- 후보: `{candidate['loss_distribution']}`, "
        f"`scale={candidate['loss_distribution_scale']}`",
        f"- 개선량 방향: `기준 {reference['loss_distribution']} Brier - "
        f"후보 {candidate['loss_distribution']} Brier`",
        "",
    ]
    if report["status"] == "unavailable":
        lines.extend([f"비교 불가: {report['decision']}", "", str(report["scope"]), ""])
        return "\n".join(lines)

    summary = report["summary"]
    lines.extend(
        [
            f"- 사용자 수: `{summary['user_count']:,}`명",
            f"- 반복 수 / seed: `{summary['bootstrap_replicates']:,}` / "
            f"`{summary['random_seed']}`",
            f"- normal 점 Brier: `{summary['point_reference_brier_score']:.6f}`",
            f"- logistic 점 Brier: `{summary['point_candidate_brier_score']:.6f}`",
            f"- 점 개선량: `{summary['point_brier_improvement']:+.6f}`",
            f"- 평균 개선량: `{summary['bootstrap_mean_brier_improvement']:+.6f}`",
            f"- 95% 구간: "
            f"`[{summary['bootstrap_lower_95_brier_improvement']:+.6f}, "
            f"{summary['bootstrap_upper_95_brier_improvement']:+.6f}]`",
            f"- 양수 개선 비율: `{summary['bootstrap_positive_improvement_rate']:.2%}`",
            "",
            str(report["decision"]),
            "",
            str(report["scope"]),
            "",
        ]
    )
    return "\n".join(lines)


def build_xgboost_aft_segment_comparison_report(
    result: XGBoostAFTExperimentResult,
    segments: pd.DataFrame,
) -> dict[str, object]:
    """AFT 이력량 세그먼트별 기준선 비교와 fallback 후보를 저장합니다."""
    required_columns = {
        "count_column",
        "count_bucket",
        "sample_count",
        "user_count",
        "outcome_known_count",
        "reference_ipcw_brier_score",
        "candidate_ipcw_brier_score",
        "brier_improvement",
        "validation_fallback_candidate",
    }
    missing_columns = required_columns - set(segments.columns)
    if missing_columns:
        raise ValueError(
            f"AFT 세그먼트 보고서 컬럼이 누락됐습니다: {sorted(missing_columns)}"
        )
    return {
        "dataset": "uci_online_retail_ii",
        "experiment_version": "xgboost_aft_segment_comparison_v1",
        "evaluation_split": "validation",
        "model": _build_xgboost_aft_candidate_identity(result),
        "reference": "train_global_event_probability",
        "brier_improvement_direction": "reference_brier_minus_aft_brier",
        "segments": dataframe_to_nullable_records(segments),
        "scope": (
            "같은 Validation 평가행에서 Train으로 추정한 상수확률과 AFT를 "
            "이력량 구간별로 비교했습니다. Brier 개선량이 0 이하인 구간은 "
            "fallback 검토 후보일 뿐 운영 규칙으로 확정하지 않으며 rolling "
            "cutoff에서 재현되는지 추가 검증해야 합니다. Test는 사용하지 않았습니다."
        ),
    }


def render_xgboost_aft_segment_comparison_report(
    report: dict[str, object],
) -> str:
    """AFT 세그먼트별 Brier·Calibration과 fallback 후보를 Markdown으로 만듭니다."""
    segment_lines = [
        (
            f"| {row['count_column']} | {row['count_bucket']} | "
            f"{row['sample_count']:,} | {row['user_count']:,} | "
            f"{row['outcome_known_count']:,} | "
            f"{_format_optional_metric(row['reference_ipcw_brier_score'])} | "
            f"{_format_optional_metric(row['candidate_ipcw_brier_score'])} | "
            f"{_format_optional_signed_metric(row['brier_improvement'])} | "
            f"{_format_optional_metric(row['candidate_expected_calibration_error'])} | "
            f"{'검토' if row['validation_fallback_candidate'] else '-'} |"
        )
        for row in report["segments"]
    ]
    model = report["model"]
    lines = [
        "# UCI XGBoost AFT 이력량 세그먼트 비교",
        "",
        f"- 모델: `{model['loss_distribution']}`, "
        f"`scale={model['loss_distribution_scale']}`, "
        f"`{model['num_boost_round']} rounds`",
        "- 개선량: `Train 상수확률 Brier - AFT Brier`",
        "",
        "| 기준 컬럼 | 구간 | 표본 | 사용자 | 정답 확인 | 기준 Brier | "
        "AFT Brier | 개선량 | AFT ECE | fallback |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        *segment_lines,
        "",
        str(report["scope"]),
        "",
    ]
    return "\n".join(lines)


def build_xgboost_aft_report(
    result: XGBoostAFTExperimentResult,
) -> dict[str, object]:
    """실험 결과를 표준 JSON으로 저장 가능한 요약 구조로 변환합니다."""
    bootstrap = result.probability.user_bootstrap
    if bootstrap is None:
        raise ValueError("XGBoost AFT 보고서에는 사용자 Bootstrap 결과가 필요합니다.")

    split = {
        key: value.isoformat() if isinstance(value, pd.Timestamp) else value
        for key, value in asdict(result.split).items()
    }
    training = result.training_summary.copy()
    trained_until = training["trained_until"]
    if not isinstance(trained_until, pd.Timestamp):
        raise TypeError("AFT 학습 종료 시각은 pandas Timestamp여야 합니다.")
    training["trained_until"] = trained_until.isoformat()
    observed_event_time = evaluate_xgboost_aft_observed_event_time(result)

    lower = float(bootstrap.summary["bootstrap_lower_95_brier_improvement"])
    upper = float(bootstrap.summary["bootstrap_upper_95_brier_improvement"])
    if lower > 0:
        decision = "AFT의 Brier 개선이 사용자 구성 변화에서도 일관되게 양수였습니다."
    elif upper < 0:
        decision = "AFT의 Brier Score가 기준선보다 일관되게 악화됐습니다."
    else:
        decision = (
            "95% Bootstrap 구간이 0을 포함해 AFT의 안정적인 우위를 확정하지 않습니다."
        )

    return {
        "dataset": "uci_online_retail_ii",
        "experiment_version": "xgboost_aft_validation_v1",
        "evaluation_split": "validation",
        "horizon_days": int(result.probability.summary["horizon_days"]),
        "split": split,
        "runtime": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("xgboost", "pandas", "numpy")},
        },
        "training": training,
        "validation_concordance": result.concordance,
        "validation_probability": result.probability.summary,
        "validation_observed_event_time": observed_event_time,
        "validation_calibration": dataframe_to_nullable_records(
            result.probability.calibration
        ),
        "validation_user_bootstrap": bootstrap.summary,
        "decision": decision,
        "scope": (
            "Train으로 모델과 전체 확률 기준선을 학습하고 Validation에서만 "
            "평가했습니다. Test는 모델 선택이 끝나기 전까지 사용하지 않습니다. "
            "Bootstrap은 고정된 모델·확률·IPCW 가중치에서 사용자 구성의 "
            "불확실성을 추정하며 재학습 불확실성과 같은 Validation을 사용한 "
            "후보 선택 과정의 불확실성·낙관성은 포함하지 않습니다."
        ),
    }


def build_xgboost_aft_bootstrap_trials_report(
    result: XGBoostAFTExperimentResult,
) -> dict[str, object]:
    """사용자 Bootstrap 반복 원자료를 요약 보고서와 분리해 반환합니다."""
    bootstrap = result.probability.user_bootstrap
    if bootstrap is None:
        raise ValueError("저장할 XGBoost AFT 사용자 Bootstrap 결과가 없습니다.")
    return {
        "dataset": "uci_online_retail_ii",
        "experiment_version": "xgboost_aft_validation_v1",
        "evaluation_split": "validation",
        "bootstrap_replicates": bootstrap.summary["bootstrap_replicates"],
        "random_seed": bootstrap.summary["random_seed"],
        "summary": bootstrap.summary,
        "trials": dataframe_to_nullable_records(bootstrap.trials),
    }


def render_xgboost_aft_report(report: dict[str, object]) -> str:
    """AFT 핵심 지표와 Bootstrap 판단을 사람이 검토할 Markdown으로 만듭니다."""
    training = report["training"]
    concordance = report["validation_concordance"]
    probability = report["validation_probability"]
    observed_time = report["validation_observed_event_time"]
    bootstrap = report["validation_user_bootstrap"]
    calibration = report["validation_calibration"]

    calibration_lines = [
        (
            f"| {row['bin_lower_bound']:.0%}~{row['bin_upper_bound']:.0%} | "
            f"{row['sample_count']:,} | {row['mean_predicted_probability']:.2%} | "
            f"{row['observed_event_rate']:.2%} | {row['calibration_gap']:+.2%} |"
        )
        for row in calibration
    ]
    maximum_gap_row = max(
        calibration,
        key=lambda row: row["absolute_calibration_gap"],
    )
    lines = [
        "# UCI XGBoost AFT Validation 평가",
        "",
        "## 학습 설정",
        "",
        f"- 학습 표본: `{training['included_sample_count']:,}`건 "
        f"(0일 제외 `{training['excluded_zero_duration_count']:,}`건)",
        f"- 피처: `{', '.join(training['feature_columns'])}`",
        f"- AFT 분포 / scale: `{training['loss_distribution']}` / "
        f"`{training['loss_distribution_scale']}`",
        f"- 부스팅 반복: `{training['num_boost_round']:,}`회",
        "",
        "## Validation 결과",
        "",
        f"- 평가 시점: `{report['horizon_days']}`일 내 동일 상품 재구매",
        f"- 평가 모집단: 원본 `{probability['source_validation_sample_count']:,}`건 "
        f"→ 0일 제외 `{probability['excluded_zero_duration_count']:,}`건 "
        f"→ AFT 평가 `{probability['aft_evaluation_sample_count']:,}`건",
        f"- 정답 확인 표본: `{probability['outcome_known_count']:,}`건",
        f"- Train 상수확률 기준선: `{training['training_reference_probability']:.2%}`",
        f"- IPCW 가중 평균 예측확률 / 실제 사건률: "
        f"`{probability['weighted_mean_predicted_probability']:.2%}` / "
        f"`{probability['weighted_observed_event_rate']:.2%}`",
        "",
        "| C-index | AFT Brier | 기준 Brier | Brier Skill | ECE | MCE |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {concordance['ipcw_concordance_index']:.6f} | "
            f"{probability['ipcw_brier_score']:.6f} | "
            f"{probability['ipcw_reference_brier_score']:.6f} | "
            f"{_format_optional_metric(probability['brier_skill_score'])} | "
            f"{probability['expected_calibration_error']:.6f} | "
            f"{probability['maximum_calibration_error']:.6f} |"
        ),
        "",
        "## 관측 재구매 시점 오차",
        "",
        "실제 재구매가 Validation 기간 안에서 관측된 행만 사용하며 검열 행은 "
        "제외합니다.",
        "",
        "| 관측 사건 | MAE | Median AE | ±7일 적중률 |",
        "| ---: | ---: | ---: | ---: |",
        f"| {observed_time['sample_count']:,} | {observed_time['mae_days']:.2f}일 | "
        f"{observed_time['median_absolute_error_days']:.2f}일 | "
        f"{observed_time['within_7_days_rate']:.2%} |",
        "",
        "## Calibration",
        "",
        "평균 예측·실제 사건률·차이는 IPCW 가중 값이며, 표본은 정답을 확인한 "
        "원시 행 수입니다.",
        "",
        "| 확률 구간 | 표본 | 평균 예측 | 실제 사건률 | 차이(%p) |",
        "| --- | ---: | ---: | ---: | ---: |",
        *calibration_lines,
        "",
        f"MCE 구간의 표본은 `{maximum_gap_row['sample_count']:,}`건이므로 "
        "MCE만으로 전체 확률 성능을 판단하지 않습니다.",
        "",
        "## 사용자 단위 Bootstrap",
        "",
        f"- 정답 확인 행을 가진 Bootstrap 대상 사용자: `{bootstrap['user_count']:,}`명",
        f"- 반복 수 / seed: `{bootstrap['bootstrap_replicates']:,}` / "
        f"`{bootstrap['random_seed']}`",
        f"- 점 개선량: `{bootstrap['point_brier_improvement']:+.6f}`",
        f"- 평균 개선량: `{bootstrap['bootstrap_mean_brier_improvement']:+.6f}`",
        f"- 95% 구간: `[{bootstrap['bootstrap_lower_95_brier_improvement']:+.6f}, "
        f"{bootstrap['bootstrap_upper_95_brier_improvement']:+.6f}]`",
        f"- 양수 개선 비율: `{bootstrap['bootstrap_positive_improvement_rate']:.2%}`",
        "",
        f"현재 `{training['loss_distribution']}`, "
        f"`scale={training['loss_distribution_scale']}`, "
        f"`{training['num_boost_round']}`회 기준 설정에서 {report['decision']}",
        "",
        str(report["scope"]),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """실제 UCI 원본으로 반복 횟수를 선택하고 최종 후보를 상세 평가합니다."""
    source = load_uci_online_retail_ii()
    classified = classify_uci_rows(source)
    events = build_uci_purchase_events(classified.rows)
    labels = build_same_product_repurchase_labels(
        events,
        observation_end_at=pd.Timestamp(events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels)
    comparison = compare_xgboost_aft_boosting_rounds(prepared)
    selected_num_boost_round = select_xgboost_aft_boosting_round(comparison)
    comparison_report = build_xgboost_aft_round_comparison_report(
        comparison,
        selected_num_boost_round=selected_num_boost_round,
    )
    comparison_markdown = render_xgboost_aft_round_comparison_report(comparison_report)

    distribution_comparison = compare_xgboost_aft_loss_distributions(
        prepared,
        num_boost_round=selected_num_boost_round,
    )
    selected_loss_distribution = select_xgboost_aft_loss_distribution(
        distribution_comparison
    )
    distribution_comparison_report = build_xgboost_aft_distribution_comparison_report(
        distribution_comparison,
        selected_loss_distribution=selected_loss_distribution,
    )
    distribution_comparison_markdown = (
        render_xgboost_aft_distribution_comparison_report(
            distribution_comparison_report
        )
    )

    logistic_scale_comparison = compare_xgboost_aft_loss_distribution_scales(
        prepared,
        loss_distribution="logistic",
        num_boost_round=selected_num_boost_round,
    )
    successful_logistic_scales = logistic_scale_comparison.loc[
        logistic_scale_comparison["status"].eq("success")
    ]
    selected_logistic_scale = (
        None
        if successful_logistic_scales.empty
        else select_xgboost_aft_loss_distribution_scale(logistic_scale_comparison)
    )
    logistic_scale_comparison_report = (
        build_xgboost_aft_logistic_scale_comparison_report(
            logistic_scale_comparison,
            selected_loss_distribution_scale=selected_logistic_scale,
        )
    )
    logistic_scale_comparison_markdown = (
        render_xgboost_aft_logistic_scale_comparison_report(
            logistic_scale_comparison_report
        )
    )

    result = evaluate_xgboost_aft_candidate(
        prepared,
        loss_distribution=selected_loss_distribution,
        num_boost_round=selected_num_boost_round,
    )
    validate_selected_xgboost_aft_distribution_result(
        distribution_comparison,
        result,
    )
    segment_comparison = summarize_xgboost_aft_probability_by_count_segments(result)
    segment_comparison_report = build_xgboost_aft_segment_comparison_report(
        result,
        segment_comparison,
    )
    segment_comparison_markdown = render_xgboost_aft_segment_comparison_report(
        segment_comparison_report
    )
    time_comparison = compare_xgboost_aft_observed_time_with_median_baselines(
        prepared,
        result,
    )
    time_comparison_report = build_xgboost_aft_observed_time_comparison_report(
        result,
        time_comparison,
    )
    time_comparison_markdown = render_xgboost_aft_observed_time_comparison_report(
        time_comparison_report
    )
    lightgbm_comparison = compare_xgboost_aft_with_lightgbm_probability(
        prepared,
        result,
    )
    lightgbm_comparison_report = build_xgboost_aft_lightgbm_comparison_report(
        result,
        lightgbm_comparison,
    )
    lightgbm_comparison_trials_report = (
        build_xgboost_aft_lightgbm_bootstrap_trials_report(
            lightgbm_comparison_report,
            lightgbm_comparison,
        )
    )
    lightgbm_comparison_markdown = render_xgboost_aft_lightgbm_comparison_report(
        lightgbm_comparison_report
    )
    logistic_candidate = None
    candidate_pair_bootstrap = None
    unavailable_reason = None
    if selected_logistic_scale is None:
        unavailable_reason = (
            "성공한 logistic scale 후보가 없어 쌍 비교를 수행하지 않았습니다."
        )
    else:
        logistic_candidate = evaluate_xgboost_aft_candidate(
            prepared,
            bootstrap_replicates=None,
            loss_distribution="logistic",
            loss_distribution_scale=selected_logistic_scale,
            num_boost_round=selected_num_boost_round,
        )
        validate_selected_xgboost_aft_scale_result(
            logistic_scale_comparison,
            logistic_candidate,
        )
        candidate_pair_bootstrap = bootstrap_xgboost_aft_candidate_pair_by_user(
            result,
            logistic_candidate,
        )
    candidate_pair_report = build_xgboost_aft_candidate_pair_bootstrap_report(
        result,
        logistic_candidate,
        candidate_pair_bootstrap,
        unavailable_reason=unavailable_reason,
    )
    candidate_pair_trials_report = (
        build_xgboost_aft_candidate_pair_bootstrap_trials_report(
            candidate_pair_report,
            candidate_pair_bootstrap,
        )
    )
    candidate_pair_markdown = render_xgboost_aft_candidate_pair_bootstrap_report(
        candidate_pair_report
    )
    report = build_xgboost_aft_report(result)
    bootstrap_trials = build_xgboost_aft_bootstrap_trials_report(result)
    markdown = render_xgboost_aft_report(report)
    comparison_json = (
        json.dumps(
            comparison_report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    distribution_comparison_json = (
        json.dumps(
            distribution_comparison_report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    logistic_scale_comparison_json = (
        json.dumps(
            logistic_scale_comparison_report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    candidate_pair_json = (
        json.dumps(
            candidate_pair_report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    candidate_pair_trials_json = (
        json.dumps(
            candidate_pair_trials_report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    segment_comparison_json = (
        json.dumps(
            segment_comparison_report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    time_comparison_json = (
        json.dumps(
            time_comparison_report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    lightgbm_comparison_json = (
        json.dumps(
            lightgbm_comparison_report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    lightgbm_comparison_trials_json = (
        json.dumps(
            lightgbm_comparison_trials_report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    report_json = (
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    bootstrap_trials_json = (
        json.dumps(
            bootstrap_trials,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )

    # 모든 계산·검증·직렬화가 성공한 뒤 보고서들을 같은 실행 결과로 교체합니다.
    write_text_atomically(
        ROUND_COMPARISON_JSON_REPORT_PATH,
        comparison_json,
    )
    write_text_atomically(
        ROUND_COMPARISON_MARKDOWN_REPORT_PATH,
        comparison_markdown,
    )
    write_text_atomically(
        DISTRIBUTION_COMPARISON_JSON_REPORT_PATH,
        distribution_comparison_json,
    )
    write_text_atomically(
        DISTRIBUTION_COMPARISON_MARKDOWN_REPORT_PATH,
        distribution_comparison_markdown,
    )
    write_text_atomically(
        LOGISTIC_SCALE_COMPARISON_JSON_REPORT_PATH,
        logistic_scale_comparison_json,
    )
    write_text_atomically(
        LOGISTIC_SCALE_COMPARISON_MARKDOWN_REPORT_PATH,
        logistic_scale_comparison_markdown,
    )
    write_text_atomically(
        CANDIDATE_PAIR_BOOTSTRAP_JSON_REPORT_PATH,
        candidate_pair_json,
    )
    write_text_atomically(
        CANDIDATE_PAIR_BOOTSTRAP_MARKDOWN_REPORT_PATH,
        candidate_pair_markdown,
    )
    write_text_atomically(
        CANDIDATE_PAIR_BOOTSTRAP_TRIALS_REPORT_PATH,
        candidate_pair_trials_json,
    )
    write_text_atomically(
        SEGMENT_COMPARISON_JSON_REPORT_PATH,
        segment_comparison_json,
    )
    write_text_atomically(
        SEGMENT_COMPARISON_MARKDOWN_REPORT_PATH,
        segment_comparison_markdown,
    )
    write_text_atomically(
        TIME_COMPARISON_JSON_REPORT_PATH,
        time_comparison_json,
    )
    write_text_atomically(
        TIME_COMPARISON_MARKDOWN_REPORT_PATH,
        time_comparison_markdown,
    )
    write_text_atomically(
        LIGHTGBM_COMPARISON_JSON_REPORT_PATH,
        lightgbm_comparison_json,
    )
    write_text_atomically(
        LIGHTGBM_COMPARISON_MARKDOWN_REPORT_PATH,
        lightgbm_comparison_markdown,
    )
    write_text_atomically(
        LIGHTGBM_COMPARISON_TRIALS_REPORT_PATH,
        lightgbm_comparison_trials_json,
    )
    write_text_atomically(
        JSON_REPORT_PATH,
        report_json,
    )
    write_text_atomically(
        BOOTSTRAP_TRIALS_REPORT_PATH,
        bootstrap_trials_json,
    )
    write_text_atomically(MARKDOWN_REPORT_PATH, markdown)
    print(markdown)


if __name__ == "__main__":
    main()
