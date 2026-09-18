"""UCI Train으로 XGBoost AFT를 학습하고 Validation에서 평가합니다.

공통 전처리·시간 분할을 재사용하되 기존 베이스라인 E2E와 실행 책임을
분리합니다. Test는 모델 선택이 끝나기 전까지 열어보지 않습니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

from .modeling.maturity_analysis import (
    add_split_ipcw_weights,
    add_split_survival_observation,
)
from .modeling.model_selection import (
    XGBoostAFTProbabilityEvaluation,
    evaluate_xgboost_aft_ipcw_concordance,
    evaluate_xgboost_aft_ipcw_probability,
)
from .modeling.probability_baseline import fit_global_event_probability_baseline
from .modeling.samples import (
    TemporalSplit,
    assign_temporal_splits,
    build_historical_interval_features,
    make_temporal_split,
)
from .modeling.xgboost_aft import (
    build_xgboost_aft_prediction_data,
    build_xgboost_aft_training_data,
    predict_xgboost_aft_duration,
    train_xgboost_aft_model,
)

HORIZON_DAYS: Final[int] = 30
CALIBRATION_BIN_COUNT: Final[int] = 10
BOOTSTRAP_REPLICATES: Final[int] = 1_000
BOOTSTRAP_RANDOM_SEED: Final[int] = 42
AFT_NUM_BOOST_ROUND: Final[int] = 5


@dataclass(frozen=True)
class XGBoostAFTExperimentResult:
    """AFT 학습 근거와 Validation 평가 결과를 한 실행 단위로 보관합니다."""

    split: TemporalSplit
    training_summary: dict[str, object]
    validation_predictions: pd.Series
    concordance: dict[str, float | int]
    probability: XGBoostAFTProbabilityEvaluation


def run_xgboost_aft_experiment(
    labels: pd.DataFrame,
    *,
    horizon_days: int = HORIZON_DAYS,
    calibration_bin_count: int = CALIBRATION_BIN_COUNT,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_random_seed: int = BOOTSTRAP_RANDOM_SEED,
    loss_distribution: str = "normal",
    loss_distribution_scale: float = 1.0,
    num_boost_round: int = AFT_NUM_BOOST_ROUND,
) -> XGBoostAFTExperimentResult:
    """같은 시간 분할에서 AFT를 학습하고 Validation 성능만 계산합니다."""
    samples = build_historical_interval_features(labels)
    split = make_temporal_split(samples)
    samples = assign_temporal_splits(samples, split)
    training = samples.loc[samples["split"].eq("train")].copy()
    validation = samples.loc[samples["split"].eq("validation")].copy()

    # AFT 학습은 고정 시점 IPCW와 분리해 생존시간·검열 정보만 사용합니다.
    survival_training = add_split_survival_observation(training)
    training_data = build_xgboost_aft_training_data(survival_training)
    training_result = train_xgboost_aft_model(
        training_data,
        loss_distribution=loss_distribution,
        loss_distribution_scale=loss_distribution_scale,
        num_boost_round=num_boost_round,
    )

    # 학습 때 확정한 피처 이름과 순서를 그대로 사용해 Validation을 예측합니다.
    prediction_data = build_xgboost_aft_prediction_data(
        validation,
        feature_columns=training_result.feature_columns,
    )
    validation_predictions = predict_xgboost_aft_duration(
        training_result,
        prediction_data,
    )

    concordance = evaluate_xgboost_aft_ipcw_concordance(
        validation,
        validation_predictions,
        horizon_days=horizon_days,
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
    probability = evaluate_xgboost_aft_ipcw_probability(
        validation,
        training_result,
        validation_predictions,
        horizon_days=horizon_days,
        training_reference_probability=(
            global_probability_model.global_event_probability
        ),
        calibration_bin_count=calibration_bin_count,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_random_seed=bootstrap_random_seed,
    )

    training_summary: dict[str, object] = {
        "training_split": "train",
        "evaluation_split": "validation",
        "trained_until": split.train_end_at,
        "source_sample_count": training_data.source_sample_count,
        "excluded_zero_duration_count": (training_data.excluded_zero_duration_count),
        "included_sample_count": training_data.included_sample_count,
        "reference_population_sample_count": len(weighted_reference_training),
        "feature_columns": list(training_result.feature_columns),
        "loss_distribution": training_result.loss_distribution,
        "loss_distribution_scale": training_result.loss_distribution_scale,
        "num_boost_round": training_result.num_boost_round,
        "training_aft_nloglik": list(training_result.training_aft_nloglik),
        "training_reference_probability": (
            global_probability_model.global_event_probability
        ),
    }
    return XGBoostAFTExperimentResult(
        split=split,
        training_summary=training_summary,
        validation_predictions=validation_predictions,
        concordance=concordance,
        probability=probability,
    )
