"""동일한 서비스 Validation 집단에서 AFT와 LightGBM을 비교합니다."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .evaluation import (
    IPCWUserBootstrapResult,
    bootstrap_ipcw_brier_pair_difference_by_user,
    evaluate_ipcw_brier_score,
    evaluate_ipcw_concordance_index,
    summarize_ipcw_calibration,
)
from .lightgbm_baseline import (
    build_lightgbm_training_data,
    predict_lightgbm_repurchase_probability,
    train_lightgbm_classifier,
)
from .maturity_analysis import add_split_ipcw_weights
from .operational_aft_input import build_service_aft_training_rows
from .operational_orders import OperationalOrderError
from .operational_temporal_split import ServiceTemporalSplit
from .xgboost_aft import (
    build_xgboost_aft_prediction_data,
    build_xgboost_aft_training_data,
    calculate_xgboost_aft_event_probability,
    predict_xgboost_aft_duration,
    train_xgboost_aft_model,
)


@dataclass(frozen=True)
class ServiceModelComparison:
    """동일 평가 행에서 계산한 지표와 사용자 단위 쌍 비교를 보관합니다."""

    summary: pd.DataFrame
    calibration: pd.DataFrame
    paired_bootstrap: IPCWUserBootstrapResult


def _evaluate_candidate(
    rows: pd.DataFrame,
    probability: pd.Series,
    *,
    model_name: str,
    reference_probability: float,
    calibration_bin_count: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """동일 IPCW 행의 확률·순위·보정 오차를 한 번에 평가합니다."""
    if not probability.index.equals(rows.index):
        raise OperationalOrderError(
            "모델 예측 행 순서가 Validation 원본과 일치하지 않습니다."
        )
    evaluation = rows.copy()
    evaluation["predicted_event_probability"] = probability
    # C-index는 일수 예측이 아니라 30일 재구매 위험의 역순을 비교합니다.
    evaluation["predicted_duration_days"] = 1.0 - probability
    brier = evaluate_ipcw_brier_score(
        evaluation, reference_probability=reference_probability
    )
    concordance = evaluate_ipcw_concordance_index(evaluation)
    calibration = summarize_ipcw_calibration(
        evaluation, bin_count=calibration_bin_count
    )
    calibration.insert(0, "model", model_name)
    return (
        {
            "model": model_name,
            "validation_sample_count": int(brier["validation_sample_count"]),
            "outcome_known_count": int(brier["outcome_known_count"]),
            "ipcw_brier_score": float(brier["ipcw_brier_score"]),
            "ipcw_reference_brier_score": float(brier["ipcw_reference_brier_score"]),
            "ipcw_c_index": float(concordance["ipcw_concordance_index"]),
            "comparable_pair_count": int(concordance["comparable_pair_count"]),
            "expected_calibration_error": float(
                calibration["weighted_absolute_gap_contribution"].sum()
            ),
            "weighted_mean_predicted_probability": float(
                calibration["mean_predicted_probability"]
                .mul(calibration["ipcw_weight_share"])
                .sum()
            ),
            "weighted_observed_event_rate": float(
                calibration["observed_event_rate"]
                .mul(calibration["ipcw_weight_share"])
                .sum()
            ),
        },
        calibration,
    )


def compare_service_aft_lightgbm(
    split: ServiceTemporalSplit,
    *,
    horizon_days: int = 30,
    aft_boost_rounds: int = 20,
    calibration_bin_count: int = 10,
    bootstrap_replicates: int = 1_000,
    bootstrap_random_seed: int = 42,
) -> ServiceModelComparison:
    """Train에서만 학습하고 두 후보를 같은 Validation 행에서 비교합니다.

    반환 지표는 입력된 시간 분할의 탐색 결과이며 독립 Test 성능이 아닙니다.
    AFT는 전체 Train의 검열 기간을, LightGBM은 정답 확인 Train을 학습합니다.
    """
    if not split.train.index.is_unique or not split.validation.index.is_unique:
        raise OperationalOrderError("서비스 비교 표본의 행 인덱스가 중복됐습니다.")
    if split.train.empty or split.validation.empty:
        raise OperationalOrderError(
            "서비스 비교에는 Train·Validation이 모두 필요합니다."
        )

    weighted_train = add_split_ipcw_weights(split.train, horizon_days=horizon_days)
    weighted_validation = add_split_ipcw_weights(
        split.validation, horizon_days=horizon_days
    )
    known_train = weighted_train.loc[weighted_train["ipcw_outcome_known"]]
    reference_probability = float(
        known_train["ipcw_event_within_horizon"]
        .astype("float64")
        .mul(known_train["ipcw_weight"])
        .sum()
        / known_train["ipcw_weight"].sum()
    )

    aft_train = build_xgboost_aft_training_data(
        build_service_aft_training_rows(split.train)
    )
    aft_model = train_xgboost_aft_model(aft_train, num_boost_round=aft_boost_rounds)
    aft_prediction_input = build_xgboost_aft_prediction_data(
        split.validation, feature_columns=aft_model.feature_columns
    )
    aft_duration = predict_xgboost_aft_duration(aft_model, aft_prediction_input)
    aft_probability = calculate_xgboost_aft_event_probability(
        aft_model, aft_duration, horizon_days=horizon_days
    )

    lightgbm_train = build_lightgbm_training_data(weighted_train)
    lightgbm_model = train_lightgbm_classifier(lightgbm_train)
    lightgbm_probability = predict_lightgbm_repurchase_probability(
        lightgbm_model, split.validation
    )
    for probability in (aft_probability, lightgbm_probability):
        if not probability.index.equals(weighted_validation.index):
            raise OperationalOrderError(
                "두 모델의 예측 행 순서가 Validation 원본과 일치하지 않습니다."
            )

    results = []
    calibrations = []
    for name, probability in (
        ("xgboost_aft", aft_probability),
        ("lightgbm", lightgbm_probability),
    ):
        metrics, calibration = _evaluate_candidate(
            weighted_validation,
            probability,
            model_name=name,
            reference_probability=reference_probability,
            calibration_bin_count=calibration_bin_count,
        )
        metrics.update(
            {
                "horizon_days": horizon_days,
                "train_sample_count": len(split.train),
                "aft_train_sample_count": aft_train.included_sample_count,
                "lightgbm_train_sample_count": len(lightgbm_train.target),
                "aft_boost_rounds": aft_boost_rounds,
            }
        )
        results.append(metrics)
        calibrations.append(calibration)

    paired_rows = weighted_validation.copy()
    paired_rows["reference_predicted_event_probability"] = aft_probability
    paired_rows["candidate_predicted_event_probability"] = lightgbm_probability
    paired_bootstrap = bootstrap_ipcw_brier_pair_difference_by_user(
        paired_rows,
        bootstrap_replicates=bootstrap_replicates,
        random_seed=bootstrap_random_seed,
    )
    return ServiceModelComparison(
        summary=pd.DataFrame(results),
        calibration=pd.concat(calibrations, ignore_index=True),
        paired_bootstrap=paired_bootstrap,
    )
