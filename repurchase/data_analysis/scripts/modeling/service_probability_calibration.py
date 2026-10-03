"""Train 내부 보정 구간에서 AFT 조건부 확률의 단조 보정 후보를 시험합니다."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .evaluation import (
    bootstrap_ipcw_brier_pair_difference_by_user,
    evaluate_ipcw_brier_score,
    summarize_ipcw_calibration,
)
from .maturity_analysis import add_split_ipcw_weights
from .operational_aft_input import build_service_aft_training_rows
from .operational_orders import OperationalOrderError
from .operational_temporal_split import ServiceTemporalSplit
from .probability_baseline import fit_global_event_probability_baseline
from .service_landmark_validation import build_service_landmark_cohort
from .xgboost_aft import (
    build_xgboost_aft_prediction_data,
    build_xgboost_aft_training_data,
    calculate_xgboost_aft_conditional_probability,
    predict_xgboost_aft_duration,
    train_xgboost_aft_model,
)


@dataclass(frozen=True)
class ServiceProbabilityCalibrationTrial:
    """동일 내부 학습 모델의 보정 전후 외부 Validation 지표입니다."""

    summary: pd.DataFrame
    calibration: pd.DataFrame
    mappings: list[dict[str, object]]
    bootstrap_summary: pd.DataFrame
    bootstrap_trials: pd.DataFrame


def _validate_split_boundaries(
    inner_split: ServiceTemporalSplit,
    outer_split: ServiceTemporalSplit,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """내부 학습·보정과 외부 평가 표본의 시간 경계를 확인합니다."""
    for name, rows in (
        ("내부 Train", inner_split.train),
        ("내부 보정", inner_split.validation),
        ("외부 Validation", outer_split.validation),
    ):
        if rows.empty or not rows.index.is_unique or "split_end_at" not in rows:
            raise OperationalOrderError(
                f"{name} 표본이 비었거나 인덱스·시간 컷이 잘못됐습니다."
            )
        if rows["split_end_at"].nunique() != 1:
            raise OperationalOrderError(f"{name}의 시간 컷이 하나가 아닙니다.")
    train_end = pd.Timestamp(inner_split.train["split_end_at"].iloc[0])
    calibration_end = pd.Timestamp(inner_split.validation["split_end_at"].iloc[0])
    outer_train_end = pd.Timestamp(outer_split.train["split_end_at"].iloc[0])
    validation_end = pd.Timestamp(outer_split.validation["split_end_at"].iloc[0])
    if not (train_end < calibration_end == outer_train_end < validation_end):
        raise OperationalOrderError(
            "내부 Train·보정·외부 Validation 시간 컷이 겹칩니다."
        )
    if (
        not inner_split.train["split"].eq("train").all()
        or not inner_split.validation["split"].eq("validation").all()
        or not outer_split.validation["split"].eq("validation").all()
        or inner_split.train["anchor_at"].gt(train_end).any()
        or not inner_split.validation["anchor_at"].gt(train_end).all()
        or inner_split.validation["anchor_at"].gt(calibration_end).any()
        or not outer_split.validation["anchor_at"].gt(calibration_end).all()
        or outer_split.validation["anchor_at"].gt(validation_end).any()
    ):
        raise OperationalOrderError("보정 실험의 표본이 시간 분할 경계를 벗어났습니다.")
    return train_end, calibration_end, validation_end


def _conditional_probability(model, rows: pd.DataFrame, window_days: int) -> pd.Series:
    """AFT 생존분포에서 표본별 조건부 재구매 확률을 계산합니다."""
    prediction_input = build_xgboost_aft_prediction_data(
        rows, feature_columns=model.feature_columns
    )
    duration = predict_xgboost_aft_duration(model, prediction_input)
    probability = calculate_xgboost_aft_conditional_probability(
        model, duration, rows["elapsed_days"], window_days=window_days
    )
    if not probability.index.equals(rows.index):
        raise OperationalOrderError("조건부 확률과 평가 표본의 행 순서가 다릅니다.")
    values = probability.to_numpy(dtype="float64")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise OperationalOrderError("조건부 확률이 유한한 0~1 범위를 벗어났습니다.")
    return probability


def compare_service_aft_isotonic_calibration(
    inner_split: ServiceTemporalSplit,
    outer_split: ServiceTemporalSplit,
    *,
    landmark_days: tuple[int, ...],
    horizon_days: int = 30,
    aft_boost_rounds: int = 20,
    aft_loss_distribution_scale: float = 1.0,
    calibration_bin_count: int = 10,
    bootstrap_replicates: int = 1_000,
    random_seed: int = 42,
) -> ServiceProbabilityCalibrationTrial:
    """내부 보정 구간만으로 Isotonic을 학습하고 바깥 Validation에서 비교합니다.

    AFT도 내부 Train만으로 학습하므로 기존 전체 Train AFT 수치와 직접 비교하지
    않습니다. 각 시점의 원본·보정 확률은 동일한 AFT와 동일한 평가 행을 공유합니다.
    """
    _validate_split_boundaries(inner_split, outer_split)
    if (
        not landmark_days
        or len(set(landmark_days)) != len(landmark_days)
        or any(
            isinstance(day, bool) or not isinstance(day, Integral) or day < 0
            for day in landmark_days
        )
    ):
        raise OperationalOrderError(
            "보정 평가 시점은 중복 없는 0 이상의 정수여야 합니다."
        )
    if (
        isinstance(horizon_days, bool)
        or not isinstance(horizon_days, Integral)
        or horizon_days < 1
    ):
        raise OperationalOrderError("보정 예측 기간은 양의 정수 일수여야 합니다.")
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, Integral)
        or bootstrap_replicates < 1
    ):
        raise OperationalOrderError("Bootstrap 반복 횟수는 양의 정수여야 합니다.")

    aft_train = build_xgboost_aft_training_data(
        build_service_aft_training_rows(inner_split.train)
    )
    model = train_xgboost_aft_model(
        aft_train,
        loss_distribution_scale=aft_loss_distribution_scale,
        num_boost_round=aft_boost_rounds,
    )
    summary_rows: list[dict[str, int | float | str]] = []
    calibration_rows: list[pd.DataFrame] = []
    mappings: list[dict[str, object]] = []
    bootstrap_summaries: list[dict[str, float | int]] = []
    bootstrap_trials: list[pd.DataFrame] = []
    for day in landmark_days:
        train_cohort = build_service_landmark_cohort(
            inner_split.train, elapsed_days=day, split_name="train"
        )
        fit_cohort = build_service_landmark_cohort(
            inner_split.validation, elapsed_days=day, split_name="validation"
        )
        evaluation_cohort = build_service_landmark_cohort(
            outer_split.validation, elapsed_days=day, split_name="validation"
        )
        weighted_train = add_split_ipcw_weights(
            train_cohort.rows, horizon_days=horizon_days
        )
        reference = fit_global_event_probability_baseline(weighted_train)
        weighted_fit = add_split_ipcw_weights(
            fit_cohort.rows, horizon_days=horizon_days
        )
        fit_probability = _conditional_probability(model, fit_cohort.rows, horizon_days)
        known_fit = weighted_fit["ipcw_outcome_known"]
        target = weighted_fit.loc[known_fit, "ipcw_event_within_horizon"]
        if target.empty or target.nunique() != 2:
            raise OperationalOrderError(
                f"{day}일 내부 보정 구간에 두 사건 부류가 없습니다."
            )
        weights = weighted_fit.loc[known_fit, "ipcw_weight"].to_numpy(dtype="float64")
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(
            fit_probability.loc[known_fit].to_numpy(dtype="float64"),
            target.astype("float64").to_numpy(),
            sample_weight=weights,
        )

        weighted_evaluation = add_split_ipcw_weights(
            evaluation_cohort.rows, horizon_days=horizon_days
        )
        raw_probability = _conditional_probability(
            model, evaluation_cohort.rows, horizon_days
        )
        calibrated_values = np.asarray(
            calibrator.predict(raw_probability.to_numpy(dtype="float64")),
            dtype="float64",
        )
        if (
            calibrated_values.shape != (len(raw_probability),)
            or not np.isfinite(calibrated_values).all()
            or ((calibrated_values < 0) | (calibrated_values > 1)).any()
        ):
            raise OperationalOrderError("보정된 확률이 유한한 0~1 범위를 벗어났습니다.")
        for candidate, probability in (
            ("raw", raw_probability),
            ("isotonic", pd.Series(calibrated_values, index=raw_probability.index)),
        ):
            evaluation = weighted_evaluation.copy()
            evaluation["predicted_event_probability"] = probability
            brier = evaluate_ipcw_brier_score(
                evaluation, reference_probability=reference.global_event_probability
            )
            calibration = summarize_ipcw_calibration(
                evaluation, bin_count=calibration_bin_count
            )
            calibration.insert(0, "candidate", candidate)
            calibration.insert(0, "elapsed_days", int(day))
            calibration_rows.append(calibration)
            summary_rows.append(
                {
                    "elapsed_days": int(day),
                    "candidate": candidate,
                    "aft_inner_train_count": int(aft_train.included_sample_count),
                    "inner_calibration_at_risk_count": len(fit_cohort.rows),
                    "inner_calibration_known_count": int(known_fit.sum()),
                    "outer_validation_at_risk_count": len(evaluation_cohort.rows),
                    "outer_validation_known_count": int(brier["outcome_known_count"]),
                    "ipcw_brier_score": float(brier["ipcw_brier_score"]),
                    "expected_calibration_error": float(
                        calibration["weighted_absolute_gap_contribution"].sum()
                    ),
                }
            )
        pair_rows = weighted_evaluation.copy()
        pair_rows["reference_predicted_event_probability"] = raw_probability
        pair_rows["candidate_predicted_event_probability"] = calibrated_values
        paired = bootstrap_ipcw_brier_pair_difference_by_user(
            pair_rows,
            bootstrap_replicates=bootstrap_replicates,
            random_seed=random_seed,
        )
        bootstrap_summaries.append({"elapsed_days": int(day), **paired.summary})
        trials = paired.trials.copy()
        trials.insert(0, "elapsed_days", int(day))
        bootstrap_trials.append(trials)
        mappings.append(
            {
                "elapsed_days": int(day),
                "method": "ipcw_weighted_isotonic",
                "out_of_bounds": "clip",
                "x_thresholds": calibrator.X_thresholds_.tolist(),
                "y_thresholds": calibrator.y_thresholds_.tolist(),
            }
        )
    return ServiceProbabilityCalibrationTrial(
        summary=pd.DataFrame(summary_rows),
        calibration=pd.concat(calibration_rows, ignore_index=True),
        mappings=mappings,
        bootstrap_summary=pd.DataFrame(bootstrap_summaries),
        bootstrap_trials=pd.concat(bootstrap_trials, ignore_index=True),
    )
