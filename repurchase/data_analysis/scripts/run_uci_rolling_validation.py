"""AFT와 LightGBM의 시간 이동 강건성을 과거 rolling cutoff에서 비교합니다.

최종 Test는 사용하지 않습니다. 이미 선택한 모델 설정과 피처를 고정하고,
겹치지 않는 과거 Validation 창에서 두 모델의 상대 성능 방향이 반복되는지만
검증합니다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isclose, isfinite
from typing import Final

import pandas as pd

from .loaders import load_uci_online_retail_ii
from .modeling.evaluation import (
    COUNT_SEGMENT_BINS,
    COUNT_SEGMENT_LABELS,
    evaluate_ipcw_brier_score,
    summarize_ipcw_probability_pair_by_count_segment,
)
from .modeling.maturity_analysis import summarize_validation_ipcw_weight_stability
from .modeling.rolling_validation import (
    RollingValidationError,
    build_expanding_rolling_splits,
)
from .modeling.samples import build_historical_interval_features, make_temporal_split
from .paths import REPORT_DIR
from .preprocessing.events import build_uci_purchase_events
from .preprocessing.labels import build_same_product_repurchase_labels
from .preprocessing.uci import classify_uci_rows
from .reporting import dataframe_to_nullable_records, write_text_atomically
from .run_uci_xgboost_aft import (
    BOOTSTRAP_RANDOM_SEED,
    BOOTSTRAP_REPLICATES,
    CALIBRATION_BIN_COUNT,
    HORIZON_DAYS,
    compare_xgboost_aft_with_lightgbm_probability,
    evaluate_xgboost_aft_candidate,
    prepare_xgboost_aft_experiment,
)

ROLLING_FOLD_COUNT: Final[int] = 3
AFT_ROLLING_LOSS_DISTRIBUTION: Final[str] = "normal"
AFT_ROLLING_LOSS_DISTRIBUTION_SCALE: Final[float] = 1.0
AFT_ROLLING_NUM_BOOST_ROUND: Final[int] = 20
ROLLING_REPORT_JSON_PATH = REPORT_DIR / "uci_repurchase_rolling_cutoff.json"
ROLLING_REPORT_MARKDOWN_PATH = REPORT_DIR / "uci_repurchase_rolling_cutoff.md"
ROLLING_BOOTSTRAP_TRIALS_PATH = (
    REPORT_DIR / "uci_repurchase_rolling_cutoff_bootstrap_trials.json"
)


@dataclass(frozen=True)
class RollingCutoffEvaluation:
    """Fold별 핵심 지표와 상세 Calibration·Bootstrap 원자료를 보관합니다."""

    folds: pd.DataFrame
    cohorts: pd.DataFrame
    calibration: pd.DataFrame
    bootstrap_trials: pd.DataFrame


def _model_metric_row(
    comparison: pd.DataFrame,
    model_candidate: str,
) -> pd.Series:
    """모델 하나의 비교 지표가 정확히 한 행인지 확인하고 반환합니다."""
    rows = comparison.loc[comparison["model_candidate"].eq(model_candidate)]
    if len(rows) != 1:
        raise RollingValidationError(
            f"{model_candidate} 비교 결과는 fold마다 한 행이어야 합니다."
        )
    return rows.iloc[0]


def _point_direction(brier_difference: float) -> str:
    """AFT Brier - LightGBM Brier의 부호를 사람이 읽을 방향으로 바꿉니다."""
    if brier_difference > 0:
        return "lightgbm_probability"
    if brier_difference < 0:
        return "xgboost_aft"
    return "tie"


def _interval_direction(lower: float, upper: float) -> str:
    """사용자 Bootstrap 95% 구간이 지지하는 모델을 반환합니다."""
    if lower > 0:
        return "lightgbm_probability"
    if upper < 0:
        return "xgboost_aft"
    return "inconclusive"


def add_fold_brier_contributions(
    cohorts: pd.DataFrame,
    paired_rows: pd.DataFrame,
    *,
    count_column: str,
    bucket_labels: pd.Series | None = None,
) -> pd.DataFrame:
    """구간별 오차 차이를 fold 전체 IPCW 분모에 대한 기여량으로 바꿉니다.

    구간 자체의 Brier는 각 구간의 가중치 합으로 나누지만, 기여량은 모든
    정답 확인 행의 가중치 합으로 나눕니다. 따라서 기여량 합은 fold 전체
    AFT Brier - LightGBM Brier와 일치해야 합니다.
    """
    required_columns = {
        count_column,
        "ipcw_outcome_known",
        "ipcw_event_within_horizon",
        "ipcw_weight",
        "reference_predicted_event_probability",
        "candidate_predicted_event_probability",
    }
    missing_columns = required_columns - set(paired_rows.columns)
    if missing_columns:
        raise RollingValidationError(
            f"구간별 Brier 기여량 필수 열이 누락됐습니다: {sorted(missing_columns)}"
        )
    if "count_bucket" not in cohorts.columns:
        raise RollingValidationError(
            "구간별 Brier 기여량에는 count_bucket이 필요합니다."
        )

    known_rows = paired_rows.loc[paired_rows["ipcw_outcome_known"]]
    if known_rows.empty:
        raise RollingValidationError("구간별 Brier 기여량에 정답 확인 표본이 없습니다.")
    weights = known_rows["ipcw_weight"].astype("float64")
    total_weight = float(weights.sum())
    if not isfinite(total_weight) or total_weight <= 0:
        raise RollingValidationError(
            "구간별 Brier 기여량의 IPCW 분모가 유효하지 않습니다."
        )

    actual = known_rows["ipcw_event_within_horizon"].astype("float64")
    aft_error = known_rows["reference_predicted_event_probability"].sub(actual).pow(2)
    lightgbm_error = (
        known_rows["candidate_predicted_event_probability"].sub(actual).pow(2)
    )
    weighted_difference = aft_error.sub(lightgbm_error).mul(weights)
    if bucket_labels is None:
        buckets = pd.cut(
            known_rows[count_column],
            bins=COUNT_SEGMENT_BINS,
            labels=COUNT_SEGMENT_LABELS,
            include_lowest=True,
        )
    else:
        if not bucket_labels.index.equals(paired_rows.index):
            raise RollingValidationError("구간 라벨과 평가 행의 인덱스가 다릅니다.")
        buckets = bucket_labels.loc[known_rows.index]
    if buckets.isna().any():
        raise RollingValidationError(
            f"{count_column}의 Brier 기여 구간을 만들 수 없습니다."
        )
    contribution_rows = pd.DataFrame(
        {
            "count_bucket": buckets.astype(str).to_numpy(),
            "weighted_difference": weighted_difference.to_numpy(),
            "known_ipcw_weight": weights.to_numpy(),
        }
    )
    by_bucket = contribution_rows.groupby("count_bucket", sort=False).sum()

    result = cohorts.copy()
    result["known_ipcw_weight_share"] = (
        result["count_bucket"]
        .map(by_bucket["known_ipcw_weight"])
        .fillna(0.0)
        .div(total_weight)
    )
    result["brier_difference_contribution"] = (
        result["count_bucket"]
        .map(by_bucket["weighted_difference"])
        .fillna(0.0)
        .div(total_weight)
    )
    if not isclose(float(result["known_ipcw_weight_share"].sum()), 1.0, abs_tol=1e-10):
        raise RollingValidationError("구간별 IPCW 가중치 비율의 합은 1이어야 합니다.")
    return result


def classify_history_irregularity(rows: pd.DataFrame) -> pd.Series:
    """미계산과 상대 MAD의 절반 이하·초과를 혼동 없이 구분합니다."""
    if "history_relative_mad" not in rows.columns:
        raise RollingValidationError("history_relative_mad가 누락됐습니다.")
    values = rows["history_relative_mad"]
    available = values.notna()
    measured = values.loc[available].astype("float64")
    if not measured.map(isfinite).all() or measured.lt(0).any():
        raise RollingValidationError(
            "상대 MAD는 결측 또는 0 이상의 유한값이어야 합니다."
        )
    # 결측은 구매 간격이 충분하지 않아 계산하지 못한 상태이지 0이 아닙니다.
    buckets = pd.Series("unavailable", index=rows.index, dtype="string")
    buckets.loc[available] = "relative_mad_le_0_5"
    buckets.loc[measured.loc[measured.gt(0.5)].index] = "relative_mad_gt_0_5"
    return buckets


def add_fold_outcome_distribution(
    cohorts: pd.DataFrame,
    paired_rows: pd.DataFrame,
    *,
    count_column: str,
    bucket_labels: pd.Series | None = None,
) -> pd.DataFrame:
    """정답 확인 행의 사건·미사건 수와 가중 사건율을 구간별로 보탭니다."""
    required = {
        count_column,
        "ipcw_outcome_known",
        "ipcw_event_within_horizon",
        "ipcw_weight",
    }
    missing = required - set(paired_rows.columns)
    if missing:
        raise RollingValidationError(f"사건 분포 진단 열 누락: {sorted(missing)}")
    if bucket_labels is None:
        buckets = pd.cut(
            paired_rows[count_column],
            bins=COUNT_SEGMENT_BINS,
            labels=COUNT_SEGMENT_LABELS,
            include_lowest=True,
        )
    else:
        if not bucket_labels.index.equals(paired_rows.index):
            raise RollingValidationError("구간 라벨과 평가 행의 인덱스가 다릅니다.")
        buckets = bucket_labels
    if buckets.isna().any():
        raise RollingValidationError("사건 분포 진단에서 표본의 구간이 누락됐습니다.")
    known = paired_rows.loc[paired_rows["ipcw_outcome_known"]].copy()
    known["bucket"] = buckets.loc[known.index].astype(str)
    known["event"] = known["ipcw_event_within_horizon"].astype("int64")
    known["weighted_event"] = known["event"].mul(known["ipcw_weight"])
    group = known.groupby("bucket", sort=False).agg(
        event_count=("event", "sum"),
        known_count=("event", "size"),
        event_weight=("weighted_event", "sum"),
        known_weight=("ipcw_weight", "sum"),
    )
    result = cohorts.copy()
    result["event_within_horizon_count"] = (
        result["count_bucket"].map(group["event_count"]).fillna(0).astype("int64")
    )
    result["no_event_within_horizon_count"] = (
        result["count_bucket"].map(group["known_count"]).fillna(0).astype("int64")
        - result["event_within_horizon_count"]
    )
    result["ipcw_weighted_event_rate"] = result["count_bucket"].map(
        group["event_weight"].div(group["known_weight"])
    )
    if (
        not result["event_within_horizon_count"]
        .add(result["no_event_within_horizon_count"])
        .equals(result["outcome_known_count"])
    ):
        raise RollingValidationError(
            "구간별 사건·미사건 합계가 정답 확인 수와 다릅니다."
        )
    return result


def summarize_history_irregularity_cohorts(
    paired_rows: pd.DataFrame,
    *,
    bucket_labels: pd.Series,
) -> pd.DataFrame:
    """불규칙성별 동일 평가 표본의 사건 분포와 두 모델의 Brier를 집계합니다."""
    required = {
        "user_id",
        "ipcw_outcome_known",
        "ipcw_event_within_horizon",
        "ipcw_weight",
        "reference_predicted_event_probability",
        "candidate_predicted_event_probability",
    }
    missing = required - set(paired_rows.columns)
    if missing:
        raise RollingValidationError(f"불규칙성 진단 열 누락: {sorted(missing)}")
    if paired_rows.empty or not bucket_labels.index.equals(paired_rows.index):
        raise RollingValidationError("불규칙성 구간과 평가 표본이 일치해야 합니다.")
    if bucket_labels.isna().any():
        raise RollingValidationError("불규칙성 진단에서 표본이 누락됐습니다.")

    segmented = paired_rows.copy()
    segmented["irregularity_bucket"] = bucket_labels
    summaries: list[dict[str, object]] = []
    for bucket, segment in segmented.groupby("irregularity_bucket", sort=True):
        known = segment.loc[segment["ipcw_outcome_known"]]
        summary: dict[str, object] = {
            "count_column": "history_relative_mad",
            "count_bucket": str(bucket),
            "sample_count": int(len(segment)),
            "user_count": int(segment["user_id"].nunique()),
            "outcome_known_count": int(len(known)),
        }
        for role, column in (
            ("reference", "reference_predicted_event_probability"),
            ("candidate", "candidate_predicted_event_probability"),
        ):
            if known.empty:
                summary[f"{role}_ipcw_brier_score"] = float("nan")
                continue
            evaluation_rows = segment.copy()
            evaluation_rows["predicted_event_probability"] = segment[column]
            score = evaluate_ipcw_brier_score(
                evaluation_rows, reference_probability=0.5
            )
            summary[f"{role}_ipcw_brier_score"] = float(score["ipcw_brier_score"])
        summary["brier_difference_aft_minus_lightgbm"] = float(
            summary["reference_ipcw_brier_score"]
        ) - float(summary["candidate_ipcw_brier_score"])
        summaries.append(summary)
    return pd.DataFrame(summaries)


def evaluate_rolling_cutoff_models(
    labels: pd.DataFrame,
    *,
    fold_count: int = ROLLING_FOLD_COUNT,
    horizon_days: int = HORIZON_DAYS,
    calibration_bin_count: int = CALIBRATION_BIN_COUNT,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_random_seed: int = BOOTSTRAP_RANDOM_SEED,
    loss_distribution: str = AFT_ROLLING_LOSS_DISTRIBUTION,
    loss_distribution_scale: float = AFT_ROLLING_LOSS_DISTRIBUTION_SCALE,
    num_boost_round: int = AFT_ROLLING_NUM_BOOST_ROUND,
) -> RollingCutoffEvaluation:
    """고정 모델 설정으로 여러 과거 cutoff의 동일 모집단 비교를 실행합니다."""
    samples = build_historical_interval_features(labels)
    base_split = make_temporal_split(samples)
    rolling_splits = build_expanding_rolling_splits(
        samples,
        fold_count=fold_count,
        base_split=base_split,
    )

    fold_records: list[dict[str, object]] = []
    cohort_tables: list[pd.DataFrame] = []
    calibration_tables: list[pd.DataFrame] = []
    bootstrap_tables: list[pd.DataFrame] = []
    for fold_index, split in enumerate(rolling_splits, start=1):
        fold_id = f"fold_{fold_index}"
        if split.validation_end_at > base_split.validation_end_at:
            raise RollingValidationError(
                f"{fold_id}: 원래 Test 시작 이후를 평가할 수 없습니다."
            )
        try:
            prepared = prepare_xgboost_aft_experiment(
                labels,
                horizon_days=horizon_days,
                split=split,
            )
            if prepared.training.empty or prepared.validation.empty:
                raise RollingValidationError(
                    "학습 또는 Validation 표본이 비어 있습니다."
                )
            result = evaluate_xgboost_aft_candidate(
                prepared,
                calibration_bin_count=calibration_bin_count,
                bootstrap_replicates=None,
                bootstrap_random_seed=bootstrap_random_seed,
                loss_distribution=loss_distribution,
                loss_distribution_scale=loss_distribution_scale,
                num_boost_round=num_boost_round,
            )
            comparison = compare_xgboost_aft_with_lightgbm_probability(
                prepared,
                result,
                calibration_bin_count=calibration_bin_count,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_random_seed=bootstrap_random_seed,
            )
        except Exception as error:
            raise RollingValidationError(
                f"{fold_id} rolling 비교에 실패했습니다: {error}"
            ) from error

        aft_metrics = _model_metric_row(
            comparison.comparison,
            "xgboost_aft",
        )
        lightgbm_metrics = _model_metric_row(
            comparison.comparison,
            "lightgbm_probability",
        )
        evaluation_validation = prepared.validation.loc[
            result.probability.rows.index
        ].copy()
        stability = summarize_validation_ipcw_weight_stability(
            evaluation_validation,
            horizon_days=horizon_days,
        ).iloc[0]
        if int(stability["event_within_horizon_count"]) == 0:
            raise RollingValidationError(
                f"{fold_id}: {horizon_days}일 내 재구매 사건이 없습니다."
            )
        if int(stability["no_event_within_horizon_count"]) == 0:
            raise RollingValidationError(
                f"{fold_id}: {horizon_days}일 내 미재구매 표본이 없습니다."
            )

        brier_difference = float(aft_metrics["ipcw_brier_score"]) - float(
            lightgbm_metrics["ipcw_brier_score"]
        )
        bootstrap = comparison.user_bootstrap.summary
        lower = float(bootstrap["bootstrap_lower_95_brier_improvement"])
        upper = float(bootstrap["bootstrap_upper_95_brier_improvement"])
        finite_values = (
            brier_difference,
            lower,
            upper,
            float(stability["ipcw_weight_max"]),
            float(stability["ipcw_effective_sample_size"]),
        )
        if not all(isfinite(value) for value in finite_values):
            raise RollingValidationError(
                f"{fold_id}: 비교 지표나 IPCW 안정성 값이 유한하지 않습니다."
            )

        fold_records.append(
            {
                "fold_id": fold_id,
                "train_start_at": split.start_at.isoformat(),
                "train_end_at": split.train_end_at.isoformat(),
                "validation_start_at": split.train_end_at.isoformat(),
                "validation_end_at": split.validation_end_at.isoformat(),
                "original_validation_end_at": (
                    base_split.validation_end_at.isoformat()
                ),
                "test_accessed": False,
                "horizon_days": horizon_days,
                "feature_columns": list(prepared.training_data.feature_columns),
                "aft_loss_distribution": loss_distribution,
                "aft_loss_distribution_scale": loss_distribution_scale,
                "aft_num_boost_round": num_boost_round,
                "bootstrap_replicates": bootstrap_replicates,
                "bootstrap_random_seed": bootstrap_random_seed,
                "common_training_source_sample_count": int(
                    aft_metrics["common_training_source_sample_count"]
                ),
                "aft_actual_training_sample_count": int(
                    aft_metrics["actual_training_sample_count"]
                ),
                "lightgbm_actual_training_sample_count": int(
                    lightgbm_metrics["actual_training_sample_count"]
                ),
                "evaluation_sample_count": int(aft_metrics["evaluation_sample_count"]),
                "evaluation_user_count": int(
                    comparison.paired_rows["user_id"].nunique()
                ),
                "outcome_known_count": int(stability["outcome_known_count"]),
                "outcome_unknown_count": int(stability["outcome_unknown_count"]),
                "outcome_known_rate": float(stability["outcome_known_rate"]),
                "event_within_horizon_count": int(
                    stability["event_within_horizon_count"]
                ),
                "no_event_within_horizon_count": int(
                    stability["no_event_within_horizon_count"]
                ),
                "ipcw_weighted_event_rate": float(
                    stability["ipcw_weighted_event_rate"]
                ),
                "ipcw_weight_median": float(stability["ipcw_weight_median"]),
                "ipcw_weight_p95": float(stability["ipcw_weight_p95"]),
                "ipcw_weight_p99": float(stability["ipcw_weight_p99"]),
                "ipcw_weight_max": float(stability["ipcw_weight_max"]),
                "ipcw_effective_sample_size": float(
                    stability["ipcw_effective_sample_size"]
                ),
                "ipcw_effective_to_known_sample_rate": float(
                    stability["ipcw_effective_to_known_sample_rate"]
                ),
                "aft_ipcw_brier_score": float(aft_metrics["ipcw_brier_score"]),
                "lightgbm_ipcw_brier_score": float(
                    lightgbm_metrics["ipcw_brier_score"]
                ),
                "ipcw_reference_brier_score": float(
                    aft_metrics["ipcw_reference_brier_score"]
                ),
                "aft_brier_skill_score": (
                    None
                    if pd.isna(aft_metrics["brier_skill_score"])
                    else float(aft_metrics["brier_skill_score"])
                ),
                "lightgbm_brier_skill_score": (
                    None
                    if pd.isna(lightgbm_metrics["brier_skill_score"])
                    else float(lightgbm_metrics["brier_skill_score"])
                ),
                "brier_difference_aft_minus_lightgbm": brier_difference,
                "aft_ipcw_concordance_index": float(
                    aft_metrics["ipcw_concordance_index"]
                ),
                "lightgbm_ipcw_concordance_index": float(
                    lightgbm_metrics["ipcw_concordance_index"]
                ),
                "aft_expected_calibration_error": float(
                    aft_metrics["expected_calibration_error"]
                ),
                "lightgbm_expected_calibration_error": float(
                    lightgbm_metrics["expected_calibration_error"]
                ),
                "aft_maximum_calibration_error": float(
                    aft_metrics["maximum_calibration_error"]
                ),
                "lightgbm_maximum_calibration_error": float(
                    lightgbm_metrics["maximum_calibration_error"]
                ),
                "bootstrap_lower_95_brier_difference": lower,
                "bootstrap_upper_95_brier_difference": upper,
                "bootstrap_lightgbm_improvement_rate": float(
                    bootstrap["bootstrap_positive_improvement_rate"]
                ),
                "point_lower_brier_model": _point_direction(brier_difference),
                "interval_supported_model": _interval_direction(lower, upper),
            }
        )

        # 상품 표본 수는 현재 fold의 Train 구매만 세어 Validation 미래 정보가
        # 진단 구간에도 섞이지 않게 합니다. 표본 수는 모델 피처가 아닙니다.
        paired_rows = comparison.paired_rows.copy()
        product_train_counts = prepared.training.groupby(
            "product_id", observed=True, sort=False
        ).size()
        paired_rows["product_train_sample_count"] = (
            paired_rows["product_id"]
            .map(product_train_counts)
            .fillna(0)
            .astype("int64")
        )
        for count_column in (
            "history_interval_count",
            "user_prior_order_count",
            "product_train_sample_count",
        ):
            cohorts = summarize_ipcw_probability_pair_by_count_segment(
                paired_rows,
                count_column=count_column,
                calibration_bin_count=calibration_bin_count,
            )
            cohorts = add_fold_brier_contributions(
                cohorts,
                paired_rows,
                count_column=count_column,
            )
            cohorts = add_fold_outcome_distribution(
                cohorts, paired_rows, count_column=count_column
            )
            if int(cohorts["sample_count"].sum()) != len(paired_rows):
                raise RollingValidationError(
                    f"{fold_id}: {count_column} 구간 표본 합계가 평가 표본과 다릅니다."
                )
            cohorts.insert(0, "fold_id", fold_id)
            cohorts["sample_rate"] = cohorts["sample_count"].div(len(paired_rows))
            # 기존 비교 함수에서 reference=AFT, candidate=LightGBM입니다.
            cohorts = cohorts.rename(
                columns={"brier_improvement": "brier_difference_aft_minus_lightgbm"}
            )
            cohort_tables.append(cohorts)

        # 상대 MAD는 적은 이력 때문에 아예 계산할 수 없는 행을 별도로 셉니다.
        irregularity_labels = classify_history_irregularity(paired_rows)
        irregularity_cohorts = summarize_history_irregularity_cohorts(
            paired_rows, bucket_labels=irregularity_labels
        )
        irregularity_cohorts = add_fold_brier_contributions(
            irregularity_cohorts,
            paired_rows,
            count_column="history_relative_mad",
            bucket_labels=irregularity_labels,
        )
        irregularity_cohorts = add_fold_outcome_distribution(
            irregularity_cohorts,
            paired_rows,
            count_column="history_relative_mad",
            bucket_labels=irregularity_labels,
        )
        if int(irregularity_cohorts["sample_count"].sum()) != len(paired_rows):
            raise RollingValidationError(
                f"{fold_id}: 불규칙성 구간 표본 합계가 평가 표본과 다릅니다."
            )
        irregularity_cohorts.insert(0, "fold_id", fold_id)
        irregularity_cohorts["sample_rate"] = irregularity_cohorts["sample_count"].div(
            len(paired_rows)
        )
        cohort_tables.append(irregularity_cohorts)

        fold_calibration = comparison.calibration.copy()
        fold_calibration.insert(0, "fold_id", fold_id)
        calibration_tables.append(fold_calibration)
        fold_trials = comparison.user_bootstrap.trials.copy()
        fold_trials.insert(0, "fold_id", fold_id)
        bootstrap_tables.append(fold_trials)

    return RollingCutoffEvaluation(
        folds=pd.DataFrame(fold_records),
        cohorts=pd.concat(cohort_tables, ignore_index=True),
        calibration=pd.concat(calibration_tables, ignore_index=True),
        bootstrap_trials=pd.concat(bootstrap_tables, ignore_index=True),
    )


def _selection_status(folds: pd.DataFrame) -> str:
    """여러 cutoff가 지지하는 결론의 강도를 보수적으로 분류합니다."""
    point_directions = set(folds["point_lower_brier_model"])
    if len(point_directions) != 1 or "tie" in point_directions:
        return "inconclusive"
    point_direction = next(iter(point_directions))
    interval_directions = set(folds["interval_supported_model"])
    if interval_directions == {point_direction}:
        return (
            "lightgbm_supported"
            if point_direction == "lightgbm_probability"
            else "aft_supported"
        )
    if interval_directions.issubset({point_direction, "inconclusive"}):
        return "direction_only"
    return "inconclusive"


def build_rolling_cutoff_report(
    evaluation: RollingCutoffEvaluation,
) -> dict[str, object]:
    """Rolling cutoff 결과를 표준 JSON으로 저장할 수 있게 요약합니다."""
    folds = evaluation.folds
    cohorts = evaluation.cohorts
    required_columns = {
        "fold_id",
        "train_end_at",
        "validation_start_at",
        "validation_end_at",
        "original_validation_end_at",
        "test_accessed",
        "horizon_days",
        "feature_columns",
        "aft_loss_distribution",
        "aft_loss_distribution_scale",
        "aft_num_boost_round",
        "brier_difference_aft_minus_lightgbm",
        "bootstrap_lower_95_brier_difference",
        "bootstrap_upper_95_brier_difference",
        "point_lower_brier_model",
        "interval_supported_model",
    }
    missing_columns = required_columns - set(folds.columns)
    if missing_columns:
        raise ValueError(
            f"Rolling cutoff 보고서 컬럼이 누락됐습니다: {sorted(missing_columns)}"
        )
    if folds.empty or folds["fold_id"].duplicated().any():
        raise ValueError("Rolling cutoff fold는 비어 있지 않고 고유해야 합니다.")
    if folds["test_accessed"].astype(bool).any():
        raise ValueError("Rolling cutoff 보고서에는 Test 결과를 포함할 수 없습니다.")
    for fold in folds.itertuples():
        fold_cohorts = cohorts.loc[cohorts["fold_id"].eq(fold.fold_id)]
        for count_column in (
            "history_interval_count",
            "user_prior_order_count",
            "product_train_sample_count",
            "history_relative_mad",
        ):
            count_rows = fold_cohorts.loc[fold_cohorts["count_column"].eq(count_column)]
            if int(count_rows["sample_count"].sum()) != fold.evaluation_sample_count:
                raise ValueError(
                    f"{fold.fold_id}: {count_column} 구간 합계가 평가 표본과 다릅니다."
                )
            if not isclose(
                float(count_rows["brier_difference_contribution"].sum()),
                fold.brier_difference_aft_minus_lightgbm,
                abs_tol=1e-10,
            ):
                raise ValueError(
                    f"{fold.fold_id}: {count_column} 구간 Brier 기여량 합계가 다릅니다."
                )
            if int(count_rows["outcome_known_count"].sum()) != fold.outcome_known_count:
                raise ValueError(
                    f"{fold.fold_id}: {count_column} 구간 정답 확인 합계가 다릅니다."
                )
            if (
                int(count_rows["event_within_horizon_count"].sum())
                != fold.event_within_horizon_count
            ):
                raise ValueError(
                    f"{fold.fold_id}: {count_column} 구간 사건 수 합계가 다릅니다."
                )
            if (
                int(count_rows["no_event_within_horizon_count"].sum())
                != fold.no_event_within_horizon_count
            ):
                raise ValueError(
                    f"{fold.fold_id}: {count_column} 구간 미사건 수 합계가 다릅니다."
                )

    train_end_at = pd.to_datetime(folds["train_end_at"], errors="raise")
    validation_start_at = pd.to_datetime(
        folds["validation_start_at"],
        errors="raise",
    )
    validation_end_at = pd.to_datetime(folds["validation_end_at"], errors="raise")
    original_validation_end_at = pd.to_datetime(
        folds["original_validation_end_at"],
        errors="raise",
    )
    if not train_end_at.equals(validation_start_at):
        raise ValueError("각 fold의 Validation은 Train 종료 직후 시작해야 합니다.")
    if validation_end_at.le(validation_start_at).any():
        raise ValueError("각 fold의 Validation 종료는 시작보다 늦어야 합니다.")
    if validation_end_at.gt(original_validation_end_at).any():
        raise ValueError("Rolling cutoff 결과가 원래 Test 구간을 침범했습니다.")
    if not validation_start_at.is_monotonic_increasing:
        raise ValueError("Rolling cutoff fold가 시간 순서로 정렬되지 않았습니다.")
    if len(folds) > 1:
        previous_validation_end = validation_end_at.iloc[:-1].reset_index(drop=True)
        next_validation_start = validation_start_at.iloc[1:].reset_index(drop=True)
        if previous_validation_end.gt(next_validation_start).any():
            raise ValueError("Rolling cutoff Validation 기간이 서로 겹칩니다.")

    differences = folds["brier_difference_aft_minus_lightgbm"].astype("float64")
    lower_bounds = folds["bootstrap_lower_95_brier_difference"].astype("float64")
    upper_bounds = folds["bootstrap_upper_95_brier_difference"].astype("float64")
    if not all(
        isfinite(value) for value in (*differences, *lower_bounds, *upper_bounds)
    ):
        raise ValueError("Rolling cutoff 차이와 신뢰구간은 유한해야 합니다.")
    if lower_bounds.gt(upper_bounds).any():
        raise ValueError("Bootstrap 95% 구간의 하한이 상한보다 클 수 없습니다.")
    expected_point_directions = differences.map(_point_direction)
    if not expected_point_directions.equals(folds["point_lower_brier_model"]):
        raise ValueError("Brier 차이와 점추정 우위 모델이 서로 모순됩니다.")
    expected_interval_directions = pd.Series(
        [
            _interval_direction(lower, upper)
            for lower, upper in zip(lower_bounds, upper_bounds, strict=True)
        ],
        index=folds.index,
    )
    if not expected_interval_directions.equals(folds["interval_supported_model"]):
        raise ValueError("Bootstrap 구간과 구간 지지 모델이 서로 모순됩니다.")

    for fixed_column in (
        "horizon_days",
        "feature_columns",
        "aft_loss_distribution",
        "aft_loss_distribution_scale",
        "aft_num_boost_round",
        "original_validation_end_at",
    ):
        if folds[fixed_column].map(str).nunique() != 1:
            raise ValueError(
                f"Rolling cutoff마다 고정 모델 조건이 다릅니다: {fixed_column}"
            )

    horizon_days = int(folds["horizon_days"].iat[0])
    status = _selection_status(folds)
    if status == "lightgbm_supported":
        decision = (
            "모든 과거 cutoff의 점추정과 사용자 Bootstrap 95% 구간이 "
            f"LightGBM의 낮은 {horizon_days}일 Brier를 지지했습니다."
        )
    elif status == "aft_supported":
        decision = (
            "모든 과거 cutoff의 점추정과 사용자 Bootstrap 95% 구간이 "
            f"AFT의 낮은 {horizon_days}일 Brier를 지지했습니다."
        )
    elif status == "direction_only":
        decision = (
            "점추정 방향은 모든 cutoff에서 같았지만 하나 이상의 사용자 "
            "Bootstrap 95% 구간이 0을 포함해 우위를 확정하지 않습니다."
        )
    else:
        decision = (
            "fold별 점추정 또는 사용자 Bootstrap 구간의 방향이 일관되지 않아 "
            "시간 강건한 단일 우위를 확인하지 못했습니다."
        )

    return {
        "dataset": "uci_online_retail_ii",
        "experiment_version": "repurchase_rolling_cutoff_v1",
        "evaluation_split": "historical_rolling_validation",
        "test_accessed": False,
        "fold_count": len(folds),
        "fixed_model_configuration": {
            "horizon_days": horizon_days,
            "feature_columns": list(folds["feature_columns"].iat[0]),
            "aft_loss_distribution": str(folds["aft_loss_distribution"].iat[0]),
            "aft_loss_distribution_scale": float(
                folds["aft_loss_distribution_scale"].iat[0]
            ),
            "aft_num_boost_round": int(folds["aft_num_boost_round"].iat[0]),
        },
        "model_selection_status": status,
        "lightgbm_lower_brier_fold_count": int(
            folds["point_lower_brier_model"].eq("lightgbm_probability").sum()
        ),
        "aft_lower_brier_fold_count": int(
            folds["point_lower_brier_model"].eq("xgboost_aft").sum()
        ),
        "brier_difference_direction": "aft_minus_lightgbm",
        "brier_difference_median": float(differences.median()),
        "brier_difference_min": float(differences.min()),
        "brier_difference_max": float(differences.max()),
        "folds": dataframe_to_nullable_records(folds),
        "cohorts": dataframe_to_nullable_records(cohorts),
        "calibration": dataframe_to_nullable_records(evaluation.calibration),
        "decision": decision,
        "scope": (
            "각 fold는 누적 Train과 일부 동일 사용자를 공유하므로 독립 반복 "
            "실험이 아닙니다. AFT는 우측검열 표본을 함께 학습하지만 LightGBM은 "
            "30일 정답을 확인할 수 있는 행만 학습하므로 실제 학습 수가 다릅니다. "
            "이 결과는 30일 확률의 시간 이동 강건성을 확인하며 정확한 구매일 "
            "성능을 확정하지 않습니다. 중앙 차이와 최솟값·최댓값은 기술 통계일 "
            "뿐 모델 선택 근거가 아니며 fold별 쌍 비교 방향을 우선합니다. 이 "
            "검증은 설정 선택 뒤 수행한 회고적 안정성 검사로 독립적인 최종 "
            "Test가 아닙니다. 모델·피처·판정 규칙을 동결하기 전까지 원래 Test는 "
            "사용하지 않았습니다. UCI는 해외 일반 소매 데이터이므로 반려동물 "
            "이커머스 일반화는 서비스 데이터에서 다시 검증해야 합니다."
        ),
    }


def build_rolling_bootstrap_trials_report(
    evaluation: RollingCutoffEvaluation,
) -> dict[str, object]:
    """Fold별 사용자 Bootstrap 반복값을 별도 JSON 구조로 만듭니다."""
    return {
        "dataset": "uci_online_retail_ii",
        "experiment_version": "repurchase_rolling_cutoff_v1",
        "test_accessed": False,
        "brier_difference_direction": "aft_minus_lightgbm",
        "trials": dataframe_to_nullable_records(evaluation.bootstrap_trials),
    }


def _format_optional_float(value: object) -> str:
    """결측 가능한 지표는 Markdown에서 N/A로 표시합니다."""
    return "N/A" if value is None or pd.isna(value) else f"{float(value):.6f}"


def render_rolling_cutoff_report(report: dict[str, object]) -> str:
    """Rolling cutoff 결과를 사람이 검토하기 쉬운 Markdown으로 만듭니다."""
    horizon_days = int(report["fixed_model_configuration"]["horizon_days"])
    minimum_effective_sample_rate = min(
        float(row["ipcw_effective_to_known_sample_rate"]) for row in report["folds"]
    )
    if minimum_effective_sample_rate >= 0.99:
        ipcw_stability_message = (
            "모든 fold에서 IPCW 유효 표본 수가 정답 확인 표본의 99% 이상으로 "
            "극단 가중치가 비교를 지배한 흔적은 없었습니다."
        )
    else:
        ipcw_stability_message = (
            "IPCW 유효 표본 수 비율의 최솟값이 "
            f"{minimum_effective_sample_rate:.2%}로 99% 미만입니다. "
            "소수의 큰 가중치가 비교 결과를 불안정하게 만들었는지 추가로 "
            "확인해야 합니다."
        )
    fold_lines = [
        (
            f"| {row['fold_id']} | {row['train_end_at'][:10]} | "
            f"{row['validation_end_at'][:10]} | "
            f"{row['evaluation_sample_count']:,} | "
            f"{row['ipcw_reference_brier_score']:.6f} | "
            f"{row['aft_ipcw_brier_score']:.6f} | "
            f"{_format_optional_float(row['aft_brier_skill_score'])} | "
            f"{row['lightgbm_ipcw_brier_score']:.6f} | "
            f"{_format_optional_float(row['lightgbm_brier_skill_score'])} | "
            f"{row['brier_difference_aft_minus_lightgbm']:+.6f} | "
            f"{row['bootstrap_lower_95_brier_difference']:+.6f}~"
            f"{row['bootstrap_upper_95_brier_difference']:+.6f} | "
            f"{row['point_lower_brier_model']} |"
        )
        for row in report["folds"]
    ]
    auxiliary_lines = [
        (
            f"| {row['fold_id']} | {row['aft_ipcw_concordance_index']:.6f} | "
            f"{row['lightgbm_ipcw_concordance_index']:.6f} | "
            f"{row['aft_expected_calibration_error']:.6f} | "
            f"{row['lightgbm_expected_calibration_error']:.6f} | "
            f"{row['aft_maximum_calibration_error']:.6f} | "
            f"{row['lightgbm_maximum_calibration_error']:.6f} |"
        )
        for row in report["folds"]
    ]
    stability_lines = [
        (
            f"| {row['fold_id']} | {row['aft_actual_training_sample_count']:,} | "
            f"{row['lightgbm_actual_training_sample_count']:,} | "
            f"{row['outcome_known_rate']:.2%} | "
            f"{row['ipcw_weighted_event_rate']:.2%} | "
            f"{row['ipcw_weight_p95']:.3f} | {row['ipcw_weight_max']:.3f} | "
            f"{row['ipcw_effective_to_known_sample_rate']:.2%} |"
        )
        for row in report["folds"]
    ]
    irregularity_lines = [
        (
            f"| {row['fold_id']} | {row['count_bucket']} | "
            f"{row['sample_count']:,} ({row['sample_rate']:.2%}) | "
            f"{row['outcome_known_count']:,} | "
            f"{row['event_within_horizon_count']:,} | "
            f"{_format_optional_float(row['ipcw_weighted_event_rate'])} | "
            f"{_format_optional_float(row['reference_ipcw_brier_score'])} | "
            f"{_format_optional_float(row['candidate_ipcw_brier_score'])} | "
            f"{row['brier_difference_contribution']:+.6f} |"
        )
        for row in report["cohorts"]
        if row["count_column"] == "history_relative_mad"
    ]
    return "\n".join(
        [
            "# UCI 재구매 모델 Rolling cutoff 시간 강건성 검증",
            "",
            f"- Fold 수: `{report['fold_count']}`",
            "- 차이 방향: `AFT Brier - LightGBM Brier`",
            f"- 판정 상태: `{report['model_selection_status']}`",
            f"- 중앙 차이: `{report['brier_difference_median']:+.6f}`",
            f"- 범위: `[{report['brier_difference_min']:+.6f}, "
            f"{report['brier_difference_max']:+.6f}]`",
            "- 원래 Test 사용: `아니요`",
            "",
            f"## {horizon_days}일 확률 성능",
            "",
            "| Fold | Train 종료 | Validation 종료 | 평가 표본 | 기준 Brier | "
            "AFT Brier | AFT Skill | LightGBM Brier | LightGBM Skill | 차이 | "
            "Bootstrap 95% | 점 우위 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | --- | --- |",
            *fold_lines,
            "",
            "## 순위·Calibration 보조 지표",
            "",
            "C-index는 순위 성능의 점추정이며 paired 신뢰구간을 계산하지 "
            "않았습니다. MCE는 일부 확률 구간에 민감하므로 단독 선택 기준으로 "
            "사용하지 않습니다.",
            "",
            "| Fold | AFT C-index | LightGBM C-index | AFT ECE | LightGBM ECE | "
            "AFT MCE | LightGBM MCE |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *auxiliary_lines,
            "",
            "## 학습 모집단·IPCW 안정성",
            "",
            "| Fold | AFT 학습 | LightGBM 학습 | 정답 확인률 | 가중 사건율 | "
            "가중치 p95 | 최대 가중치 | ESS/정답 확인 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *stability_lines,
            "",
            f"{ipcw_stability_message} 다만 사건율과 "
            "표본 구성이 시점마다 달라 성능 변화의 원인을 하나로 단정하지 "
            "않습니다.",
            "",
            "## 과거 구매 간격 불규칙성별 사건·오차",
            "",
            "상대 MAD = 과거 간격의 중앙값 절대편차 / 과거 간격 중앙값. "
            "간격이 부족해 계산하지 못한 표본은 unavailable로 유지합니다. "
            "0.5는 편차가 대표 간격의 절반을 넘는지 보기 위한 고정 진단 "
            "경계일 뿐 학습·모델 선택 임계값이 아닙니다. 사건 수는 정답 확인 "
            "행만 세고, 검열로 정답을 모르는 행은 미사건으로 세지 않습니다. "
            "각 구간 Brier는 구간 내 IPCW 평균이며 기여량은 fold 전체 "
            "IPCW 분모를 쓰므로, 기여량을 더해야 전체 차이가 됩니다.",
            "",
            "| Fold | 상대 MAD 구간 | 평가 표본 (비율) | 정답 확인 | 사건 | "
            "IPCW 가중 사건율 | AFT Brier | LightGBM Brier | 전체 차이 기여 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *irregularity_lines,
            "",
            str(report["decision"]),
            "",
            str(report["scope"]),
            "",
        ]
    )


def main() -> None:
    """실제 UCI 데이터에서 세 과거 cutoff의 고정 모델 비교를 실행합니다."""
    source = load_uci_online_retail_ii()
    classified = classify_uci_rows(source)
    events = build_uci_purchase_events(classified.rows)
    labels = build_same_product_repurchase_labels(
        events,
        observation_end_at=pd.Timestamp(events["ordered_at"].max()),
    )
    evaluation = evaluate_rolling_cutoff_models(labels)
    report = build_rolling_cutoff_report(evaluation)
    trials_report = build_rolling_bootstrap_trials_report(evaluation)
    markdown = render_rolling_cutoff_report(report)
    write_text_atomically(
        ROLLING_REPORT_JSON_PATH,
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    write_text_atomically(ROLLING_REPORT_MARKDOWN_PATH, markdown)
    write_text_atomically(
        ROLLING_BOOTSTRAP_TRIALS_PATH,
        json.dumps(
            trials_report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )
    print(markdown)


if __name__ == "__main__":
    main()
