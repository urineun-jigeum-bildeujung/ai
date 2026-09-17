"""UCI 원본부터 시간 분할·베이스라인 학습·평가·예측까지 실행합니다.

전처리 E2E 산출물을 실제 모델 입력으로 연결하고, Train 시점과 Validation
시점에서 순차적으로 중앙값 베이스라인을 학습합니다. 결과에는 전역 기준과
계층형 기준의 성능, fallback 사용 비율, 현재 시점 예측 예시를 함께 기록합니다.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Final

import pandas as pd

from .loaders import load_uci_online_retail_ii
from .modeling.baseline import (
    HierarchicalMedianModel,
    attach_hierarchical_prior_features,
    fit_hierarchical_median_baseline,
    predict_global_median_baseline,
    predict_hierarchical_median_baseline,
)
from .modeling.error_analysis import (
    build_random_product_concentration_trials,
    compare_observed_hhi_to_random_trials,
    select_largest_error_rows,
    summarize_fixed_cohort_prior_support,
    summarize_largest_error_product_concentration,
    summarize_largest_error_tail,
    summarize_largest_error_tail_by_history_count,
    summarize_prior_support_by_log2_bucket,
    summarize_random_product_concentration_trials,
    summarize_user_product_error_variability,
    summarize_user_product_errors_by_anchor_month,
    summarize_user_product_errors_by_history_count,
    summarize_user_product_errors_by_product,
    summarize_user_product_errors_by_product_sample_count,
    summarize_user_product_errors_by_product_sample_count_bucket,
    summarize_user_product_errors_by_user_prior_order_count,
    summarize_user_product_errors_by_user_prior_order_count_bucket,
)
from .modeling.evaluation import evaluate_predictions
from .modeling.maturity_analysis import (
    compare_validation_common_followup_candidates,
    compare_validation_common_followup_monthly_composition,
    merge_validation_maturity_and_quality,
    summarize_matured_prediction_quality_by_anchor_month,
    summarize_validation_followup_distribution,
    summarize_validation_ipcw_weight_stability,
    summarize_validation_label_maturity_by_anchor_month,
)
from .modeling.model_selection import (
    evaluate_ipcw_probability_candidates,
    evaluate_ipcw_shrinkage_candidates,
    evaluate_shrinkage_candidates,
)
from .modeling.samples import (
    assign_temporal_splits,
    build_historical_interval_features,
    make_temporal_split,
)
from .paths import REPORT_DIR
from .preprocessing.events import build_uci_purchase_events
from .preprocessing.labels import build_same_product_repurchase_labels
from .preprocessing.uci import classify_uci_rows
from .reporting import dataframe_to_nullable_records, write_text_atomically

JSON_REPORT_PATH = REPORT_DIR / "uci_baseline_e2e_evaluation.json"
MARKDOWN_REPORT_PATH = REPORT_DIR / "uci_baseline_e2e_evaluation.md"
PRODUCT_CONCENTRATION_TRIALS_REPORT_PATH = (
    REPORT_DIR / "uci_baseline_e2e_product_concentration_trials.json"
)
TOP_ERROR_CONTRIBUTOR_COUNT: Final[int] = 10
# 1% 결과가 극소수 표본에만 좌우되는지 확인하기 위해 5% 결과도 함께 비교합니다.
TAIL_ERROR_RATES: Final[tuple[float, ...]] = (0.01, 0.05)
# 개인 이력 1~2건 구간의 과신을 완화하는 약한~강한 수축 후보를 비교합니다.
SHRINKAGE_STRENGTH_CANDIDATES: Final[tuple[float, ...]] = (1.0, 2.0, 4.0, 8.0)
# 상품별 확률 근거량을 전체 확률과 얼마나 강하게 섞을지 Validation에서 비교합니다.
PROBABILITY_SMOOTHING_STRENGTH_CANDIDATES: Final[tuple[float, ...]] = (
    1.0,
    2.0,
    4.0,
    8.0,
)
# 모델 후보 비교와 prior 분석이 동일한 기존 최악 표본을 사용하도록 고정합니다.
MODEL_SELECTION_TAIL_RATE: Final[float] = 0.05
# 같은 크기의 무작위 표본을 충분히 반복해 HHI 비교 분포를 만듭니다.
PRODUCT_CONCENTRATION_RANDOM_TRIAL_COUNT: Final[int] = 1_000
# 같은 코드와 데이터에서 동일한 무작위 비교 결과를 재현하도록 시드를 고정합니다.
PRODUCT_CONCENTRATION_RANDOM_SEED: Final[int] = 42
# 실제 집중도가 무작위 상단 5%에 있는지를 판단하는 사전 기준입니다.
PRODUCT_CONCENTRATION_SIGNIFICANCE_LEVEL: Final[float] = 0.05
# 실제 Validation 분위수와 2주·4주·월 단위의 서비스 해석을 함께 비교합니다.
COMMON_FOLLOWUP_HORIZON_CANDIDATES: Final[tuple[int, ...]] = (14, 28, 30, 60, 90)
# 후보 비교 결과를 바탕으로 IPCW의 1차 고정 평가 시점을 30일로 설정합니다.
PRIMARY_IPCW_HORIZON_DAYS: Final[int] = 30
# 0~100% 확률을 10%p 단위로 나눠 확률 보정 상태를 확인합니다.
CALIBRATION_BIN_COUNT: Final[int] = 10


@dataclass(frozen=True)
class BaselineCycleResult:
    """E2E 요약과 별도 보존할 무작위 분석 원자료를 함께 전달합니다."""

    summary: dict[str, Any]
    product_concentration_trials: pd.DataFrame


def _isoformat(timestamp: pd.Timestamp) -> str:
    """보고서의 모든 시각을 동일한 ISO 8601 문자열로 변환합니다."""
    return pd.Timestamp(timestamp).isoformat()


def _format_optional_days(value: object) -> str:
    """계산 가능한 일수는 소수 둘째 자리로, 없는 값은 계산 불가로 표시합니다."""
    if value is None:
        return "계산 불가"
    return f"{float(value):.2f}"


def _format_optional_rate(value: object) -> str:
    """계산 가능한 비율은 백분율로, 없는 값은 계산 불가로 표시합니다."""
    if value is None:
        return "계산 불가"
    return f"{float(value):.2%}"


def _format_optional_percentage_point(value: object) -> str:
    """계산 가능한 비중 변화는 부호 있는 퍼센트포인트로 표시합니다."""
    if value is None:
        return "계산 불가"
    return f"{float(value):+.2%}p"


def _model_summary(model: HierarchicalMedianModel) -> dict[str, object]:
    """거대한 상품별 통계 대신 재현에 필요한 모델 핵심 정보만 요약합니다."""
    return {
        "trained_until": _isoformat(model.trained_until),
        "global_median_days": model.global_median_days,
        "global_observation_count": model.global_observation_count,
        "product_model_count": len(model.product_median_days),
    }


def _evaluate_stage(
    samples: pd.DataFrame,
    model: HierarchicalMedianModel,
) -> dict[str, object]:
    """같은 평가 표본에서 전역 중앙값과 계층형 중앙값의 성능을 비교합니다."""
    global_predictions = predict_global_median_baseline(model, samples)
    hierarchical_predictions = predict_hierarchical_median_baseline(model, samples)
    global_evaluation = evaluate_predictions(global_predictions)
    hierarchical_evaluation = evaluate_predictions(hierarchical_predictions)
    history_count_analysis: list[dict[str, object]] = []
    variability_analysis: dict[str, float | int | None] | None = None
    tail_error_analysis: list[dict[str, object]] = []
    product_analysis: dict[str, object] | None = None
    monthly_analysis: list[dict[str, object]] = []
    has_user_product_history = hierarchical_predictions["prediction_source"].eq(
        "user_product_history"
    )
    if has_user_product_history.any():
        for tail_rate in TAIL_ERROR_RATES:
            tail_summary: dict[str, object] = summarize_largest_error_tail(
                hierarchical_predictions,
                tail_rate=tail_rate,
            )
            tail_summary["by_history_count"] = (
                summarize_largest_error_tail_by_history_count(
                    hierarchical_predictions,
                    tail_rate=tail_rate,
                ).to_dict(orient="records")
            )
            tail_error_analysis.append(tail_summary)
        history_count_analysis = summarize_user_product_errors_by_history_count(
            hierarchical_predictions
        ).to_dict(orient="records")
        product_summary = summarize_user_product_errors_by_product(
            hierarchical_predictions
        )
        top_products = product_summary.head(TOP_ERROR_CONTRIBUTOR_COUNT)
        product_analysis = {
            "product_count": int(len(product_summary)),
            "top_contributors": top_products.to_dict(orient="records"),
            "top_contributor_error_share": float(
                top_products["absolute_error_share"].sum()
            ),
        }
        monthly_analysis = summarize_user_product_errors_by_anchor_month(
            hierarchical_predictions
        ).to_dict(orient="records")
    has_variability = (
        has_user_product_history
        & hierarchical_predictions["history_relative_mad"].notna()
    )
    if has_variability.any():
        variability_analysis = summarize_user_product_error_variability(
            hierarchical_predictions
        )
    global_mae = float(global_evaluation["overall"]["mae_days"])
    hierarchical_mae = float(hierarchical_evaluation["overall"]["mae_days"])
    return {
        "sample_count": int(len(samples)),
        "global_baseline": global_evaluation,
        "hierarchical_baseline": hierarchical_evaluation,
        "user_product_error_tail": tail_error_analysis,
        "user_product_error_by_history_count": history_count_analysis,
        "user_product_variability_analysis": variability_analysis,
        "user_product_error_by_product": product_analysis,
        "user_product_error_by_anchor_month": monthly_analysis,
        "mae_improvement_days": global_mae - hierarchical_mae,
        "mae_improvement_rate": (
            None if global_mae == 0 else (global_mae - hierarchical_mae) / global_mae
        ),
    }


def _split_summary(samples: pd.DataFrame) -> dict[str, object]:
    """시간 구간별 전체·관측·구간 내 정답 확정 표본 수를 계산합니다."""
    summary: dict[str, object] = {}
    for split_name, split_rows in samples.groupby("split", observed=True, sort=False):
        summary[str(split_name)] = {
            "sample_count": int(len(split_rows)),
            "observed_outcome_count": int(split_rows["event_observed"].sum()),
            "matured_outcome_count": int(
                split_rows["outcome_available_by_split_end"].sum()
            ),
            "unmatured_or_censored_count": int(
                (~split_rows["outcome_available_by_split_end"]).sum()
            ),
        }
    return summary


def _analyze_validation_prior_support(
    reference_predictions: pd.DataFrame,
    model: HierarchicalMedianModel,
) -> pd.DataFrame:
    """Validation 고정 꼬리에서 prior 관측 수의 과대표집 여부를 분석합니다."""
    fixed_tail_rows = select_largest_error_rows(
        reference_predictions,
        tail_rate=MODEL_SELECTION_TAIL_RATE,
    )
    prior_enriched_predictions = attach_hierarchical_prior_features(
        model,
        reference_predictions,
    )
    summary = summarize_fixed_cohort_prior_support(
        prior_enriched_predictions,
        fixed_tail_rows,
    )
    return summary


def _build_current_prediction(
    samples: pd.DataFrame,
    model: HierarchicalMedianModel,
    observation_end_at: pd.Timestamp,
) -> dict[str, object]:
    """관측 종료 시점에 개인 이력이 있는 활성 사용자·상품 한 건을 예측합니다."""
    candidates = samples.loc[
        samples["is_right_censored"] & samples["history_interval_count"].gt(0)
    ].copy()
    if candidates.empty:
        candidates = samples.loc[samples["is_right_censored"]].copy()
    if candidates.empty:
        raise ValueError("현재 시점 예측에 사용할 우측검열 표본이 없습니다.")

    # 가장 최근 구매를 선택해 매 실행에서 동일한 운영 예시가 나오도록 합니다.
    example = candidates.sort_values(
        ["anchor_at", "user_id", "product_id"],
        kind="stable",
    ).tail(1)
    prediction = predict_hierarchical_median_baseline(model, example).iloc[0]
    # 일수의 부동소수점 오차가 나노초 단위 꼬리로 출력되지 않도록 초로 반올림합니다.
    predicted_duration_seconds = round(
        float(prediction["predicted_duration_days"]) * 86_400
    )
    predicted_date = prediction["anchor_at"] + pd.to_timedelta(
        predicted_duration_seconds,
        unit="s",
    )
    return {
        "user_id": str(prediction["user_id"]),
        "product_id": str(prediction["product_id"]),
        "last_purchase_at": _isoformat(prediction["anchor_at"]),
        "predicted_duration_days": float(prediction["predicted_duration_days"]),
        "predicted_purchase_at": _isoformat(predicted_date),
        "days_remaining_at_observation_end": float(
            (predicted_date - observation_end_at).total_seconds() / 86_400
        ),
        "prediction_source": str(prediction["prediction_source"]),
        "prediction_observation_count": int(prediction["prediction_observation_count"]),
    }


def run_baseline_cycle(labels: pd.DataFrame) -> BaselineCycleResult:
    """재구매 라벨부터 시간 분할·순차 학습·평가·현재 예측을 실행합니다."""
    samples = build_historical_interval_features(labels)
    split = make_temporal_split(samples)
    samples = assign_temporal_splits(samples, split)

    # 라벨 성숙률의 분모가 사라지지 않도록 전체 Validation 모집단을 먼저 보존합니다.
    validation_population = samples.loc[samples["split"].eq("validation")].copy()
    validation_followup_distribution = summarize_validation_followup_distribution(
        validation_population
    )
    validation_common_followup_candidates = (
        compare_validation_common_followup_candidates(
            validation_population,
            horizon_days_candidates=COMMON_FOLLOWUP_HORIZON_CANDIDATES,
        )
    )
    validation_common_followup_monthly_composition = (
        compare_validation_common_followup_monthly_composition(
            validation_population,
            horizon_days_candidates=COMMON_FOLLOWUP_HORIZON_CANDIDATES,
        )
    )
    validation_ipcw_weight_stability = summarize_validation_ipcw_weight_stability(
        validation_population,
        horizon_days=PRIMARY_IPCW_HORIZON_DAYS,
    )
    training_population = samples.loc[samples["split"].eq("train")].copy()
    validation_ipcw_probability_evaluation = evaluate_ipcw_probability_candidates(
        training_population,
        validation_population,
        product_smoothing_strengths=(PROBABILITY_SMOOTHING_STRENGTH_CANDIDATES),
        horizon_days=PRIMARY_IPCW_HORIZON_DAYS,
        calibration_bin_count=CALIBRATION_BIN_COUNT,
    )
    validation_ipcw_probability_comparison = (
        validation_ipcw_probability_evaluation.comparison
    )
    validation_ipcw_probability_calibration = (
        validation_ipcw_probability_evaluation.calibration
    )
    # 실제 오차는 평가 마감일까지 다음 구매 정답이 확인된 표본에서만 계산합니다.
    validation_rows = validation_population.loc[
        validation_population["outcome_available_by_split_end"]
    ].copy()
    test_rows = samples.loc[
        samples["split"].eq("test") & samples["outcome_available_by_split_end"]
    ]
    if validation_rows.empty or test_rows.empty:
        raise ValueError("Validation 또는 Test에 평가 가능한 관측 표본이 없습니다.")

    train_model = fit_hierarchical_median_baseline(
        samples,
        trained_until=split.train_end_at,
    )
    test_model = fit_hierarchical_median_baseline(
        samples,
        trained_until=split.validation_end_at,
    )
    validation_shrinkage_candidates = evaluate_shrinkage_candidates(
        validation_rows,
        train_model,
        shrinkage_strengths=SHRINKAGE_STRENGTH_CANDIDATES,
        tail_rate=MODEL_SELECTION_TAIL_RATE,
    )
    validation_ipcw_candidate_evaluation = evaluate_ipcw_shrinkage_candidates(
        validation_population,
        train_model,
        shrinkage_strengths=SHRINKAGE_STRENGTH_CANDIDATES,
        horizon_days=PRIMARY_IPCW_HORIZON_DAYS,
    )
    validation_ipcw_candidate_comparison = (
        validation_ipcw_candidate_evaluation.comparison
    )
    validation_ipcw_binary_evaluation = (
        validation_ipcw_candidate_evaluation.reference_binary_evaluation
    )
    validation_ipcw_concordance_evaluation = (
        validation_ipcw_candidate_evaluation.reference_concordance_evaluation
    )
    # 같은 Validation 기준 예측을 원인별 분석에서 공유해 계산 기준을 통일합니다.
    validation_reference_predictions = predict_hierarchical_median_baseline(
        train_model,
        validation_rows,
    )
    validation_maturity_analysis = merge_validation_maturity_and_quality(
        summarize_validation_label_maturity_by_anchor_month(validation_population),
        summarize_matured_prediction_quality_by_anchor_month(
            validation_reference_predictions
        ),
    )
    validation_prior_support_analysis = _analyze_validation_prior_support(
        validation_reference_predictions,
        train_model,
    )
    validation_prior_support_bucket_analysis = summarize_prior_support_by_log2_bucket(
        validation_prior_support_analysis
    )
    validation_product_concentration_analysis = (
        summarize_largest_error_product_concentration(
            validation_reference_predictions,
            tail_rate=MODEL_SELECTION_TAIL_RATE,
        )
    )
    validation_product_sample_count_analysis = (
        summarize_user_product_errors_by_product_sample_count(
            validation_reference_predictions,
            tail_rate=MODEL_SELECTION_TAIL_RATE,
        )
    )
    validation_product_sample_count_bucket_analysis = (
        summarize_user_product_errors_by_product_sample_count_bucket(
            validation_reference_predictions,
            tail_rate=MODEL_SELECTION_TAIL_RATE,
        )
    )
    validation_user_prior_order_count_analysis = (
        summarize_user_product_errors_by_user_prior_order_count(
            validation_reference_predictions,
            tail_rate=MODEL_SELECTION_TAIL_RATE,
        )
    )
    validation_user_prior_order_count_bucket_analysis = (
        summarize_user_product_errors_by_user_prior_order_count_bucket(
            validation_reference_predictions,
            tail_rate=MODEL_SELECTION_TAIL_RATE,
        )
    )
    tail_sample_count = int(
        validation_product_concentration_analysis["largest_error_tail"][
            "total_sample_count"
        ]
    )
    validation_product_concentration_random_trials = (
        build_random_product_concentration_trials(
            validation_reference_predictions,
            sample_count=tail_sample_count,
            trial_count=PRODUCT_CONCENTRATION_RANDOM_TRIAL_COUNT,
            random_seed=PRODUCT_CONCENTRATION_RANDOM_SEED,
        )
    )
    observed_tail_hhi = float(
        validation_product_concentration_analysis["largest_error_tail"]["hhi"]
    )
    validation_product_concentration_random_baseline = {
        "sample_count_per_trial": tail_sample_count,
        "random_seed": PRODUCT_CONCENTRATION_RANDOM_SEED,
        "distribution": summarize_random_product_concentration_trials(
            validation_product_concentration_random_trials
        ),
        "observed_comparison": compare_observed_hhi_to_random_trials(
            validation_product_concentration_random_trials,
            observed_hhi=observed_tail_hhi,
        ),
    }
    observation_end_at = pd.Timestamp(samples["anchor_at"].max())

    invariants = {
        "temporal_boundaries_ordered": bool(
            split.start_at < split.train_end_at < split.validation_end_at < split.end_at
        ),
        "validation_anchors_after_train": bool(
            validation_population["anchor_at"].gt(split.train_end_at).all()
        ),
        "test_anchors_after_validation": bool(
            test_rows["anchor_at"].gt(split.validation_end_at).all()
        ),
        "validation_outcomes_matured": bool(
            validation_rows["next_same_product_at"].le(split.validation_end_at).all()
        ),
        "validation_population_preserved": bool(
            int(validation_maturity_analysis["validation_sample_count"].sum())
            == len(validation_population)
        ),
        "validation_matured_population_preserved": bool(
            int(validation_maturity_analysis["matured_sample_count"].sum())
            == len(validation_rows)
        ),
        "common_followup_validation_population_preserved": bool(
            validation_common_followup_candidates["validation_sample_count"]
            .eq(len(validation_population))
            .all()
        ),
        "common_followup_coverage_counts_balanced": bool(
            (
                validation_common_followup_candidates["eligible_sample_count"]
                + validation_common_followup_candidates["ineligible_sample_count"]
            )
            .eq(len(validation_population))
            .all()
        ),
        "common_followup_outcome_counts_balanced": bool(
            (
                validation_common_followup_candidates["event_within_horizon_count"]
                + validation_common_followup_candidates["no_event_within_horizon_count"]
            )
            .eq(validation_common_followup_candidates["eligible_sample_count"])
            .all()
        ),
        "ipcw_validation_population_preserved": bool(
            (
                validation_ipcw_weight_stability.loc[0, "outcome_known_count"]
                + validation_ipcw_weight_stability.loc[0, "outcome_unknown_count"]
            )
            == len(validation_population)
        ),
        "ipcw_known_outcomes_balanced": bool(
            (
                validation_ipcw_weight_stability.loc[0, "event_within_horizon_count"]
                + validation_ipcw_weight_stability.loc[
                    0, "no_event_within_horizon_count"
                ]
            )
            == validation_ipcw_weight_stability.loc[0, "outcome_known_count"]
        ),
        "ipcw_weighted_outcome_mass_balanced": bool(
            math.isclose(
                validation_ipcw_weight_stability.loc[0, "ipcw_weighted_event_mass"]
                + validation_ipcw_weight_stability.loc[
                    0, "ipcw_weighted_no_event_mass"
                ],
                validation_ipcw_weight_stability.loc[0, "ipcw_weight_sum"],
            )
        ),
        "ipcw_binary_evaluation_population_preserved": bool(
            validation_ipcw_binary_evaluation["validation_sample_count"]
            == len(validation_population)
        ),
        "ipcw_binary_evaluation_known_population_preserved": bool(
            validation_ipcw_binary_evaluation["outcome_known_count"]
            == validation_ipcw_weight_stability.loc[0, "outcome_known_count"]
        ),
        "ipcw_binary_error_mass_balanced": bool(
            math.isclose(
                validation_ipcw_binary_evaluation["ipcw_weighted_false_positive_mass"]
                + validation_ipcw_binary_evaluation[
                    "ipcw_weighted_false_negative_mass"
                ],
                validation_ipcw_binary_evaluation["ipcw_weighted_error_mass"],
            )
        ),
        "ipcw_binary_confusion_mass_balanced": bool(
            math.isclose(
                validation_ipcw_binary_evaluation["ipcw_weighted_true_positive_mass"]
                + validation_ipcw_binary_evaluation["ipcw_weighted_true_negative_mass"]
                + validation_ipcw_binary_evaluation["ipcw_weighted_false_positive_mass"]
                + validation_ipcw_binary_evaluation[
                    "ipcw_weighted_false_negative_mass"
                ],
                validation_ipcw_binary_evaluation["ipcw_weight_sum"],
            )
        ),
        "ipcw_concordance_population_preserved": bool(
            validation_ipcw_concordance_evaluation["validation_sample_count"]
            == len(validation_population)
        ),
        "ipcw_concordance_pair_counts_balanced": bool(
            validation_ipcw_concordance_evaluation["concordant_pair_count"]
            + validation_ipcw_concordance_evaluation["tied_pair_count"]
            + validation_ipcw_concordance_evaluation["discordant_pair_count"]
            == validation_ipcw_concordance_evaluation["comparable_pair_count"]
        ),
        "ipcw_concordance_index_in_unit_interval": bool(
            0 <= validation_ipcw_concordance_evaluation["ipcw_concordance_index"] <= 1
        ),
        "ipcw_candidate_validation_population_preserved": bool(
            validation_ipcw_candidate_comparison["validation_sample_count"]
            .eq(len(validation_population))
            .all()
        ),
        "ipcw_candidate_known_population_preserved": bool(
            validation_ipcw_candidate_comparison["outcome_known_count"]
            .eq(validation_ipcw_binary_evaluation["outcome_known_count"])
            .all()
        ),
        "ipcw_candidate_reference_differences_zero": bool(
            validation_ipcw_candidate_comparison.iloc[0][
                [
                    "ipcw_weighted_binary_accuracy_difference_vs_reference",
                    "ipcw_weighted_balanced_accuracy_difference_vs_reference",
                    "ipcw_concordance_index_difference_vs_reference",
                ]
            ]
            .dropna()
            .eq(0)
            .all()
        ),
        "ipcw_probability_validation_population_preserved": bool(
            validation_ipcw_probability_comparison["validation_sample_count"]
            .eq(len(validation_population))
            .all()
        ),
        "ipcw_probability_known_population_preserved": bool(
            validation_ipcw_probability_comparison["outcome_known_count"]
            .eq(validation_ipcw_binary_evaluation["outcome_known_count"])
            .all()
        ),
        "ipcw_probability_reference_scores_consistent": bool(
            math.isclose(
                validation_ipcw_probability_comparison.iloc[0]["ipcw_brier_score"],
                validation_ipcw_probability_comparison.iloc[0][
                    "ipcw_reference_brier_score"
                ],
            )
            and (
                pd.isna(
                    validation_ipcw_probability_comparison.iloc[0]["brier_skill_score"]
                )
                or math.isclose(
                    validation_ipcw_probability_comparison.iloc[0]["brier_skill_score"],
                    0.0,
                )
            )
        ),
        "ipcw_probability_calibration_population_preserved": bool(
            validation_ipcw_probability_calibration.groupby(
                ["model_candidate", "product_smoothing_strength"],
                dropna=False,
            )["sample_count"]
            .sum()
            .eq(validation_ipcw_binary_evaluation["outcome_known_count"])
            .all()
        ),
        "ipcw_probability_calibration_weight_share_balanced": bool(
            validation_ipcw_probability_calibration.groupby(
                ["model_candidate", "product_smoothing_strength"],
                dropna=False,
            )["ipcw_weight_share"]
            .sum()
            .map(lambda value: math.isclose(value, 1.0))
            .all()
        ),
        "product_sample_count_detail_population_preserved": bool(
            validation_product_sample_count_analysis["overall_sample_count"].sum()
            == validation_product_concentration_analysis["overall_personalized"][
                "total_sample_count"
            ]
        ),
        "product_sample_count_detail_tail_preserved": bool(
            validation_product_sample_count_analysis["tail_sample_count"].sum()
            == validation_product_concentration_analysis["largest_error_tail"][
                "total_sample_count"
            ]
        ),
        "product_sample_count_bucket_population_preserved": bool(
            validation_product_sample_count_bucket_analysis[
                "overall_sample_count"
            ].sum()
            == validation_product_sample_count_analysis["overall_sample_count"].sum()
        ),
        "product_sample_count_bucket_tail_preserved": bool(
            validation_product_sample_count_bucket_analysis["tail_sample_count"].sum()
            == validation_product_sample_count_analysis["tail_sample_count"].sum()
        ),
        "user_prior_order_count_detail_population_preserved": bool(
            validation_user_prior_order_count_analysis["overall_sample_count"].sum()
            == validation_product_concentration_analysis["overall_personalized"][
                "total_sample_count"
            ]
        ),
        "user_prior_order_count_detail_tail_preserved": bool(
            validation_user_prior_order_count_analysis["tail_sample_count"].sum()
            == validation_product_concentration_analysis["largest_error_tail"][
                "total_sample_count"
            ]
        ),
        "user_prior_order_count_bucket_population_preserved": bool(
            validation_user_prior_order_count_bucket_analysis[
                "overall_sample_count"
            ].sum()
            == validation_user_prior_order_count_analysis["overall_sample_count"].sum()
        ),
        "user_prior_order_count_bucket_tail_preserved": bool(
            validation_user_prior_order_count_bucket_analysis["tail_sample_count"].sum()
            == validation_user_prior_order_count_analysis["tail_sample_count"].sum()
        ),
        "test_outcomes_matured": bool(
            test_rows["next_same_product_at"].le(split.end_at).all()
        ),
    }
    failed_invariants = [name for name, passed in invariants.items() if not passed]
    if failed_invariants:
        raise RuntimeError(
            f"베이스라인 E2E 불변조건을 위반했습니다: {failed_invariants}"
        )

    summary = {
        "dataset": "uci_online_retail_ii",
        "prediction_scope": "same_user_same_product",
        "split": {
            "start_at": _isoformat(split.start_at),
            "train_end_at": _isoformat(split.train_end_at),
            "validation_end_at": _isoformat(split.validation_end_at),
            "end_at": _isoformat(split.end_at),
            "train_fraction": split.train_fraction,
            "validation_fraction": split.validation_fraction,
            "test_fraction": split.test_fraction,
        },
        "split_summary": _split_summary(samples),
        "validation_model": _model_summary(train_model),
        "test_model": _model_summary(test_model),
        # Test를 보지 않고 선택 근거를 남기기 위해 Validation 결과만 기록합니다.
        "validation_shrinkage_candidates": validation_shrinkage_candidates.to_dict(
            orient="records"
        ),
        "validation_prior_support_analysis": validation_prior_support_analysis.to_dict(
            orient="records"
        ),
        "validation_prior_support_bucket_analysis": (
            validation_prior_support_bucket_analysis.to_dict(orient="records")
        ),
        "validation_product_concentration_analysis": (
            validation_product_concentration_analysis
        ),
        "validation_product_sample_count_analysis": dataframe_to_nullable_records(
            validation_product_sample_count_analysis
        ),
        "validation_product_sample_count_bucket_analysis": (
            dataframe_to_nullable_records(
                validation_product_sample_count_bucket_analysis
            )
        ),
        "validation_user_prior_order_count_analysis": (
            dataframe_to_nullable_records(validation_user_prior_order_count_analysis)
        ),
        "validation_user_prior_order_count_bucket_analysis": (
            dataframe_to_nullable_records(
                validation_user_prior_order_count_bucket_analysis
            )
        ),
        "validation_product_concentration_random_baseline": (
            validation_product_concentration_random_baseline
        ),
        "validation_label_maturity_analysis": dataframe_to_nullable_records(
            validation_maturity_analysis
        ),
        "validation_followup_distribution": dataframe_to_nullable_records(
            validation_followup_distribution
        ),
        "validation_common_followup_candidates": dataframe_to_nullable_records(
            validation_common_followup_candidates
        ),
        "validation_common_followup_monthly_composition": (
            dataframe_to_nullable_records(
                validation_common_followup_monthly_composition
            )
        ),
        "validation_ipcw_weight_stability": dataframe_to_nullable_records(
            validation_ipcw_weight_stability
        ),
        "validation_ipcw_binary_evaluation": validation_ipcw_binary_evaluation,
        "validation_ipcw_concordance_evaluation": (
            validation_ipcw_concordance_evaluation
        ),
        "validation_ipcw_candidate_comparison": dataframe_to_nullable_records(
            validation_ipcw_candidate_comparison
        ),
        "validation_ipcw_probability_comparison": dataframe_to_nullable_records(
            validation_ipcw_probability_comparison
        ),
        "validation_ipcw_probability_calibration": dataframe_to_nullable_records(
            validation_ipcw_probability_calibration
        ),
        "validation_evaluation": _evaluate_stage(validation_rows, train_model),
        "test_evaluation": _evaluate_stage(test_rows, test_model),
        "current_prediction_example": _build_current_prediction(
            samples,
            fit_hierarchical_median_baseline(
                samples,
                trained_until=observation_end_at,
            ),
            observation_end_at,
        ),
        "invariants": invariants,
    }
    return BaselineCycleResult(
        summary=summary,
        product_concentration_trials=(
            validation_product_concentration_random_trials.copy()
        ),
    )


def render_markdown(summary: dict[str, Any]) -> str:
    """시간 분할·모델 성능·현재 예측을 사람이 검토하기 쉬운 표로 변환합니다."""
    split = summary["split"]
    validation = summary["validation_evaluation"]
    test = summary["test_evaluation"]
    shrinkage_candidates = summary["validation_shrinkage_candidates"]
    prior_support_analysis = summary["validation_prior_support_analysis"]
    prior_support_bucket_analysis = summary["validation_prior_support_bucket_analysis"]
    product_concentration = summary["validation_product_concentration_analysis"]
    product_sample_count_bucket_analysis = summary[
        "validation_product_sample_count_bucket_analysis"
    ]
    user_prior_order_count_bucket_analysis = summary[
        "validation_user_prior_order_count_bucket_analysis"
    ]
    concentration_random_baseline = summary[
        "validation_product_concentration_random_baseline"
    ]
    maturity_analysis = summary["validation_label_maturity_analysis"]
    followup_distribution = summary["validation_followup_distribution"]
    common_followup_candidates = summary["validation_common_followup_candidates"]
    common_followup_monthly_composition = summary[
        "validation_common_followup_monthly_composition"
    ]
    ipcw_weight_stability = summary["validation_ipcw_weight_stability"][0]
    ipcw_binary_evaluation = summary["validation_ipcw_binary_evaluation"]
    ipcw_concordance_evaluation = summary["validation_ipcw_concordance_evaluation"]
    ipcw_candidate_comparison = summary["validation_ipcw_candidate_comparison"]
    ipcw_probability_comparison = summary["validation_ipcw_probability_comparison"]
    ipcw_probability_calibration = summary["validation_ipcw_probability_calibration"]
    ipcw_candidate_comparison_lines = []
    for candidate in ipcw_candidate_comparison:
        candidate_label = (
            "기존 계층형"
            if candidate["shrinkage_strength"] is None
            else f"수축 k={float(candidate['shrinkage_strength']):g}"
        )
        ipcw_candidate_comparison_lines.append(
            f"| {candidate_label} | "
            f"{_format_optional_rate(candidate['ipcw_weighted_precision'])} | "
            f"{_format_optional_rate(candidate['ipcw_weighted_recall'])} | "
            f"{_format_optional_rate(candidate['ipcw_weighted_balanced_accuracy'])} | "
            f"{_format_optional_percentage_point(candidate['ipcw_weighted_balanced_accuracy_difference_vs_reference'])} | "
            f"{_format_optional_rate(candidate['ipcw_weighted_f1'])} | "
            f"{_format_optional_rate(candidate['ipcw_concordance_index'])} | "
            f"{_format_optional_percentage_point(candidate['ipcw_concordance_index_difference_vs_reference'])} |"
        )
    ipcw_probability_comparison_lines = []
    for candidate in ipcw_probability_comparison:
        candidate_label = (
            "전체 확률 기준선"
            if candidate["product_smoothing_strength"] is None
            else f"상품 확률 k={float(candidate['product_smoothing_strength']):g}"
        )
        ipcw_probability_comparison_lines.append(
            f"| {candidate_label} | "
            f"{candidate['training_global_event_probability']:.2%} | "
            f"{candidate['product_prediction_rate']:.2%} | "
            f"{candidate['ipcw_brier_score']:.6f} | "
            f"{candidate['ipcw_reference_brier_score']:.6f} | "
            f"{_format_optional_rate(candidate['brier_skill_score'])} | "
            f"{candidate['expected_calibration_error']:.2%} | "
            f"{candidate['maximum_calibration_error']:.2%} |"
        )
    best_probability_candidate = min(
        ipcw_probability_comparison,
        key=lambda candidate: candidate["ipcw_brier_score"],
    )
    best_smoothing_strength = best_probability_candidate["product_smoothing_strength"]
    best_probability_candidate_label = (
        "전체 확률 기준선"
        if best_smoothing_strength is None
        else f"상품 확률 k={float(best_smoothing_strength):g}"
    )
    best_calibration_rows = [
        row
        for row in ipcw_probability_calibration
        if row["model_candidate"] == best_probability_candidate["model_candidate"]
        and (
            row["product_smoothing_strength"] == best_smoothing_strength
            or (
                row["product_smoothing_strength"] is None
                and best_smoothing_strength is None
            )
        )
    ]
    best_calibration_lines = [
        f"| {row['bin_lower_bound']:.0%}~{row['bin_upper_bound']:.0%} | "
        f"{row['sample_count']:,} | {row['ipcw_weight_share']:.2%} | "
        f"{row['mean_predicted_probability']:.2%} | "
        f"{row['observed_event_rate']:.2%} | "
        f"{_format_optional_percentage_point(row['calibration_gap'])} |"
        for row in best_calibration_rows
    ]
    current = summary["current_prediction_example"]
    lines = [
        "# UCI 재구매 예측 1차 학습 E2E 결과",
        "",
        f"- 예측 범위: `{summary['prediction_scope']}`",
        f"- Train 종료: `{split['train_end_at']}`",
        f"- Validation 종료: `{split['validation_end_at']}`",
        f"- Test 종료: `{split['end_at']}`",
        "",
        "## 시간 분할 표본",
        "",
        "| 구간 | 전체 | 관측 정답 | 구간 내 확정 정답 | 미확정·검열 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ("train", "validation", "test"):
        values = summary["split_summary"].get(name, {})
        lines.append(
            f"| {name} | {values.get('sample_count', 0):,} | "
            f"{values.get('observed_outcome_count', 0):,} | "
            f"{values.get('matured_outcome_count', 0):,} | "
            f"{values.get('unmatured_or_censored_count', 0):,} |"
        )

    lines.extend(
        [
            "",
            "## Validation 관찰 가능 기간 분포",
            "",
            "| 분위수 | 관찰 가능 기간(일) |",
            "| ---: | ---: |",
        ]
    )
    for row in followup_distribution:
        lines.append(
            f"| {row['quantile']:.0%} | {row['available_followup_days']:.2f} |"
        )
    lines.extend(
        [
            "",
            "- 미관측·검열 표본을 제외하지 않은 전체 Validation 모집단에서 "
            "계산했습니다.",
            "- 분위수는 공통 관찰 기간 후보를 찾는 근거이며, 최종 후보의 실제 "
            "평가 가능 비율은 별도로 비교합니다.",
        ]
    )

    lines.extend(
        [
            "",
            "## Validation 공통 관찰 기간 후보별 구매 월 구성",
            "",
            "| 관찰 기간 | 구매 월 | 전체 | 평가 가능 | 평가 가능률 | 적용 전 비중 | "
            "적용 후 비중 | 비중 변화 |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in common_followup_monthly_composition:
        lines.append(
            f"| {row['horizon_days']:.0f}일 | {row['anchor_month']} | "
            f"{row['validation_sample_count']:,} | {row['eligible_sample_count']:,} | "
            f"{row['eligible_rate']:.2%} | "
            f"{row['validation_sample_share']:.2%} | "
            f"{_format_optional_rate(row['eligible_sample_share'])} | "
            f"{_format_optional_percentage_point(row['eligible_share_shift'])} |"
        )
    lines.extend(
        [
            "",
            "- 비중 변화가 양수이면 공통 관찰 기간 적용 후 해당 월이 과대표현되고, "
            "음수이면 과소표현됩니다.",
            "- 월별 절대 표본 수와 평가 가능률을 함께 확인해 소표본의 높은 비율을 "
            "과대해석하지 않습니다.",
        ]
    )

    lines.extend(
        [
            "",
            "## Validation 공통 관찰 기간 후보 비교",
            "",
            "| 관찰 기간 | 평가 가능 | 평가 제외 | 평가 가능률 | 기간 내 재구매 | "
            "기간 내 미재구매 | 기간 내 재구매율 | 재구매 간격 평균(일) | "
            "재구매 간격 중앙값(일) | 재구매 간격 25~75%(일) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in common_followup_candidates:
        lines.append(
            f"| {row['horizon_days']:.0f}일 | {row['eligible_sample_count']:,} | "
            f"{row['ineligible_sample_count']:,} | {row['eligible_rate']:.2%} | "
            f"{row['event_within_horizon_count']:,} | "
            f"{row['no_event_within_horizon_count']:,} | "
            f"{_format_optional_rate(row['event_within_horizon_rate'])} | "
            f"{_format_optional_days(row['event_duration_mean_days'])} | "
            f"{_format_optional_days(row['event_duration_median_days'])} | "
            f"{_format_optional_days(row['event_duration_q25_days'])}~"
            f"{_format_optional_days(row['event_duration_q75_days'])} |"
        )
    lines.extend(
        [
            "",
            "- 기간 내 재구매율의 분모는 전체 Validation이 아니라 해당 기간을 "
            "충분히 관찰한 평가 가능 표본입니다.",
            "- 28일과 30일은 25% 분위수 근처에서 이틀 차이의 민감도를 확인하기 "
            "위한 비교 후보이며, 결과 확인 후 하나를 선택합니다.",
        ]
    )

    lines.extend(
        [
            "",
            "## Validation 30일 IPCW 원시 가중치 안정성",
            "",
            "| 전체 표본 | 결과 확인 | 결과 불명 | 확인률 | 30일 내 재구매 | "
            "30일까지 미재구매 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {ipcw_weight_stability['validation_sample_count']:,} | "
            f"{ipcw_weight_stability['outcome_known_count']:,} | "
            f"{ipcw_weight_stability['outcome_unknown_count']:,} | "
            f"{ipcw_weight_stability['outcome_known_rate']:.2%} | "
            f"{ipcw_weight_stability['event_within_horizon_count']:,} | "
            f"{ipcw_weight_stability['no_event_within_horizon_count']:,} |",
            "",
            "| 결과 확인 표본의 단순 재구매율 | IPCW 보정 재구매율 | 차이 |",
            "| ---: | ---: | ---: |",
            f"| {ipcw_weight_stability['unweighted_known_event_rate']:.2%} | "
            f"{ipcw_weight_stability['ipcw_weighted_event_rate']:.2%} | "
            f"{_format_optional_percentage_point(ipcw_weight_stability['ipcw_event_rate_difference'])} |",
            "",
            "| 가중치 합/전체 | 평균 | 중앙값 | P90 | P95 | P99 | 최댓값 | "
            "ESS | ESS/결과 확인 | ESS/전체 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {ipcw_weight_stability['ipcw_weight_sum_ratio']:.3f} | "
            f"{ipcw_weight_stability['ipcw_weight_mean']:.3f} | "
            f"{ipcw_weight_stability['ipcw_weight_median']:.3f} | "
            f"{ipcw_weight_stability['ipcw_weight_p90']:.3f} | "
            f"{ipcw_weight_stability['ipcw_weight_p95']:.3f} | "
            f"{ipcw_weight_stability['ipcw_weight_p99']:.3f} | "
            f"{ipcw_weight_stability['ipcw_weight_max']:.3f} | "
            f"{ipcw_weight_stability['ipcw_effective_sample_size']:.2f} | "
            f"{ipcw_weight_stability['ipcw_effective_to_known_sample_rate']:.2%} | "
            f"{ipcw_weight_stability['ipcw_effective_to_validation_sample_rate']:.2%} |",
            "",
            "- 결과 불명 표본은 30일 전에 검열되어 가중치 0으로 평가식에서 제외됩니다.",
            "- 아직 상한을 적용하지 않은 원시 가중치이며, P99·최댓값·ESS를 함께 "
            "보고 상한 필요성을 판단합니다.",
            "",
            "### 계층형 중앙값 모델의 30일 IPCW 이진 평가",
            "",
            "| 30일 내 재구매 예측 | 단순 오류율 | IPCW 오류율 | IPCW 정확도 | "
            "거짓양성 질량 | 거짓음성 질량 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {ipcw_binary_evaluation['predicted_event_count']:,} | "
            f"{ipcw_binary_evaluation['unweighted_binary_error_rate']:.2%} | "
            f"{ipcw_binary_evaluation['ipcw_weighted_binary_error_rate']:.2%} | "
            f"{ipcw_binary_evaluation['ipcw_weighted_binary_accuracy']:.2%} | "
            f"{ipcw_binary_evaluation['ipcw_weighted_false_positive_mass']:.2f} | "
            f"{ipcw_binary_evaluation['ipcw_weighted_false_negative_mass']:.2f} |",
            "",
            "| 가중 TP | 가중 TN | 가중 FP | 가중 FN |",
            "| ---: | ---: | ---: | ---: |",
            f"| {ipcw_binary_evaluation['ipcw_weighted_true_positive_mass']:.2f} | "
            f"{ipcw_binary_evaluation['ipcw_weighted_true_negative_mass']:.2f} | "
            f"{ipcw_binary_evaluation['ipcw_weighted_false_positive_mass']:.2f} | "
            f"{ipcw_binary_evaluation['ipcw_weighted_false_negative_mass']:.2f} |",
            "",
            "| Precision | Recall | Specificity | Balanced Accuracy | F1 |",
            "| ---: | ---: | ---: | ---: | ---: |",
            f"| {_format_optional_rate(ipcw_binary_evaluation['ipcw_weighted_precision'])} | "
            f"{_format_optional_rate(ipcw_binary_evaluation['ipcw_weighted_recall'])} | "
            f"{_format_optional_rate(ipcw_binary_evaluation['ipcw_weighted_specificity'])} | "
            f"{_format_optional_rate(ipcw_binary_evaluation['ipcw_weighted_balanced_accuracy'])} | "
            f"{_format_optional_rate(ipcw_binary_evaluation['ipcw_weighted_f1'])} |",
            "",
            "| 계층형 모델 정확도 | 항상 미재구매 정확도 | 정확도 차이 |",
            "| ---: | ---: | ---: |",
            f"| {ipcw_binary_evaluation['ipcw_weighted_binary_accuracy']:.2%} | "
            f"{ipcw_binary_evaluation['always_no_event_accuracy']:.2%} | "
            f"{_format_optional_percentage_point(ipcw_binary_evaluation['accuracy_difference_vs_always_no_event'])} |",
            "",
            "- 예상 일수가 30일 이하이면 기간 내 재구매로 변환한 이진 "
            "베이스라인입니다.",
            "- 확률 예측이 아니므로 정식 Brier Score가 아니라 IPCW 가중 이진 "
            "오류율로 기록합니다.",
            "- 재구매가 적은 불균형 데이터에서는 정확도가 높아도 재구매를 전혀 "
            "잡지 못할 수 있어 항상 미재구매 기준선과 Balanced Accuracy를 함께 "
            "비교합니다.",
            "- 현재는 거짓양성과 거짓음성 비용을 동일하게 취급하되 두 오류 질량을 "
            "분리해 보존합니다.",
            "",
            "### 계층형 중앙값 모델의 30일 IPCW C-index",
            "",
            "| 사건 기준 표본 | 비교 기여 사건 | 비교 가능 쌍 | 일반 C-index | IPCW C-index |",
            "| ---: | ---: | ---: | ---: | ---: |",
            f"| {ipcw_concordance_evaluation['event_reference_count']:,} | "
            f"{ipcw_concordance_evaluation['contributing_event_count']:,} | "
            f"{ipcw_concordance_evaluation['comparable_pair_count']:,} | "
            f"{ipcw_concordance_evaluation['unweighted_concordance_index']:.2%} | "
            f"{ipcw_concordance_evaluation['ipcw_concordance_index']:.2%} |",
            "",
            "- C-index는 정확한 날짜 오차가 아니라 더 빨리 재구매할 사용자를 "
            "먼저 정렬했는지 평가합니다.",
            "- 0.5는 무작위 순위 수준이며 1에 가까울수록 비교 가능한 쌍의 순서를 "
            "더 정확하게 맞혔다는 뜻입니다.",
            "- IPCW C-index는 먼저 발생한 사건 시점의 검열 생존확률 제곱의 "
            "역수로 각 비교 쌍을 보정합니다.",
            "",
            "### 기존 계층형 모델과 수축 후보의 동일 조건 비교",
            "",
            "| 후보 | Precision | Recall | Balanced Accuracy | 기준 대비 | F1 | "
            "IPCW C-index | 기준 대비 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *ipcw_candidate_comparison_lines,
            "",
            "- 모든 후보는 동일한 Validation 표본·30일 시점·검열 가중치로 "
            "비교했습니다.",
            "- 기준 대비 값은 기존 계층형 중앙값 모델과의 퍼센트포인트 차이입니다.",
            "- 단일 지표만으로 후보를 확정하지 않고 순위 성능과 이진 판별 성능의 "
            "변화를 함께 확인합니다.",
            "",
            "### 30일 재구매 확률 후보의 IPCW Brier Score",
            "",
            "| 후보 | Train 전체 확률 | 상품 확률 적용률 | IPCW Brier | "
            "전체 확률 Brier | Brier Skill Score | ECE | 최대 구간 오차 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *ipcw_probability_comparison_lines,
            "",
            "- 모든 확률 후보는 Train에서만 학습하고 동일한 Validation 표본과 "
            "IPCW 가중치로 비교했습니다.",
            "- Brier Score는 0에 가까울수록 확률 예측이 실제 결과에 가깝습니다.",
            "- Brier Skill Score가 양수이면 Train 전체 재구매율만 사용하는 "
            "기준선보다 개선됐고, 음수이면 기준선보다 나쁩니다.",
            "",
            f"#### Brier 기준 현재 최저 후보 Calibration: {best_probability_candidate_label}",
            "",
            "| 예측 확률 구간 | 표본 수 | IPCW 비중 | 평균 예측 확률 | "
            "실제 재구매율 | 예측-실제 차이 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *best_calibration_lines,
            "",
            "- 예측-실제 차이가 양수이면 과대평가, 음수이면 과소평가입니다.",
            "- ECE는 각 구간의 절대 차이에 전체 IPCW 비중을 곱해 합산한 "
            "평균 Calibration 오차입니다.",
            "- 최대 구간 오차는 표본이 매우 적은 구간에서 크게 흔들릴 수 있으므로 "
            "반드시 표본 수와 IPCW 비중을 함께 확인합니다.",
            "- 현재 최저 Brier 후보는 Validation 기준 결과이며 Test 확인 전까지 "
            "최종 모델로 확정하지 않습니다.",
        ]
    )

    lines.extend(
        [
            "",
            "## Validation 월별 라벨 성숙도와 조건부 오차",
            "",
            "| 구매 기준 월 | 전체 표본 | 성숙 표본 | 미성숙·검열 | 성숙률 | "
            "관찰 가능 기간 중앙값(일) | 실제 간격 평균(일) | "
            "실제 간격 중앙값(일) | MAE(일) | 중앙 절대오차(일) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in maturity_analysis:
        lines.append(
            f"| {row['anchor_month']} | {row['validation_sample_count']:,} | "
            f"{row['matured_sample_count']:,} | "
            f"{row['unmatured_sample_count']:,} | {row['maturity_rate']:.2%} | "
            f"{row['median_available_followup_days']:.2f} | "
            f"{_format_optional_days(row['mean_target_duration_days'])} | "
            f"{_format_optional_days(row['median_target_duration_days'])} | "
            f"{_format_optional_days(row['mae_days'])} | "
            f"{_format_optional_days(row['median_absolute_error_days'])} |"
        )
    lines.extend(
        [
            "",
            "- MAE와 실제 재구매 간격은 Validation 종료일까지 다음 구매가 확인된 "
            "성숙 표본에서만 계산한 조건부 결과입니다.",
            "- 평가 마감에 가까운 월은 관찰 가능 기간이 짧아 긴 재구매 간격이 "
            "미성숙 상태로 남을 수 있으므로, 낮은 MAE를 모델 개선으로 단독 "
            "해석하지 않습니다.",
            "- 성숙 표본이 없는 월의 실제 간격과 오차는 0이 아닌 `계산 불가`로 "
            "표시합니다.",
        ]
    )

    lines.extend(
        [
            "",
            "## 모델 비교",
            "",
            "| 구간 | 모델 | MAE(일) | 중앙 절대오차(일) | ±3일 | ±7일 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for stage_name, evaluation in (("Validation", validation), ("Test", test)):
        for model_name, key in (
            ("전역 중앙값", "global_baseline"),
            ("계층형 중앙값", "hierarchical_baseline"),
        ):
            metrics = evaluation[key]["overall"]
            lines.append(
                f"| {stage_name} | {model_name} | {metrics['mae_days']:.2f} | "
                f"{metrics['median_absolute_error_days']:.2f} | "
                f"{metrics['within_3_days_rate']:.2%} | "
                f"{metrics['within_7_days_rate']:.2%} |"
            )

    if shrinkage_candidates:
        lines.extend(
            [
                "",
                "## Validation 수축 강도 후보 비교",
                "",
                "| k | 평균 개인 가중치 | MAE(일) | 중앙 절대오차(일) | ±7일 | "
                "상위 5% MAE(일) | 상위 5% 오차 기여율 | 늦은 예측 |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for candidate in shrinkage_candidates:
            lines.append(
                f"| {candidate['shrinkage_strength']:.0f} | "
                f"{candidate['mean_personal_history_weight']:.2%} | "
                f"{candidate['mae_days']:.2f} | "
                f"{candidate['median_absolute_error_days']:.2f} | "
                f"{candidate['within_7_days_rate']:.2%} | "
                f"{candidate['tail_mae_days']:.2f} | "
                f"{candidate['tail_absolute_error_share']:.2%} | "
                f"{candidate['tail_late_prediction_rate']:.2%} |"
            )

        lines.extend(
            [
                "",
                "## Validation 기존 최악 5% 고정 코호트 재평가",
                "",
                "| k | 기존 MAE(일) | 후보 MAE(일) | 개선(일) | 개선 표본 | "
                "악화 표본 | 늦은 예측 |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for candidate in shrinkage_candidates:
            lines.append(
                f"| {candidate['shrinkage_strength']:.0f} | "
                f"{candidate['fixed_cohort_reference_mae_days']:.2f} | "
                f"{candidate['fixed_cohort_candidate_mae_days']:.2f} | "
                f"{candidate['fixed_cohort_mae_improvement_days']:.2f} | "
                f"{candidate['fixed_cohort_improved_sample_count']:,}건 "
                f"({candidate['fixed_cohort_improved_sample_rate']:.2%}) | "
                f"{candidate['fixed_cohort_worsened_sample_count']:,}건 "
                f"({candidate['fixed_cohort_worsened_sample_rate']:.2%}) | "
                f"{candidate['fixed_cohort_late_prediction_rate']:.2%} |"
            )

    single_product_prior = next(
        (
            row
            for row in prior_support_analysis
            if row["prior_source"] == "product_history"
            and row["prior_observation_count"] == 1
        ),
        None,
    )
    if prior_support_bucket_analysis:
        lines.extend(
            [
                "",
                "## Validation prior 지지 표본 로그 구간 분석",
                "",
            ]
        )
        if single_product_prior is not None:
            ratio = single_product_prior["tail_overrepresentation_ratio"]
            if math.isclose(ratio, 1.0):
                distribution_text = "비슷하게"
                hypothesis_text = (
                    "관측 수 1건이 고정 꼬리에 특별히 더 많이 나타난다는 근거는 "
                    "관찰되지 않았습니다."
                )
            elif ratio > 1:
                distribution_text = "많이"
                hypothesis_text = (
                    "관측 수 1건이 고정 꼬리에 과대표집되었지만, 이 결과만으로 "
                    "꼬리오차의 원인이라고 판단할 수는 없습니다."
                )
            else:
                distribution_text = "적게"
                hypothesis_text = (
                    "따라서 관측 수 1건이 꼬리오차의 일반적인 원인이라는 가설은 "
                    "이번 Validation 분할에서 지지되지 않았습니다."
                )
            lines.extend(
                [
                    "### 상품 prior 관측 1건 가설 검증",
                    "",
                    f"- 전체 개인화 표본 중 **{single_product_prior['overall_sample_count']:,}건 "
                    f"({single_product_prior['overall_sample_rate']:.2%})**, 기존 최악 5% "
                    f"고정 꼬리 중 **{single_product_prior['fixed_tail_sample_count']:,}건 "
                    f"({single_product_prior['fixed_tail_sample_rate']:.2%})**입니다.",
                    f"- 과대표집 배율은 **{ratio:.2f}배**로, 관측 수 1건 상품 prior가 "
                    f"고정 꼬리에서 전체 분포보다 {distribution_text} 나타났습니다.",
                    f"- {hypothesis_text}",
                    "",
                ]
            )
        lines.extend(
            [
                "| prior 출처 | 관측 수 구간 | 전체 표본 | 전체 비율 | "
                "고정 꼬리 표본 | 꼬리 비율 | 과대표집 배율 |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in prior_support_bucket_analysis:
            lines.append(
                f"| `{row['prior_source']}` | `{row['prior_support_bucket']}` | "
                f"{row['overall_sample_count']:,} | {row['overall_sample_rate']:.2%} | "
                f"{row['fixed_tail_sample_count']:,} | "
                f"{row['fixed_tail_sample_rate']:.2%} | "
                f"{row['tail_overrepresentation_ratio']:.2f}배 |"
            )
        lines.extend(
            [
                "",
                "- 과대표집 배율은 전체 분포 대비 고정 꼬리에서 얼마나 자주 "
                "나타났는지를 비교한 값입니다.",
                "- 배율이 크더라도 실제 표본 수가 적으면 우연에 민감하므로, 배율과 "
                "전체·꼬리 표본 수를 함께 해석해야 합니다.",
                "- 이 표는 prior 관측 수 구간과 꼬리 동반 빈도의 기술적 연관을 "
                "보여주며, 특정 구간이 큰 오차의 원인이라는 인과관계를 증명하지 "
                "않습니다.",
            ]
        )

    lines.extend(
        [
            "",
            "## Validation 개인화 예측 상품 집중도",
            "",
            f"- 분석 요청 비율: **{product_concentration['requested_tail_rate']:.2%}**",
            f"- 실제 선택: **{product_concentration['largest_error_tail']['total_sample_count']:,} / "
            f"{product_concentration['overall_personalized']['total_sample_count']:,}건 "
            f"({product_concentration['actual_tail_sample_rate']:.3%})**",
            "",
            "| 구분 | 표본 수 | 실제 상품 수 | Top-1 점유율 | Top-5 점유율 | "
            "HHI | 유효 상품 수 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, key in (
        ("전체 개인화", "overall_personalized"),
        ("큰 절대오차 꼬리", "largest_error_tail"),
    ):
        concentration = product_concentration[key]
        lines.append(
            f"| {label} | {concentration['total_sample_count']:,} | "
            f"{concentration['unique_product_count']:,} | "
            f"{concentration['top1_share']:.2%} | "
            f"{concentration['top5_share']:.2%} | "
            f"{concentration['hhi']:.4f} | "
            f"{concentration['effective_product_count']:.2f} |"
        )
    lines.extend(
        [
            "",
            "- HHI가 높고 유효 상품 수가 작을수록 표본이 일부 상품에 더 "
            "집중되어 있음을 뜻합니다.",
            "- 집중도 차이는 오차가 특정 상품에 함께 나타나는지를 보여주지만, "
            "그 상품이 오차의 원인임을 단독으로 증명하지는 않습니다.",
        ]
    )

    if product_sample_count_bucket_analysis:
        lines.extend(
            [
                "",
                "## Validation 상품 표본 수 구간별 오차",
                "",
                "| 상품 표본 수 | 상품 수 | 전체 표본 | MAE(일) | "
                "Median AE(일) | 최악 5% 표본 | 꼬리 포함률 | 과대표집 배율 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in product_sample_count_bucket_analysis:
            lines.append(
                f"| `{row['product_sample_count_bucket']}` | "
                f"{row['unique_product_count']:,} | "
                f"{row['overall_sample_count']:,} | {row['mae_days']:.2f} | "
                f"{row['median_absolute_error_days']:.2f} | "
                f"{row['tail_sample_count']:,} | "
                f"{row['tail_membership_rate']:.2%} | "
                f"{row['tail_overrepresentation_ratio']:.2f}배 |"
            )
        lines.extend(
            [
                "",
                "- 상품 표본 수는 오차 평가가 가능한 개인화 Validation에서 "
                "해당 상품이 등장한 횟수이며, 모델 입력 피처가 아닌 사후 진단 "
                "기준입니다.",
                "- 꼬리 포함률은 각 구간 자체에서 최악 5%에 들어간 비율이고, "
                "과대표집 배율은 해당 구간의 전체 비중 대비 최악 5% 비중을 "
                "비교한 값입니다.",
                "- 배율과 오차가 높더라도 상품 수와 표본 수가 작으면 우연에 "
                "민감하므로 함께 확인해야 합니다.",
            ]
        )

    if user_prior_order_count_bucket_analysis:
        lines.extend(
            [
                "",
                "## Validation 사용자 과거 주문 수 구간별 오차",
                "",
                "| 이전 주문 수 | 사용자 수 | 전체 표본 | MAE(일) | "
                "Median AE(일) | 최악 5% 표본 | 꼬리 포함률 | 과대표집 배율 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in user_prior_order_count_bucket_analysis:
            lines.append(
                f"| `{row['user_prior_order_count_bucket']}` | "
                f"{row['unique_user_count']:,} | "
                f"{row['overall_sample_count']:,} | {row['mae_days']:.2f} | "
                f"{row['median_absolute_error_days']:.2f} | "
                f"{row['tail_sample_count']:,} | "
                f"{row['tail_membership_rate']:.2%} | "
                f"{row['tail_overrepresentation_ratio']:.2f}배 |"
            )
        lines.extend(
            [
                "",
                "- 이전 주문 수는 각 예측 시점보다 엄격하게 앞선 사용자의 고유 "
                "주문 수이며, 같은 주문의 여러 상품은 한 번만 계산합니다.",
                "- 같은 시각의 서로 다른 주문은 서로의 과거로 계산하지 않아 미래 "
                "정보와 임의 정렬 순서의 영향을 차단합니다.",
                "- 사용자 수와 표본 수를 함께 확인해 소수 사용자의 반복 행이 "
                "구간 결과를 지배하는지 구분해야 합니다.",
            ]
        )

    random_distribution = concentration_random_baseline["distribution"]
    observed_comparison = concentration_random_baseline["observed_comparison"]
    observed_percentile = observed_comparison["observed_hhi_empirical_percentile"]
    upper_tail_p_value = observed_comparison["monte_carlo_upper_tail_p_value"]
    if upper_tail_p_value <= PRODUCT_CONCENTRATION_SIGNIFICANCE_LEVEL:
        concentration_interpretation = (
            "실제 꼬리 HHI가 무작위 비교 분포의 상단 5%에 있어, 큰 오차가 "
            "특정 상품에 집중됐다는 가설을 후속 분석할 근거가 관찰됐습니다."
        )
    elif observed_percentile <= PRODUCT_CONCENTRATION_SIGNIFICANCE_LEVEL:
        concentration_interpretation = (
            "실제 꼬리 HHI가 무작위 비교 분포의 하단 5%에 있어, 큰 오차가 "
            "특정 상품에 집중됐다는 가설은 이번 Validation에서 지지되지 "
            "않았습니다."
        )
    else:
        concentration_interpretation = (
            "실제 꼬리 HHI가 무작위 비교 분포의 중앙 범위에 있어, 상품 "
            "집중도가 일반적인 무작위 변동과 다르다는 근거는 관찰되지 "
            "않았습니다."
        )
    lines.extend(
        [
            "",
            "### 동일 표본 수 무작위 기준선",
            "",
            f"- 각 반복에서 개인화 표본 **{concentration_random_baseline['sample_count_per_trial']:,}건**을 "
            f"비복원 추출하고, 고정 시드 `{concentration_random_baseline['random_seed']}`로 "
            f"**{random_distribution['trial_count']:,}회** 반복했습니다.",
            "",
            "| 실제 꼬리 HHI | 무작위 평균 | 무작위 중앙값 | 무작위 5백분위 | "
            "무작위 95백분위 | 실제 HHI 백분위 | 상단 꼬리 p-value |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {observed_comparison['observed_hhi']:.6f} | "
            f"{random_distribution['hhi_mean']:.6f} | "
            f"{random_distribution['hhi_median']:.6f} | "
            f"{random_distribution['hhi_p05']:.6f} | "
            f"{random_distribution['hhi_p95']:.6f} | "
            f"{observed_percentile:.2%} | {upper_tail_p_value:.4f} |",
            "",
            f"- 무작위 반복 중 **{observed_comparison['random_hhi_at_least_observed_rate']:.2%}**가 "
            "실제 꼬리 HHI 이상이었습니다.",
            f"- {concentration_interpretation}",
            "- 이 p-value는 모델이 맞을 확률이 아니라, 동일 표본 수 무작위 "
            "추출에서도 실제만큼 높은 집중도가 얼마나 자주 나타나는지를 "
            "뜻합니다.",
        ]
    )

    test_sources = test["hierarchical_baseline"]["by_prediction_source"]
    lines.extend(
        [
            "",
            "## Test fallback 사용 결과",
            "",
            "| 예측 근거 | 표본 | 비율 | MAE(일) | 중앙 절대오차(일) | ±7일 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for source in ("user_product_history", "product_history", "global_history"):
        metrics = test_sources.get(source)
        if metrics is None:
            continue
        lines.append(
            f"| `{source}` | {metrics['sample_count']:,} | "
            f"{metrics['sample_rate']:.2%} | {metrics['mae_days']:.2f} | "
            f"{metrics['median_absolute_error_days']:.2f} | "
            f"{metrics['within_7_days_rate']:.2%} |"
        )

    history_count_analysis = test["user_product_error_by_history_count"]
    tail_error_analysis = test["user_product_error_tail"]
    if tail_error_analysis:
        lines.extend(
            [
                "",
                "## Test 개인 이력 꼬리오차 기여도",
                "",
                "| 요청 상위 비율 | 실제 표본 | 실제 비율 | 오차 기여율 | "
                "늦은 예측 | 빠른 예측 |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for metrics in tail_error_analysis:
            lines.append(
                f"| {metrics['requested_tail_rate']:.0%} | "
                f"{metrics['tail_sample_count']:,} | "
                f"{metrics['actual_tail_sample_rate']:.2%} | "
                f"{metrics['absolute_error_share']:.2%} | "
                f"{metrics['late_prediction_rate']:.2%} | "
                f"{metrics['early_prediction_rate']:.2%} |"
            )

    if history_count_analysis:
        lines.extend(
            [
                "",
                "## Test 개인 이력 개수별 오차",
                "",
                "| 과거 간격 수 | 표본 | MAE(일) | 중앙 절대오차(일) | "
                "평균 방향 오차(일) |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for metrics in history_count_analysis:
            lines.append(
                f"| {metrics['history_interval_count']:,} | "
                f"{metrics['sample_count']:,} | {metrics['mae_days']:.2f} | "
                f"{metrics['median_absolute_error_days']:.2f} | "
                f"{metrics['mean_prediction_error_days']:+.2f} |"
            )

    variability_analysis = test["user_product_variability_analysis"]
    if variability_analysis is not None:
        spearman = variability_analysis["spearman_relative_mad_absolute_error"]
        spearman_text = "계산 불가" if spearman is None else f"{spearman:+.4f}"
        lines.extend(
            [
                "",
                "## Test 개인 이력 변동성과 오차의 관계",
                "",
                f"- 분석 표본: **{variability_analysis['sample_count']:,}건**",
                f"- 상대 MAD 중앙값: "
                f"**{variability_analysis['median_relative_mad']:.4f}**",
                f"- 상대 MAD와 절대오차의 Spearman 순위 상관: **{spearman_text}**",
                "- 이 값은 두 변수의 단조 관계를 나타내며 인과관계를 증명하지 "
                "않습니다.",
            ]
        )

    product_analysis = test["user_product_error_by_product"]
    if product_analysis is not None:
        lines.extend(
            [
                "",
                "## Test 개인 이력 오차 기여 상위 상품",
                "",
                f"- 분석 상품: **{product_analysis['product_count']:,}개**",
                f"- 상위 {TOP_ERROR_CONTRIBUTOR_COUNT}개 상품의 전체 오차 기여율: "
                f"**{product_analysis['top_contributor_error_share']:.2%}**",
                "",
                "| 상품 | 표본 | MAE(일) | 총 절대오차(일) | 오차 기여율 |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for metrics in product_analysis["top_contributors"]:
            lines.append(
                f"| `{metrics['product_id']}` | {metrics['sample_count']:,} | "
                f"{metrics['mae_days']:.2f} | "
                f"{metrics['total_absolute_error_days']:.2f} | "
                f"{metrics['absolute_error_share']:.2%} |"
            )

    monthly_analysis = test["user_product_error_by_anchor_month"]
    if monthly_analysis:
        lines.extend(
            [
                "",
                "## Test 개인 이력 월별 오차",
                "",
                "| 예측 기준 월 | 표본 | MAE(일) | 중앙 절대오차(일) | "
                "평균 방향 오차(일) | 오차 기여율 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for metrics in monthly_analysis:
            lines.append(
                f"| {metrics['anchor_month']} | {metrics['sample_count']:,} | "
                f"{metrics['mae_days']:.2f} | "
                f"{metrics['median_absolute_error_days']:.2f} | "
                f"{metrics['mean_prediction_error_days']:+.2f} | "
                f"{metrics['absolute_error_share']:.2%} |"
            )

    global_test = test["global_baseline"]["overall"]
    hierarchical_test = test["hierarchical_baseline"]["overall"]
    mae_change = hierarchical_test["mae_days"] - global_test["mae_days"]
    median_error_change = (
        hierarchical_test["median_absolute_error_days"]
        - global_test["median_absolute_error_days"]
    )
    seven_day_change = (
        hierarchical_test["within_7_days_rate"] - global_test["within_7_days_rate"]
    )
    best_shrinkage_mae = (
        min(shrinkage_candidates, key=lambda candidate: candidate["mae_days"])
        if shrinkage_candidates
        else None
    )
    best_shrinkage_within_seven_days = (
        max(
            shrinkage_candidates,
            key=lambda candidate: candidate["within_7_days_rate"],
        )
        if shrinkage_candidates
        else None
    )
    lines.extend(
        [
            "",
            "## 결과 해석",
            "",
            f"- 계층형 중앙값의 Test MAE는 전역 중앙값보다 "
            f"**{abs(mae_change):.2f}일 "
            f"{'높았습니다' if mae_change >= 0 else '낮았습니다'}.**",
            f"- 중앙 절대오차는 **{abs(median_error_change):.2f}일 "
            f"{'증가' if median_error_change >= 0 else '감소'}**했고, "
            f"±7일 적중률은 **{abs(seven_day_change):.2%}p "
            f"{'증가' if seven_day_change >= 0 else '감소'}**했습니다.",
            "- 개인 이력은 일반적인 표본을 더 가깝게 맞혔지만 일부 큰 오차가 "
            "평균 절대오차를 끌어올렸습니다.",
            "- 이 결과만으로 계층형 모델을 최종 채택하지 않습니다. 전역 기준보다 "
            "좋아진 지표와 나빠진 지표를 함께 다음 모델의 비교 기준으로 사용합니다.",
        ]
    )
    if best_shrinkage_mae and best_shrinkage_within_seven_days:
        lines.extend(
            [
                f"- Validation 후보 중 MAE는 `k={best_shrinkage_mae['shrinkage_strength']:.0f}`에서 "
                f"**{best_shrinkage_mae['mae_days']:.2f}일**로 가장 낮았지만, ±7일 "
                f"적중률은 `k={best_shrinkage_within_seven_days['shrinkage_strength']:.0f}`의 "
                f"**{best_shrinkage_within_seven_days['within_7_days_rate']:.2%}**가 가장 "
                "높아 단일 지표로 수축 강도를 확정하지 않습니다.",
                "- 수축 중앙값은 큰 오차를 완화하는 비교 기준으로 유지합니다. 다음 "
                "실험은 이력 수·변동성·상품·시기의 비선형 상호작용을 학습하는 "
                "LightGBM을 우선하고, 우측검열까지 사용하는 생존분석은 별도 비교 "
                "실험으로 분리합니다.",
            ]
        )
    lines.extend(
        [
            "",
            "## 현재 평가의 한계",
            "",
            "- UCI에 신뢰할 수 있는 상품군 정보가 없어 동일 SKU 재구매만 평가했습니다.",
            "- MAE 평가는 다음 구매가 관측된 표본만 사용하며 우측검열 표본을 직접 "
            "학습하지 않습니다.",
            "- Validation은 해당 구간 종료 전 정답이 확정된 표본만 사용하므로 긴 "
            "구매 간격이 상대적으로 덜 포함될 수 있습니다.",
            "- 70%·15%·15% 단일 시간 분할 결과이므로 계절별 안정성을 별도로 "
            "검증해야 합니다.",
            "- 수축 강도는 Validation의 MAE·중앙 절대오차·±7일 적중률 간 "
            "우선순위를 정한 뒤 확정해야 합니다.",
        ]
    )

    lines.extend(
        [
            "",
            "## 현재 시점 예측 예시",
            "",
            f"- 사용자: `{current['user_id']}`",
            f"- 상품: `{current['product_id']}`",
            f"- 마지막 구매: `{current['last_purchase_at']}`",
            f"- 예상 주기: **{current['predicted_duration_days']:.2f}일**",
            f"- 예상 구매 시점: `{current['predicted_purchase_at']}`",
            f"- 사용 근거: `{current['prediction_source']}` "
            f"({current['prediction_observation_count']:,}개 관측)",
            "",
            "## 불변조건",
            "",
        ]
    )
    lines.extend(
        f"- [{'x' if passed else ' '}] `{name}`"
        for name, passed in summary["invariants"].items()
    )
    lines.append("")
    return "\n".join(lines)


def build_product_concentration_trials_report(
    result: BaselineCycleResult,
) -> dict[str, object]:
    """무작위 상품 집중도 원자료를 재현 정보와 함께 저장 구조로 변환합니다."""
    summary = result.summary
    concentration = summary["validation_product_concentration_analysis"]
    random_baseline = summary["validation_product_concentration_random_baseline"]

    return {
        "dataset": summary["dataset"],
        "evaluation_split": "validation",
        "requested_tail_rate": concentration["requested_tail_rate"],
        "actual_tail_sample_rate": concentration["actual_tail_sample_rate"],
        "sample_count_per_trial": random_baseline["sample_count_per_trial"],
        "random_seed": random_baseline["random_seed"],
        "distribution": random_baseline["distribution"],
        "observed_comparison": random_baseline["observed_comparison"],
        "trials": result.product_concentration_trials.to_dict(orient="records"),
    }


def main() -> None:
    """실제 UCI 원본을 읽어 모델 E2E를 실행하고 JSON·Markdown을 저장합니다."""
    source = load_uci_online_retail_ii()
    classified = classify_uci_rows(source)
    events = build_uci_purchase_events(classified.rows)
    labels = build_same_product_repurchase_labels(
        events,
        observation_end_at=pd.Timestamp(events["ordered_at"].max()),
    )
    result = run_baseline_cycle(labels)
    summary = result.summary
    random_trials_report = build_product_concentration_trials_report(result)
    write_text_atomically(
        JSON_REPORT_PATH,
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    write_text_atomically(
        PRODUCT_CONCENTRATION_TRIALS_REPORT_PATH,
        json.dumps(
            random_trials_report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )
    write_text_atomically(MARKDOWN_REPORT_PATH, render_markdown(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
