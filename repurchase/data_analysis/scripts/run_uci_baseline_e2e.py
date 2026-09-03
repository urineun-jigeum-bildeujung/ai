"""UCI 원본부터 시간 분할·베이스라인 학습·평가·예측까지 실행합니다.

전처리 E2E 산출물을 실제 모델 입력으로 연결하고, Train 시점과 Validation
시점에서 순차적으로 중앙값 베이스라인을 학습합니다. 결과에는 전역 기준과
계층형 기준의 성능, fallback 사용 비율, 현재 시점 예측 예시를 함께 기록합니다.
"""

from __future__ import annotations

import json
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
    select_largest_error_rows,
    summarize_fixed_cohort_prior_support,
    summarize_largest_error_tail,
    summarize_largest_error_tail_by_history_count,
    summarize_user_product_error_variability,
    summarize_user_product_errors_by_anchor_month,
    summarize_user_product_errors_by_history_count,
    summarize_user_product_errors_by_product,
)
from .modeling.evaluation import evaluate_predictions
from .modeling.model_selection import evaluate_shrinkage_candidates
from .modeling.samples import (
    assign_temporal_splits,
    build_historical_interval_features,
    make_temporal_split,
)
from .paths import REPORT_DIR
from .preprocessing.events import build_uci_purchase_events
from .preprocessing.labels import build_same_product_repurchase_labels
from .preprocessing.uci import classify_uci_rows
from .reporting import write_text_atomically

JSON_REPORT_PATH = REPORT_DIR / "uci_baseline_e2e_evaluation.json"
MARKDOWN_REPORT_PATH = REPORT_DIR / "uci_baseline_e2e_evaluation.md"
TOP_ERROR_CONTRIBUTOR_COUNT: Final[int] = 10
# 1% 결과가 극소수 표본에만 좌우되는지 확인하기 위해 5% 결과도 함께 비교합니다.
TAIL_ERROR_RATES: Final[tuple[float, ...]] = (0.01, 0.05)
# 개인 이력 1~2건 구간의 과신을 완화하는 약한~강한 수축 후보를 비교합니다.
SHRINKAGE_STRENGTH_CANDIDATES: Final[tuple[float, ...]] = (1.0, 2.0, 4.0, 8.0)
# 모델 후보 비교와 prior 분석이 동일한 기존 최악 표본을 사용하도록 고정합니다.
MODEL_SELECTION_TAIL_RATE: Final[float] = 0.05


def _isoformat(timestamp: pd.Timestamp) -> str:
    """보고서의 모든 시각을 동일한 ISO 8601 문자열로 변환합니다."""
    return pd.Timestamp(timestamp).isoformat()


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
    validation_rows: pd.DataFrame,
    model: HierarchicalMedianModel,
) -> list[dict[str, object]]:
    """Validation 고정 꼬리에서 prior 관측 수의 과대표집 여부를 분석합니다."""
    reference_predictions = predict_hierarchical_median_baseline(
        model,
        validation_rows,
    )
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
    return summary.to_dict(orient="records")


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


def run_baseline_cycle(labels: pd.DataFrame) -> dict[str, Any]:
    """재구매 라벨부터 시간 분할·순차 학습·평가·현재 예측을 실행합니다."""
    samples = build_historical_interval_features(labels)
    split = make_temporal_split(samples)
    samples = assign_temporal_splits(samples, split)

    validation_rows = samples.loc[
        samples["split"].eq("validation") & samples["outcome_available_by_split_end"]
    ]
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
    validation_prior_support_analysis = _analyze_validation_prior_support(
        validation_rows,
        train_model,
    )
    observation_end_at = pd.Timestamp(samples["anchor_at"].max())

    invariants = {
        "temporal_boundaries_ordered": bool(
            split.start_at < split.train_end_at < split.validation_end_at < split.end_at
        ),
        "validation_anchors_after_train": bool(
            validation_rows["anchor_at"].gt(split.train_end_at).all()
        ),
        "test_anchors_after_validation": bool(
            test_rows["anchor_at"].gt(split.validation_end_at).all()
        ),
        "validation_outcomes_matured": bool(
            validation_rows["next_same_product_at"].le(split.validation_end_at).all()
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

    return {
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
        "validation_prior_support_analysis": validation_prior_support_analysis,
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


def render_markdown(summary: dict[str, Any]) -> str:
    """시간 분할·모델 성능·현재 예측을 사람이 검토하기 쉬운 표로 변환합니다."""
    split = summary["split"]
    validation = summary["validation_evaluation"]
    test = summary["test_evaluation"]
    shrinkage_candidates = summary["validation_shrinkage_candidates"]
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


def main() -> None:
    """실제 UCI 원본을 읽어 모델 E2E를 실행하고 JSON·Markdown을 저장합니다."""
    source = load_uci_online_retail_ii()
    classified = classify_uci_rows(source)
    events = build_uci_purchase_events(classified.rows)
    labels = build_same_product_repurchase_labels(
        events,
        observation_end_at=pd.Timestamp(events["ordered_at"].max()),
    )
    summary = run_baseline_cycle(labels)
    write_text_atomically(
        JSON_REPORT_PATH,
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    write_text_atomically(MARKDOWN_REPORT_PATH, render_markdown(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
