"""작은 구매 시퀀스로 시간 분할부터 현재 예측까지 한 사이클을 검증합니다."""

from __future__ import annotations

import json
from itertools import pairwise

import pandas as pd
import pytest

from scripts.preprocessing.labels import build_same_product_repurchase_labels
from scripts.run_uci_baseline_e2e import (
    COMMON_FOLLOWUP_HORIZON_CANDIDATES,
    PRIMARY_IPCW_HORIZON_DAYS,
    SHRINKAGE_STRENGTH_CANDIDATES,
    _format_optional_days,
    build_product_concentration_trials_report,
    render_markdown,
    run_baseline_cycle,
)


def make_purchase_events() -> pd.DataFrame:
    """두 사용자·상품의 반복 구매를 전체 관측 기간에 걸쳐 생성합니다."""
    rows: list[dict[str, object]] = []
    for user_id, product_id, start_at, interval_days, event_count in (
        ("u1", "p1", "2026-01-01", 5, 24),
        ("u2", "p2", "2026-01-03", 8, 16),
    ):
        for index in range(event_count):
            rows.append(
                {
                    "user_id": user_id,
                    "order_id": f"{user_id}-o{index:02d}",
                    "product_id": product_id,
                    "ordered_at": pd.Timestamp(start_at)
                    + pd.Timedelta(days=interval_days * index),
                }
            )
    return pd.DataFrame(rows)


def test_baseline_cycle_connects_split_training_evaluation_and_prediction() -> None:
    """작은 고정 표본에서 두 평가 구간과 현재 시점 예측이 모두 생성됩니다."""
    events = make_purchase_events()
    labels = build_same_product_repurchase_labels(
        events,
        observation_end_at=pd.Timestamp(events["ordered_at"].max()),
    )

    result = run_baseline_cycle(labels)
    summary = result.summary

    assert summary["prediction_scope"] == "same_user_same_product"
    assert summary["validation_evaluation"]["sample_count"] > 0
    assert summary["test_evaluation"]["sample_count"] > 0
    assert [
        candidate["shrinkage_strength"]
        for candidate in summary["validation_shrinkage_candidates"]
    ] == [1.0, 2.0, 4.0, 8.0]
    prior_support_analysis = summary["validation_prior_support_analysis"]
    assert prior_support_analysis
    assert (
        sum(row["overall_sample_count"] for row in prior_support_analysis)
        == summary["validation_shrinkage_candidates"][0]["personal_sample_count"]
    )
    assert (
        sum(row["fixed_tail_sample_count"] for row in prior_support_analysis)
        == summary["validation_shrinkage_candidates"][0]["fixed_cohort_sample_count"]
    )
    assert sum(
        row["overall_sample_rate"] for row in prior_support_analysis
    ) == pytest.approx(1.0)
    assert sum(
        row["fixed_tail_sample_rate"] for row in prior_support_analysis
    ) == pytest.approx(1.0)
    prior_support_bucket_analysis = summary["validation_prior_support_bucket_analysis"]
    assert prior_support_bucket_analysis
    assert all(
        isinstance(row["prior_support_bucket"], str)
        for row in prior_support_bucket_analysis
    )
    assert sum(
        row["overall_sample_count"] for row in prior_support_bucket_analysis
    ) == sum(row["overall_sample_count"] for row in prior_support_analysis)
    assert sum(
        row["fixed_tail_sample_count"] for row in prior_support_bucket_analysis
    ) == sum(row["fixed_tail_sample_count"] for row in prior_support_analysis)
    assert sum(
        row["overall_sample_rate"] for row in prior_support_bucket_analysis
    ) == pytest.approx(1.0)
    assert sum(
        row["fixed_tail_sample_rate"] for row in prior_support_bucket_analysis
    ) == pytest.approx(1.0)
    product_concentration = summary["validation_product_concentration_analysis"]
    overall_concentration = product_concentration["overall_personalized"]
    tail_concentration = product_concentration["largest_error_tail"]
    reference_candidate = summary["validation_shrinkage_candidates"][0]
    assert product_concentration["requested_tail_rate"] == pytest.approx(0.05)
    assert (
        overall_concentration["total_sample_count"]
        == reference_candidate["personal_sample_count"]
    )
    assert (
        tail_concentration["total_sample_count"]
        == reference_candidate["fixed_cohort_sample_count"]
    )
    assert product_concentration["actual_tail_sample_rate"] == pytest.approx(
        tail_concentration["total_sample_count"]
        / overall_concentration["total_sample_count"]
    )
    product_sample_count_analysis = summary["validation_product_sample_count_analysis"]
    product_sample_count_bucket_analysis = summary[
        "validation_product_sample_count_bucket_analysis"
    ]
    assert product_sample_count_analysis
    assert product_sample_count_bucket_analysis
    assert (
        sum(row["overall_sample_count"] for row in product_sample_count_analysis)
        == overall_concentration["total_sample_count"]
    )
    assert (
        sum(row["tail_sample_count"] for row in product_sample_count_analysis)
        == tail_concentration["total_sample_count"]
    )
    assert (
        sum(row["overall_sample_count"] for row in product_sample_count_bucket_analysis)
        == overall_concentration["total_sample_count"]
    )
    assert (
        sum(row["tail_sample_count"] for row in product_sample_count_bucket_analysis)
        == tail_concentration["total_sample_count"]
    )
    assert all(
        isinstance(row["product_sample_count_bucket"], str)
        for row in product_sample_count_bucket_analysis
    )
    user_prior_order_count_analysis = summary[
        "validation_user_prior_order_count_analysis"
    ]
    user_prior_order_count_bucket_analysis = summary[
        "validation_user_prior_order_count_bucket_analysis"
    ]
    assert user_prior_order_count_analysis
    assert user_prior_order_count_bucket_analysis
    assert (
        sum(row["overall_sample_count"] for row in user_prior_order_count_analysis)
        == overall_concentration["total_sample_count"]
    )
    assert (
        sum(row["tail_sample_count"] for row in user_prior_order_count_analysis)
        == tail_concentration["total_sample_count"]
    )
    assert (
        sum(
            row["overall_sample_count"]
            for row in user_prior_order_count_bucket_analysis
        )
        == overall_concentration["total_sample_count"]
    )
    assert (
        sum(row["tail_sample_count"] for row in user_prior_order_count_bucket_analysis)
        == tail_concentration["total_sample_count"]
    )
    assert all(
        isinstance(row["user_prior_order_count_bucket"], str)
        for row in user_prior_order_count_bucket_analysis
    )
    random_baseline = summary["validation_product_concentration_random_baseline"]
    assert (
        random_baseline["sample_count_per_trial"]
        == tail_concentration["total_sample_count"]
    )
    assert random_baseline["distribution"]["trial_count"] == 1_000
    assert "trials" not in random_baseline
    assert len(result.product_concentration_trials) == 1_000
    assert random_baseline["observed_comparison"]["observed_hhi"] == pytest.approx(
        tail_concentration["hhi"]
    )
    assert (
        0.0
        <= random_baseline["observed_comparison"]["monte_carlo_upper_tail_p_value"]
        <= 1.0
    )
    maturity_analysis = summary["validation_label_maturity_analysis"]
    assert maturity_analysis
    assert (
        sum(row["validation_sample_count"] for row in maturity_analysis)
        == (summary["split_summary"]["validation"]["sample_count"])
    )
    assert (
        sum(row["matured_sample_count"] for row in maturity_analysis)
        == (summary["validation_evaluation"]["sample_count"])
    )
    assert all(
        row["validation_sample_count"]
        == row["matured_sample_count"] + row["unmatured_sample_count"]
        for row in maturity_analysis
    )
    assert all(0.0 <= row["maturity_rate"] <= 1.0 for row in maturity_analysis)
    followup_distribution = summary["validation_followup_distribution"]
    assert [row["quantile"] for row in followup_distribution] == [
        0.0,
        0.1,
        0.25,
        0.5,
        0.75,
        0.9,
        1.0,
    ]
    assert all(
        left["available_followup_days"] <= right["available_followup_days"]
        for left, right in pairwise(followup_distribution)
    )
    common_followup_candidates = summary["validation_common_followup_candidates"]
    assert [row["horizon_days"] for row in common_followup_candidates] == list(
        COMMON_FOLLOWUP_HORIZON_CANDIDATES
    )
    validation_sample_count = summary["split_summary"]["validation"]["sample_count"]
    assert all(
        row["validation_sample_count"] == validation_sample_count
        for row in common_followup_candidates
    )
    assert all(
        row["eligible_sample_count"] + row["ineligible_sample_count"]
        == validation_sample_count
        for row in common_followup_candidates
    )
    assert all(
        row["event_within_horizon_count"] + row["no_event_within_horizon_count"]
        == row["eligible_sample_count"]
        for row in common_followup_candidates
    )
    monthly_composition = summary["validation_common_followup_monthly_composition"]
    assert sorted({row["horizon_days"] for row in monthly_composition}) == list(
        COMMON_FOLLOWUP_HORIZON_CANDIDATES
    )
    assert all(
        row["validation_sample_count"]
        == row["eligible_sample_count"] + row["ineligible_sample_count"]
        for row in monthly_composition
    )
    ipcw_weight_stability = summary["validation_ipcw_weight_stability"][0]
    assert ipcw_weight_stability["horizon_days"] == PRIMARY_IPCW_HORIZON_DAYS
    assert (
        ipcw_weight_stability["outcome_known_count"]
        + ipcw_weight_stability["outcome_unknown_count"]
        == validation_sample_count
    )
    assert (
        ipcw_weight_stability["event_within_horizon_count"]
        + ipcw_weight_stability["no_event_within_horizon_count"]
        == ipcw_weight_stability["outcome_known_count"]
    )
    assert (
        ipcw_weight_stability["ipcw_weight_max"]
        >= ipcw_weight_stability["ipcw_weight_median"]
    )
    assert ipcw_weight_stability["ipcw_weighted_event_mass"] + ipcw_weight_stability[
        "ipcw_weighted_no_event_mass"
    ] == pytest.approx(ipcw_weight_stability["ipcw_weight_sum"])
    assert (
        ipcw_weight_stability["ipcw_effective_sample_size"]
        <= ipcw_weight_stability["outcome_known_count"]
    )
    ipcw_binary_evaluation = summary["validation_ipcw_binary_evaluation"]
    assert ipcw_binary_evaluation["horizon_days"] == PRIMARY_IPCW_HORIZON_DAYS
    assert ipcw_binary_evaluation["validation_sample_count"] == validation_sample_count
    assert (
        ipcw_binary_evaluation["outcome_known_count"]
        == ipcw_weight_stability["outcome_known_count"]
    )
    assert ipcw_binary_evaluation[
        "ipcw_weighted_false_positive_mass"
    ] + ipcw_binary_evaluation["ipcw_weighted_false_negative_mass"] == pytest.approx(
        ipcw_binary_evaluation["ipcw_weighted_error_mass"]
    )
    assert ipcw_binary_evaluation[
        "ipcw_weighted_true_positive_mass"
    ] + ipcw_binary_evaluation[
        "ipcw_weighted_true_negative_mass"
    ] + ipcw_binary_evaluation[
        "ipcw_weighted_false_positive_mass"
    ] + ipcw_binary_evaluation["ipcw_weighted_false_negative_mass"] == pytest.approx(
        ipcw_binary_evaluation["ipcw_weight_sum"]
    )
    ipcw_concordance_evaluation = summary["validation_ipcw_concordance_evaluation"]
    assert (
        ipcw_concordance_evaluation["validation_sample_count"]
        == validation_sample_count
    )
    assert (
        ipcw_concordance_evaluation["concordant_pair_count"]
        + ipcw_concordance_evaluation["tied_pair_count"]
        + ipcw_concordance_evaluation["discordant_pair_count"]
        == ipcw_concordance_evaluation["comparable_pair_count"]
    )
    assert 0 <= ipcw_concordance_evaluation["ipcw_concordance_index"] <= 1
    ipcw_candidate_comparison = summary["validation_ipcw_candidate_comparison"]
    assert len(ipcw_candidate_comparison) == 1 + len(SHRINKAGE_STRENGTH_CANDIDATES)
    assert all(
        candidate["validation_sample_count"] == validation_sample_count
        for candidate in ipcw_candidate_comparison
    )
    assert (
        ipcw_candidate_comparison[0][
            "ipcw_weighted_balanced_accuracy_difference_vs_reference"
        ]
        is None
    )
    assert ipcw_candidate_comparison[0][
        "ipcw_concordance_index_difference_vs_reference"
    ] == pytest.approx(0.0)
    assert "product_concentration_analysis" not in summary["test_evaluation"]
    assert (
        summary["test_evaluation"]["hierarchical_baseline"]["overall"]["mae_days"] == 0
    )
    assert summary["current_prediction_example"]["prediction_source"] == (
        "user_product_history"
    )
    history_count_analysis = summary["test_evaluation"][
        "user_product_error_by_history_count"
    ]
    tail_error_analysis = summary["test_evaluation"]["user_product_error_tail"]
    assert [row["requested_tail_rate"] for row in tail_error_analysis] == [0.01, 0.05]
    assert all(row["tail_sample_count"] > 0 for row in tail_error_analysis)
    assert all(row["by_history_count"] for row in tail_error_analysis)
    assert all(
        history_row["overall_sample_count"] > 0
        for row in tail_error_analysis
        for history_row in row["by_history_count"]
    )
    assert history_count_analysis
    assert all(row["sample_count"] > 0 for row in history_count_analysis)
    variability_analysis = summary["test_evaluation"][
        "user_product_variability_analysis"
    ]
    assert variability_analysis is not None
    assert variability_analysis["sample_count"] > 0
    product_analysis = summary["test_evaluation"]["user_product_error_by_product"]
    assert product_analysis is not None
    assert product_analysis["product_count"] > 0
    assert product_analysis["top_contributors"]
    assert summary["test_evaluation"]["user_product_error_by_anchor_month"]
    assert all(summary["invariants"].values())

    random_trials_report = build_product_concentration_trials_report(result)
    assert random_trials_report["dataset"] == "uci_online_retail_ii"
    assert random_trials_report["evaluation_split"] == "validation"
    assert (
        random_trials_report["sample_count_per_trial"]
        == tail_concentration["total_sample_count"]
    )
    assert len(random_trials_report["trials"]) == 1_000

    # 실제 보고서 저장과 동일한 조건으로 pandas 전용 타입과 NaN 잔존을 검사합니다.
    serialized_summary = json.dumps(
        summary,
        ensure_ascii=False,
        allow_nan=False,
    )
    assert json.loads(serialized_summary) == summary
    serialized_trials_report = json.dumps(
        random_trials_report,
        ensure_ascii=False,
        allow_nan=False,
    )
    assert json.loads(serialized_trials_report) == random_trials_report

    markdown = render_markdown(summary)

    assert "Validation 관찰 가능 기간 분포" in markdown
    assert "미관측·검열 표본을 제외하지 않은 전체 Validation" in markdown
    assert "Validation 공통 관찰 기간 후보 비교" in markdown
    assert "기간 내 재구매율의 분모는 전체 Validation이 아니라" in markdown
    assert "재구매 간격 중앙값(일)" in markdown
    assert "Validation 공통 관찰 기간 후보별 구매 월 구성" in markdown
    assert "적용 전 비중" in markdown
    assert "적용 후 해당 월이 과대표현" in markdown
    assert "Validation 30일 IPCW 원시 가중치 안정성" in markdown
    assert "ESS/결과 확인" in markdown
    assert "아직 상한을 적용하지 않은 원시 가중치" in markdown
    assert "IPCW 보정 재구매율" in markdown
    assert "계층형 중앙값 모델의 30일 IPCW 이진 평가" in markdown
    assert "Balanced Accuracy" in markdown
    assert "항상 미재구매 정확도" in markdown
    assert "정식 Brier Score가 아니라" in markdown
    assert "계층형 중앙값 모델의 30일 IPCW C-index" in markdown
    assert "비교 가능 쌍" in markdown
    assert "0.5는 무작위 순위 수준" in markdown
    assert "기존 계층형 모델과 수축 후보의 동일 조건 비교" in markdown
    assert "수축 k=8" in markdown
    assert "동일한 Validation 표본·30일 시점·검열 가중치" in markdown
    assert "Validation 월별 라벨 성숙도와 조건부 오차" in markdown
    assert "관찰 가능 기간 중앙값(일)" in markdown
    assert "성숙 표본에서만 계산한 조건부 결과" in markdown
    assert "낮은 MAE를 모델 개선으로 단독 해석하지 않습니다" in markdown
    assert "Validation 수축 강도 후보 비교" in markdown
    assert "상위 5% MAE(일)" in markdown
    assert "Validation 기존 최악 5% 고정 코호트 재평가" in markdown
    assert "Validation prior 지지 표본 로그 구간 분석" in markdown
    assert "Validation 개인화 예측 상품 집중도" in markdown
    assert "실제 상품 수" in markdown
    assert "Top-1 점유율" in markdown
    assert "Top-5 점유율" in markdown
    assert "유효 상품 수" in markdown
    assert "Validation 상품 표본 수 구간별 오차" in markdown
    assert "꼬리 포함률" in markdown
    assert "모델 입력 피처가 아닌 사후 진단 기준" in markdown
    assert "Validation 사용자 과거 주문 수 구간별 오차" in markdown
    assert "같은 주문의 여러 상품은 한 번만 계산" in markdown
    assert "소수 사용자의 반복 행" in markdown
    expected_selection_text = (
        f"실제 선택: **{tail_concentration['total_sample_count']:,} / "
        f"{overall_concentration['total_sample_count']:,}건 "
        f"({product_concentration['actual_tail_sample_rate']:.3%})**"
    )
    assert expected_selection_text in markdown
    assert "동일 표본 수 무작위 기준선" in markdown
    assert "실제 HHI 백분위" in markdown
    assert "상단 꼬리 p-value" in markdown
    assert "p-value는 모델이 맞을 확률이 아니라" in markdown
    has_single_product_prior = any(
        row["prior_source"] == "product_history" and row["prior_observation_count"] == 1
        for row in prior_support_analysis
    )
    assert ("상품 prior 관측 1건 가설 검증" in markdown) is (has_single_product_prior)
    assert "배율과 전체·꼬리 표본 수를 함께 해석" in markdown
    assert "인과관계를 증명하지 않습니다" in markdown
    assert "악화 표본" in markdown
    assert "## 결과 해석" in markdown
    assert "단일 지표로 수축 강도를 확정하지 않습니다" in markdown
    assert "LightGBM을 우선" in markdown
    assert "## 현재 평가의 한계" in markdown
    assert "Test fallback 사용 결과" in markdown
    assert "Test 개인 이력 꼬리오차 기여도" in markdown
    assert "Test 개인 이력 개수별 오차" in markdown
    assert "Test 개인 이력 변동성과 오차의 관계" in markdown
    assert "Test 개인 이력 오차 기여 상위 상품" in markdown
    assert "Test 개인 이력 월별 오차" in markdown


def test_format_optional_days_distinguishes_missing_from_zero() -> None:
    """계산 불가 상태를 실제 오차 0일과 다른 문구로 표시합니다."""
    assert _format_optional_days(None) == "계산 불가"
    assert _format_optional_days(0.0) == "0.00"
