"""작은 구매 시퀀스로 시간 분할부터 현재 예측까지 한 사이클을 검증합니다."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.preprocessing.labels import build_same_product_repurchase_labels
from scripts.run_uci_baseline_e2e import (
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
