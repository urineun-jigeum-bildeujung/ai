"""작은 구매 시퀀스로 시간 분할부터 현재 예측까지 한 사이클을 검증합니다."""

from __future__ import annotations

import pandas as pd

from scripts.preprocessing.labels import build_same_product_repurchase_labels
from scripts.run_uci_baseline_e2e import render_markdown, run_baseline_cycle


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

    summary = run_baseline_cycle(labels)

    assert summary["prediction_scope"] == "same_user_same_product"
    assert summary["validation_evaluation"]["sample_count"] > 0
    assert summary["test_evaluation"]["sample_count"] > 0
    assert [
        candidate["shrinkage_strength"]
        for candidate in summary["validation_shrinkage_candidates"]
    ] == [1.0, 2.0, 4.0, 8.0]
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

    markdown = render_markdown(summary)

    assert "Validation 수축 강도 후보 비교" in markdown
    assert "상위 5% MAE(일)" in markdown
    assert "Validation 기존 최악 5% 고정 코호트 재평가" in markdown
    assert "악화 표본" in markdown
    assert "## 결과 해석" in markdown
    assert "## 현재 평가의 한계" in markdown
    assert "Test fallback 사용 결과" in markdown
    assert "Test 개인 이력 꼬리오차 기여도" in markdown
    assert "Test 개인 이력 개수별 오차" in markdown
    assert "Test 개인 이력 변동성과 오차의 관계" in markdown
    assert "Test 개인 이력 오차 기여 상위 상품" in markdown
    assert "Test 개인 이력 월별 오차" in markdown
