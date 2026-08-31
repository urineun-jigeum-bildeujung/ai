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
    assert (
        summary["test_evaluation"]["hierarchical_baseline"]["overall"]["mae_days"] == 0
    )
    assert summary["current_prediction_example"]["prediction_source"] == (
        "user_product_history"
    )
    assert all(summary["invariants"].values())

    markdown = render_markdown(summary)
    assert "## 결과 해석" in markdown
    assert "## 현재 평가의 한계" in markdown
    assert "Test fallback 사용 결과" in markdown
