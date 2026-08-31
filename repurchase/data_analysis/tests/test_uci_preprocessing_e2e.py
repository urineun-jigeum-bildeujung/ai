"""UCI 행 분류부터 재구매 라벨까지 단계 연결을 작은 표본으로 검증합니다."""

from __future__ import annotations

import pandas as pd

from scripts.validate_uci_preprocessing_e2e import run_uci_preprocessing_e2e
from tests.test_uci_preprocessing import make_uci_fixture


def make_e2e_fixture() -> pd.DataFrame:
    """기존 품질 표본에 실제 동일 상품 재구매 한 건을 추가합니다."""
    source = make_uci_fixture()
    repeated_purchase = source.loc[source["source_row_id"].eq("r1")].iloc[0].copy()
    repeated_purchase["source_row_id"] = "r8"
    repeated_purchase["order_id"] = "o8"
    repeated_purchase["ordered_at"] = pd.Timestamp("2026-01-11")
    source.loc[len(source)] = repeated_purchase
    return source


def test_uci_preprocessing_e2e_preserves_stage_contracts() -> None:
    """각 단계 건수가 이어지고 모든 단계·파이프라인 불변조건이 통과합니다."""
    summary = run_uci_preprocessing_e2e(make_e2e_fixture())

    assert summary["cleaning"]["source_rows"] == 8
    assert summary["cleaning"]["accepted_repurchase_rows"] == 5
    assert summary["purchase_events"]["purchase_event_count"] == 4
    assert summary["repurchase_labels"]["label_count"] == 4
    assert summary["repurchase_labels"]["observed_repurchase_count"] == 1
    assert all(summary["pipeline_invariants"].values())
