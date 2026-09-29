"""Rolling 모델 우위의 사용자·상품 집중도 진단을 작은 수치로 검증합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.rolling_diagnostics import (
    summarize_entity_contribution_concentration,
)
from scripts.modeling.rolling_validation import RollingValidationError


@pytest.fixture
def paired_rows() -> pd.DataFrame:
    """한 사용자의 양·음 오차 상쇄와 fold 외 구간의 분모를 동시에 검증합니다."""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u3", "u4", "u6", "u5"],
            "product_id": ["p1", "p1", "p1", "p2", "p3", "p1", "p4"],
            "history_interval_count": [8, 8, 8, 8, 8, 8, 1],
            "ipcw_outcome_known": [True, True, True, True, True, False, True],
            "ipcw_event_within_horizon": pd.array(
                [True, True, True, True, True, pd.NA, True], dtype="boolean"
            ),
            "ipcw_weight": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            "reference_predicted_event_probability": [
                0.5,
                1.0,
                0.6,
                0.8,
                1.0,
                0.4,
                0.5,
            ],
            "candidate_predicted_event_probability": [
                1.0,
                0.5,
                1.0,
                1.0,
                0.7,
                0.5,
                1.0,
            ],
        }
    )


def test_concentration_uses_net_entity_contribution_and_fold_denominator(
    paired_rows: pd.DataFrame,
) -> None:
    """같은 사용자 안의 이득·손실을 상쇄하고 다른 구간 가중치도 분모에 둡니다."""
    result = summarize_entity_contribution_concentration(
        paired_rows,
        count_column="history_interval_count",
        count_bucket="8-15",
        entity_column="user_id",
    )
    assert result["sample_count"] == 6
    assert result["known_sample_count"] == 5
    assert result["entity_count"] == 5
    # u1의 +0.25와 -0.25는 서로 상쇄되어 양의 기여 사용자로 세지 않습니다.
    assert result["positive_entity_count"] == 2
    assert result["positive_contribution_total"] == pytest.approx(0.20 / 6)
    assert result["negative_contribution_total"] == pytest.approx(-0.09 / 6)
    assert result["net_contribution_total"] == pytest.approx(0.11 / 6)
    assert result["positive_top1_share"] == pytest.approx(0.8)
    assert result["positive_top5_share"] == pytest.approx(1.0)
    assert result["positive_hhi"] == pytest.approx(0.68)
    assert result["positive_effective_entity_count"] == pytest.approx(1 / 0.68)
    assert result["row_top1_share"] == pytest.approx(2 / 6)
    assert result["positive_top1_entity_row_share"] == pytest.approx(1 / 6)


def test_product_concentration_counts_unknown_rows_without_scoring_them(
    paired_rows: pd.DataFrame,
) -> None:
    """같은 상품의 검열 행은 구성 비율에 남지만 Brier 기여는 하지 않습니다."""
    result = summarize_entity_contribution_concentration(
        paired_rows,
        count_column="history_interval_count",
        count_bucket="8-15",
        entity_column="product_id",
    )
    assert result["entity_count"] == 3
    assert result["row_top1_share"] == pytest.approx(4 / 6)
    assert result["positive_top1_share"] == pytest.approx(0.8)
    assert result["positive_top1_entity_row_share"] == pytest.approx(4 / 6)


def test_concentration_keeps_null_when_no_entity_favors_lightgbm(
    paired_rows: pd.DataFrame,
) -> None:
    """양의 순기여가 전혀 없으면 집중도를 0으로 꾸며내지 않습니다."""
    negative = paired_rows.iloc[[4]].copy()
    result = summarize_entity_contribution_concentration(
        negative,
        count_column="history_interval_count",
        count_bucket="8-15",
        entity_column="user_id",
    )
    assert result["positive_entity_count"] == 0
    assert result["positive_top1_share"] is None
    assert result["positive_hhi"] is None
    assert result["positive_effective_entity_count"] is None


def test_concentration_rejects_missing_entity_or_bucket(
    paired_rows: pd.DataFrame,
) -> None:
    """행을 조용히 버리는 결측 식별자와 빈 분석 구간을 거절합니다."""
    broken = paired_rows.copy()
    broken.loc[0, "user_id"] = None
    with pytest.raises(RollingValidationError, match="식별자"):
        summarize_entity_contribution_concentration(
            broken,
            count_column="history_interval_count",
            count_bucket="8-15",
            entity_column="user_id",
        )
    with pytest.raises(RollingValidationError, match="구간이 비었습니다"):
        summarize_entity_contribution_concentration(
            paired_rows,
            count_column="history_interval_count",
            count_bucket="128+",
            entity_column="user_id",
        )
