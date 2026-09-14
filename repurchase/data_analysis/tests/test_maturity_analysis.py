"""관찰 가능 기간 계산이 원본을 보존하고 잘못된 입력을 거절하는지 검증합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.maturity_analysis import (
    add_available_followup_days,
    merge_validation_maturity_and_quality,
    summarize_matured_prediction_quality_by_anchor_month,
    summarize_validation_label_maturity_by_anchor_month,
)


def test_add_available_followup_days_calculates_days_without_mutation() -> None:
    """구매 시점부터 평가 마감일까지의 일수를 계산하되 원본은 변경하지 않습니다."""
    rows = pd.DataFrame(
        {
            "anchor_at": pd.to_datetime(["2026-08-01", "2026-08-15"]),
            "split_end_at": pd.to_datetime(["2026-08-20", "2026-08-20"]),
        }
    )

    result = add_available_followup_days(rows)

    assert result["available_followup_days"].tolist() == [19.0, 5.0]
    assert "available_followup_days" not in rows.columns


def test_add_available_followup_days_rejects_missing_required_column() -> None:
    """필수 날짜 열이 없으면 불완전한 계산 대신 명확한 오류를 냅니다."""
    rows = pd.DataFrame({"anchor_at": pd.to_datetime(["2026-08-01"])})

    with pytest.raises(ValueError, match="split_end_at"):
        add_available_followup_days(rows)


def test_add_available_followup_days_rejects_empty_rows() -> None:
    """분석 표본이 없으면 의미 없는 빈 결과를 만들지 않습니다."""
    rows = pd.DataFrame(columns=["anchor_at", "split_end_at"])

    with pytest.raises(ValueError, match="표본이 없습니다"):
        add_available_followup_days(rows)


def test_add_available_followup_days_rejects_reversed_time() -> None:
    """평가 마감이 구매 시점보다 빠른 잘못된 시간 관계를 거절합니다."""
    rows = pd.DataFrame(
        {
            "anchor_at": pd.to_datetime(["2026-08-21"]),
            "split_end_at": pd.to_datetime(["2026-08-20"]),
        }
    )

    with pytest.raises(ValueError, match="이를 수 없습니다"):
        add_available_followup_days(rows)


def test_summarize_validation_label_maturity_by_anchor_month() -> None:
    """월별 전체·성숙·미성숙 표본 수와 성숙률을 같은 모집단에서 계산합니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation"] * 5,
            "anchor_at": pd.to_datetime(
                [
                    "2026-05-01",
                    "2026-05-10",
                    "2026-05-20",
                    "2026-06-01",
                    "2026-06-15",
                ]
            ),
            "split_end_at": pd.to_datetime(["2026-08-20"] * 5),
            "outcome_available_by_split_end": [True, True, False, True, False],
        }
    )

    result = summarize_validation_label_maturity_by_anchor_month(rows)

    assert result["anchor_month"].tolist() == ["2026-05", "2026-06"]
    assert result["validation_sample_count"].tolist() == [3, 2]
    assert result["matured_sample_count"].tolist() == [2, 1]
    assert result["unmatured_sample_count"].tolist() == [1, 1]
    assert result["maturity_rate"].tolist() == pytest.approx([2 / 3, 1 / 2])


def test_maturity_summary_rejects_non_validation_rows() -> None:
    """다른 시간 구간이 섞이면 Validation 모집단 분석을 중단합니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation", "test"],
            "anchor_at": pd.to_datetime(["2026-05-01", "2026-08-21"]),
            "split_end_at": pd.to_datetime(["2026-08-20", "2026-12-01"]),
            "outcome_available_by_split_end": [True, True],
        }
    )

    with pytest.raises(ValueError, match="Validation 표본만"):
        summarize_validation_label_maturity_by_anchor_month(rows)


def test_summarize_matured_prediction_quality_by_anchor_month() -> None:
    """정답이 확인된 표본의 실제 간격과 예측 오차를 구매 월별로 계산합니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation"] * 3,
            "anchor_at": pd.to_datetime(["2026-05-01", "2026-05-10", "2026-06-01"]),
            "outcome_available_by_split_end": [True, True, True],
            "target_duration_days": [10.0, 30.0, 20.0],
            "predicted_duration_days": [12.0, 20.0, 26.0],
        }
    )

    result = summarize_matured_prediction_quality_by_anchor_month(rows)

    assert result["anchor_month"].tolist() == ["2026-05", "2026-06"]
    assert result["evaluable_sample_count"].tolist() == [2, 1]
    assert result["mean_target_duration_days"].tolist() == [20.0, 20.0]
    assert result["median_target_duration_days"].tolist() == [20.0, 20.0]
    assert result["mae_days"].tolist() == [6.0, 6.0]
    assert result["median_absolute_error_days"].tolist() == [6.0, 6.0]


def test_prediction_quality_summary_rejects_unmatured_rows() -> None:
    """정답이 확인되지 않은 표본이 섞이면 오차 계산 전에 실행을 중단합니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation"],
            "anchor_at": pd.to_datetime(["2026-05-01"]),
            "outcome_available_by_split_end": [False],
            "target_duration_days": [10.0],
            "predicted_duration_days": [12.0],
        }
    )

    with pytest.raises(ValueError, match="정답이 확인된 표본"):
        summarize_matured_prediction_quality_by_anchor_month(rows)


def test_merge_validation_maturity_and_quality_preserves_unmatured_month() -> None:
    """성숙 표본이 없는 월도 왼쪽 성숙도 결과에서 삭제하지 않습니다."""
    maturity_summary = pd.DataFrame(
        {
            "anchor_month": ["2026-05", "2026-06", "2026-08"],
            "matured_sample_count": [2, 1, 0],
            "maturity_rate": [2 / 3, 1 / 2, 0.0],
        }
    )
    quality_summary = pd.DataFrame(
        {
            "anchor_month": ["2026-05", "2026-06"],
            "evaluable_sample_count": [2, 1],
            "mae_days": [6.0, 6.0],
        }
    )

    result = merge_validation_maturity_and_quality(
        maturity_summary,
        quality_summary,
    )

    assert result["anchor_month"].tolist() == ["2026-05", "2026-06", "2026-08"]
    assert result["evaluable_sample_count"].tolist() == [2, 1, 0]
    assert pd.isna(result.loc[2, "mae_days"])


def test_merge_rejects_mismatched_matured_and_evaluable_counts() -> None:
    """같은 월의 성숙 표본 수와 평가 가능 표본 수가 다르면 병합을 거절합니다."""
    maturity_summary = pd.DataFrame(
        {
            "anchor_month": ["2026-05"],
            "matured_sample_count": [2],
        }
    )
    quality_summary = pd.DataFrame(
        {
            "anchor_month": ["2026-05"],
            "evaluable_sample_count": [1],
        }
    )

    with pytest.raises(ValueError, match="일치하지 않습니다"):
        merge_validation_maturity_and_quality(maturity_summary, quality_summary)


def test_merge_rejects_quality_month_missing_from_maturity_summary() -> None:
    """전체 성숙도 모집단에 없는 월별 예측 결과는 조용히 버리지 않습니다."""
    maturity_summary = pd.DataFrame(
        {
            "anchor_month": ["2026-05"],
            "matured_sample_count": [2],
        }
    )
    quality_summary = pd.DataFrame(
        {
            "anchor_month": ["2026-06"],
            "evaluable_sample_count": [1],
        }
    )

    with pytest.raises(ValueError, match="성숙도 결과에 없는"):
        merge_validation_maturity_and_quality(maturity_summary, quality_summary)


@pytest.mark.parametrize(
    "outcome_available",
    [
        [True, None],
        ["True", "False"],
    ],
)
def test_maturity_summary_rejects_invalid_outcome_available(
    outcome_available: list[object],
) -> None:
    """결측값이나 문자열로 표현된 정답 확인 여부를 boolean처럼 사용하지 않습니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation", "validation"],
            "anchor_at": pd.to_datetime(["2026-05-01", "2026-05-02"]),
            "split_end_at": pd.to_datetime(["2026-08-20", "2026-08-20"]),
            "outcome_available_by_split_end": outcome_available,
        }
    )

    with pytest.raises(ValueError, match="boolean"):
        summarize_validation_label_maturity_by_anchor_month(rows)
