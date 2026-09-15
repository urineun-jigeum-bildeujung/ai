"""관찰 가능 기간 계산이 원본을 보존하고 잘못된 입력을 거절하는지 검증합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.maturity_analysis import (
    add_available_followup_days,
    add_validation_common_followup_eligibility,
    add_validation_event_within_horizon,
    add_validation_ipcw_weights,
    add_validation_survival_observation,
    compare_validation_common_followup_candidates,
    compare_validation_common_followup_monthly_composition,
    estimate_validation_censoring_survival_curve,
    merge_validation_maturity_and_quality,
    summarize_matured_prediction_quality_by_anchor_month,
    summarize_validation_common_followup_by_anchor_month,
    summarize_validation_common_followup_outcomes,
    summarize_validation_followup_distribution,
    summarize_validation_ipcw_weight_stability,
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


def test_add_validation_survival_observation_uses_only_split_known_information() -> (
    None
):
    """분할 뒤에 발생한 재구매를 보지 않고 당시 관찰 가능 기간으로 검열합니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation"] * 3,
            "anchor_at": pd.to_datetime(["2026-07-01", "2026-07-20", "2026-08-01"]),
            "split_end_at": pd.to_datetime(["2026-08-20"] * 3),
            "outcome_available_by_split_end": [True, False, False],
            # 두 번째 행의 45일은 전체 데이터에서는 보이지만 Validation 종료 뒤의 미래입니다.
            "target_duration_days": [20.0, 45.0, float("nan")],
        }
    )

    result = add_validation_survival_observation(rows)

    assert result["available_followup_days"].tolist() == [50.0, 31.0, 19.0]
    assert result["survival_observed_duration_days"].tolist() == [20.0, 31.0, 19.0]
    assert result["survival_event_observed"].tolist() == [True, False, False]
    assert result["survival_right_censored"].tolist() == [False, True, True]
    assert "survival_observed_duration_days" not in rows.columns


def test_survival_observation_rejects_event_after_available_followup() -> None:
    """분할 종료 전에 확인됐다는 표본의 간격이 관찰 기간보다 길면 거절합니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation"],
            "anchor_at": pd.to_datetime(["2026-08-01"]),
            "split_end_at": pd.to_datetime(["2026-08-20"]),
            "outcome_available_by_split_end": [True],
            "target_duration_days": [20.0],
        }
    )

    with pytest.raises(ValueError, match="관찰 가능 기간을 넘을 수 없습니다"):
        add_validation_survival_observation(rows)


def test_estimate_validation_censoring_survival_curve() -> None:
    """시점별 위험집단과 검열 수로 검열 생존확률을 누적 계산합니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation"] * 4,
            "anchor_at": pd.to_datetime(
                ["2026-08-18", "2026-08-17", "2026-08-16", "2026-08-15"]
            ),
            "split_end_at": pd.to_datetime(["2026-08-20"] * 4),
            "outcome_available_by_split_end": [True, False, True, False],
            "target_duration_days": [2.0, float("nan"), 4.0, float("nan")],
        }
    )

    result = estimate_validation_censoring_survival_curve(rows)

    assert result["survival_observed_duration_days"].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert result["at_risk_count"].tolist() == [4, 3, 2, 1]
    assert result["censoring_event_count"].tolist() == [0, 1, 0, 1]
    assert result["repurchase_event_count"].tolist() == [1, 0, 1, 0]
    assert result["censoring_survival_step"].tolist() == pytest.approx(
        [1.0, 2 / 3, 1.0, 0.0]
    )
    assert result["censoring_survival_probability_before"].tolist() == pytest.approx(
        [1.0, 1.0, 2 / 3, 2 / 3]
    )
    assert result["censoring_survival_probability"].tolist() == pytest.approx(
        [1.0, 2 / 3, 2 / 3, 0.0]
    )
    assert "survival_observed_duration_days" not in rows.columns


def test_add_validation_ipcw_weights_distinguishes_known_and_unknown_outcomes() -> None:
    """사건·고정 시점까지의 무사건·조기 검열에 서로 다른 가중치를 적용합니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation"] * 4,
            "anchor_at": pd.to_datetime(
                ["2026-08-18", "2026-08-17", "2026-08-16", "2026-08-15"]
            ),
            "split_end_at": pd.to_datetime(["2026-08-20"] * 4),
            "outcome_available_by_split_end": [True, False, True, False],
            "target_duration_days": [2.0, float("nan"), 4.0, float("nan")],
        }
    )

    result = add_validation_ipcw_weights(rows, horizon_days=4)

    assert result["ipcw_event_within_horizon"].tolist() == [True, pd.NA, True, False]
    assert result["ipcw_outcome_known"].tolist() == [True, False, True, True]
    assert result["ipcw_censoring_survival_probability"].tolist() == [
        1.0,
        pd.NA,
        pytest.approx(2 / 3),
        pytest.approx(2 / 3),
    ]
    assert result["ipcw_weight"].tolist() == pytest.approx([1.0, 0.0, 1.5, 1.5])
    assert "ipcw_weight" not in rows.columns


def test_summarize_validation_ipcw_weight_stability() -> None:
    """원시 가중치의 꼬리와 가중치 집중에 따른 유효 표본 감소를 요약합니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation"] * 4,
            "anchor_at": pd.to_datetime(
                ["2026-08-18", "2026-08-17", "2026-08-16", "2026-08-15"]
            ),
            "split_end_at": pd.to_datetime(["2026-08-20"] * 4),
            "outcome_available_by_split_end": [True, False, True, False],
            "target_duration_days": [2.0, float("nan"), 4.0, float("nan")],
        }
    )

    result = summarize_validation_ipcw_weight_stability(rows, horizon_days=4).iloc[0]

    assert result["validation_sample_count"] == 4
    assert result["outcome_known_count"] == 3
    assert result["outcome_unknown_count"] == 1
    assert result["outcome_known_rate"] == 0.75
    assert result["event_within_horizon_count"] == 2
    assert result["no_event_within_horizon_count"] == 1
    assert result["unweighted_known_event_rate"] == pytest.approx(2 / 3)
    assert result["ipcw_weighted_event_mass"] == pytest.approx(2.5)
    assert result["ipcw_weighted_no_event_mass"] == pytest.approx(1.5)
    assert result["ipcw_weighted_event_rate"] == pytest.approx(2.5 / 4)
    assert result["ipcw_event_rate_difference"] == pytest.approx((2.5 / 4) - (2 / 3))
    assert result["ipcw_weight_sum"] == pytest.approx(4.0)
    assert result["ipcw_weight_sum_ratio"] == pytest.approx(1.0)
    assert result["ipcw_weight_mean"] == pytest.approx(4 / 3)
    assert result["ipcw_weight_median"] == pytest.approx(1.5)
    assert result["ipcw_weight_p90"] == pytest.approx(1.5)
    assert result["ipcw_weight_p95"] == pytest.approx(1.5)
    assert result["ipcw_weight_p99"] == pytest.approx(1.5)
    assert result["ipcw_weight_max"] == pytest.approx(1.5)
    assert result["ipcw_effective_sample_size"] == pytest.approx(16 / 5.5)
    assert result["ipcw_effective_to_known_sample_rate"] == pytest.approx(
        (16 / 5.5) / 3
    )
    assert result["ipcw_effective_to_validation_sample_rate"] == pytest.approx(
        (16 / 5.5) / 4
    )


def test_summarize_validation_followup_distribution() -> None:
    """관찰 가능 기간을 정렬했을 때 주요 분위수의 일수를 계산합니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation"] * 5,
            "anchor_at": pd.to_datetime(
                ["2026-08-10", "2026-07-31", "2026-07-21", "2026-07-11", "2026-07-01"]
            ),
            "split_end_at": pd.to_datetime(["2026-08-20"] * 5),
        }
    )

    result = summarize_validation_followup_distribution(rows)

    assert result["quantile"].tolist() == [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    assert result["available_followup_days"].tolist() == [
        10.0,
        14.0,
        20.0,
        30.0,
        40.0,
        46.0,
        50.0,
    ]


def test_followup_distribution_rejects_non_validation_rows() -> None:
    """후보 탐색에 Test 표본이 섞이면 분포 계산을 중단합니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation", "test"],
            "anchor_at": pd.to_datetime(["2026-08-10", "2026-09-01"]),
            "split_end_at": pd.to_datetime(["2026-08-20", "2026-12-01"]),
        }
    )

    with pytest.raises(ValueError, match="Validation 표본만"):
        summarize_validation_followup_distribution(rows)


def test_add_validation_common_followup_eligibility_marks_boundary() -> None:
    """공통 관찰 기간보다 길거나 같은 표본만 평가 후보로 표시합니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation", "validation", "validation"],
            "anchor_at": pd.to_datetime(["2026-07-20", "2026-07-21", "2026-08-01"]),
            "split_end_at": pd.to_datetime(["2026-08-20"] * 3),
        }
    )

    result = add_validation_common_followup_eligibility(rows, horizon_days=30)

    assert result["available_followup_days"].tolist() == [31.0, 30.0, 19.0]
    assert result["common_followup_horizon_days"].tolist() == [30, 30, 30]
    assert result["common_followup_eligible"].tolist() == [True, True, False]
    assert "common_followup_eligible" not in rows.columns


@pytest.mark.parametrize("horizon_days", [0, -1, 30.5, True])
def test_common_followup_eligibility_rejects_invalid_horizon(
    horizon_days: object,
) -> None:
    """공통 관찰 기간은 양의 정수 일수만 허용합니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation"],
            "anchor_at": pd.to_datetime(["2026-07-20"]),
            "split_end_at": pd.to_datetime(["2026-08-20"]),
        }
    )

    with pytest.raises(ValueError, match="양의 정수"):
        add_validation_common_followup_eligibility(
            rows,
            horizon_days=horizon_days,  # type: ignore[arg-type]
        )


def test_common_followup_eligibility_rejects_non_validation_rows() -> None:
    """평가 방법 선택 단계에 Test 표본이 섞이면 실행을 중단합니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation", "test"],
            "anchor_at": pd.to_datetime(["2026-07-20", "2026-08-21"]),
            "split_end_at": pd.to_datetime(["2026-08-20", "2026-12-01"]),
        }
    )

    with pytest.raises(ValueError, match="Validation 표본만"):
        add_validation_common_followup_eligibility(rows, horizon_days=30)


def test_summarize_validation_common_followup_outcomes() -> None:
    """평가 가능 범위와 그 안에서 확인된 기간 내 재구매율을 계산합니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation"] * 4,
            "anchor_at": pd.to_datetime(
                ["2026-07-01", "2026-07-20", "2026-07-21", "2026-08-01"]
            ),
            "split_end_at": pd.to_datetime(["2026-08-20"] * 4),
            "outcome_available_by_split_end": [True, False, False, False],
            "target_duration_days": [20.0, 45.0, float("nan"), float("nan")],
        }
    )

    result = summarize_validation_common_followup_outcomes(rows, horizon_days=30)

    assert result.to_dict(orient="records") == [
        {
            "horizon_days": 30,
            "validation_sample_count": 4,
            "eligible_sample_count": 3,
            "ineligible_sample_count": 1,
            "eligible_rate": 0.75,
            "event_within_horizon_count": 1,
            "no_event_within_horizon_count": 2,
            "event_within_horizon_rate": 1 / 3,
            "event_duration_mean_days": 20.0,
            "event_duration_q25_days": 20.0,
            "event_duration_median_days": 20.0,
            "event_duration_q75_days": 20.0,
        }
    ]


def test_common_followup_outcomes_keep_empty_candidate_as_na() -> None:
    """평가 가능 표본이 없는 후보도 비교할 수 있도록 비율을 NA로 남깁니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation"],
            "anchor_at": pd.to_datetime(["2026-08-01"]),
            "split_end_at": pd.to_datetime(["2026-08-20"]),
            "outcome_available_by_split_end": [False],
            "target_duration_days": [float("nan")],
        }
    )

    result = summarize_validation_common_followup_outcomes(rows, horizon_days=30)

    assert result.loc[0, "eligible_sample_count"] == 0
    assert result.loc[0, "ineligible_sample_count"] == 1
    assert result.loc[0, "eligible_rate"] == 0.0
    assert result.loc[0, "event_within_horizon_count"] == 0
    assert result.loc[0, "no_event_within_horizon_count"] == 0
    assert pd.isna(result.loc[0, "event_within_horizon_rate"])
    assert pd.isna(result.loc[0, "event_duration_mean_days"])
    assert pd.isna(result.loc[0, "event_duration_q25_days"])
    assert pd.isna(result.loc[0, "event_duration_median_days"])
    assert pd.isna(result.loc[0, "event_duration_q75_days"])
    assert str(result["event_within_horizon_rate"].dtype) == "Float64"


def test_compare_validation_common_followup_candidates() -> None:
    """후보별 결과를 동일한 Validation에서 계산해 관찰 기간순으로 정렬합니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation"] * 4,
            "anchor_at": pd.to_datetime(
                ["2026-07-01", "2026-07-20", "2026-07-21", "2026-08-01"]
            ),
            "split_end_at": pd.to_datetime(["2026-08-20"] * 4),
            "outcome_available_by_split_end": [True, False, False, False],
            "target_duration_days": [20.0, 45.0, float("nan"), float("nan")],
        }
    )

    result = compare_validation_common_followup_candidates(
        rows,
        horizon_days_candidates=[30, 14, 60],
    )

    assert result["horizon_days"].tolist() == [14, 30, 60]
    assert result["validation_sample_count"].tolist() == [4, 4, 4]
    assert result["eligible_sample_count"].tolist() == [4, 3, 0]
    assert result["ineligible_sample_count"].tolist() == [0, 1, 4]
    assert result["eligible_rate"].tolist() == [1.0, 0.75, 0.0]
    assert result["event_within_horizon_count"].tolist() == [0, 1, 0]
    assert result["no_event_within_horizon_count"].tolist() == [4, 2, 0]
    assert result.loc[0, "event_within_horizon_rate"] == 0.0
    assert result.loc[1, "event_within_horizon_rate"] == pytest.approx(1 / 3)
    assert pd.isna(result.loc[2, "event_within_horizon_rate"])
    assert pd.isna(result.loc[0, "event_duration_median_days"])
    assert result.loc[1, "event_duration_median_days"] == 20.0
    assert pd.isna(result.loc[2, "event_duration_median_days"])


def test_common_followup_outcomes_summarize_observed_event_durations() -> None:
    """미재구매를 가짜 간격으로 채우지 않고 실제 사건의 분포만 계산합니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation"] * 4,
            "anchor_at": pd.to_datetime(["2026-07-01"] * 4),
            "split_end_at": pd.to_datetime(["2026-08-20"] * 4),
            "outcome_available_by_split_end": [True] * 4,
            "target_duration_days": [10.0, 20.0, 30.0, 45.0],
        }
    )

    result = summarize_validation_common_followup_outcomes(rows, horizon_days=30)

    assert result.loc[0, "event_within_horizon_count"] == 3
    assert result.loc[0, "no_event_within_horizon_count"] == 1
    assert result.loc[0, "event_duration_mean_days"] == 20.0
    assert result.loc[0, "event_duration_q25_days"] == 15.0
    assert result.loc[0, "event_duration_median_days"] == 20.0
    assert result.loc[0, "event_duration_q75_days"] == 25.0


def test_common_followup_candidate_comparison_rejects_empty_candidates() -> None:
    """후보가 없으면 의미 없는 빈 비교표를 만들지 않습니다."""
    rows = pd.DataFrame()

    with pytest.raises(ValueError, match="하나 이상"):
        compare_validation_common_followup_candidates(
            rows,
            horizon_days_candidates=[],
        )


def test_common_followup_candidate_comparison_rejects_duplicates() -> None:
    """중복 후보를 자동 제거하지 않고 입력 오류로 알립니다."""
    rows = pd.DataFrame()

    with pytest.raises(ValueError, match="중복"):
        compare_validation_common_followup_candidates(
            rows,
            horizon_days_candidates=[14, 30, 30],
        )


def test_summarize_common_followup_by_anchor_month_tracks_share_shift() -> None:
    """월별 보존율과 전체 대비 평가 가능 표본의 월 비중 변화를 계산합니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation"] * 4,
            "anchor_at": pd.to_datetime(
                ["2026-07-01", "2026-07-25", "2026-08-01", "2026-08-10"]
            ),
            "split_end_at": pd.to_datetime(["2026-08-20"] * 4),
        }
    )

    result = summarize_validation_common_followup_by_anchor_month(
        rows,
        horizon_days=30,
    )

    assert result["anchor_month"].tolist() == ["2026-07", "2026-08"]
    assert result["validation_sample_count"].tolist() == [2, 2]
    assert result["eligible_sample_count"].tolist() == [1, 0]
    assert result["ineligible_sample_count"].tolist() == [1, 2]
    assert result["eligible_rate"].tolist() == [0.5, 0.0]
    assert result["validation_sample_share"].tolist() == [0.5, 0.5]
    assert result["eligible_sample_share"].tolist() == [1.0, 0.0]
    assert result["eligible_share_shift"].tolist() == [0.5, -0.5]


def test_common_followup_monthly_summary_keeps_no_eligible_share_as_na() -> None:
    """평가 가능 표본이 전혀 없으면 월별 평가 표본 비중을 NA로 남깁니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation", "validation"],
            "anchor_at": pd.to_datetime(["2026-07-25", "2026-08-10"]),
            "split_end_at": pd.to_datetime(["2026-08-20", "2026-08-20"]),
        }
    )

    result = summarize_validation_common_followup_by_anchor_month(
        rows,
        horizon_days=60,
    )

    assert result["eligible_sample_count"].tolist() == [0, 0]
    assert result["eligible_rate"].tolist() == [0.0, 0.0]
    assert result["validation_sample_share"].tolist() == [0.5, 0.5]
    assert result["eligible_sample_share"].isna().all()
    assert result["eligible_share_shift"].isna().all()


def test_compare_common_followup_monthly_composition() -> None:
    """후보별 월 구성을 같은 Validation에서 관찰 기간과 월 순서로 정렬합니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation"] * 4,
            "anchor_at": pd.to_datetime(
                ["2026-07-01", "2026-07-25", "2026-08-01", "2026-08-10"]
            ),
            "split_end_at": pd.to_datetime(["2026-08-20"] * 4),
        }
    )

    result = compare_validation_common_followup_monthly_composition(
        rows,
        horizon_days_candidates=[30, 14],
    )

    assert result[["horizon_days", "anchor_month"]].to_records(
        index=False
    ).tolist() == [
        (14, "2026-07"),
        (14, "2026-08"),
        (30, "2026-07"),
        (30, "2026-08"),
    ]
    assert result["validation_sample_count"].tolist() == [2, 2, 2, 2]
    assert result["eligible_sample_count"].tolist() == [2, 1, 1, 0]


def test_add_validation_event_within_horizon_preserves_three_states() -> None:
    """기간 내 재구매와 미재구매를 구분하고 관찰 부족은 NA로 보존합니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation"] * 4,
            "anchor_at": pd.to_datetime(
                ["2026-07-01", "2026-07-01", "2026-07-01", "2026-08-01"]
            ),
            "split_end_at": pd.to_datetime(["2026-08-20"] * 4),
            "outcome_available_by_split_end": [True, True, False, False],
            "target_duration_days": [20.0, 45.0, float("nan"), float("nan")],
        }
    )

    result = add_validation_event_within_horizon(rows, horizon_days=30)

    assert bool(result.loc[0, "event_within_horizon"])
    assert not bool(result.loc[1, "event_within_horizon"])
    assert not bool(result.loc[2, "event_within_horizon"])
    assert pd.isna(result.loc[3, "event_within_horizon"])
    assert str(result["event_within_horizon"].dtype) == "boolean"
    assert "event_within_horizon" not in rows.columns


def test_event_within_horizon_rejects_observed_event_without_duration() -> None:
    """확인된 사건의 실제 간격이 없으면 False로 오인하지 않고 거절합니다."""
    rows = pd.DataFrame(
        {
            "split": ["validation"],
            "anchor_at": pd.to_datetime(["2026-07-01"]),
            "split_end_at": pd.to_datetime(["2026-08-20"]),
            "outcome_available_by_split_end": [True],
            "target_duration_days": [float("nan")],
        }
    )

    with pytest.raises(ValueError, match="실제 재구매 간격"):
        add_validation_event_within_horizon(rows, horizon_days=30)


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
