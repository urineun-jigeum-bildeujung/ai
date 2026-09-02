"""재구매 오차 분석 함수의 행 단위 계산을 검증합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.error_analysis import (
    add_error_columns,
    build_fixed_cohort_comparison_rows,
    compare_error_on_fixed_cohort,
    select_largest_error_rows,
    summarize_largest_error_tail,
    summarize_largest_error_tail_by_history_count,
    summarize_user_product_error_variability,
    summarize_user_product_errors_by_anchor_month,
    summarize_user_product_errors_by_history_count,
    summarize_user_product_errors_by_prior_count,
    summarize_user_product_errors_by_product,
)


def test_add_error_columns_without_mutating_input() -> None:
    """방향·절대오차를 계산하되 원본 DataFrame은 변경하지 않는지 확인합니다."""
    rows = pd.DataFrame(
        {
            "target_duration_days": [30.0, 50.0],
            "predicted_duration_days": [28.0, 70.0],
        }
    )

    result = add_error_columns(rows)

    assert result["prediction_error_days"].tolist() == [-2.0, 20.0]
    assert result["absolute_error_days"].tolist() == [2.0, 20.0]
    assert result["is_invalid_prediction"].tolist() == [False, False]
    assert "prediction_error_days" not in rows.columns
    assert "absolute_error_days" not in rows.columns
    assert "is_invalid_prediction" not in rows.columns


def test_personal_error_summary_accepts_original_and_shrunk_predictions() -> None:
    """기존·수축 개인화 예측은 포함하고 상품 fallback 예측은 제외합니다."""
    rows = pd.DataFrame(
        {
            "target_duration_days": [10.0, 20.0, 30.0],
            "predicted_duration_days": [12.0, 24.0, 100.0],
            "prediction_source": [
                "user_product_history",
                "shrunk_user_product_history",
                "product_history",
            ],
            "history_interval_count": [1, 2, 0],
        }
    )

    result = summarize_user_product_errors_by_history_count(rows)

    assert result["history_interval_count"].tolist() == [1, 2]
    assert result["sample_count"].tolist() == [1, 1]
    assert result["mae_days"].tolist() == [2.0, 4.0]


def test_add_error_columns_rejects_missing_required_column() -> None:
    """필수 열이 누락되면 계산 전에 명확한 오류를 반환하는지 확인합니다."""
    # 실제 구매 간격은 있지만 모델 예측값은 없는 잘못된 입력을 만듭니다.
    rows = pd.DataFrame(
        {
            "target_duration_days": [30.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="predicted_duration_days",
    ):
        add_error_columns(rows)


@pytest.mark.parametrize(
    ("target_value", "predicted_value", "expected_message"),
    [
        ("30일", 28.0, "숫자형"),
        (None, 28.0, "결측값"),
        (30.0, float("inf"), "유한한 숫자"),
        (-1.0, 28.0, "음수일 수 없습니다"),
    ],
)
def test_add_error_columns_rejects_unusable_values(
    target_value: object,
    predicted_value: object,
    expected_message: str,
) -> None:
    """숫자로 비교할 수 없는 입력과 잘못된 실제 기간을 거부하는지 확인합니다."""
    rows = pd.DataFrame(
        {
            "target_duration_days": [target_value],
            "predicted_duration_days": [predicted_value],
        }
    )

    with pytest.raises(ValueError, match=expected_message):
        add_error_columns(rows)


def test_add_error_columns_keeps_negative_prediction_as_model_failure() -> None:
    """음수 예측을 제거하지 않고 오차와 실패 표시를 함께 보존합니다."""
    rows = pd.DataFrame(
        {
            "target_duration_days": [30.0],
            "predicted_duration_days": [-5.0],
        }
    )

    result = add_error_columns(rows)

    assert result["prediction_error_days"].tolist() == [-35.0]
    assert result["absolute_error_days"].tolist() == [35.0]
    assert result["is_invalid_prediction"].tolist() == [True]


def test_add_error_columns_preserves_early_and_late_direction() -> None:
    """빠른 예측은 음수, 늦은 예측은 양수로 구분하는지 확인합니다."""
    rows = pd.DataFrame(
        {
            "target_duration_days": [30.0, 30.0],
            "predicted_duration_days": [20.0, 40.0],
        }
    )

    result = add_error_columns(rows)

    assert result["prediction_error_days"].tolist() == [-10.0, 10.0]


def test_select_largest_error_rows_uses_absolute_error_size() -> None:
    """빠른·늦은 방향과 관계없이 절대오차가 큰 개인 이력을 선택합니다."""
    rows = pd.DataFrame(
        {
            "prediction_source": [
                "user_product_history",
                "user_product_history",
                "user_product_history",
                "product_history",
            ],
            "target_duration_days": [120.0, 20.0, 30.0, 30.0],
            "predicted_duration_days": [20.0, 100.0, 35.0, 300.0],
        }
    )

    result = select_largest_error_rows(rows, tail_rate=0.5)

    # 개인 이력 3개의 50%는 1.5개이므로 올림한 2개가 선택됩니다.
    assert len(result) == 2
    # -100일과 +80일이 선택되어 오차의 방향과 크기가 함께 보존됩니다.
    assert result["prediction_error_days"].tolist() == [-100.0, 80.0]
    assert result["absolute_error_days"].tolist() == [100.0, 80.0]


def test_compare_error_on_fixed_cohort_tracks_same_reference_failures() -> None:
    """후보별 최악 행이 달라도 기준 모델의 고정 실패 표본만 다시 비교합니다."""
    sample_ids = {
        "user_id": ["u1", "u2", "u3"],
        "order_id": ["o1", "o2", "o3"],
        "product_id": ["p1", "p2", "p3"],
    }
    reference_rows = pd.DataFrame(
        {
            **sample_ids,
            "prediction_source": ["user_product_history"] * 3,
            "target_duration_days": [30.0, 30.0, 30.0],
            "predicted_duration_days": [130.0, 70.0, 32.0],
        }
    )
    candidate_rows = pd.DataFrame(
        {
            **sample_ids,
            "prediction_source": ["shrunk_user_product_history"] * 3,
            "target_duration_days": [30.0, 30.0, 30.0],
            "predicted_duration_days": [50.0, 20.0, 300.0],
        }
    )
    # 기준 모델의 오차는 100·40·2일이므로 상위 50%는 u1과 u2입니다.
    fixed_cohort = select_largest_error_rows(reference_rows, tail_rate=0.5)

    comparison_rows = build_fixed_cohort_comparison_rows(
        reference_rows,
        candidate_rows,
        fixed_cohort,
    )

    result = compare_error_on_fixed_cohort(
        reference_rows,
        candidate_rows,
        fixed_cohort,
    )

    assert result["cohort_sample_count"] == 2
    assert result["reference_mae_days"] == 70.0
    assert result["candidate_mae_days"] == 15.0
    assert result["mae_improvement_days"] == 55.0
    assert result["improved_sample_count"] == 2
    assert result["improved_sample_rate"] == 1.0
    assert result["worsened_sample_count"] == 0
    assert result["worsened_sample_rate"] == 0.0
    assert result["candidate_late_prediction_count"] == 1
    assert result["candidate_late_prediction_rate"] == 0.5
    assert result["candidate_early_prediction_count"] == 1
    assert result["candidate_early_prediction_rate"] == 0.5
    assert comparison_rows["user_id"].tolist() == ["u1", "u2"]
    assert comparison_rows["absolute_error_improvement_days"].tolist() == [80.0, 30.0]
    assert comparison_rows["comparison_outcome"].tolist() == [
        "IMPROVED",
        "IMPROVED",
    ]


def test_summarize_user_product_errors_by_prior_count() -> None:
    """개인화 예측의 prior 관측 수별 표본 비중과 오차를 계산합니다."""
    rows = pd.DataFrame(
        {
            "prediction_source": ["shrunk_user_product_history"] * 3,
            "prior_source": ["product_history"] * 3,
            "prior_observation_count": [1, 1, 10],
            "target_duration_days": [30.0, 30.0, 30.0],
            "predicted_duration_days": [60.0, 20.0, 40.0],
        }
    )

    result = summarize_user_product_errors_by_prior_count(rows)

    assert result["prior_observation_count"].tolist() == [1, 10]
    assert result["sample_count"].tolist() == [2, 1]
    assert result["sample_rate"].tolist() == pytest.approx([2 / 3, 1 / 3])
    assert result["mae_days"].tolist() == [20.0, 10.0]
    assert result["median_absolute_error_days"].tolist() == [20.0, 10.0]
    assert result["mean_prediction_error_days"].tolist() == [10.0, 10.0]


@pytest.mark.parametrize("tail_rate", [0.0, -0.1, 1.1])
def test_select_largest_error_rows_rejects_invalid_rate(tail_rate: float) -> None:
    """0% 이하 또는 100%를 넘는 꼬리 비율을 거부합니다."""
    rows = pd.DataFrame(
        {
            "prediction_source": ["user_product_history"],
            "target_duration_days": [30.0],
            "predicted_duration_days": [40.0],
        }
    )

    with pytest.raises(ValueError, match="tail_rate"):
        select_largest_error_rows(rows, tail_rate=tail_rate)


def test_summarize_largest_error_tail() -> None:
    """꼬리 표본의 전체 오차 기여도와 예측 방향 비율을 계산합니다."""
    rows = pd.DataFrame(
        {
            "prediction_source": [
                "user_product_history",
                "user_product_history",
                "user_product_history",
                "user_product_history",
                "product_history",
            ],
            "target_duration_days": [120.0, 20.0, 30.0, 30.0, 30.0],
            "predicted_duration_days": [20.0, 100.0, 50.0, 30.0, 300.0],
        }
    )

    result = summarize_largest_error_tail(rows, tail_rate=0.5)

    # 개인 이력 4개 중 절대오차가 큰 상위 2개(-100일, +80일)를 사용합니다.
    assert result["tail_sample_count"] == 2
    assert result["actual_tail_sample_rate"] == 0.5
    # 전체 절대오차 200일 중 꼬리 표본이 만든 오차는 180일입니다.
    assert result["tail_absolute_error_days"] == 180.0
    assert result["absolute_error_share"] == pytest.approx(0.9)
    assert result["late_prediction_count"] == 1
    assert result["late_prediction_rate"] == 0.5
    assert result["early_prediction_count"] == 1
    assert result["early_prediction_rate"] == 0.5
    assert result["exact_prediction_count"] == 0
    assert result["exact_prediction_rate"] == 0.0


def test_summarize_largest_error_tail_by_history_count() -> None:
    """이력 개수별 전체 비율과 꼬리 비율을 비교해 과대표집을 계산합니다."""
    rows = pd.DataFrame(
        {
            "prediction_source": ["user_product_history"] * 8,
            "history_interval_count": [1, 1, 1, 1, 2, 2, 2, 2],
            "target_duration_days": [30.0] * 8,
            "predicted_duration_days": [
                130.0,
                120.0,
                31.0,
                31.0,
                40.0,
                39.0,
                38.0,
                37.0,
            ],
        }
    )

    result = summarize_largest_error_tail_by_history_count(rows, tail_rate=0.25)
    by_history_count = result.set_index("history_interval_count")

    # 전체에서는 두 이력 구간이 각각 절반이지만 큰 오차 2개는 모두 이력 1개입니다.
    assert by_history_count.loc[1, "overall_sample_rate"] == 0.5
    assert by_history_count.loc[1, "tail_sample_count"] == 2
    assert by_history_count.loc[1, "tail_sample_rate"] == 1.0
    assert by_history_count.loc[1, "tail_membership_rate"] == 0.5
    assert by_history_count.loc[1, "tail_overrepresentation_ratio"] == 2.0
    assert by_history_count.loc[1, "tail_absolute_error_share"] == 1.0
    # 이력 2개 표본도 결과에 보존하되 꼬리에 포함되지 않았음을 0으로 표시합니다.
    assert by_history_count.loc[2, "tail_sample_count"] == 0
    assert by_history_count.loc[2, "tail_overrepresentation_ratio"] == 0.0


def test_summarize_user_product_errors_by_history_count() -> None:
    """개인 이력 개수별 표본 수와 오차 지표를 올바르게 요약하는지 확인합니다."""
    rows = pd.DataFrame(
        {
            "prediction_source": [
                "user_product_history",
                "user_product_history",
                "user_product_history",
                "product_history",
            ],
            "history_interval_count": [1, 1, 2, 0],
            "target_duration_days": [30.0, 50.0, 30.0, 40.0],
            "predicted_duration_days": [20.0, 70.0, 35.0, 60.0],
        }
    )

    result = summarize_user_product_errors_by_history_count(rows)
    by_history_count = result.set_index("history_interval_count")

    assert by_history_count.loc[1, "sample_count"] == 2
    assert by_history_count.loc[1, "mae_days"] == 15.0
    assert by_history_count.loc[1, "median_absolute_error_days"] == 15.0
    assert by_history_count.loc[1, "mean_prediction_error_days"] == 5.0
    assert by_history_count.loc[2, "sample_count"] == 1
    assert by_history_count.loc[2, "mae_days"] == 5.0


def test_summarize_user_product_error_variability() -> None:
    """상대 MAD와 절대오차의 순서가 같을 때 양의 순위 상관을 반환합니다."""
    rows = pd.DataFrame(
        {
            "prediction_source": [
                "user_product_history",
                "user_product_history",
                "user_product_history",
                "user_product_history",
                "product_history",
            ],
            "history_relative_mad": [0.1, 0.2, 0.4, None, 0.9],
            "target_duration_days": [30.0, 30.0, 30.0, 30.0, 30.0],
            "predicted_duration_days": [32.0, 40.0, 50.0, 35.0, 100.0],
        }
    )

    result = summarize_user_product_error_variability(rows)

    assert result["sample_count"] == 3
    assert result["median_relative_mad"] == 0.2
    assert result["spearman_relative_mad_absolute_error"] == pytest.approx(1.0)


def make_error_contributor_rows() -> pd.DataFrame:
    """상품·월별 오차 기여도 검증에 사용할 작은 예측 결과를 만듭니다."""
    return pd.DataFrame(
        {
            "prediction_source": [
                "user_product_history",
                "user_product_history",
                "user_product_history",
                "product_history",
            ],
            "product_id": ["p1", "p1", "p2", "p3"],
            "anchor_at": pd.to_datetime(
                ["2026-01-01", "2026-02-01", "2026-01-15", "2026-01-20"]
            ),
            "target_duration_days": [30.0, 30.0, 30.0, 30.0],
            "predicted_duration_days": [20.0, 50.0, 35.0, 130.0],
        }
    )


def test_summarize_user_product_errors_by_product() -> None:
    """개인 이력의 전체 절대오차 기여도가 큰 상품부터 정렬합니다."""
    result = summarize_user_product_errors_by_product(make_error_contributor_rows())
    by_product = result.set_index("product_id")

    assert result["product_id"].tolist() == ["p1", "p2"]
    assert by_product.loc["p1", "sample_count"] == 2
    assert by_product.loc["p1", "total_absolute_error_days"] == 30.0
    assert by_product.loc["p1", "mae_days"] == 15.0
    assert by_product.loc["p1", "absolute_error_share"] == pytest.approx(30 / 35)


def test_summarize_user_product_errors_by_anchor_month() -> None:
    """상품 전체 이력 예측을 제외하고 예측 기준 월별 오차를 요약합니다."""
    result = summarize_user_product_errors_by_anchor_month(
        make_error_contributor_rows()
    )
    by_month = result.set_index("anchor_month")

    assert by_month.loc["2026-01", "sample_count"] == 2
    assert by_month.loc["2026-01", "mae_days"] == 7.5
    assert by_month.loc["2026-01", "mean_prediction_error_days"] == -2.5
    assert by_month.loc["2026-02", "sample_count"] == 1
    assert by_month.loc["2026-02", "mae_days"] == 20.0
