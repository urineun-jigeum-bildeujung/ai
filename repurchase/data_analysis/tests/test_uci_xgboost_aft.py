"""작은 구매 이력으로 XGBoost AFT Validation 실행 흐름을 검증합니다."""

from __future__ import annotations

import json
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import xgboost

from scripts.modeling.samples import (
    assign_temporal_splits,
    build_historical_interval_features,
)
from scripts.modeling.xgboost_aft import (
    XGBoostAFTError,
    XGBoostAFTNumericalPredictionError,
)
from scripts.preprocessing.labels import build_same_product_repurchase_labels
from scripts.run_uci_xgboost_aft import (
    bootstrap_xgboost_aft_candidate_pair_by_user,
    build_xgboost_aft_bootstrap_trials_report,
    build_xgboost_aft_candidate_pair_bootstrap_report,
    build_xgboost_aft_candidate_pair_bootstrap_trials_report,
    build_xgboost_aft_distribution_comparison_report,
    build_xgboost_aft_lightgbm_bootstrap_trials_report,
    build_xgboost_aft_lightgbm_comparison_report,
    build_xgboost_aft_logistic_scale_comparison_report,
    build_xgboost_aft_observed_time_comparison_report,
    build_xgboost_aft_report,
    build_xgboost_aft_round_comparison_report,
    build_xgboost_aft_segment_comparison_report,
    compare_xgboost_aft_boosting_rounds,
    compare_xgboost_aft_loss_distribution_scales,
    compare_xgboost_aft_loss_distributions,
    compare_xgboost_aft_observed_time_with_median_baselines,
    compare_xgboost_aft_with_lightgbm_probability,
    evaluate_xgboost_aft_candidate,
    evaluate_xgboost_aft_observed_event_time,
    prepare_xgboost_aft_experiment,
    render_xgboost_aft_candidate_pair_bootstrap_report,
    render_xgboost_aft_distribution_comparison_report,
    render_xgboost_aft_lightgbm_comparison_report,
    render_xgboost_aft_logistic_scale_comparison_report,
    render_xgboost_aft_observed_time_comparison_report,
    render_xgboost_aft_report,
    render_xgboost_aft_round_comparison_report,
    render_xgboost_aft_segment_comparison_report,
    run_xgboost_aft_experiment,
    select_xgboost_aft_boosting_round,
    select_xgboost_aft_loss_distribution,
    select_xgboost_aft_loss_distribution_scale,
    summarize_xgboost_aft_probability_by_count_segments,
    validate_selected_xgboost_aft_distribution_result,
    validate_selected_xgboost_aft_result,
    validate_selected_xgboost_aft_scale_result,
)


def test_run_xgboost_aft_experiment_uses_train_and_validation_only(
    uci_e2e_purchase_events: pd.DataFrame,
) -> None:
    """공통 시간 분할로 학습하고 Test를 열지 않은 평가 결과를 반환합니다."""
    events = uci_e2e_purchase_events
    labels = build_same_product_repurchase_labels(
        events,
        observation_end_at=pd.Timestamp(events["ordered_at"].max()),
    )

    result = run_xgboost_aft_experiment(
        labels,
        bootstrap_replicates=20,
        bootstrap_random_seed=7,
    )
    samples = assign_temporal_splits(
        build_historical_interval_features(labels),
        result.split,
    )
    validation_index = samples.index[samples["split"].eq("validation")]
    test_index = samples.index[samples["split"].eq("test")]

    assert result.validation_predictions.index.equals(validation_index)
    assert result.validation_predictions.index.intersection(test_index).empty
    assert result.training_summary["training_split"] == "train"
    assert result.training_summary["evaluation_split"] == "validation"
    assert result.training_summary["trained_until"] == result.split.train_end_at
    assert result.training_summary["source_sample_count"] == int(
        samples["split"].eq("train").sum()
    )
    assert result.training_summary["num_boost_round"] == 5
    assert len(result.training_summary["training_aft_nloglik"]) == 5
    assert (
        result.training_summary["reference_population_sample_count"]
        == (result.training_summary["included_sample_count"])
    )
    assert result.concordance["source_validation_sample_count"] == len(validation_index)
    assert result.probability.summary["source_validation_sample_count"] == len(
        validation_index
    )
    assert (
        result.concordance["excluded_zero_duration_count"]
        == (result.probability.summary["excluded_zero_duration_count"])
    )
    assert (
        result.concordance["aft_evaluation_sample_count"]
        == (result.probability.summary["aft_evaluation_sample_count"])
    )
    assert result.probability.user_bootstrap is not None
    assert result.probability.user_bootstrap.summary["bootstrap_replicates"] == 20
    assert len(result.probability.user_bootstrap.trials) == 20

    report = build_xgboost_aft_report(result)
    trials_report = build_xgboost_aft_bootstrap_trials_report(result)
    markdown = render_xgboost_aft_report(report)
    assert report["evaluation_split"] == "validation"
    assert report["runtime"]["xgboost"] == xgboost.__version__
    assert report["training"]["trained_until"] == result.split.train_end_at.isoformat()
    assert len(trials_report["trials"]) == 20
    assert "C-index" in markdown
    assert "IPCW 가중 평균 예측확률 / 실제 사건률" in markdown
    assert "관측 재구매 시점 오차" in markdown
    assert "기준 설정에서" in markdown
    assert "사용자 단위 Bootstrap" in markdown
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    json.dumps(trials_report, ensure_ascii=False, allow_nan=False)

    undefined_skill_report = report.copy()
    undefined_skill_probability = report["validation_probability"].copy()
    undefined_skill_probability["brier_skill_score"] = None
    undefined_skill_report["validation_probability"] = undefined_skill_probability
    assert "N/A" in render_xgboost_aft_report(undefined_skill_report)


def test_evaluate_xgboost_aft_observed_event_time_excludes_censored_rows(
    uci_e2e_purchase_events: pd.DataFrame,
) -> None:
    """미래 target이 있어도 Validation에서 검열된 행은 시점 오차에서 제외합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels, horizon_days=14)
    result = evaluate_xgboost_aft_candidate(
        prepared,
        num_boost_round=2,
        bootstrap_replicates=None,
    )

    metrics = evaluate_xgboost_aft_observed_event_time(result)
    event_rows = result.probability.rows.loc[
        result.probability.rows["survival_event_observed"]
    ]

    assert metrics["sample_count"] == len(event_rows)
    assert metrics["sample_count"] < len(result.probability.rows)
    assert metrics["mae_days"] >= 0.0
    assert metrics["median_absolute_error_days"] >= 0.0
    assert 0.0 <= metrics["within_7_days_rate"] <= 1.0


def test_compare_xgboost_aft_observed_time_candidates_uses_same_cohort(
    uci_e2e_purchase_events: pd.DataFrame,
) -> None:
    """AFT와 중앙값 후보가 같은 0일 제외 Train·관측 Validation을 사용합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels, horizon_days=14)
    result = evaluate_xgboost_aft_candidate(
        prepared,
        num_boost_round=2,
        bootstrap_replicates=None,
    )
    training_before = prepared.training.copy(deep=True)
    validation_before = prepared.validation.copy(deep=True)

    comparison = compare_xgboost_aft_observed_time_with_median_baselines(
        prepared,
        result,
        shrinkage_strengths=(1.0, 4.0),
    )
    observed_count = int(result.probability.rows["survival_event_observed"].sum())

    assert comparison["candidate_name"].tolist() == [
        "xgboost_aft",
        "hierarchical_median",
        "shrunk_hierarchical_median",
        "shrunk_hierarchical_median",
    ]
    assert comparison["shrinkage_strength"].isna().tolist() == [
        True,
        True,
        False,
        False,
    ]
    assert comparison["sample_count"].eq(observed_count).all()
    assert comparison["mae_days"].ge(0.0).all()
    assert comparison["within_7_days_rate"].between(0.0, 1.0).all()
    aft_row = comparison.loc[comparison["candidate_name"].eq("xgboost_aft")].iloc[0]
    observed_metrics = evaluate_xgboost_aft_observed_event_time(result)
    assert aft_row["sample_count"] == observed_metrics["sample_count"]
    assert aft_row["mae_days"] == pytest.approx(observed_metrics["mae_days"])
    assert aft_row["median_absolute_error_days"] == pytest.approx(
        observed_metrics["median_absolute_error_days"]
    )
    assert aft_row["within_7_days_rate"] == pytest.approx(
        observed_metrics["within_7_days_rate"]
    )
    assert aft_row["mae_improvement_vs_aft_days"] == pytest.approx(0.0)
    assert aft_row["median_ae_improvement_vs_aft_days"] == pytest.approx(0.0)
    assert aft_row["within_7_days_rate_improvement_vs_aft"] == pytest.approx(0.0)

    report = build_xgboost_aft_observed_time_comparison_report(result, comparison)
    markdown = render_xgboost_aft_observed_time_comparison_report(report)
    assert report["evaluation_split"] == "validation"
    assert report["evaluation_sample_count"] == observed_count
    assert report["training_population"]["aft_actual_training_sample_count"] == len(
        prepared.training_data.row_index
    )
    assert (
        report["training_population"][
            "median_actual_observed_event_training_sample_count"
        ]
        <= report["training_population"]["aft_actual_training_sample_count"]
    )
    assert len(report["candidates"]) == 4
    assert "정확한 구매 시점 예측 역할" in report["scope"]
    assert "AFT 대비 개선량은 양수" in markdown
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    pd.testing.assert_frame_equal(prepared.training, training_before)
    pd.testing.assert_frame_equal(prepared.validation, validation_before)


def test_compare_xgboost_aft_observed_time_rejects_mixed_experiment_rows(
    uci_e2e_purchase_events: pd.DataFrame,
) -> None:
    """인덱스만 같고 구매 키나 정답이 다른 AFT 결과는 비교 전에 거절합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels, horizon_days=14)
    result = evaluate_xgboost_aft_candidate(
        prepared,
        num_boost_round=2,
        bootstrap_replicates=None,
    )

    for column, changed_value in (
        ("user_id", "MIXED_USER"),
        (
            "survival_event_observed",
            not bool(result.probability.rows["survival_event_observed"].iat[0]),
        ),
        (
            "target_duration_days",
            float(result.probability.rows["target_duration_days"].dropna().iat[0])
            + 100.0,
        ),
    ):
        changed_rows = result.probability.rows.copy()
        if column == "target_duration_days":
            changed_index = changed_rows["target_duration_days"].first_valid_index()
        else:
            changed_index = changed_rows.index[0]
        changed_rows.loc[changed_index, column] = changed_value
        changed_probability = replace(result.probability, rows=changed_rows)
        changed_result = replace(result, probability=changed_probability)

        with pytest.raises(ValueError, match=column):
            compare_xgboost_aft_observed_time_with_median_baselines(
                prepared,
                changed_result,
            )

    changed_horizon = replace(prepared, horizon_days=30)
    with pytest.raises(ValueError, match="horizon_days"):
        compare_xgboost_aft_with_lightgbm_probability(
            changed_horizon,
            result,
            bootstrap_replicates=2,
        )

    changed_training_data = replace(
        prepared.training_data,
        feature_columns=prepared.training_data.feature_columns[:-1],
    )
    changed_features = replace(prepared, training_data=changed_training_data)
    with pytest.raises(ValueError, match="feature_columns"):
        compare_xgboost_aft_with_lightgbm_probability(
            changed_features,
            result,
            bootstrap_replicates=2,
        )

    changed_validation = prepared.validation.copy()
    feature_column = prepared.training_data.feature_columns[0]
    changed_validation.loc[changed_validation.index[0], feature_column] += 100
    changed_feature_values = replace(prepared, validation=changed_validation)
    with pytest.raises(ValueError, match=feature_column):
        compare_xgboost_aft_with_lightgbm_probability(
            changed_feature_values,
            result,
            bootstrap_replicates=2,
        )


def test_compare_xgboost_aft_observed_time_candidates_rejects_invalid_strengths(
    uci_e2e_purchase_events: pd.DataFrame,
) -> None:
    """비어 있거나 중복된 수축 후보는 비교 학습 전에 명확하게 거절합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels, horizon_days=14)
    result = evaluate_xgboost_aft_candidate(
        prepared,
        num_boost_round=2,
        bootstrap_replicates=None,
    )

    for strengths in ((), (1.0, 1.0)):
        with pytest.raises(ValueError):
            compare_xgboost_aft_observed_time_with_median_baselines(
                prepared,
                result,
                shrinkage_strengths=strengths,
            )


def test_compare_xgboost_aft_with_lightgbm_uses_same_probability_cohort(
    uci_e2e_purchase_events: pd.DataFrame,
) -> None:
    """같은 피처·Train 원본·Validation에서 AFT와 LightGBM을 쌍 비교합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels, horizon_days=14)
    result = evaluate_xgboost_aft_candidate(
        prepared,
        num_boost_round=2,
        bootstrap_replicates=None,
    )
    probability_rows_before = result.probability.rows.copy(deep=True)

    comparison = compare_xgboost_aft_with_lightgbm_probability(
        prepared,
        result,
        bootstrap_replicates=20,
        bootstrap_random_seed=7,
    )

    assert comparison.comparison["model_candidate"].tolist() == [
        "xgboost_aft",
        "lightgbm_probability",
    ]
    for column in (
        "feature_columns",
        "common_training_source_sample_count",
        "evaluation_sample_count",
        "outcome_known_count",
        "ipcw_reference_brier_score",
    ):
        assert comparison.comparison[column].map(str).nunique() == 1
    assert comparison.comparison["ipcw_brier_score"].between(0.0, 1.0).all()
    assert comparison.comparison["ipcw_concordance_index"].between(0.0, 1.0).all()
    assert set(comparison.calibration["model_candidate"]) == {
        "xgboost_aft",
        "lightgbm_probability",
    }
    for count_column in comparison.segments["count_column"].unique():
        segment_rows = comparison.segments.loc[
            comparison.segments["count_column"].eq(count_column)
        ]
        assert segment_rows["sample_count"].sum() == len(result.probability.rows)
        assert segment_rows["outcome_known_count"].sum() == int(
            result.probability.rows["ipcw_outcome_known"].sum()
        )
    assert set(comparison.segments["validation_lower_brier_model"]).issubset(
        {"xgboost_aft", "lightgbm_probability"}
    )
    assert len(comparison.paired_rows) == len(result.probability.rows)
    assert comparison.user_bootstrap.summary["bootstrap_replicates"] == 20
    aft_brier, lightgbm_brier = comparison.comparison["ipcw_brier_score"]
    assert comparison.user_bootstrap.summary[
        "point_brier_improvement"
    ] == pytest.approx(aft_brier - lightgbm_brier)

    report = build_xgboost_aft_lightgbm_comparison_report(result, comparison)
    trials_report = build_xgboost_aft_lightgbm_bootstrap_trials_report(
        report,
        comparison,
    )
    markdown = render_xgboost_aft_lightgbm_comparison_report(report)
    assert report["evaluation_split"] == "validation"
    assert report["brier_improvement_direction"] == ("aft_brier_minus_lightgbm_brier")
    assert len(report["comparison"]) == 2
    assert len(report["segments"]) == len(comparison.segments)
    assert len(trials_report["trials"]) == 20
    assert "예상 일수 해석이 아닙니다" in report["scope"]
    assert "사용자 단위 paired Bootstrap" in markdown
    assert "이력량 구간별 비교" in markdown
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    json.dumps(trials_report, ensure_ascii=False, allow_nan=False)
    pd.testing.assert_frame_equal(result.probability.rows, probability_rows_before)


def test_summarize_xgboost_aft_probability_by_count_segments_preserves_population(
    uci_e2e_purchase_events: pd.DataFrame,
) -> None:
    """각 이력량 기준의 구간 합이 동일 AFT Validation 모집단을 보존합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels, horizon_days=14)
    result = evaluate_xgboost_aft_candidate(
        prepared,
        num_boost_round=2,
        bootstrap_replicates=None,
    )
    original_rows = result.probability.rows.copy(deep=True)

    segments = summarize_xgboost_aft_probability_by_count_segments(result)

    assert set(segments["count_column"]) == {
        "history_interval_count",
        "user_prior_order_count",
    }
    for count_column in segments["count_column"].unique():
        column_rows = segments.loc[segments["count_column"].eq(count_column)]
        assert column_rows["sample_count"].sum() == len(result.probability.rows)
        assert column_rows["outcome_known_count"].sum() == int(
            result.probability.rows["ipcw_outcome_known"].sum()
        )
    assert segments["validation_fallback_candidate"].dtype == bool
    pd.testing.assert_frame_equal(result.probability.rows, original_rows)

    report = build_xgboost_aft_segment_comparison_report(result, segments)
    markdown = render_xgboost_aft_segment_comparison_report(report)
    assert len(report["segments"]) == len(segments)
    assert "fallback 검토 후보" in report["scope"]
    assert "Train 상수확률 Brier - AFT Brier" in markdown
    json.dumps(report, ensure_ascii=False, allow_nan=False)

    unknown_segments = segments.copy()
    unknown_index = unknown_segments.index[0]
    unknown_segments.loc[unknown_index, "outcome_known_count"] = 0
    for column in (
        "reference_ipcw_brier_score",
        "candidate_ipcw_brier_score",
        "brier_improvement",
        "candidate_expected_calibration_error",
    ):
        unknown_segments.loc[unknown_index, column] = np.nan
    unknown_report = build_xgboost_aft_segment_comparison_report(
        result,
        unknown_segments,
    )
    unknown_markdown = render_xgboost_aft_segment_comparison_report(unknown_report)
    assert "N/A" in unknown_markdown
    json.dumps(unknown_report, ensure_ascii=False, allow_nan=False)


def test_evaluate_xgboost_aft_candidate_can_skip_bootstrap(
    uci_e2e_purchase_events: pd.DataFrame,
) -> None:
    """후보 탐색에서는 공통 데이터를 재사용하고 Bootstrap을 생략합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels, horizon_days=30)

    result = evaluate_xgboost_aft_candidate(
        prepared,
        num_boost_round=5,
        bootstrap_replicates=None,
    )

    assert prepared.horizon_days == 30
    assert prepared.prediction_data.feature_columns == (
        prepared.training_data.feature_columns
    )
    assert result.probability.user_bootstrap is None


def test_reusing_prepared_experiment_does_not_change_shared_data(
    uci_e2e_purchase_events: pd.DataFrame,
) -> None:
    """후보 A·B·A 실행 뒤에도 공통 데이터와 동일 후보 결과를 보존합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels, horizon_days=14)
    validation_before = prepared.validation.copy(deep=True)
    lower_bound_before = prepared.training_data.matrix.get_float_info(
        "label_lower_bound"
    ).copy()
    upper_bound_before = prepared.training_data.matrix.get_float_info(
        "label_upper_bound"
    ).copy()

    first_a = evaluate_xgboost_aft_candidate(
        prepared,
        num_boost_round=2,
        bootstrap_replicates=None,
    )
    evaluate_xgboost_aft_candidate(
        prepared,
        num_boost_round=3,
        bootstrap_replicates=None,
    )
    second_a = evaluate_xgboost_aft_candidate(
        prepared,
        num_boost_round=2,
        bootstrap_replicates=None,
    )

    pd.testing.assert_frame_equal(prepared.validation, validation_before)
    np.testing.assert_array_equal(
        prepared.training_data.matrix.get_float_info("label_lower_bound"),
        lower_bound_before,
    )
    np.testing.assert_array_equal(
        prepared.training_data.matrix.get_float_info("label_upper_bound"),
        upper_bound_before,
    )
    pd.testing.assert_series_equal(
        first_a.validation_predictions,
        second_a.validation_predictions,
    )
    assert first_a.training_summary == second_a.training_summary
    assert first_a.concordance == second_a.concordance
    assert first_a.probability.summary == second_a.probability.summary


def test_wrapper_matches_explicit_prepare_and_evaluate_flow(
    uci_e2e_purchase_events: pd.DataFrame,
) -> None:
    """기존 실행 함수와 새 두 단계 실행이 같은 설정에서 같은 결과를 냅니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels, horizon_days=14)
    explicit = evaluate_xgboost_aft_candidate(
        prepared,
        loss_distribution_scale=2.0,
        num_boost_round=3,
        bootstrap_replicates=None,
    )
    wrapped = run_xgboost_aft_experiment(
        labels,
        horizon_days=14,
        loss_distribution_scale=2.0,
        num_boost_round=3,
        bootstrap_replicates=None,
    )

    assert explicit.split == wrapped.split
    assert explicit.training_summary == wrapped.training_summary
    pd.testing.assert_series_equal(
        explicit.validation_predictions,
        wrapped.validation_predictions,
    )
    assert explicit.concordance == wrapped.concordance
    assert explicit.probability.summary == wrapped.probability.summary
    pd.testing.assert_frame_equal(
        explicit.probability.calibration,
        wrapped.probability.calibration,
    )


def test_compare_xgboost_aft_boosting_rounds_uses_fixed_evaluation_cohort(
    uci_e2e_purchase_events: pd.DataFrame,
) -> None:
    """반복 횟수 후보마다 같은 기준선과 Validation 평가 표본을 사용합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels, horizon_days=14)

    comparison = compare_xgboost_aft_boosting_rounds(
        prepared,
        round_candidates=(2, 3),
    )

    assert comparison["num_boost_round"].tolist() == [2, 3]
    for fixed_column in (
        "ipcw_reference_brier_score",
        "aft_evaluation_sample_count",
        "horizon_days",
        "validation_sample_count",
        "outcome_known_count",
        "ipcw_weight_sum",
        "reference_probability",
    ):
        assert comparison[fixed_column].nunique() == 1
    assert comparison["final_training_aft_nloglik"].notna().all()
    assert comparison["ipcw_brier_score"].notna().all()

    selected = select_xgboost_aft_boosting_round(comparison)
    selected_result = evaluate_xgboost_aft_candidate(
        prepared,
        num_boost_round=selected,
        bootstrap_replicates=None,
    )
    validate_selected_xgboost_aft_result(comparison, selected_result)

    different_round = next(
        round_count
        for round_count in comparison["num_boost_round"]
        if round_count != selected
    )
    different_result = evaluate_xgboost_aft_candidate(
        prepared,
        num_boost_round=int(different_round),
        bootstrap_replicates=None,
    )
    with pytest.raises(RuntimeError, match="반복 횟수"):
        validate_selected_xgboost_aft_result(comparison, different_result)

    report = build_xgboost_aft_round_comparison_report(
        comparison,
        selected_num_boost_round=selected,
    )
    markdown = render_xgboost_aft_round_comparison_report(report)
    assert report["selected_num_boost_round"] == selected
    assert report["horizon_days"] == 14
    assert len(report["candidates"]) == 2
    assert "Validation 선택 후보" in markdown
    assert "Test 성능이나 배포 가능성" in markdown
    assert "14일 horizon" in markdown
    assert "선택 과정의 불확실성" in markdown
    json.dumps(report, ensure_ascii=False, allow_nan=False)

    undefined_skill_report = report.copy()
    undefined_skill_report["candidates"] = [
        {**row, "brier_skill_score": None} for row in report["candidates"]
    ]
    assert "N/A" in render_xgboost_aft_round_comparison_report(undefined_skill_report)

    wrong_selection = next(
        round_count
        for round_count in comparison["num_boost_round"]
        if round_count != selected
    )
    with pytest.raises(ValueError, match="선택 규칙과 다릅니다"):
        build_xgboost_aft_round_comparison_report(
            comparison,
            selected_num_boost_round=int(wrong_selection),
        )


def test_compare_xgboost_aft_boosting_rounds_rejects_invalid_candidates(
    uci_e2e_purchase_events: pd.DataFrame,
) -> None:
    """비어 있거나 양의 정수가 아니거나 중복된 후보는 학습 전에 거절합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels)

    for round_candidates in ((), (0,), (-1,), (2.5,), (True,), (2, 2)):
        with pytest.raises(ValueError):
            compare_xgboost_aft_boosting_rounds(
                prepared,
                round_candidates=round_candidates,
            )


def test_compare_xgboost_aft_loss_distributions_uses_fixed_evaluation_cohort(
    uci_e2e_purchase_events: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """손실분포 후보마다 같은 반복 수·scale·Validation 표본을 사용합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels, horizon_days=14)
    evaluation_calls: list[tuple[object, dict[str, object]]] = []

    def record_evaluation_call(
        prepared_argument: object,
        **kwargs: object,
    ) -> object:
        evaluation_calls.append((prepared_argument, kwargs.copy()))
        return evaluate_xgboost_aft_candidate(prepared_argument, **kwargs)

    monkeypatch.setattr(
        "scripts.run_uci_xgboost_aft.evaluate_xgboost_aft_candidate",
        record_evaluation_call,
    )

    comparison = compare_xgboost_aft_loss_distributions(
        prepared,
        distribution_candidates=("normal", "logistic", "extreme"),
        loss_distribution_scale=1.0,
        num_boost_round=2,
    )

    assert comparison["loss_distribution"].tolist() == [
        "normal",
        "logistic",
        "extreme",
    ]
    assert len(comparison) == 3
    for fixed_column in (
        "loss_distribution_scale",
        "num_boost_round",
        "ipcw_reference_brier_score",
        "aft_evaluation_sample_count",
        "horizon_days",
        "validation_sample_count",
        "outcome_known_count",
        "ipcw_weight_sum",
        "reference_probability",
    ):
        assert comparison[fixed_column].nunique() == 1
    assert comparison["loss_distribution_scale"].iat[0] == 1.0
    assert comparison["num_boost_round"].iat[0] == 2
    assert comparison["ipcw_brier_score"].notna().all()
    assert comparison["ipcw_concordance_index"].notna().all()
    assert [
        call_kwargs["loss_distribution"] for _, call_kwargs in evaluation_calls
    ] == ["normal", "logistic", "extreme"]
    assert all(
        prepared_argument is prepared for prepared_argument, _ in evaluation_calls
    )
    assert all(
        call_kwargs["bootstrap_replicates"] is None
        for _, call_kwargs in evaluation_calls
    )


def test_compare_xgboost_aft_loss_distribution_scales_changes_only_scale(
    uci_e2e_purchase_events: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """같은 logistic 설정에서 scale만 바꾸고 기존 실패 기록 규칙을 재사용합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels, horizon_days=14)
    comparison_calls: list[dict[str, object]] = []
    original_compare = compare_xgboost_aft_loss_distributions

    def record_comparison_call(
        prepared_argument: object,
        **kwargs: object,
    ) -> pd.DataFrame:
        comparison_calls.append(kwargs.copy())
        return original_compare(prepared_argument, **kwargs)

    monkeypatch.setattr(
        "scripts.run_uci_xgboost_aft.compare_xgboost_aft_loss_distributions",
        record_comparison_call,
    )

    comparison = compare_xgboost_aft_loss_distribution_scales(
        prepared,
        loss_distribution="logistic",
        num_boost_round=2,
        scale_candidates=(0.5, 1.0, 2.0),
    )

    assert comparison["loss_distribution_scale"].tolist() == [0.5, 1.0, 2.0]
    assert comparison["loss_distribution"].eq("logistic").all()
    assert comparison["num_boost_round"].eq(2).all()
    assert [call["loss_distribution_scale"] for call in comparison_calls] == [
        0.5,
        1.0,
        2.0,
    ]
    assert all(
        call["distribution_candidates"] == ("logistic",) for call in comparison_calls
    )


@pytest.mark.parametrize(
    "statuses",
    [
        ("success", "failed"),
        ("failed", "success"),
        ("success", "success"),
        ("failed", "failed"),
    ],
)
def test_compare_xgboost_aft_loss_distribution_scales_preserves_schema_without_future_warning(
    uci_e2e_purchase_events: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    statuses: tuple[str, str],
) -> None:
    """성공·실패 순서와 무관하게 null·실제 0과 결과 스키마를 보존합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels)

    def build_comparison_row(
        prepared_argument: object,
        **kwargs: object,
    ) -> pd.DataFrame:
        scale = float(kwargs["loss_distribution_scale"])
        status = statuses[0] if scale == 1.0 else statuses[1]
        success = status == "success"
        return pd.DataFrame(
            [
                {
                    "loss_distribution": "logistic",
                    "loss_distribution_scale": scale,
                    "num_boost_round": 20,
                    "status": status,
                    "failure_reason": None if success else "수치 실패",
                    "prediction_sample_count": 100,
                    "invalid_prediction_count": 0 if success else 80,
                    "nan_count": 0,
                    "positive_infinity_count": 0 if success else 60,
                    "negative_infinity_count": 0,
                    "nonpositive_finite_count": 0 if success else 20,
                    "final_training_aft_nloglik": 2.0 if success else None,
                    "minimum_training_aft_nloglik": 2.0 if success else None,
                    "ipcw_concordance_index": 0.75 if success else None,
                    "ipcw_brier_score": 0.08 if success else None,
                    "ipcw_reference_brier_score": 0.10 if success else None,
                    "brier_skill_score": 0.20 if success else None,
                    "expected_calibration_error": 0.05 if success else None,
                    "maximum_calibration_error": 0.10 if success else None,
                    "weighted_calibration_gap": 0.01 if success else None,
                    "horizon_days": 30,
                    "validation_sample_count": 90 if success else None,
                    "outcome_known_count": 70 if success else None,
                    "ipcw_weight_sum": 90.0 if success else None,
                    "reference_probability": 0.10 if success else None,
                    "aft_evaluation_sample_count": 90 if success else None,
                }
            ]
        )

    monkeypatch.setattr(
        "scripts.run_uci_xgboost_aft.compare_xgboost_aft_loss_distributions",
        build_comparison_row,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        comparison = compare_xgboost_aft_loss_distribution_scales(
            prepared,
            loss_distribution="logistic",
            num_boost_round=20,
            scale_candidates=(1.0, 2.0),
        )

    assert len(comparison.columns) == 26
    assert comparison["status"].tolist() == list(statuses)
    assert comparison["invalid_prediction_count"].tolist() == [
        0 if status == "success" else 80 for status in statuses
    ]
    failed_metrics = comparison.loc[
        comparison["status"].eq("failed"), "ipcw_brier_score"
    ]
    assert failed_metrics.isna().all()


def test_compare_xgboost_aft_loss_distribution_scales_rejects_changed_cohort(
    uci_e2e_purchase_events: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """scale 외 Validation 모집단이 달라지면 비교 결과 생성을 중단합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels)
    original_compare = compare_xgboost_aft_loss_distributions

    def change_one_prediction_count(
        prepared_argument: object,
        **kwargs: object,
    ) -> pd.DataFrame:
        result = original_compare(prepared_argument, **kwargs)
        if kwargs["loss_distribution_scale"] == 2.0:
            result = result.copy()
            result["prediction_sample_count"] += 1
        return result

    monkeypatch.setattr(
        "scripts.run_uci_xgboost_aft.compare_xgboost_aft_loss_distributions",
        change_one_prediction_count,
    )

    with pytest.raises(RuntimeError, match="prediction_sample_count"):
        compare_xgboost_aft_loss_distribution_scales(
            prepared,
            loss_distribution="logistic",
            num_boost_round=2,
            scale_candidates=(1.0, 2.0),
        )


@pytest.mark.parametrize(
    "scale_candidates",
    [(), (1.0, 1.0), (0.0,), (-1.0,), (float("inf"),), (float("nan"),), (True,)],
)
def test_compare_xgboost_aft_loss_distribution_scales_rejects_invalid_candidates_before_training(
    uci_e2e_purchase_events: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    scale_candidates: tuple[object, ...],
) -> None:
    """빈 값·중복·비양수·비유한 scale은 학습 전에 거절합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels)

    def fail_if_training_starts(*args: object, **kwargs: object) -> None:
        raise AssertionError("잘못된 scale 후보를 검사하기 전에 학습이 시작됐습니다.")

    monkeypatch.setattr(
        "scripts.run_uci_xgboost_aft.compare_xgboost_aft_loss_distributions",
        fail_if_training_starts,
    )

    with pytest.raises(ValueError):
        compare_xgboost_aft_loss_distribution_scales(
            prepared,
            loss_distribution="logistic",
            num_boost_round=2,
            scale_candidates=scale_candidates,
        )


def test_select_xgboost_aft_loss_distribution_scale_ignores_failed_candidates() -> None:
    """수치 실패 scale은 지표 선택에서 제외하고 성공 후보만 비교합니다."""
    comparison = pd.DataFrame(
        {
            "loss_distribution": ["logistic"] * 4,
            "loss_distribution_scale": [0.5, 1.0, 2.0, 4.0],
            "num_boost_round": [20] * 4,
            "status": ["failed", "failed", "success", "success"],
            "ipcw_brier_score": [None, None, 0.08, 0.09],
            "ipcw_concordance_index": [None, None, 0.75, 0.80],
            "expected_calibration_error": [None, None, 0.06, 0.04],
            "weighted_calibration_gap": [None, None, 0.05, 0.03],
        }
    )

    selected = select_xgboost_aft_loss_distribution_scale(comparison)

    assert selected == 2.0


def test_select_xgboost_aft_loss_distribution_scale_prefers_default_on_exact_tie() -> (
    None
):
    """모든 지표가 같으면 배수상 기본 scale 1에 가장 가까운 값을 선택합니다."""
    comparison = pd.DataFrame(
        {
            "loss_distribution_scale": [2.0, 1.0, 0.5],
            "status": ["success"] * 3,
            "ipcw_brier_score": [0.08] * 3,
            "ipcw_concordance_index": [0.75] * 3,
            "expected_calibration_error": [0.06] * 3,
            "weighted_calibration_gap": [0.05] * 3,
        }
    )

    selected = select_xgboost_aft_loss_distribution_scale(comparison)

    assert selected == 1.0


def test_build_xgboost_aft_logistic_scale_report_preserves_failures() -> None:
    """scale 보고서는 실패 진단을 null 안전 JSON과 Markdown에 함께 남깁니다."""
    comparison = pd.DataFrame(
        {
            "loss_distribution": ["logistic", "logistic"],
            "loss_distribution_scale": [1.0, 2.0],
            "num_boost_round": [20, 20],
            "horizon_days": [30, 30],
            "status": ["failed", "success"],
            "prediction_sample_count": [100, 100],
            "invalid_prediction_count": [80, 0],
            "nan_count": [0, 0],
            "positive_infinity_count": [60, 0],
            "negative_infinity_count": [0, 0],
            "nonpositive_finite_count": [20, 0],
            "ipcw_brier_score": [None, 0.08],
            "ipcw_concordance_index": [None, 0.75],
            "expected_calibration_error": [None, 0.06],
            "maximum_calibration_error": [None, 0.15],
            "weighted_calibration_gap": [None, 0.05],
        }
    )

    report = build_xgboost_aft_logistic_scale_comparison_report(
        comparison,
        selected_loss_distribution_scale=2.0,
    )
    markdown = render_xgboost_aft_logistic_scale_comparison_report(report)

    assert report["selected_loss_distribution_scale"] == 2.0
    assert report["candidates"][0]["ipcw_brier_score"] is None
    assert "80/100" in markdown
    assert "scale=1.0" in markdown
    json.dumps(report, ensure_ascii=False, allow_nan=False)


def test_build_xgboost_aft_logistic_scale_report_allows_all_failed_diagnostic() -> None:
    """모든 scale 실패도 선택값 없이 기록해 최종 normal 평가를 막지 않습니다."""
    comparison = pd.DataFrame(
        {
            "loss_distribution": ["logistic", "logistic"],
            "loss_distribution_scale": [0.5, 1.0],
            "num_boost_round": [20, 20],
            "horizon_days": [30, 30],
            "status": ["failed", "failed"],
            "prediction_sample_count": [100, 100],
            "invalid_prediction_count": [100, 80],
            "nan_count": [0, 0],
            "positive_infinity_count": [100, 60],
            "negative_infinity_count": [0, 0],
            "nonpositive_finite_count": [0, 20],
            "ipcw_brier_score": [None, None],
            "ipcw_concordance_index": [None, None],
            "expected_calibration_error": [None, None],
            "maximum_calibration_error": [None, None],
            "weighted_calibration_gap": [None, None],
        }
    )

    report = build_xgboost_aft_logistic_scale_comparison_report(
        comparison,
        selected_loss_distribution_scale=None,
    )
    markdown = render_xgboost_aft_logistic_scale_comparison_report(report)

    assert report["selected_loss_distribution_scale"] is None
    assert "선택 scale: `N/A`" in markdown
    assert "100/100" in markdown
    json.dumps(report, ensure_ascii=False, allow_nan=False)


def test_bootstrap_xgboost_aft_candidate_pair_uses_same_users_and_rows(
    uci_e2e_purchase_events: pd.DataFrame,
) -> None:
    """두 AFT 후보의 동일 Validation 행을 사용자 단위로 함께 재표본합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels, horizon_days=14)
    reference = evaluate_xgboost_aft_candidate(
        prepared,
        loss_distribution="normal",
        loss_distribution_scale=1.0,
        num_boost_round=2,
        bootstrap_replicates=None,
    )
    candidate = evaluate_xgboost_aft_candidate(
        prepared,
        loss_distribution="extreme",
        loss_distribution_scale=1.0,
        num_boost_round=2,
        bootstrap_replicates=None,
    )

    bootstrap = bootstrap_xgboost_aft_candidate_pair_by_user(
        reference,
        candidate,
        bootstrap_replicates=20,
        random_seed=7,
    )

    assert bootstrap.summary["bootstrap_replicates"] == 20
    assert bootstrap.summary["random_seed"] == 7
    assert bootstrap.summary["point_reference_brier_score"] == pytest.approx(
        reference.probability.summary["ipcw_brier_score"]
    )
    assert bootstrap.summary["point_candidate_brier_score"] == pytest.approx(
        candidate.probability.summary["ipcw_brier_score"]
    )
    assert len(bootstrap.trials) == 20


@pytest.mark.parametrize("changed_column", ["ipcw_weight", "ipcw_horizon_days"])
def test_bootstrap_xgboost_aft_candidate_pair_rejects_changed_ipcw_condition(
    uci_e2e_purchase_events: pd.DataFrame,
    changed_column: str,
) -> None:
    """두 후보의 평가 시점·검열 가중치가 다르면 Bootstrap 전에 거절합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels, horizon_days=14)
    reference = evaluate_xgboost_aft_candidate(
        prepared,
        loss_distribution="normal",
        num_boost_round=2,
        bootstrap_replicates=None,
    )
    candidate = evaluate_xgboost_aft_candidate(
        prepared,
        loss_distribution="logistic",
        loss_distribution_scale=2.0,
        num_boost_round=2,
        bootstrap_replicates=None,
    )
    candidate.probability.rows.loc[
        candidate.probability.rows.index[0], changed_column
    ] += 1.0

    with pytest.raises(ValueError, match=changed_column):
        bootstrap_xgboost_aft_candidate_pair_by_user(
            reference,
            candidate,
            bootstrap_replicates=20,
        )


def test_bootstrap_xgboost_aft_candidate_pair_rejects_changed_summary_horizon(
    uci_e2e_purchase_events: pd.DataFrame,
) -> None:
    """행이 같아도 요약 평가 horizon이 다르면 쌍 비교하지 않습니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels, horizon_days=14)
    reference = evaluate_xgboost_aft_candidate(
        prepared,
        loss_distribution="normal",
        num_boost_round=2,
        bootstrap_replicates=None,
    )
    candidate = evaluate_xgboost_aft_candidate(
        prepared,
        loss_distribution="logistic",
        loss_distribution_scale=2.0,
        num_boost_round=2,
        bootstrap_replicates=None,
    )
    candidate.probability.summary["horizon_days"] = 30

    with pytest.raises(ValueError, match="요약 평가 horizon"):
        bootstrap_xgboost_aft_candidate_pair_by_user(
            reference,
            candidate,
            bootstrap_replicates=20,
        )


def test_validate_selected_xgboost_aft_scale_result_matches_comparison(
    uci_e2e_purchase_events: pd.DataFrame,
) -> None:
    """paired 비교용 재학습 후보가 scale 선택 당시 지표와 같은지 확인합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels, horizon_days=14)
    comparison = compare_xgboost_aft_loss_distribution_scales(
        prepared,
        loss_distribution="logistic",
        num_boost_round=2,
        scale_candidates=(1.0, 2.0),
    )
    selected_scale = select_xgboost_aft_loss_distribution_scale(comparison)
    selected_result = evaluate_xgboost_aft_candidate(
        prepared,
        loss_distribution="logistic",
        loss_distribution_scale=selected_scale,
        num_boost_round=2,
        bootstrap_replicates=None,
    )

    validate_selected_xgboost_aft_scale_result(comparison, selected_result)

    comparison.loc[
        comparison["loss_distribution_scale"].eq(selected_scale),
        "ipcw_brier_score",
    ] -= 0.001
    with pytest.raises(RuntimeError, match="ipcw_brier_score"):
        validate_selected_xgboost_aft_scale_result(comparison, selected_result)


def test_build_xgboost_aft_candidate_pair_bootstrap_report_records_direction(
    uci_e2e_purchase_events: pd.DataFrame,
) -> None:
    """후보 쌍 보고서는 개선량 방향·요약·반복 원자료를 함께 연결합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels, horizon_days=14)
    reference = evaluate_xgboost_aft_candidate(
        prepared,
        loss_distribution="normal",
        num_boost_round=2,
        bootstrap_replicates=None,
    )
    candidate = evaluate_xgboost_aft_candidate(
        prepared,
        loss_distribution="logistic",
        loss_distribution_scale=2.0,
        num_boost_round=2,
        bootstrap_replicates=None,
    )
    bootstrap = bootstrap_xgboost_aft_candidate_pair_by_user(
        reference,
        candidate,
        bootstrap_replicates=20,
        random_seed=7,
    )

    report = build_xgboost_aft_candidate_pair_bootstrap_report(
        reference,
        candidate,
        bootstrap,
    )
    trials_report = build_xgboost_aft_candidate_pair_bootstrap_trials_report(
        report,
        bootstrap,
    )
    markdown = render_xgboost_aft_candidate_pair_bootstrap_report(report)

    assert report["status"] == "complete"
    assert report["improvement_direction"] == ("reference_brier_minus_candidate_brier")
    assert len(trials_report["trials"]) == 20
    assert "기준 normal Brier - 후보 logistic Brier" in markdown
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    json.dumps(trials_report, ensure_ascii=False, allow_nan=False)


def test_build_xgboost_aft_candidate_pair_bootstrap_report_records_unavailable(
    uci_e2e_purchase_events: pd.DataFrame,
) -> None:
    """logistic이 모두 실패하면 빈 반복 결과와 비교 불가 사유를 기록합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels, horizon_days=14)
    reference = evaluate_xgboost_aft_candidate(
        prepared,
        loss_distribution="normal",
        num_boost_round=2,
        bootstrap_replicates=None,
    )

    report = build_xgboost_aft_candidate_pair_bootstrap_report(
        reference,
        None,
        None,
        unavailable_reason="성공한 logistic scale 후보가 없습니다.",
    )
    trials_report = build_xgboost_aft_candidate_pair_bootstrap_trials_report(
        report,
        None,
    )
    markdown = render_xgboost_aft_candidate_pair_bootstrap_report(report)

    assert report["status"] == "unavailable"
    assert report["candidate"]["loss_distribution_scale"] is None
    assert trials_report["trials"] == []
    assert "비교 불가" in markdown
    json.dumps(report, ensure_ascii=False, allow_nan=False)


def test_compare_xgboost_aft_loss_distributions_rejects_invalid_candidates_before_training(
    uci_e2e_purchase_events: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """빈 값·중복·미지원 후보는 일부 후보를 학습하기 전에 거절합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels)

    def fail_if_training_starts(*args: object, **kwargs: object) -> None:
        raise AssertionError("잘못된 후보를 모두 검사하기 전에 학습이 시작됐습니다.")

    monkeypatch.setattr(
        "scripts.run_uci_xgboost_aft.evaluate_xgboost_aft_candidate",
        fail_if_training_starts,
    )

    for distribution_candidates in (
        (),
        ("normal", "normal"),
        ("normal", "unknown"),
        ("normal", 1),
    ):
        with pytest.raises(ValueError):
            compare_xgboost_aft_loss_distributions(
                prepared,
                distribution_candidates=distribution_candidates,
                num_boost_round=2,
            )


def test_compare_xgboost_aft_loss_distributions_records_numerical_failure(
    uci_e2e_purchase_events: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """수치 예측 실패만 후보 실패로 남기고 정상 후보의 선택을 계속합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels)
    original_evaluate = evaluate_xgboost_aft_candidate

    def fail_logistic_candidate(
        prepared_argument: object,
        **kwargs: object,
    ) -> object:
        if kwargs["loss_distribution"] == "logistic":
            raise XGBoostAFTNumericalPredictionError(
                sample_count=100,
                nan_count=0,
                positive_infinity_count=60,
                negative_infinity_count=0,
                nonpositive_finite_count=20,
            )
        return original_evaluate(prepared_argument, **kwargs)

    monkeypatch.setattr(
        "scripts.run_uci_xgboost_aft.evaluate_xgboost_aft_candidate",
        fail_logistic_candidate,
    )

    comparison = compare_xgboost_aft_loss_distributions(
        prepared,
        num_boost_round=2,
    )

    failed = comparison.loc[comparison["loss_distribution"].eq("logistic")].iloc[0]
    assert failed["status"] == "failed"
    assert failed["prediction_sample_count"] == 100
    assert failed["invalid_prediction_count"] == 80
    assert failed["positive_infinity_count"] == 60
    assert failed["nonpositive_finite_count"] == 20
    assert pd.isna(failed["validation_sample_count"])
    assert pd.isna(failed["ipcw_brier_score"])
    selected = select_xgboost_aft_loss_distribution(comparison)
    assert selected in {
        "normal",
        "extreme",
    }
    report = build_xgboost_aft_distribution_comparison_report(
        comparison,
        selected_loss_distribution=selected,
    )
    markdown = render_xgboost_aft_distribution_comparison_report(report)
    assert "logistic" in markdown
    assert "실패" in markdown
    assert "80/100" in markdown
    assert "+inf 60" in markdown
    json.dumps(report, ensure_ascii=False, allow_nan=False)


def test_compare_xgboost_aft_loss_distributions_propagates_contract_error(
    uci_e2e_purchase_events: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """피처·정렬 같은 일반 계약 오류를 후보 실패로 숨기지 않습니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels)

    def raise_contract_error(*args: object, **kwargs: object) -> None:
        raise XGBoostAFTError("공통 데이터 계약 오류")

    monkeypatch.setattr(
        "scripts.run_uci_xgboost_aft.evaluate_xgboost_aft_candidate",
        raise_contract_error,
    )

    with pytest.raises(XGBoostAFTError, match="공통 데이터 계약 오류"):
        compare_xgboost_aft_loss_distributions(
            prepared,
            num_boost_round=2,
        )


def test_select_xgboost_aft_loss_distribution_rejects_all_failed_candidates(
    uci_e2e_purchase_events: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """모든 분포가 수치 실패하면 임의 후보를 고르지 않고 명확히 거절합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels)

    def fail_every_candidate(*args: object, **kwargs: object) -> None:
        raise XGBoostAFTNumericalPredictionError(
            sample_count=100,
            nan_count=0,
            positive_infinity_count=100,
            negative_infinity_count=0,
            nonpositive_finite_count=0,
        )

    monkeypatch.setattr(
        "scripts.run_uci_xgboost_aft.evaluate_xgboost_aft_candidate",
        fail_every_candidate,
    )
    comparison = compare_xgboost_aft_loss_distributions(
        prepared,
        num_boost_round=2,
    )

    assert comparison["status"].eq("failed").all()
    with pytest.raises(ValueError, match="선택할 AFT 후보 결과가 없습니다"):
        select_xgboost_aft_loss_distribution(comparison)


def test_select_xgboost_aft_loss_distribution_prioritizes_validation_brier() -> None:
    """학습 손실이 낮아도 Validation Brier가 가장 낮은 손실분포를 선택합니다."""
    comparison = pd.DataFrame(
        {
            "loss_distribution": ["normal", "logistic", "extreme"],
            "final_training_aft_nloglik": [2.50, 2.40, 2.30],
            "ipcw_brier_score": [0.08, 0.09, 0.10],
            "ipcw_concordance_index": [0.70, 0.80, 0.90],
            "expected_calibration_error": [0.10, 0.05, 0.01],
            "weighted_calibration_gap": [0.05, 0.02, 0.01],
        }
    )

    selected = select_xgboost_aft_loss_distribution(comparison)

    assert selected == "normal"


def test_select_xgboost_aft_loss_distribution_keeps_normal_on_exact_tie() -> None:
    """모든 Validation 지표가 같으면 행 순서와 무관하게 기준 분포를 유지합니다."""
    comparison = pd.DataFrame(
        {
            "loss_distribution": ["extreme", "normal", "logistic"],
            "ipcw_brier_score": [0.08, 0.08, 0.08],
            "ipcw_concordance_index": [0.80, 0.80, 0.80],
            "expected_calibration_error": [0.05, 0.05, 0.05],
            "weighted_calibration_gap": [0.01, 0.01, 0.01],
        }
    )

    selected = select_xgboost_aft_loss_distribution(comparison)

    assert selected == "normal"
    assert "distribution_preference" not in comparison.columns


def test_select_xgboost_aft_loss_distribution_rejects_invalid_candidates() -> None:
    """누락·중복·미지원 손실분포가 있는 선택표는 명확히 거절합니다."""
    valid = pd.DataFrame(
        {
            "loss_distribution": ["normal", "logistic"],
            "ipcw_brier_score": [0.08, 0.09],
            "ipcw_concordance_index": [0.80, 0.79],
            "expected_calibration_error": [0.05, 0.06],
            "weighted_calibration_gap": [0.01, -0.02],
        }
    )

    with pytest.raises(ValueError, match="필수 컬럼"):
        select_xgboost_aft_loss_distribution(valid.drop(columns="ipcw_brier_score"))
    with pytest.raises(ValueError, match="중복된 손실분포"):
        select_xgboost_aft_loss_distribution(
            valid.assign(loss_distribution=["normal", "normal"])
        )
    with pytest.raises(ValueError, match="normal, logistic, extreme"):
        select_xgboost_aft_loss_distribution(
            valid.assign(loss_distribution=["normal", "unknown"])
        )
    with pytest.raises(ValueError, match="normal, logistic, extreme"):
        select_xgboost_aft_loss_distribution(
            valid.assign(loss_distribution=["normal", 1])
        )


def test_build_xgboost_aft_distribution_comparison_report_records_fixed_conditions(
    uci_e2e_purchase_events: pd.DataFrame,
) -> None:
    """손실분포 선택 결과에 고정 조건과 Validation 한계를 함께 기록합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    prepared = prepare_xgboost_aft_experiment(labels, horizon_days=14)
    comparison = compare_xgboost_aft_loss_distributions(
        prepared,
        num_boost_round=2,
    )
    selected = select_xgboost_aft_loss_distribution(comparison)

    report = build_xgboost_aft_distribution_comparison_report(
        comparison,
        selected_loss_distribution=selected,
    )
    markdown = render_xgboost_aft_distribution_comparison_report(report)

    assert report["selected_loss_distribution"] == selected
    assert report["num_boost_round"] == 2
    assert report["loss_distribution_scale"] == 1.0
    assert report["horizon_days"] == 14
    assert len(report["candidates"]) == 3
    assert "Validation 선택 후보" in markdown
    assert "Test 성능이나 배포 가능성" in markdown
    assert "paired Bootstrap" in markdown
    json.dumps(report, ensure_ascii=False, allow_nan=False)

    selected_result = evaluate_xgboost_aft_candidate(
        prepared,
        bootstrap_replicates=None,
        loss_distribution=selected,
        loss_distribution_scale=1.0,
        num_boost_round=2,
    )
    validate_selected_xgboost_aft_distribution_result(
        comparison,
        selected_result,
    )

    wrong_selection = next(
        distribution
        for distribution in comparison["loss_distribution"]
        if distribution != selected
    )
    with pytest.raises(ValueError, match="선택 규칙과 다릅니다"):
        build_xgboost_aft_distribution_comparison_report(
            comparison,
            selected_loss_distribution=str(wrong_selection),
        )
    wrong_result = evaluate_xgboost_aft_candidate(
        prepared,
        bootstrap_replicates=None,
        loss_distribution=str(wrong_selection),
        loss_distribution_scale=1.0,
        num_boost_round=2,
    )
    with pytest.raises(RuntimeError, match="손실분포"):
        validate_selected_xgboost_aft_distribution_result(
            comparison,
            wrong_result,
        )


def test_select_xgboost_aft_boosting_round_prioritizes_validation_brier() -> None:
    """다른 지표가 좋아도 Validation Brier가 더 낮은 후보를 먼저 선택합니다."""
    comparison = pd.DataFrame(
        {
            "num_boost_round": [5, 20],
            "ipcw_brier_score": [0.09, 0.10],
            "ipcw_concordance_index": [0.60, 0.90],
            "expected_calibration_error": [0.20, 0.05],
            "weighted_calibration_gap": [0.15, 0.01],
        }
    )

    selected = select_xgboost_aft_boosting_round(comparison)

    assert selected == 5


def test_select_xgboost_aft_boosting_round_uses_declared_tie_breakers() -> None:
    """Brier 동률이면 C-index·ECE·절대 편향·낮은 비용 순으로 선택합니다."""
    comparison = pd.DataFrame(
        {
            "num_boost_round": [5, 20, 50, 100, 200],
            "ipcw_brier_score": [0.10] * 5,
            "ipcw_concordance_index": [0.70, 0.80, 0.80, 0.80, 0.80],
            "expected_calibration_error": [0.05, 0.20, 0.10, 0.10, 0.10],
            "weighted_calibration_gap": [0.01, 0.01, 0.10, -0.05, -0.05],
        }
    )

    selected = select_xgboost_aft_boosting_round(comparison)

    assert selected == 100


def test_select_xgboost_aft_boosting_round_uses_absolute_gap_and_lower_cost() -> None:
    """부호가 아닌 편향 크기를 비교하고 완전 동률이면 낮은 반복 수를 선택합니다."""
    common_metrics = {
        "ipcw_brier_score": [0.10, 0.10],
        "ipcw_concordance_index": [0.80, 0.80],
        "expected_calibration_error": [0.05, 0.05],
    }
    gap_comparison = pd.DataFrame(
        {
            "num_boost_round": [50, 100],
            **common_metrics,
            "weighted_calibration_gap": [-0.40, 0.10],
        }
    )
    cost_comparison = pd.DataFrame(
        {
            "num_boost_round": [200, 100],
            **common_metrics,
            "weighted_calibration_gap": [0.10, 0.10],
        }
    )

    assert select_xgboost_aft_boosting_round(gap_comparison) == 100
    assert select_xgboost_aft_boosting_round(gap_comparison.iloc[::-1]) == 100
    assert select_xgboost_aft_boosting_round(cost_comparison) == 100


def test_select_xgboost_aft_boosting_round_rejects_invalid_metrics() -> None:
    """누락·중복·결측 지표가 있는 비교표는 선택 전에 명확히 거절합니다."""
    valid = pd.DataFrame(
        {
            "num_boost_round": [5, 20],
            "ipcw_brier_score": [0.10, 0.11],
            "ipcw_concordance_index": [0.70, 0.71],
            "expected_calibration_error": [0.05, 0.04],
            "weighted_calibration_gap": [0.01, -0.01],
        }
    )

    with pytest.raises(ValueError, match="필수 컬럼"):
        select_xgboost_aft_boosting_round(valid.drop(columns="ipcw_brier_score"))
    with pytest.raises(ValueError, match="중복된 반복 횟수"):
        select_xgboost_aft_boosting_round(valid.assign(num_boost_round=[5, 5]))
    with pytest.raises(ValueError, match="0보다 큰 정수"):
        select_xgboost_aft_boosting_round(valid.assign(num_boost_round=[2.5, 20]))
    with pytest.raises(ValueError, match="유한한 실수"):
        select_xgboost_aft_boosting_round(
            valid.assign(ipcw_brier_score=[0.10, float("nan")])
        )
    with pytest.raises(ValueError, match="유한한 실수"):
        select_xgboost_aft_boosting_round(valid.assign(ipcw_brier_score=[False, True]))
    with pytest.raises(ValueError, match="유한한 실수"):
        select_xgboost_aft_boosting_round(valid.assign(ipcw_brier_score=[-0.10, 0.10]))
    with pytest.raises(ValueError, match="유한한 실수"):
        select_xgboost_aft_boosting_round(
            valid.assign(ipcw_concordance_index=[2.0, 0.70])
        )
