"""작은 구매 이력으로 XGBoost AFT Validation 실행 흐름을 검증합니다."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.modeling.samples import (
    assign_temporal_splits,
    build_historical_interval_features,
)
from scripts.preprocessing.labels import build_same_product_repurchase_labels
from scripts.run_uci_xgboost_aft import (
    build_xgboost_aft_bootstrap_trials_report,
    build_xgboost_aft_report,
    build_xgboost_aft_round_comparison_report,
    compare_xgboost_aft_boosting_rounds,
    evaluate_xgboost_aft_candidate,
    prepare_xgboost_aft_experiment,
    render_xgboost_aft_report,
    render_xgboost_aft_round_comparison_report,
    run_xgboost_aft_experiment,
    select_xgboost_aft_boosting_round,
    validate_selected_xgboost_aft_result,
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
    assert report["training"]["trained_until"] == result.split.train_end_at.isoformat()
    assert len(trials_report["trials"]) == 20
    assert "C-index" in markdown
    assert "IPCW 가중 평균 예측확률 / 실제 사건률" in markdown
    assert "기준 설정에서" in markdown
    assert "사용자 단위 Bootstrap" in markdown
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    json.dumps(trials_report, ensure_ascii=False, allow_nan=False)

    undefined_skill_report = report.copy()
    undefined_skill_probability = report["validation_probability"].copy()
    undefined_skill_probability["brier_skill_score"] = None
    undefined_skill_report["validation_probability"] = undefined_skill_probability
    assert "N/A" in render_xgboost_aft_report(undefined_skill_report)


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
