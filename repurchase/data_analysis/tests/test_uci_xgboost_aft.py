"""작은 구매 이력으로 XGBoost AFT Validation 실행 흐름을 검증합니다."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.modeling.samples import (
    assign_temporal_splits,
    build_historical_interval_features,
)
from scripts.preprocessing.labels import build_same_product_repurchase_labels
from scripts.run_uci_xgboost_aft import (
    build_xgboost_aft_bootstrap_trials_report,
    build_xgboost_aft_report,
    evaluate_xgboost_aft_candidate,
    prepare_xgboost_aft_experiment,
    render_xgboost_aft_report,
    run_xgboost_aft_experiment,
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
    prepared = prepare_xgboost_aft_experiment(labels)
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
