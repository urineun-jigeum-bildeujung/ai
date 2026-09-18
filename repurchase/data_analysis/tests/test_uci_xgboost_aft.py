"""작은 구매 이력으로 XGBoost AFT Validation 실행 흐름을 검증합니다."""

from __future__ import annotations

import json

import pandas as pd

from scripts.modeling.samples import (
    assign_temporal_splits,
    build_historical_interval_features,
)
from scripts.preprocessing.labels import build_same_product_repurchase_labels
from scripts.run_uci_xgboost_aft import (
    build_xgboost_aft_bootstrap_trials_report,
    build_xgboost_aft_report,
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
    assert "사용자 단위 Bootstrap" in markdown
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    json.dumps(trials_report, ensure_ascii=False, allow_nan=False)
