"""AFT·LightGBM rolling cutoff 시간 강건성 실행을 검증합니다."""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from scripts.modeling.rolling_validation import RollingValidationError
from scripts.modeling.samples import (
    build_historical_interval_features,
    make_temporal_split,
)
from scripts.preprocessing.labels import build_same_product_repurchase_labels
from scripts.run_uci_rolling_validation import (
    RollingCutoffEvaluation,
    build_rolling_bootstrap_trials_report,
    build_rolling_cutoff_report,
    evaluate_rolling_cutoff_models,
    render_rolling_cutoff_report,
)


@pytest.fixture
def rolling_evaluation(
    uci_e2e_purchase_events: pd.DataFrame,
) -> RollingCutoffEvaluation:
    """작은 UCI 구매 이력에서 두 cutoff를 빠르게 비교합니다."""
    # 두 과거 Validation 창에 관찰 기간이 충분한 단발 구매를 하나씩 넣어
    # 사건과 미사건이 모두 있는 확률 평가 조건을 만듭니다.
    events = pd.concat(
        [
            uci_e2e_purchase_events,
            pd.DataFrame(
                [
                    {
                        "user_id": "u4",
                        "order_id": "u4-o00",
                        "product_id": "p4",
                        "ordered_at": pd.Timestamp("2026-03-10"),
                    },
                    {
                        "user_id": "u5",
                        "order_id": "u5-o00",
                        "product_id": "p5",
                        "ordered_at": pd.Timestamp("2026-03-28"),
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    labels = build_same_product_repurchase_labels(
        events,
        observation_end_at=pd.Timestamp(events["ordered_at"].max()),
    )
    return evaluate_rolling_cutoff_models(
        labels,
        fold_count=2,
        horizon_days=14,
        calibration_bin_count=4,
        bootstrap_replicates=20,
        bootstrap_random_seed=7,
        num_boost_round=2,
    )


def test_rolling_cutoff_models_keep_test_closed_and_conditions_fixed(
    uci_e2e_purchase_events: pd.DataFrame,
    rolling_evaluation: RollingCutoffEvaluation,
) -> None:
    """모든 fold가 Test 이전의 같은 모델·평가 계약을 사용합니다."""
    folds = rolling_evaluation.folds
    assert folds["fold_id"].tolist() == ["fold_1", "fold_2"]
    assert not folds["test_accessed"].any()
    assert folds["horizon_days"].eq(14).all()
    assert folds["aft_loss_distribution"].eq("normal").all()
    assert folds["aft_loss_distribution_scale"].eq(1.0).all()
    assert folds["aft_num_boost_round"].eq(2).all()
    assert folds["feature_columns"].map(tuple).nunique() == 1
    assert folds["evaluation_sample_count"].gt(0).all()
    assert folds["outcome_known_count"].gt(0).all()
    assert folds["event_within_horizon_count"].gt(0).all()
    assert folds["no_event_within_horizon_count"].gt(0).all()
    assert folds["ipcw_effective_sample_size"].gt(0).all()
    assert (
        folds["point_lower_brier_model"]
        .isin(["xgboost_aft", "lightgbm_probability", "tie"])
        .all()
    )
    assert (
        folds["interval_supported_model"]
        .isin(["xgboost_aft", "lightgbm_probability", "inconclusive"])
        .all()
    )

    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )
    samples = build_historical_interval_features(labels)
    base_split = make_temporal_split(samples)
    assert folds["validation_end_at"].iat[-1] == (
        base_split.validation_end_at.isoformat()
    )
    assert (
        pd.to_datetime(folds["validation_end_at"])
        .le(base_split.validation_end_at)
        .all()
    )

    for fold_id in folds["fold_id"]:
        fold_calibration = rolling_evaluation.calibration.loc[
            rolling_evaluation.calibration["fold_id"].eq(fold_id)
        ]
        assert set(fold_calibration["model_candidate"]) == {
            "xgboost_aft",
            "lightgbm_probability",
        }
        assert (
            len(
                rolling_evaluation.bootstrap_trials.loc[
                    rolling_evaluation.bootstrap_trials["fold_id"].eq(fold_id)
                ]
            )
            == 20
        )


def test_rolling_cutoff_report_is_standard_json_and_explains_scope(
    rolling_evaluation: RollingCutoffEvaluation,
) -> None:
    """요약과 반복 원자료에 NaN 없이 Test 미사용·해석 범위를 남깁니다."""
    report = build_rolling_cutoff_report(rolling_evaluation)
    trials_report = build_rolling_bootstrap_trials_report(rolling_evaluation)
    markdown = render_rolling_cutoff_report(report)

    assert report["evaluation_split"] == "historical_rolling_validation"
    assert report["test_accessed"] is False
    assert report["fold_count"] == 2
    assert report["brier_difference_direction"] == "aft_minus_lightgbm"
    assert report["model_selection_status"] in {
        "aft_supported",
        "lightgbm_supported",
        "direction_only",
        "inconclusive",
    }
    assert "독립 반복 실험이 아닙니다" in report["scope"]
    assert "원래 Test는 사용하지 않았습니다" in report["scope"]
    assert "원래 Test 사용: `아니요`" in markdown
    assert len(trials_report["trials"]) == 40
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    json.dumps(trials_report, ensure_ascii=False, allow_nan=False)


def test_rolling_cutoff_decision_requires_consistent_intervals(
    rolling_evaluation: RollingCutoffEvaluation,
) -> None:
    """점 방향만 같을 때와 구간까지 같은 방향일 때를 구분합니다."""
    direction_only_folds = rolling_evaluation.folds.copy()
    direction_only_folds["brier_difference_aft_minus_lightgbm"] = [0.01, 0.02]
    direction_only_folds["point_lower_brier_model"] = "lightgbm_probability"
    direction_only_folds["interval_supported_model"] = "inconclusive"
    direction_only_folds["bootstrap_lower_95_brier_difference"] = -0.01
    direction_only_folds["bootstrap_upper_95_brier_difference"] = 0.03
    direction_only = replace(rolling_evaluation, folds=direction_only_folds)
    assert (
        build_rolling_cutoff_report(direction_only)["model_selection_status"]
        == "direction_only"
    )

    supported_folds = direction_only_folds.copy()
    supported_folds["interval_supported_model"] = "lightgbm_probability"
    supported_folds["bootstrap_lower_95_brier_difference"] = 0.001
    supported = replace(rolling_evaluation, folds=supported_folds)
    assert (
        build_rolling_cutoff_report(supported)["model_selection_status"]
        == "lightgbm_supported"
    )

    mixed_folds = supported_folds.copy()
    mixed_folds.loc[mixed_folds.index[-1], "point_lower_brier_model"] = "xgboost_aft"
    mixed_folds.loc[
        mixed_folds.index[-1], "brier_difference_aft_minus_lightgbm"
    ] = -0.01
    mixed_folds.loc[
        mixed_folds.index[-1], "bootstrap_lower_95_brier_difference"
    ] = -0.02
    mixed_folds.loc[
        mixed_folds.index[-1], "bootstrap_upper_95_brier_difference"
    ] = -0.001
    mixed_folds.loc[mixed_folds.index[-1], "interval_supported_model"] = "xgboost_aft"
    mixed = replace(rolling_evaluation, folds=mixed_folds)
    assert (
        build_rolling_cutoff_report(mixed)["model_selection_status"] == "inconclusive"
    )

    conflicting_folds = supported_folds.copy()
    conflicting_folds.loc[
        conflicting_folds.index[0], "bootstrap_lower_95_brier_difference"
    ] = -0.03
    conflicting_folds.loc[
        conflicting_folds.index[0], "bootstrap_upper_95_brier_difference"
    ] = -0.001
    conflicting_folds.loc[conflicting_folds.index[0], "interval_supported_model"] = (
        "xgboost_aft"
    )
    conflicting = replace(rolling_evaluation, folds=conflicting_folds)
    assert (
        build_rolling_cutoff_report(conflicting)["model_selection_status"]
        == "inconclusive"
    )


def test_rolling_cutoff_report_rejects_test_result(
    rolling_evaluation: RollingCutoffEvaluation,
) -> None:
    """Test가 한 fold라도 섞이면 보고서를 만들기 전에 실패합니다."""
    changed_folds = rolling_evaluation.folds.copy()
    changed_folds.loc[changed_folds.index[0], "test_accessed"] = True
    changed = replace(rolling_evaluation, folds=changed_folds)

    with pytest.raises(ValueError, match="Test 결과"):
        build_rolling_cutoff_report(changed)

    boundary_changed_folds = rolling_evaluation.folds.copy()
    boundary_changed_folds.loc[boundary_changed_folds.index[0], "validation_end_at"] = (
        "2099-01-01T00:00:00"
    )
    boundary_changed = replace(
        rolling_evaluation,
        folds=boundary_changed_folds,
    )

    with pytest.raises(ValueError, match="Test 구간"):
        build_rolling_cutoff_report(boundary_changed)


def test_rolling_cutoff_report_rejects_derived_direction_mismatch(
    rolling_evaluation: RollingCutoffEvaluation,
) -> None:
    """수치의 부호와 저장된 우위 모델이 다르면 보고서를 거절합니다."""
    changed_folds = rolling_evaluation.folds.copy()
    current = changed_folds.loc[
        changed_folds.index[0],
        "point_lower_brier_model",
    ]
    changed_folds.loc[
        changed_folds.index[0],
        "point_lower_brier_model",
    ] = "xgboost_aft" if current == "lightgbm_probability" else "lightgbm_probability"
    changed = replace(rolling_evaluation, folds=changed_folds)

    with pytest.raises(ValueError, match="점추정 우위"):
        build_rolling_cutoff_report(changed)


def test_rolling_cutoff_models_report_fold_id_for_invalid_evaluation_class(
    uci_e2e_purchase_events: pd.DataFrame,
) -> None:
    """한 fold의 평가 정답이 단일 클래스이면 fold 이름과 함께 실패합니다."""
    labels = build_same_product_repurchase_labels(
        uci_e2e_purchase_events,
        observation_end_at=pd.Timestamp(uci_e2e_purchase_events["ordered_at"].max()),
    )

    with pytest.raises(RollingValidationError, match=r"fold_1.*미재구매"):
        evaluate_rolling_cutoff_models(
            labels,
            fold_count=2,
            horizon_days=14,
            calibration_bin_count=4,
            bootstrap_replicates=5,
            bootstrap_random_seed=7,
            num_boost_round=2,
        )
