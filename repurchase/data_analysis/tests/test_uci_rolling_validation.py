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
    add_fold_brier_contributions,
    add_fold_outcome_distribution,
    build_rolling_bootstrap_trials_report,
    build_rolling_concentration_report,
    build_rolling_cutoff_report,
    build_rolling_focus_bootstrap_report,
    build_rolling_focus_bootstrap_trials_report,
    build_rolling_user_composition_report,
    classify_history_irregularity,
    evaluate_rolling_cutoff_models,
    render_rolling_concentration_report,
    render_rolling_cutoff_report,
    render_rolling_focus_bootstrap_report,
    render_rolling_user_composition_report,
    summarize_history_irregularity_cohorts,
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
    concentration_report = build_rolling_concentration_report(rolling_evaluation)
    focus_report = build_rolling_focus_bootstrap_report(rolling_evaluation)
    focus_trials = build_rolling_focus_bootstrap_trials_report(rolling_evaluation)
    markdown = render_rolling_cutoff_report(report)
    concentration_markdown = render_rolling_concentration_report(concentration_report)
    focus_markdown = render_rolling_focus_bootstrap_report(focus_report)

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
    assert "## 14일 확률 성능" in markdown
    assert "원래 Test 사용: `아니요`" in markdown
    assert len(trials_report["trials"]) == 40
    assert report["cohorts"]
    assert concentration_report["test_accessed"] is False
    assert "원래 Test는 사용하지 않았습니다" in concentration_report["scope"]
    assert "재구매 Rolling 구간별 모델 우위 집중도" in concentration_markdown
    assert focus_report["test_accessed"] is False
    assert focus_trials["test_accessed"] is False
    assert "구간 내부" in focus_report["scope"]
    assert "사용자 Bootstrap" in focus_markdown
    assert "uci_repurchase_rolling_focus_bootstrap_trials.json.gz" in focus_markdown
    assert "과거 구매 간격 불규칙성별 사건·오차" in markdown
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    json.dumps(trials_report, ensure_ascii=False, allow_nan=False)
    json.dumps(concentration_report, ensure_ascii=False, allow_nan=False)
    json.dumps(focus_report, ensure_ascii=False, allow_nan=False)
    json.dumps(focus_trials, ensure_ascii=False, allow_nan=False)


def test_rolling_report_formats_cohort_event_rate_as_percent(
    rolling_evaluation: RollingCutoffEvaluation,
) -> None:
    """구간별 가중 사건율도 전체 fold 표와 같은 백분율로 표시합니다."""
    report = build_rolling_cutoff_report(rolling_evaluation)
    irregularity_rows = [
        row
        for row in report["cohorts"]
        if row["count_column"] == "history_relative_mad"
    ]
    assert len(irregularity_rows) >= 2
    irregularity_rows[0]["ipcw_weighted_event_rate"] = 0.25
    irregularity_rows[1]["ipcw_weighted_event_rate"] = None

    markdown = render_rolling_cutoff_report(report)
    lines = markdown.splitlines()
    for row, expected in zip(irregularity_rows[:2], ["25.00%", "N/A"], strict=True):
        prefix = f"| {row['fold_id']} | {row['count_bucket']} |"
        line = next(line for line in lines if line.startswith(prefix))
        assert f"| {expected} |" in line


def test_concentration_report_requires_both_entities_and_matching_cohort() -> None:
    """사용자·상품 중 하나가 없거나 순기여가 원래 구간과 다르면 거절합니다."""
    concentration = pd.DataFrame(
        [
            {
                "fold_id": "fold_1",
                "count_column": "history_interval_count",
                "count_bucket": "8-15",
                "entity_column": entity_column,
                "sample_count": 10,
                "known_sample_count": 8,
                "net_contribution_total": 0.02,
            }
            for entity_column in ("user_id", "product_id")
        ]
    )
    evaluation = RollingCutoffEvaluation(
        folds=pd.DataFrame({"fold_id": ["fold_1"], "test_accessed": [False]}),
        cohorts=pd.DataFrame(
            {
                "fold_id": ["fold_1"],
                "count_column": ["history_interval_count"],
                "count_bucket": ["8-15"],
                "sample_count": [10],
                "outcome_known_count": [8],
                "brier_difference_contribution": [0.02],
                "brier_difference_aft_minus_lightgbm": [0.02],
            }
        ),
        calibration=pd.DataFrame(),
        bootstrap_trials=pd.DataFrame(),
        concentration=concentration,
        focus_bootstrap=pd.DataFrame(),
        focus_bootstrap_trials=pd.DataFrame(),
    )
    assert len(build_rolling_concentration_report(evaluation)["results"]) == 2
    with pytest.raises(ValueError, match="누락"):
        build_rolling_concentration_report(
            replace(evaluation, concentration=concentration.iloc[:1])
        )
    changed = concentration.copy()
    changed.loc[0, "net_contribution_total"] = 0.03
    with pytest.raises(ValueError, match="순기여 합계"):
        build_rolling_concentration_report(replace(evaluation, concentration=changed))


def test_focus_bootstrap_report_matches_existing_segment_and_trials() -> None:
    """구간 점추정과 반복 원자료가 기존 구간 성능과 일치해야 합니다."""
    focus = pd.DataFrame(
        [
            {
                "fold_id": "fold_1",
                "count_column": "history_interval_count",
                "count_bucket": "8-15",
                "sample_count": 10,
                "outcome_known_count": 8,
                "known_user_count": 3,
                "bootstrap_replicates": 3,
                "bootstrap_random_seed": 7,
                "status": "evaluated",
                "point_brier_difference_aft_minus_lightgbm": 0.02,
                "bootstrap_lower_95_brier_difference": -0.0085,
                "bootstrap_upper_95_brier_difference": 0.0295,
                "bootstrap_lightgbm_improvement_rate": 2 / 3,
            }
        ]
    )
    trials = pd.DataFrame(
        {
            "fold_id": ["fold_1"] * 3,
            "count_column": ["history_interval_count"] * 3,
            "count_bucket": ["8-15"] * 3,
            "replicate_index": [0, 1, 2],
            "brier_improvement": [-0.01, 0.02, 0.03],
        }
    )
    evaluation = RollingCutoffEvaluation(
        folds=pd.DataFrame({"fold_id": ["fold_1"], "test_accessed": [False]}),
        cohorts=pd.DataFrame(
            {
                "fold_id": ["fold_1"],
                "count_column": ["history_interval_count"],
                "count_bucket": ["8-15"],
                "sample_count": [10],
                "outcome_known_count": [8],
                "brier_difference_aft_minus_lightgbm": [0.02],
            }
        ),
        calibration=pd.DataFrame(),
        bootstrap_trials=pd.DataFrame(),
        concentration=pd.DataFrame(),
        focus_bootstrap=focus,
        focus_bootstrap_trials=trials,
    )
    report = build_rolling_focus_bootstrap_report(evaluation)
    assert len(report["results"]) == 1
    changed = focus.copy()
    changed.loc[0, "point_brier_difference_aft_minus_lightgbm"] = 0.04
    with pytest.raises(ValueError, match="점추정"):
        build_rolling_focus_bootstrap_report(
            replace(evaluation, focus_bootstrap=changed)
        )
    with pytest.raises(ValueError, match="반복 원자료 수"):
        build_rolling_focus_bootstrap_report(
            replace(evaluation, focus_bootstrap_trials=trials.iloc[:2])
        )
    changed_interval = focus.copy()
    changed_interval.loc[0, "bootstrap_upper_95_brier_difference"] = 0.04
    with pytest.raises(ValueError, match="요약값"):
        build_rolling_focus_bootstrap_report(
            replace(evaluation, focus_bootstrap=changed_interval)
        )


def test_rolling_cohorts_partition_each_fold_without_losing_samples(
    rolling_evaluation: RollingCutoffEvaluation,
) -> None:
    """각 이력 구간을 합하면 같은 fold의 원래 평가 표본이 됩니다."""
    columns = {
        "history_interval_count",
        "user_prior_order_count",
        "product_train_sample_count",
        "history_relative_mad",
    }
    cohorts = rolling_evaluation.cohorts
    assert set(cohorts["count_column"]) == columns
    for fold in rolling_evaluation.folds.itertuples():
        for count_column in columns:
            selected = cohorts.loc[
                cohorts["fold_id"].eq(fold.fold_id)
                & cohorts["count_column"].eq(count_column)
            ]
            assert int(selected["sample_count"].sum()) == fold.evaluation_sample_count
            assert selected["sample_rate"].sum() == pytest.approx(1.0)
            assert selected["outcome_known_count"].le(selected["sample_count"]).all()
            assert selected["event_within_horizon_count"].sum() == (
                fold.event_within_horizon_count
            )
            assert selected["no_event_within_horizon_count"].sum() == (
                fold.no_event_within_horizon_count
            )
            assert selected["known_ipcw_weight_share"].sum() == pytest.approx(1.0)
            assert selected["brier_difference_contribution"].sum() == pytest.approx(
                fold.brier_difference_aft_minus_lightgbm
            )
        irregularity = cohorts.loc[
            cohorts["fold_id"].eq(fold.fold_id)
            & cohorts["count_column"].eq("history_relative_mad")
        ]
        assert irregularity["outcome_known_count"].sum() == fold.outcome_known_count
        assert irregularity["event_within_horizon_count"].sum() == (
            fold.event_within_horizon_count
        )
        assert irregularity["no_event_within_horizon_count"].sum() == (
            fold.no_event_within_horizon_count
        )


def test_irregularity_distinguishes_missing_zero_and_boundary() -> None:
    """계산 불가를 규칙적인 간격 0과 합치지 않고 0.5 경계를 고정합니다."""
    rows = pd.DataFrame({"history_relative_mad": [float("nan"), 0, 0.5, 0.6]})
    assert classify_history_irregularity(rows).tolist() == [
        "unavailable",
        "relative_mad_le_0_5",
        "relative_mad_le_0_5",
        "relative_mad_gt_0_5",
    ]
    for invalid in (-0.1, float("inf"), float("-inf")):
        with pytest.raises(RollingValidationError, match="유한값"):
            classify_history_irregularity(
                pd.DataFrame({"history_relative_mad": [invalid]})
            )


def test_irregularity_event_distribution_preserves_unknown_outcomes() -> None:
    """검열로 정답이 불명인 행은 사건도 미사건도 아니며 표본에는 남깁니다."""
    rows = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "history_relative_mad": [None, 0.2, 0.7],
            "ipcw_horizon_days": [30, 30, 30],
            "ipcw_outcome_known": [True, False, True],
            "ipcw_event_within_horizon": pd.array(
                [True, pd.NA, False], dtype="boolean"
            ),
            "ipcw_weight": [2.0, 0.0, 1.0],
            "reference_predicted_event_probability": [0.8, 0.4, 0.2],
            "candidate_predicted_event_probability": [0.6, 0.3, 0.1],
        }
    )
    labels = classify_history_irregularity(rows)
    cohorts = summarize_history_irregularity_cohorts(rows, bucket_labels=labels)
    cohorts = add_fold_brier_contributions(
        cohorts, rows, count_column="history_relative_mad", bucket_labels=labels
    )
    cohorts = add_fold_outcome_distribution(
        cohorts, rows, count_column="history_relative_mad", bucket_labels=labels
    ).set_index("count_bucket")
    assert cohorts["sample_count"].sum() == 3
    assert cohorts["outcome_known_count"].sum() == 2
    assert cohorts["event_within_horizon_count"].sum() == 1
    assert cohorts["no_event_within_horizon_count"].sum() == 1
    assert cohorts.loc["unavailable", "ipcw_weighted_event_rate"] == 1.0
    assert pd.isna(cohorts.loc["relative_mad_le_0_5", "ipcw_weighted_event_rate"])
    assert cohorts["known_ipcw_weight_share"].sum() == pytest.approx(1)
    assert cohorts["brier_difference_contribution"].sum() == pytest.approx(
        (2 * (0.2**2 - 0.4**2) + (0.2**2 - 0.1**2)) / 3
    )


def test_fold_brier_contributions_use_the_full_fold_denominator() -> None:
    """구간 평균이 아니라 전체 IPCW 분모에 대한 오차 기여량을 계산합니다."""
    paired_rows = pd.DataFrame(
        {
            "history_interval_count": [0, 0, 1, 1],
            "ipcw_outcome_known": [True, True, True, False],
            "ipcw_event_within_horizon": pd.array(
                [True, False, True, pd.NA], dtype="boolean"
            ),
            "ipcw_weight": [1.0, 2.0, 1.0, 0.0],
            "reference_predicted_event_probability": [0.8, 0.3, 0.4, 0.9],
            "candidate_predicted_event_probability": [0.6, 0.2, 0.7, 0.1],
        }
    )
    cohorts = pd.DataFrame({"count_bucket": ["0", "1"]})

    result = add_fold_brier_contributions(
        cohorts, paired_rows, count_column="history_interval_count"
    )

    # 세 정답 확인 행의 가중치 합은 4이고, 검열로 정답을 모르는 행은 0만 기여합니다.
    assert result["known_ipcw_weight_share"].tolist() == pytest.approx([0.75, 0.25])
    assert result["brier_difference_contribution"].tolist() == pytest.approx(
        [-0.005, 0.0675]
    )
    assert result["brier_difference_contribution"].sum() == pytest.approx(0.0625)
    assert "brier_difference_contribution" not in cohorts.columns


def test_rolling_report_rejects_missing_cohort_samples(
    rolling_evaluation: RollingCutoffEvaluation,
) -> None:
    """구간 행이 빠져 평가 모집단이 축소되면 보고서를 만들지 않습니다."""
    cohorts = rolling_evaluation.cohorts.copy()
    selected = cohorts.loc[
        cohorts["fold_id"].eq("fold_1")
        & cohorts["count_column"].eq("history_interval_count")
    ]
    cohorts = cohorts.drop(index=selected.index[0])

    with pytest.raises(ValueError, match="구간 합계"):
        build_rolling_cutoff_report(replace(rolling_evaluation, cohorts=cohorts))


def test_rolling_report_rejects_inconsistent_cohort_contribution(
    rolling_evaluation: RollingCutoffEvaluation,
) -> None:
    """구간 기여량과 전체 Brier 차이가 맞지 않으면 보고서를 거절합니다."""
    cohorts = rolling_evaluation.cohorts.copy()
    cohorts.loc[cohorts.index[0], "brier_difference_contribution"] += 0.01

    with pytest.raises(ValueError, match="Brier 기여량 합계"):
        build_rolling_cutoff_report(replace(rolling_evaluation, cohorts=cohorts))


def test_rolling_report_rejects_irregularity_event_count_mismatch(
    rolling_evaluation: RollingCutoffEvaluation,
) -> None:
    """사건 수가 전체 fold와 맞지 않으면 요약 보고서를 만들지 않습니다."""
    cohorts = rolling_evaluation.cohorts.copy()
    selected = cohorts.index[
        cohorts["count_column"].eq("history_relative_mad")
        & cohorts["fold_id"].eq("fold_1")
    ]
    cohorts.loc[selected[0], "event_within_horizon_count"] += 1
    with pytest.raises(ValueError, match="사건 수 합계"):
        build_rolling_cutoff_report(replace(rolling_evaluation, cohorts=cohorts))


def _cohorts_for_changed_fold_differences(
    evaluation: RollingCutoffEvaluation,
    changed_folds: pd.DataFrame,
) -> pd.DataFrame:
    """판정 분기 테스트의 가상 Brier 차이에 구간 합계도 맞춥니다."""
    cohorts = evaluation.cohorts.copy()
    for original, changed in zip(
        evaluation.folds.itertuples(), changed_folds.itertuples(), strict=True
    ):
        difference_change = (
            changed.brier_difference_aft_minus_lightgbm
            - original.brier_difference_aft_minus_lightgbm
        )
        for count_column in cohorts["count_column"].unique():
            selected = cohorts.loc[
                cohorts["fold_id"].eq(original.fold_id)
                & cohorts["count_column"].eq(count_column)
            ]
            cohorts.loc[selected.index[0], "brier_difference_contribution"] += (
                difference_change
            )
    return cohorts


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
    direction_only = replace(
        rolling_evaluation,
        folds=direction_only_folds,
        cohorts=_cohorts_for_changed_fold_differences(
            rolling_evaluation, direction_only_folds
        ),
    )
    assert (
        build_rolling_cutoff_report(direction_only)["model_selection_status"]
        == "direction_only"
    )

    supported_folds = direction_only_folds.copy()
    supported_folds["interval_supported_model"] = "lightgbm_probability"
    supported_folds["bootstrap_lower_95_brier_difference"] = 0.001
    supported = replace(
        rolling_evaluation,
        folds=supported_folds,
        cohorts=_cohorts_for_changed_fold_differences(
            rolling_evaluation, supported_folds
        ),
    )
    assert (
        build_rolling_cutoff_report(supported)["model_selection_status"]
        == "lightgbm_supported"
    )
    assert "14일 Brier" in build_rolling_cutoff_report(supported)["decision"]

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
    mixed = replace(
        rolling_evaluation,
        folds=mixed_folds,
        cohorts=_cohorts_for_changed_fold_differences(rolling_evaluation, mixed_folds),
    )
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
    conflicting = replace(
        rolling_evaluation,
        folds=conflicting_folds,
        cohorts=_cohorts_for_changed_fold_differences(
            rolling_evaluation, conflicting_folds
        ),
    )
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


def test_rolling_cutoff_markdown_reports_low_ipcw_effective_sample_rate(
    rolling_evaluation: RollingCutoffEvaluation,
) -> None:
    """IPCW 유효 표본 비율이 낮으면 안정적이라는 고정 문구를 쓰지 않습니다."""
    changed_folds = rolling_evaluation.folds.copy()
    changed_folds.loc[
        changed_folds.index[0],
        "ipcw_effective_to_known_sample_rate",
    ] = 0.5
    changed = replace(rolling_evaluation, folds=changed_folds)

    markdown = render_rolling_cutoff_report(build_rolling_cutoff_report(changed))

    assert "최솟값이 50.00%" in markdown
    assert "99% 미만" in markdown
    assert "모든 fold에서 IPCW 유효 표본 수가" not in markdown


def test_rolling_user_composition_report_preserves_original_scores(
    rolling_evaluation: RollingCutoffEvaluation,
) -> None:
    """사용자 집계가 원래 fold Brier와 같고 식별자 없이 직렬화됩니다."""
    report = build_rolling_user_composition_report(rolling_evaluation)
    assert report["test_accessed"] is False
    assert len(report["user_diagnostics"]) == 2
    assert len(report["user_groups"]) == 4
    assert len(report["user_overlaps"]) == 1
    assert len(report["user_bootstrap"]) == 1
    assert "user_id" not in json.dumps(report)
    markdown = render_rolling_user_composition_report(report)
    assert "사용자 구성·중복 진단" in markdown
    assert "fold_1 → fold_2" in markdown


def test_user_composition_report_rejects_missing_fold_pair(
    rolling_evaluation: RollingCutoffEvaluation,
) -> None:
    """보고서 생성 직전에도 fold 쌍 누락을 검출합니다."""
    changed = replace(
        rolling_evaluation,
        user_overlaps=rolling_evaluation.user_overlaps.iloc[0:0],
    )
    with pytest.raises(ValueError, match="fold 쌍"):
        build_rolling_user_composition_report(changed)
