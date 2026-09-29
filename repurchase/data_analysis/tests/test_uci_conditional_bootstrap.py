"""조건부 확률의 사용자 Bootstrap 요약과 경계 실패를 검증합니다."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from scripts.modeling.evaluation import RepurchaseEvaluationError
from scripts.modeling.landmark_validation import LandmarkValidationError
from scripts.preprocessing.labels import build_same_product_repurchase_labels
from scripts.run_uci_conditional_bootstrap import (
    build_conditional_bootstrap_report,
    render_conditional_bootstrap,
)
from scripts.run_uci_xgboost_aft import prepare_xgboost_aft_experiment


def _prepared_experiment(events: pd.DataFrame):
    """작은 고정 구매 사건에서 공통 AFT 학습·평가 분할을 준비합니다."""
    labels = build_same_product_repurchase_labels(
        events, observation_end_at=pd.Timestamp(events["ordered_at"].max())
    )
    return prepare_xgboost_aft_experiment(labels)


def _events_with_second_known_user(events: pd.DataFrame) -> pd.DataFrame:
    """7일 이후 평가에도 다른 사용자가 남도록 고정 구매 이력을 추가합니다."""
    second_user = events.loc[events["user_id"].eq("u2")].copy()
    second_user["user_id"] = "u4"
    second_user["order_id"] = second_user["order_id"].str.replace(
        "u2-", "u4-", regex=False
    )
    return pd.concat([events, second_user], ignore_index=True)


def test_bootstrap_reuses_point_scores_and_keeps_trials_separate(
    uci_e2e_purchase_events: pd.DataFrame,
) -> None:
    """점추정과 구간은 동일한 평가 행에서 계산하고 반복 자료는 별도 보관합니다."""
    prepared = _prepared_experiment(
        _events_with_second_known_user(uci_e2e_purchase_events)
    )

    report, trials = build_conditional_bootstrap_report(
        prepared, landmark_days=(0, 7), bootstrap_replicates=20, bootstrap_random_seed=7
    )
    repeated, repeated_trials = build_conditional_bootstrap_report(
        prepared, landmark_days=(0, 7), bootstrap_replicates=20, bootstrap_random_seed=7
    )

    assert report == repeated
    assert trials == repeated_trials
    assert report["split"] == "validation"
    assert len(trials) == 40
    assert {row["elapsed_days"] for row in trials} == {0, 7}
    assert all("bootstrap_trials" not in item for item in report["landmarks"])
    for item in report["landmarks"]:
        score = item["brier"]
        bootstrap = item["user_bootstrap"]
        assert bootstrap["point_brier_improvement"] == pytest.approx(
            score["ipcw_reference_brier_score"] - score["ipcw_brier_score"]
        )
        assert bootstrap["outcome_known_count"] == score["outcome_known_count"]
        assert 0 <= bootstrap["bootstrap_positive_improvement_rate"] <= 1
    assert "Test는 평가하지 않았습니다" in render_conditional_bootstrap(report)


def test_bootstrap_refuses_one_validation_user(
    uci_e2e_purchase_events: pd.DataFrame,
) -> None:
    """한 사용자만 남으면 사용자 간 표본 변동을 추정하지 않습니다."""
    prepared = _prepared_experiment(uci_e2e_purchase_events)
    first_user = prepared.validation["user_id"].iloc[0]
    one_user = replace(
        prepared,
        validation=prepared.validation.loc[
            prepared.validation["user_id"].eq(first_user)
        ].copy(),
    )

    with pytest.raises(RepurchaseEvaluationError, match="2명 이상"):
        build_conditional_bootstrap_report(
            one_user, landmark_days=(0,), bootstrap_replicates=10
        )


def test_bootstrap_refuses_missing_validation_user(
    uci_e2e_purchase_events: pd.DataFrame,
) -> None:
    """사용자 ID가 없는 행을 다른 사용자로 묵시적으로 합치지 않습니다."""
    prepared = _prepared_experiment(uci_e2e_purchase_events)
    invalid_validation = prepared.validation.copy()
    invalid_validation.loc[invalid_validation.index[0], "user_id"] = pd.NA
    invalid = replace(prepared, validation=invalid_validation)

    with pytest.raises(LandmarkValidationError, match="식별자"):
        build_conditional_bootstrap_report(
            invalid, landmark_days=(0,), bootstrap_replicates=10
        )


@pytest.mark.parametrize("replicates", [0, -1, True])
def test_bootstrap_refuses_invalid_replicates(
    uci_e2e_purchase_events: pd.DataFrame,
    replicates: object,
) -> None:
    """실험 반복 수를 자동 변환하지 않고 공통 평가 계약으로 거절합니다."""
    prepared = _prepared_experiment(uci_e2e_purchase_events)

    with pytest.raises(RepurchaseEvaluationError, match="반복 수"):
        build_conditional_bootstrap_report(
            prepared, landmark_days=(0,), bootstrap_replicates=replicates
        )
