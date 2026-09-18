"""Validation 기반 수축 강도 후보 비교 로직을 검증합니다."""

from __future__ import annotations

from math import log
from statistics import NormalDist

import pandas as pd
import pytest

from scripts.modeling import model_selection
from scripts.modeling.baseline import HierarchicalMedianModel
from scripts.modeling.evaluation import bootstrap_ipcw_brier_pair_difference_by_user
from scripts.modeling.model_selection import (
    LIGHTGBM_FEATURE_SETS,
    build_lightgbm_feature_pair_predictions,
    build_paired_probability_predictions,
    evaluate_ipcw_probability_candidates,
    evaluate_ipcw_shrinkage_candidates,
    evaluate_lightgbm_feature_sets,
    evaluate_lightgbm_probability_candidate,
    evaluate_shrinkage_candidates,
    evaluate_xgboost_aft_ipcw_brier,
    evaluate_xgboost_aft_ipcw_concordance,
    evaluate_xgboost_aft_ipcw_probability,
)
from scripts.modeling.xgboost_aft import (
    build_xgboost_aft_training_data,
    train_xgboost_aft_model,
)


def make_model() -> HierarchicalMedianModel:
    """상품 prior와 전체 fallback을 가진 작은 고정 모델을 만듭니다."""
    return HierarchicalMedianModel(
        trained_until=pd.Timestamp("2026-01-31"),
        global_median_days=20.0,
        global_observation_count=100,
        product_median_days={"p1": 30.0},
        product_observation_counts={"p1": 50},
    )


def make_validation_samples() -> pd.DataFrame:
    """수축이 강할수록 상품 prior에 가까워지는 Validation 표본을 만듭니다."""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "order_id": ["o1", "o2"],
            "product_id": ["p1", "new"],
            "history_median_days": [120.0, None],
            "history_interval_count": [1, 0],
            "target_duration_days": [30.0, 20.0],
        }
    )


def make_ipcw_validation_samples() -> pd.DataFrame:
    """동일한 검열 조건에서 기존 모델과 수축 후보를 비교할 표본을 만듭니다."""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "order_id": ["o1", "o2", "o3", "o4"],
            "product_id": ["p1", "new", "p1", "new"],
            "history_median_days": [12.0, None, 3.0, None],
            "history_interval_count": [1, 0, 2, 0],
            "split": ["validation"] * 4,
            "anchor_at": pd.to_datetime(
                ["2026-08-18", "2026-08-17", "2026-08-16", "2026-08-15"]
            ),
            "split_end_at": pd.to_datetime(["2026-08-20"] * 4),
            "outcome_available_by_split_end": [True, False, True, False],
            "target_duration_days": [2.0, float("nan"), 4.0, float("nan")],
        }
    )


def make_ipcw_probability_samples(split: str) -> pd.DataFrame:
    """Train 학습과 Validation 평가에 공통으로 사용할 작은 확률 표본을 만듭니다."""
    split_end = "2026-01-20" if split == "train" else "2026-02-20"
    anchor_month = "2026-01" if split == "train" else "2026-02"
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "order_id": [f"{split}-o1", f"{split}-o2", f"{split}-o3", f"{split}-o4"],
            "product_id": ["p1", "p2", "p1", "p2"],
            "split": [split] * 4,
            "anchor_at": pd.to_datetime(
                [
                    f"{anchor_month}-18",
                    f"{anchor_month}-17",
                    f"{anchor_month}-16",
                    f"{anchor_month}-15",
                ]
            ),
            "split_end_at": pd.to_datetime([split_end] * 4),
            "outcome_available_by_split_end": [True, False, True, False],
            "target_duration_days": [2.0, float("nan"), 4.0, float("nan")],
            "history_interval_count": [0, 1, 2, 3],
            "history_median_days": [float("nan"), 2.0, 3.0, 4.0],
            "history_relative_mad": [float("nan"), float("nan"), 0.2, 0.1],
            "user_prior_order_count": [0, 2, 3, 4],
        }
    )


def make_probability_pair_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """인덱스는 다르지만 구매 키 순서가 같은 두 모델의 예측표를 만듭니다."""
    weighted = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "order_id": ["o1", "o2", "o3"],
            "product_id": ["p1", "p2", "p1"],
            "ipcw_outcome_known": [True, True, False],
            "ipcw_event_within_horizon": pd.array(
                [True, False, pd.NA], dtype="boolean"
            ),
            "ipcw_weight": [1.2, 1.0, 0.0],
        },
        index=[10, 20, 30],
    )
    reference = weighted[["user_id", "order_id", "product_id"]].copy()
    candidate = reference.copy()
    reference.index = [100, 200, 300]
    candidate.index = [1000, 2000, 3000]
    reference["predicted_event_probability"] = [0.6, 0.3, 0.4]
    candidate["predicted_event_probability"] = [0.7, 0.2, 0.5]
    return weighted, reference, candidate


def make_xgboost_aft_concordance_samples() -> pd.DataFrame:
    """0일 한 건과 순위를 비교할 수 있는 세 건의 AFT 평가 표본을 만듭니다."""
    return pd.DataFrame(
        {
            "user_id": ["u30", "u10", "u20", "u40"],
            "split": ["validation"] * 4,
            "anchor_at": pd.to_datetime(
                ["2026-01-08", "2026-01-10", "2026-01-06", "2026-01-04"]
            ),
            "split_end_at": pd.to_datetime(["2026-01-10"] * 4),
            "outcome_available_by_split_end": pd.array(
                [True, False, False, True],
                dtype="boolean",
            ),
            "target_duration_days": [2.0, float("nan"), float("nan"), 6.0],
        },
        index=[30, 10, 20, 40],
    )


def make_xgboost_aft_training_samples() -> pd.DataFrame:
    """AFT 확률 변환에 사용할 작은 Train 학습 표본을 만듭니다."""
    rows = make_ipcw_probability_samples("train")
    rows["survival_observed_duration_days"] = [2.0, 3.0, 4.0, 5.0]
    rows["survival_event_observed"] = pd.array(
        [True, False, True, False],
        dtype="boolean",
    )
    return rows


def test_evaluate_xgboost_aft_ipcw_concordance_reuses_common_metric() -> None:
    """AFT 예측을 인덱스로 정렬하고 0일 제외 뒤 공통 IPCW C-index를 계산합니다."""
    samples = make_xgboost_aft_concordance_samples()
    predictions = pd.Series(
        [6.0, 4.0, 2.0, 1.0],
        index=[40, 20, 30, 10],
        name="predicted_duration_days",
    )

    result = evaluate_xgboost_aft_ipcw_concordance(
        samples,
        predictions,
        horizon_days=6,
    )

    assert result["source_validation_sample_count"] == 4
    assert result["excluded_zero_duration_count"] == 1
    assert result["aft_evaluation_sample_count"] == 3
    assert result["validation_sample_count"] == 3
    assert result["comparable_pair_count"] == 2
    assert result["ipcw_concordance_index"] == pytest.approx(1.0)


def test_evaluate_xgboost_aft_ipcw_concordance_rejects_unmatched_prediction() -> None:
    """평가 원본에서 한 행이 빠진 AFT 예측을 C-index에 전달하지 않습니다."""
    predictions = pd.Series(
        [2.0, 4.0, 6.0],
        index=[30, 20, 40],
        name="predicted_duration_days",
    )

    with pytest.raises(ValueError, match="인덱스 집합"):
        evaluate_xgboost_aft_ipcw_concordance(
            make_xgboost_aft_concordance_samples(),
            predictions,
            horizon_days=6,
        )


def test_prepare_xgboost_aft_ipcw_excludes_zero_before_weighting() -> None:
    """0일 검열 표본이 남은 AFT 평가 집단의 IPCW 분모를 바꾸지 못하게 합니다."""
    predictions = pd.Series(
        [2.0, 1.0, 4.0, 6.0],
        index=[30, 10, 20, 40],
        name="predicted_duration_days",
    )

    evaluation = model_selection._prepare_xgboost_aft_ipcw_evaluation_rows(
        make_xgboost_aft_concordance_samples(),
        predictions,
        horizon_days=6,
    )

    assert evaluation.source_sample_count == 4
    assert evaluation.excluded_zero_duration_count == 1
    assert evaluation.rows.index.tolist() == [30, 20, 40]
    assert evaluation.rows.loc[40, "ipcw_censoring_survival_probability"] == 0.5
    assert evaluation.rows.loc[40, "ipcw_weight"] == pytest.approx(2.0)


def test_evaluate_xgboost_aft_ipcw_brier_reuses_filtered_cohort() -> None:
    """사건·비사건·조기검열을 구분해 IPCW Brier를 손계산 값과 비교합니다."""
    training_result = train_xgboost_aft_model(
        build_xgboost_aft_training_data(make_xgboost_aft_training_samples())
    )
    predictions = pd.Series(
        [2.0, 1.0, 4.0, 6.0],
        index=[30, 10, 20, 40],
        name="predicted_duration_days",
    )

    result = evaluate_xgboost_aft_ipcw_brier(
        make_xgboost_aft_concordance_samples(),
        training_result,
        predictions,
        horizon_days=5,
        training_reference_probability=0.5,
    )

    event_probability = NormalDist().cdf(log(5.0 / 2.0))
    no_event_probability = NormalDist().cdf(log(5.0 / 6.0))
    expected_brier = (
        (1.0 - event_probability) ** 2 + 2.0 * (0.0 - no_event_probability) ** 2
    ) / 3.0

    assert result["source_validation_sample_count"] == 4
    assert result["excluded_zero_duration_count"] == 1
    assert result["aft_evaluation_sample_count"] == 3
    assert result["validation_sample_count"] == 3
    assert result["outcome_known_count"] == 2
    assert result["ipcw_weight_sum"] == pytest.approx(3.0)
    assert result["ipcw_brier_score"] == pytest.approx(expected_brier)
    assert result["reference_probability"] == 0.5


def test_evaluate_xgboost_aft_ipcw_probability_uses_same_rows_for_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Brier와 Calibration이 같은 AFT 확률·IPCW 평가 행을 사용합니다."""
    training_result = train_xgboost_aft_model(
        build_xgboost_aft_training_data(make_xgboost_aft_training_samples())
    )
    predictions = pd.Series(
        [2.0, 1.0, 4.0, 6.0],
        index=[30, 10, 20, 40],
        name="predicted_duration_days",
    )
    # 알려진 확률을 사용해 Calibration과 Brier의 수동 계산값을 검증합니다.
    monkeypatch.setattr(
        model_selection,
        "calculate_xgboost_aft_event_probability",
        lambda training_result, predicted_duration, *, horizon_days: pd.Series(
            [0.4, 0.1, 0.9],
            index=predicted_duration.index,
            name="predicted_event_probability",
        ),
    )

    result = evaluate_xgboost_aft_ipcw_probability(
        make_xgboost_aft_concordance_samples(),
        training_result,
        predictions,
        horizon_days=5,
        training_reference_probability=0.5,
        calibration_bin_count=2,
        bootstrap_replicates=100,
        bootstrap_random_seed=42,
    )
    brier_only = evaluate_xgboost_aft_ipcw_brier(
        make_xgboost_aft_concordance_samples(),
        training_result,
        predictions,
        horizon_days=5,
        training_reference_probability=0.5,
    )

    summary = result.summary
    calibration = result.calibration
    assert summary["source_validation_sample_count"] == 4
    assert summary["excluded_zero_duration_count"] == 1
    assert summary["aft_evaluation_sample_count"] == 3
    assert summary["outcome_known_count"] == 2
    assert calibration["sample_count"].sum() == 2
    assert calibration["ipcw_weight_sum"].sum() == pytest.approx(3.0)
    assert calibration["ipcw_weight_share"].sum() == pytest.approx(1.0)
    assert summary["ipcw_brier_score"] == pytest.approx(0.66)
    for key in (
        "source_validation_sample_count",
        "excluded_zero_duration_count",
        "aft_evaluation_sample_count",
        "horizon_days",
        "outcome_known_count",
        "ipcw_weight_sum",
        "ipcw_brier_score",
        "reference_probability",
        "ipcw_reference_brier_score",
        "brier_skill_score",
    ):
        assert summary[key] == brier_only[key]
    assert summary["requested_calibration_bin_count"] == 2
    assert summary["expected_calibration_error"] == pytest.approx(0.8)
    assert summary["maximum_calibration_error"] == pytest.approx(0.9)
    assert summary["nonempty_calibration_bin_count"] == 2
    assert result.user_bootstrap is not None
    bootstrap_summary = result.user_bootstrap.summary
    assert bootstrap_summary["bootstrap_replicates"] == 100
    assert bootstrap_summary["user_count"] == 2
    assert bootstrap_summary["point_candidate_brier_score"] == pytest.approx(0.66)
    assert bootstrap_summary["point_reference_brier_score"] == pytest.approx(0.25)
    assert bootstrap_summary["point_brier_improvement"] == pytest.approx(-0.41)
    assert bootstrap_summary["bootstrap_positive_improvement_rate"] == 0.0


def test_evaluate_xgboost_aft_ipcw_probability_skips_optional_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """반복 수가 없으면 사용자 식별자나 Bootstrap 실행을 요구하지 않습니다."""
    training_result = train_xgboost_aft_model(
        build_xgboost_aft_training_data(make_xgboost_aft_training_samples())
    )
    validation_samples = make_xgboost_aft_concordance_samples().drop(columns="user_id")
    predictions = pd.Series(
        [6.0, 4.0, 2.0, 1.0],
        index=[40, 20, 30, 10],
        name="predicted_duration_days",
    )

    def fail_if_called(*args: object, **kwargs: object) -> None:
        pytest.fail("요청하지 않은 사용자 Bootstrap이 실행됐습니다.")

    monkeypatch.setattr(
        model_selection,
        "bootstrap_ipcw_brier_difference_by_user",
        fail_if_called,
    )

    result = evaluate_xgboost_aft_ipcw_probability(
        validation_samples,
        training_result,
        predictions,
        horizon_days=5,
        training_reference_probability=0.5,
        calibration_bin_count=2,
    )

    assert result.user_bootstrap is None


def test_probability_pair_preserves_keys_weights_and_unknown_outcomes() -> None:
    """인덱스가 달라도 같은 구매에 확률을 붙이고 미관측 정답과 원본을 보존합니다."""
    weighted, reference, candidate = make_probability_pair_inputs()
    originals = [rows.copy(deep=True) for rows in (weighted, reference, candidate)]

    paired = build_paired_probability_predictions(weighted, reference, candidate)

    pd.testing.assert_frame_equal(paired.loc[:, weighted.columns], weighted)
    assert paired["reference_predicted_event_probability"].tolist() == [0.6, 0.3, 0.4]
    assert paired["candidate_predicted_event_probability"].tolist() == [0.7, 0.2, 0.5]
    paired.loc[10, "ipcw_weight"] = 9.0
    paired.loc[10, "reference_predicted_event_probability"] = 0.0
    for rows, original in zip((weighted, reference, candidate), originals, strict=True):
        pd.testing.assert_frame_equal(rows, original)


@pytest.mark.parametrize("role", ["reference", "candidate"])
@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("reordered", "순서"),
        ("missing_row", "표본 수"),
        ("duplicate_key", "중복"),
        ("missing_probability", "predicted_event_probability"),
        ("missing_key", "누락"),
    ],
)
def test_probability_pair_rejects_misaligned_predictions(
    role: str, problem: str, message: str
) -> None:
    """어느 모델이든 구매 누락·중복·순서 불일치가 있으면 자동 병합하지 않습니다."""
    weighted, reference, candidate = make_probability_pair_inputs()
    predictions = {"reference": reference, "candidate": candidate}
    rows = predictions[role]
    if problem == "reordered":
        rows = rows.iloc[::-1]
    elif problem == "missing_row":
        rows = rows.iloc[:-1]
    elif problem == "duplicate_key":
        rows = rows.iloc[[0, 0, 2]]
    elif problem == "missing_probability":
        rows = rows.drop(columns="predicted_event_probability")
    else:
        rows = rows.drop(columns="order_id")
    predictions[role] = rows

    with pytest.raises(ValueError, match=message):
        build_paired_probability_predictions(
            weighted,
            predictions["reference"],
            predictions["candidate"],
        )


def test_probability_pair_rejects_matching_missing_keys() -> None:
    """모든 표의 같은 위치에 결측 키가 있어도 유효한 구매 식별자로 인정하지 않습니다."""
    weighted, reference, candidate = make_probability_pair_inputs()
    for rows in (weighted, reference, candidate):
        rows.loc[rows.index[0], "order_id"] = None

    with pytest.raises(ValueError, match="식별자.*결측"):
        build_paired_probability_predictions(weighted, reference, candidate)


def test_probability_pair_rejects_duplicate_probability_columns() -> None:
    """같은 확률 이름이 두 열을 가리키는 모호한 입력을 거절합니다."""
    weighted, reference, candidate = make_probability_pair_inputs()
    candidate = pd.concat(
        [candidate, candidate[["predicted_event_probability"]]], axis=1
    )

    with pytest.raises(ValueError, match="열 이름.*중복"):
        build_paired_probability_predictions(weighted, reference, candidate)


def test_evaluate_shrinkage_candidates_preserves_candidates_and_metrics() -> None:
    """후보 순서를 보존하고 동일 표본의 성능·평균 개인 가중치를 반환합니다."""
    result = evaluate_shrinkage_candidates(
        make_validation_samples(),
        make_model(),
        shrinkage_strengths=(1.0, 4.0, 8.0),
    )

    assert result["shrinkage_strength"].tolist() == [1.0, 4.0, 8.0]
    assert result["sample_count"].tolist() == [2, 2, 2]
    assert result["personal_sample_count"].tolist() == [1, 1, 1]
    assert result["mean_personal_history_weight"].tolist() == pytest.approx(
        [0.5, 0.2, 1 / 9]
    )
    # 이 표본에서는 prior가 정답과 같아 수축을 강하게 할수록 MAE가 감소합니다.
    assert result["mae_days"].is_monotonic_decreasing
    # 개인화 표본이 한 건뿐이므로 상위 5% 선택 시 그 한 건이 전체 오차를 차지합니다.
    assert result["tail_sample_count"].tolist() == [1, 1, 1]
    assert result["tail_absolute_error_days"].tolist() == pytest.approx(
        [45.0, 18.0, 10.0]
    )
    assert result["tail_mae_days"].tolist() == pytest.approx([45.0, 18.0, 10.0])
    assert result["tail_absolute_error_share"].tolist() == [1.0, 1.0, 1.0]
    assert result["tail_late_prediction_rate"].tolist() == [1.0, 1.0, 1.0]
    assert result["fixed_cohort_reference_mae_days"].tolist() == [90.0] * 3
    assert result["fixed_cohort_candidate_mae_days"].tolist() == pytest.approx(
        [45.0, 18.0, 10.0]
    )
    assert result["fixed_cohort_mae_improvement_days"].tolist() == pytest.approx(
        [45.0, 72.0, 80.0]
    )
    assert result["fixed_cohort_improved_sample_rate"].tolist() == [1.0] * 3


def test_evaluate_ipcw_shrinkage_candidates_uses_same_population_and_reference() -> (
    None
):
    """기존 모델과 모든 수축 후보를 같은 표본·검열 가중치에서 비교합니다."""
    evaluation = evaluate_ipcw_shrinkage_candidates(
        make_ipcw_validation_samples(),
        make_model(),
        shrinkage_strengths=(1.0, 4.0),
        horizon_days=4,
    )
    result = evaluation.comparison

    assert result["model_candidate"].tolist() == [
        "hierarchical_median",
        "shrunk_hierarchical_median",
        "shrunk_hierarchical_median",
    ]
    assert result["shrinkage_strength"].isna().tolist() == [True, False, False]
    assert result["validation_sample_count"].tolist() == [4, 4, 4]
    assert result["outcome_known_count"].nunique() == 1
    assert result.iloc[0][
        "ipcw_weighted_balanced_accuracy_difference_vs_reference"
    ] == pytest.approx(0.0)
    assert result.iloc[0][
        "ipcw_concordance_index_difference_vs_reference"
    ] == pytest.approx(0.0)
    assert evaluation.reference_binary_evaluation["validation_sample_count"] == 4
    assert evaluation.reference_concordance_evaluation["validation_sample_count"] == 4


def test_evaluate_ipcw_probability_candidates_uses_train_and_shared_validation() -> (
    None
):
    """Train 확률만 학습하고 모든 후보를 같은 Validation·기준선으로 비교합니다."""
    evaluation = evaluate_ipcw_probability_candidates(
        make_ipcw_probability_samples("train"),
        make_ipcw_probability_samples("validation"),
        product_smoothing_strengths=(1.0, 4.0),
        horizon_days=4,
        calibration_bin_count=2,
        bootstrap_product_smoothing_strength=4.0,
        bootstrap_replicates=100,
        bootstrap_random_seed=42,
    )
    result = evaluation.comparison

    assert result["model_candidate"].tolist() == [
        "global_event_probability",
        "hierarchical_event_probability",
        "hierarchical_event_probability",
    ]
    assert result["product_smoothing_strength"].isna().tolist() == [
        True,
        False,
        False,
    ]
    assert result["evaluation_sample_count"].tolist() == [4, 4, 4]
    assert result["outcome_known_count"].nunique() == 1
    assert result["horizon_days"].tolist() == [4, 4, 4]
    assert result.iloc[0]["product_prediction_rate"] == pytest.approx(0.0)
    assert result.iloc[1:]["product_prediction_rate"].tolist() == [1.0, 1.0]
    assert result.iloc[0]["brier_skill_score"] == pytest.approx(0.0)
    assert result["nonempty_calibration_bin_count"].between(1, 2).all()

    calibration = evaluation.calibration
    assert set(calibration["model_candidate"]) == {
        "global_event_probability",
        "hierarchical_event_probability",
    }
    assert calibration.groupby(
        ["model_candidate", "product_smoothing_strength"],
        dropna=False,
    )["sample_count"].sum().tolist() == [3, 3, 3]
    assert calibration.groupby(
        ["model_candidate", "product_smoothing_strength"],
        dropna=False,
    )["ipcw_weight_share"].sum().tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert evaluation.user_bootstrap is not None
    assert evaluation.user_bootstrap.summary["bootstrap_replicates"] == 100
    assert evaluation.user_bootstrap.summary["user_count"] == 3
    assert len(evaluation.user_bootstrap.trials) == 100


def test_evaluate_ipcw_probability_candidates_rejects_unknown_bootstrap_strength() -> (
    None
):
    """비교하지 않은 수축 강도의 Bootstrap을 요청하면 명확히 거절합니다."""
    with pytest.raises(ValueError, match="후보 집합"):
        evaluate_ipcw_probability_candidates(
            make_ipcw_probability_samples("train"),
            make_ipcw_probability_samples("validation"),
            product_smoothing_strengths=(1.0, 4.0),
            horizon_days=4,
            bootstrap_product_smoothing_strength=8.0,
        )


def test_evaluate_lightgbm_probability_candidate_uses_train_and_validation() -> None:
    """Train으로만 학습한 LightGBM을 동일한 Validation IPCW 기준으로 평가합니다."""
    evaluation = evaluate_lightgbm_probability_candidate(
        make_ipcw_probability_samples("train"),
        make_ipcw_probability_samples("validation"),
        horizon_days=4,
        calibration_bin_count=2,
        bootstrap_reference_product_smoothing_strength=4.0,
        bootstrap_replicates=100,
        bootstrap_random_seed=42,
    )
    result = evaluation.comparison.iloc[0]

    assert result["model_candidate"] == "lightgbm_probability"
    assert pd.isna(result["product_smoothing_strength"])
    assert result["evaluation_sample_count"] == 4
    assert result["outcome_known_count"] == 3
    assert result["horizon_days"] == 4
    assert 0 <= result["ipcw_brier_score"] <= 1
    assert 0 <= result["expected_calibration_error"] <= 1

    calibration = evaluation.calibration
    assert calibration["model_candidate"].eq("lightgbm_probability").all()
    assert calibration["sample_count"].sum() == 3
    assert calibration["ipcw_weight_share"].sum() == pytest.approx(1.0)
    assert evaluation.user_bootstrap is not None
    assert evaluation.user_bootstrap.summary["bootstrap_replicates"] == 100
    assert evaluation.user_bootstrap.summary["user_count"] == 3
    assert len(evaluation.user_bootstrap.trials) == 100


def test_lightgbm_feature_comparison_preserves_rows_targets_and_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """결측이 있는 표본까지 같은 행·정답·가중치로 학습하고 같은 행을 평가합니다."""
    training = make_ipcw_probability_samples("train")
    validation = make_ipcw_probability_samples("validation")
    original_training = training.copy(deep=True)
    original_validation = validation.copy(deep=True)
    training_inputs = []
    prediction_inputs = []
    original_train = model_selection.train_lightgbm_classifier
    original_predict = model_selection.predict_lightgbm_repurchase_probability

    def capture_training(data):
        """실제 학습에 전달되는 표본·정답·가중치를 후보별로 보관합니다."""
        training_inputs.append(data)
        return original_train(data)

    def capture_prediction(model, rows):
        """실제 예측에 전달되는 모든 행의 동일성을 확인하도록 보관합니다."""
        prediction_inputs.append(rows.copy(deep=True))
        return original_predict(model, rows)

    monkeypatch.setattr(model_selection, "train_lightgbm_classifier", capture_training)
    monkeypatch.setattr(
        model_selection, "predict_lightgbm_repurchase_probability", capture_prediction
    )
    result = evaluate_lightgbm_feature_sets(training, validation, horizon_days=4)

    assert result.comparison["feature_set"].tolist() == [
        name for name, _ in LIGHTGBM_FEATURE_SETS
    ]
    assert result.comparison["evaluation_sample_count"].tolist() == [4, 4, 4]
    assert result.comparison["outcome_known_count"].tolist() == [3, 3, 3]
    assert pd.isna(result.comparison.iloc[0]["brier_improvement_vs_previous"])
    assert result.comparison["training_sample_count"].tolist() == [3, 3, 3]
    assert len(training_inputs) == len(prediction_inputs) == 3
    for data, (_, columns), predicted_rows in zip(
        training_inputs, LIGHTGBM_FEATURE_SETS, prediction_inputs, strict=True
    ):
        assert list(data.features.columns) == list(columns)
        assert data.features.index.tolist() == [0, 2, 3]
        pd.testing.assert_series_equal(data.target, training_inputs[0].target)
        pd.testing.assert_series_equal(
            data.sample_weight, training_inputs[0].sample_weight
        )
        pd.testing.assert_frame_equal(predicted_rows, original_validation)
    assert pd.isna(training_inputs[2].features.loc[0, "history_relative_mad"])
    pd.testing.assert_frame_equal(training, original_training)
    pd.testing.assert_frame_equal(validation, original_validation)


def test_lightgbm_feature_pair_shares_samples_and_trains_each_candidate_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B/C의 정답·가중치·행을 고정하고 결측과 미관측 표본도 보존합니다."""
    training = make_ipcw_probability_samples("train")
    validation = make_ipcw_probability_samples("validation")
    originals = [rows.copy(deep=True) for rows in (training, validation)]
    training_inputs = []
    prediction_inputs = []
    prediction_outputs = []
    weight_calls = []
    original_train = model_selection.train_lightgbm_classifier
    original_predict = model_selection.predict_lightgbm_repurchase_probability
    original_weight = model_selection.add_split_ipcw_weights

    def capture_training(data):
        """학습 호출 횟수와 실제 입력을 기록합니다."""
        training_inputs.append(data)
        return original_train(data)

    def capture_prediction(model, rows):
        """예측 대상과 실제 확률을 기록해 반환표와 대조합니다."""
        prediction_inputs.append(rows.copy(deep=True))
        result = original_predict(model, rows)
        prediction_outputs.append(result)
        return result

    def capture_weight(rows, *, horizon_days):
        """가중치를 후보별로 반복 계산하지 않는지 기록합니다."""
        weight_calls.append((rows["split"].iloc[0], horizon_days))
        return original_weight(rows, horizon_days=horizon_days)

    monkeypatch.setattr(model_selection, "train_lightgbm_classifier", capture_training)
    monkeypatch.setattr(
        model_selection, "predict_lightgbm_repurchase_probability", capture_prediction
    )
    monkeypatch.setattr(model_selection, "add_split_ipcw_weights", capture_weight)
    result = build_lightgbm_feature_pair_predictions(
        training,
        validation,
        reference_feature_columns=LIGHTGBM_FEATURE_SETS[1][1],
        candidate_feature_columns=LIGHTGBM_FEATURE_SETS[2][1],
        horizon_days=4,
    )

    assert len(training_inputs) == len(prediction_inputs) == 2
    assert weight_calls == [("train", 4), ("validation", 4)]
    for data, (_, columns), predicted_rows in zip(
        training_inputs, LIGHTGBM_FEATURE_SETS[1:], prediction_inputs, strict=True
    ):
        assert data.horizon_days == 4
        assert tuple(data.features.columns) == columns
        assert data.features.index.tolist() == [0, 2, 3]
        pd.testing.assert_series_equal(data.target, training_inputs[0].target)
        pd.testing.assert_series_equal(
            data.sample_weight, training_inputs[0].sample_weight
        )
        pd.testing.assert_frame_equal(predicted_rows, originals[1])
    assert pd.isna(training_inputs[1].features.loc[0, "history_relative_mad"])
    expected = original_weight(validation, horizon_days=4)
    pd.testing.assert_frame_equal(result.loc[:, expected.columns], expected)
    assert len(result) == 4
    assert result["product_train_outcome_count"].tolist() == [2, 1, 2, 1]
    assert not result.loc[1, "ipcw_outcome_known"]
    assert pd.isna(result.loc[1, "ipcw_event_within_horizon"])
    for role, values in zip(
        ("reference", "candidate"), prediction_outputs, strict=True
    ):
        assert result[f"{role}_predicted_event_probability"].tolist() == values.tolist()
    for rows, original in zip((training, validation), originals, strict=True):
        pd.testing.assert_frame_equal(rows, original)


def test_lightgbm_feature_pair_bootstrap_reuses_predictions_without_retraining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """학습 연결부터 사용자 재표집까지 검증하되 시험용 확률로 역할을 구분합니다."""
    training = make_ipcw_probability_samples("train")
    validation = make_ipcw_probability_samples("validation")
    # 한 사용자의 구매 두 건이 있어도 서로 다른 사용자 두 명으로 세면 안 됩니다.
    validation.loc[2, "user_id"] = "u1"
    training_calls = []
    prediction_calls = []
    original_train = model_selection.train_lightgbm_classifier

    def capture_training(data):
        """실제 학습을 수행하면서 호출 횟수를 기록합니다."""
        training_calls.append(data)
        return original_train(data)

    def supply_test_probabilities(model, rows):
        """두 후보에 다른 확률을 넣어 역할 교환이나 한 후보의 재사용을 탐지합니다."""
        prediction_calls.append(tuple(model.feature_name_))
        values = (
            [0.8, 0.5, 0.3, 0.2]
            if "history_relative_mad" not in model.feature_name_
            else [0.9, 0.5, 0.8, 0.6]
        )
        return pd.Series(values, index=rows.index, dtype="float64")

    monkeypatch.setattr(model_selection, "train_lightgbm_classifier", capture_training)
    monkeypatch.setattr(
        model_selection,
        "predict_lightgbm_repurchase_probability",
        supply_test_probabilities,
    )
    paired = build_lightgbm_feature_pair_predictions(
        training,
        validation,
        reference_feature_columns=LIGHTGBM_FEATURE_SETS[1][1],
        candidate_feature_columns=LIGHTGBM_FEATURE_SETS[2][1],
        horizon_days=4,
    )
    original_paired = paired.copy(deep=True)
    result = bootstrap_ipcw_brier_pair_difference_by_user(
        paired, bootstrap_replicates=100, random_seed=42
    )
    repeated = bootstrap_ipcw_brier_pair_difference_by_user(
        paired, bootstrap_replicates=100, random_seed=42
    )

    # 확인된 정답은 [1, 1, 0]입니다. 공통 가중치로 수계산한 값과 대조합니다.
    weights = paired.loc[[0, 2, 3], "ipcw_weight"].tolist()
    expected_b = sum(w * e for w, e in zip(weights, [0.04, 0.49, 0.04], strict=True))
    expected_c = sum(w * e for w, e in zip(weights, [0.01, 0.04, 0.36], strict=True))
    assert result.summary["point_reference_brier_score"] == pytest.approx(
        expected_b / sum(weights)
    )
    assert result.summary["point_candidate_brier_score"] == pytest.approx(
        expected_c / sum(weights)
    )
    assert result.summary["point_brier_improvement"] == pytest.approx(
        (expected_b - expected_c) / sum(weights)
    )
    assert result.summary["user_count"] == 2
    assert result.summary["outcome_known_count"] == 3
    assert len(result.trials) == 100
    # C가 좋아지는 사용자와 나빠지는 사용자를 모두 포함해 개선을 강제하지 않습니다.
    assert result.trials["brier_improvement"].gt(0).any()
    assert result.trials["brier_improvement"].lt(0).any()
    assert len(training_calls) == len(prediction_calls) == 2
    assert result.summary == repeated.summary
    pd.testing.assert_frame_equal(result.trials, repeated.trials)
    pd.testing.assert_frame_equal(paired, original_paired)


@pytest.mark.parametrize("invalid_input", ["test", "train_split", "candidate_feature"])
def test_lightgbm_feature_pair_rejects_invalid_inputs_before_training(
    monkeypatch: pytest.MonkeyPatch, invalid_input: str
) -> None:
    """Test 유입·Train 오지정·정답 피처 유입은 학습을 시작하기 전에 거절합니다."""
    training = make_ipcw_probability_samples("train")
    validation = make_ipcw_probability_samples("validation")
    candidate_columns = LIGHTGBM_FEATURE_SETS[2][1]
    if invalid_input == "test":
        validation["split"] = "test"
        message = "Validation 표본만"
    elif invalid_input == "train_split":
        training["split"] = "validation"
        message = "Train 표본만"
    else:
        candidate_columns = ("target_duration_days",)
        message = "피처"

    def reject_training(data):
        """잘못된 입력으로 학습에 도달하면 테스트를 실패시킵니다."""
        pytest.fail("입력 검증 전에 학습이 실행됐습니다.")

    monkeypatch.setattr(model_selection, "train_lightgbm_classifier", reject_training)
    with pytest.raises(ValueError, match=message):
        build_lightgbm_feature_pair_predictions(
            training,
            validation,
            reference_feature_columns=LIGHTGBM_FEATURE_SETS[1][1],
            candidate_feature_columns=candidate_columns,
            horizon_days=4,
        )


def test_lightgbm_feature_comparison_rejects_test_split() -> None:
    """피처 선택 실험에 Test 표본을 전달하면 평가 전에 거절합니다."""
    with pytest.raises(ValueError, match="Validation 표본만"):
        evaluate_lightgbm_feature_sets(
            make_ipcw_probability_samples("train"),
            make_ipcw_probability_samples("test"),
        )


@pytest.mark.parametrize(
    "strengths",
    [(), (1.0, 1.0)],
)
def test_evaluate_shrinkage_candidates_rejects_unusable_candidate_set(
    strengths: tuple[float, ...],
) -> None:
    """비어 있거나 중복된 후보 집합으로 불필요한 실험을 실행하지 않습니다."""
    with pytest.raises(ValueError, match="후보|중복"):
        evaluate_shrinkage_candidates(
            make_validation_samples(),
            make_model(),
            shrinkage_strengths=strengths,
        )
