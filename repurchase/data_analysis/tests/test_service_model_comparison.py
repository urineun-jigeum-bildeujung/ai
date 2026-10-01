"""서비스 모델 비교가 동일 평가 행과 검열 가중치를 유지하는지 확인합니다."""

from __future__ import annotations

import pandas as pd
import pytest

import scripts.modeling.service_model_comparison as service_comparison
from scripts.modeling.operational_orders import OperationalOrderError
from scripts.modeling.operational_temporal_split import (
    ServiceTemporalSplit,
    _attach_evaluation_contract,
)
from scripts.modeling.service_model_comparison import (
    _canonical_service_train_order,
    _evaluate_candidate,
    compare_service_aft_lightgbm,
    select_service_aft_boost_rounds,
    summarize_service_brier_attribution,
)


def _weighted_validation() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u3"],
            "survival_observed_duration_days": [5.0, 10.0, 30.0, 35.0],
            "survival_event_observed": [True, True, False, False],
            "ipcw_horizon_days": [30] * 4,
            "ipcw_censoring_survival_probability": [1.0] * 4,
            "ipcw_outcome_known": [True] * 4,
            "ipcw_event_within_horizon": [True, True, False, False],
            "ipcw_weight": [1.0] * 4,
        },
        index=pd.Index([10, 20, 30, 40]),
    )


def test_candidate_reports_same_cohort_for_brier_concordance_calibration() -> None:
    rows = _weighted_validation()
    probability = pd.Series([0.9, 0.8, 0.1, 0.2], index=rows.index)

    summary, calibration = _evaluate_candidate(
        rows,
        probability,
        model_name="candidate",
        reference_probability=0.5,
        calibration_bin_count=10,
    )

    assert summary["validation_sample_count"] == 4
    assert summary["outcome_known_count"] == 4
    assert summary["ipcw_brier_score"] == pytest.approx(0.025)
    assert summary["ipcw_reference_brier_score"] == pytest.approx(0.25)
    assert summary["ipcw_c_index"] == pytest.approx(1.0)
    assert calibration["sample_count"].sum() == 4
    assert calibration["model"].eq("candidate").all()


def test_candidate_rejects_same_labels_in_different_order() -> None:
    rows = _weighted_validation()
    probability = pd.Series([0.8, 0.9, 0.1, 0.2], index=pd.Index([20, 10, 30, 40]))

    with pytest.raises(OperationalOrderError, match="행 순서"):
        _evaluate_candidate(
            rows,
            probability,
            model_name="candidate",
            reference_probability=0.5,
            calibration_bin_count=10,
        )


def test_brier_attribution_reconciles_history_and_product_group() -> None:
    """두 분해 축의 전역 기여도 합계가 같은 전체 Brier 차이로 돌아옵니다."""
    rows = _weighted_validation()
    rows["target_id"] = ["g1", "g1", "g2", "g2"]
    rows["history_interval_count"] = [0, 0, 1, 2]
    rows["ipcw_event_within_horizon"] = [True, False, True, False]
    rows["ipcw_weight"] = [1.0, 2.0, 1.0, 2.0]
    rows["reference_predicted_event_probability"] = [0.2, 0.2, 0.8, 0.8]
    rows["candidate_predicted_event_probability"] = [0.5, 0.1, 0.9, 0.4]

    attribution = summarize_service_brier_attribution(rows)
    history = attribution.loc[
        attribution["segment_kind"].eq("history_interval_count")
    ].set_index("segment_value")
    products = attribution.loc[
        attribution["segment_kind"].eq("product_group")
    ].set_index("segment_value")

    assert history["outcome_known_count"].sum() == 4
    assert products["outcome_known_count"].sum() == 4
    assert history["global_brier_contribution"].sum() == pytest.approx(0.24)
    assert products["global_brier_contribution"].sum() == pytest.approx(0.24)
    assert history.loc["0", "global_brier_contribution"] == pytest.approx(0.075)
    assert history.loc["1", "global_brier_contribution"] == pytest.approx(0.005)
    assert history.loc["2+", "global_brier_contribution"] == pytest.approx(0.16)


@pytest.mark.parametrize("invalid", [None, -1, 1.5, float("inf")])
def test_brier_attribution_rejects_invalid_history_count(invalid: object) -> None:
    rows = _weighted_validation()
    rows["target_id"] = ["g1"] * 4
    rows["history_interval_count"] = [0, 1, 2, invalid]
    rows["reference_predicted_event_probability"] = [0.2] * 4
    rows["candidate_predicted_event_probability"] = [0.3] * 4

    with pytest.raises(OperationalOrderError, match="과거 구매 간격 수"):
        summarize_service_brier_attribution(rows)


def _service_rows(*, split_name: str) -> pd.DataFrame:
    train = split_name == "train"
    count = 8 if train else 6
    anchors = pd.date_range(
        "2026-01-01" if train else "2026-03-02",
        periods=count,
        freq="D",
        tz="UTC",
    )
    split_end = pd.Timestamp(
        "2026-03-01T00:00:00Z" if train else "2026-05-01T00:00:00Z"
    )
    event = [True, True, True, True, False, False, False, False][:count]
    duration = [5.0, 10.0, 15.0, 20.0] + [
        float((split_end - anchor).days) for anchor in anchors[4:]
    ]
    rows = pd.DataFrame(
        {
            "user_id": [f"u{i // 2}" for i in range(count)],
            "pet_id": [f"p{i // 2}" for i in range(count)],
            "target_id": [f"g{i % 2}" for i in range(count)],
            "order_id": [f"o{i}" for i in range(count)],
            "anchor_at": anchors,
            "duration_days": duration,
            "event_observed": event,
            "is_right_censored": [not value for value in event],
            "feature_generation_version": [2] * count,
            "history_interval_count": list(range(count)),
            "history_median_days": [float(value) for value in range(count)],
            "history_relative_mad": [0.0] * count,
            "user_prior_order_count": list(range(count)),
        },
        index=pd.Index(range(100, 100 + count) if train else range(200, 200 + count)),
    )
    return _attach_evaluation_contract(
        rows, split_name=split_name, split_end_at=split_end
    )


def test_aft_round_selection_uses_inner_validation_only() -> None:
    """후보 모두 같은 내부 정답 확인 행으로 점수를 매깁니다."""
    split = ServiceTemporalSplit(
        train=_service_rows(split_name="train"),
        validation=_service_rows(split_name="validation"),
    )

    result = select_service_aft_boost_rounds(split, candidate_rounds=(3, 2))

    assert result.candidates["num_boost_round"].tolist() == [2, 3]
    assert result.candidates["validation_sample_count"].nunique() == 1
    assert result.candidates["outcome_known_count"].nunique() == 1
    assert result.selected_rounds == int(
        result.candidates.sort_values(["ipcw_brier_score", "num_boost_round"])[
            "num_boost_round"
        ].iloc[0]
    )


def test_aft_round_selection_tie_prefers_fewer_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """동일 점수일 때 실행 순서에 관계없이 단순한 후보를 선택합니다."""
    split = ServiceTemporalSplit(
        train=_service_rows(split_name="train"),
        validation=_service_rows(split_name="validation"),
    )

    def equal_brier(*args: object, **kwargs: object) -> dict[str, int | float]:
        return {
            "ipcw_brier_score": 0.25,
            "validation_sample_count": 6,
            "outcome_known_count": 6,
        }

    monkeypatch.setattr(service_comparison, "evaluate_ipcw_brier_score", equal_brier)

    result = select_service_aft_boost_rounds(split, candidate_rounds=(3, 2))

    assert result.selected_rounds == 2


def test_aft_round_selection_prefers_lower_brier_over_fewer_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """동점이 아니라면 반복 횟수가 큰 후보라도 낮은 Brier를 선택합니다."""
    split = ServiceTemporalSplit(
        train=_service_rows(split_name="train"),
        validation=_service_rows(split_name="validation"),
    )
    scores = iter((0.3, 0.2))

    def candidate_brier(*args: object, **kwargs: object) -> dict[str, int | float]:
        return {
            "ipcw_brier_score": next(scores),
            "validation_sample_count": 6,
            "outcome_known_count": 6,
        }

    monkeypatch.setattr(
        service_comparison, "evaluate_ipcw_brier_score", candidate_brier
    )

    result = select_service_aft_boost_rounds(split, candidate_rounds=(3, 2))

    assert result.selected_rounds == 3


@pytest.mark.parametrize("candidates", [(1,), (0, 2), (2, 2), (2, 1.5)])
def test_aft_round_selection_rejects_invalid_candidates(
    candidates: tuple[object, ...],
) -> None:
    split = ServiceTemporalSplit(
        train=_service_rows(split_name="train"),
        validation=_service_rows(split_name="validation"),
    )

    with pytest.raises(OperationalOrderError, match="반복 횟수 후보"):
        select_service_aft_boost_rounds(split, candidate_rounds=candidates)


def test_aft_round_selection_rejects_overlapping_inner_anchors() -> None:
    """내부 검증 앵커가 내부 학습 컷 이전이면 평가를 시작하지 않습니다."""
    validation = _service_rows(split_name="validation")
    validation.loc[validation.index[0], "anchor_at"] = pd.Timestamp(
        "2026-02-01T00:00:00Z"
    )
    split = ServiceTemporalSplit(
        train=_service_rows(split_name="train"), validation=validation
    )

    with pytest.raises(OperationalOrderError, match="내부 시간 분할"):
        select_service_aft_boost_rounds(split, candidate_rounds=(2, 3))


def test_comparison_trains_both_models_and_preserves_validation_count() -> None:
    split = ServiceTemporalSplit(
        train=_service_rows(split_name="train"),
        validation=_service_rows(split_name="validation"),
    )
    comparison = compare_service_aft_lightgbm(
        split,
        aft_boost_rounds=2,
        bootstrap_replicates=20,
    )

    assert comparison.summary["model"].tolist() == ["xgboost_aft", "lightgbm"]
    assert comparison.summary["validation_sample_count"].tolist() == [6, 6]
    assert comparison.summary["outcome_known_count"].tolist() == [6, 6]
    assert comparison.paired_bootstrap.summary["outcome_known_count"] == 6
    assert comparison.calibration.groupby("model")["sample_count"].sum().to_dict() == {
        "xgboost_aft": 6,
        "lightgbm": 6,
    }
    for segment_kind in ("history_interval_count", "product_group"):
        contribution = comparison.brier_attribution.loc[
            comparison.brier_attribution["segment_kind"].eq(segment_kind),
            "global_brier_contribution",
        ].sum()
        assert contribution == pytest.approx(
            comparison.paired_bootstrap.summary["point_brier_improvement"]
        )

    shuffled_train = split.train.sample(frac=1, random_state=7).copy()
    repeated = compare_service_aft_lightgbm(
        ServiceTemporalSplit(train=shuffled_train, validation=split.validation),
        aft_boost_rounds=2,
        bootstrap_replicates=20,
    )
    pd.testing.assert_frame_equal(comparison.summary, repeated.summary)


def test_training_order_is_stable_across_integer_and_string_ids() -> None:
    rows = _service_rows(split_name="train")
    rows["order_id"] = [11, 2, 3, 4, 5, 6, 7, 8]
    rows["anchor_at"] = pd.Timestamp("2026-01-01T00:00:00Z")
    numeric = _canonical_service_train_order(rows)
    string = rows.sample(frac=1, random_state=7).copy()
    string["order_id"] = string["order_id"].astype("string")
    string = _canonical_service_train_order(string)

    assert numeric.index.tolist() == string.index.tolist()
    assert numeric["order_id"].astype("string").tolist() == string["order_id"].tolist()


def test_training_order_rejects_duplicate_event_key() -> None:
    rows = _service_rows(split_name="train")
    rows.loc[101, ["anchor_at", "user_id", "pet_id", "target_id", "order_id"]] = (
        rows.loc[100, ["anchor_at", "user_id", "pet_id", "target_id", "order_id"]]
    )

    with pytest.raises(OperationalOrderError, match="사건 키"):
        _canonical_service_train_order(rows)
