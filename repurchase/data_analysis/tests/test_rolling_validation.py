"""Rolling cutoff 시간 경계와 Test 봉인 조건을 검증합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.maturity_analysis import add_split_survival_observation
from scripts.modeling.rolling_validation import (
    RollingValidationError,
    build_expanding_rolling_splits,
    validate_temporal_split_boundaries,
)
from scripts.modeling.samples import (
    TemporalSplit,
    assign_temporal_splits,
    build_historical_interval_features,
    make_temporal_split,
)
from scripts.preprocessing.labels import build_same_product_repurchase_labels
from scripts.run_uci_xgboost_aft import prepare_xgboost_aft_experiment


def _build_daily_samples() -> pd.DataFrame:
    """100일 관측 범위를 가진 작은 시간 경계 검증용 표본을 만듭니다."""
    anchor_at = pd.date_range("2024-01-01", periods=101, freq="D")
    return pd.DataFrame(
        {
            "anchor_at": anchor_at,
            "next_same_product_at": pd.NaT,
            "event_observed": False,
        }
    )


def test_build_expanding_rolling_splits_keeps_original_test_closed() -> None:
    """40→55, 55→70, 70→85%만 사용하고 마지막 15%는 열지 않습니다."""
    samples = _build_daily_samples()
    base_split = make_temporal_split(samples)

    rolling_splits = build_expanding_rolling_splits(
        samples,
        fold_count=3,
        base_split=base_split,
    )

    assert len(rolling_splits) == 3
    assert [split.train_fraction for split in rolling_splits] == pytest.approx(
        [0.40, 0.55, 0.70]
    )
    assert [split.validation_fraction for split in rolling_splits] == pytest.approx(
        [0.15, 0.15, 0.15]
    )
    assert rolling_splits[-1].train_end_at == base_split.train_end_at
    assert rolling_splits[-1].validation_end_at == base_split.validation_end_at

    base_test_index = samples.index[
        samples["anchor_at"].gt(base_split.validation_end_at)
    ]
    validation_indices = []
    previous_training_index = None
    for split in rolling_splits:
        assigned = assign_temporal_splits(samples, split)
        training_index = assigned.index[assigned["split"].eq("train")]
        validation_index = assigned.index[assigned["split"].eq("validation")]
        used_index = training_index.union(validation_index)

        assert used_index.intersection(base_test_index).empty
        if previous_training_index is not None:
            assert previous_training_index.difference(training_index).empty
        previous_training_index = training_index
        validation_indices.append(validation_index)

    for left_index, right_index in zip(
        validation_indices,
        validation_indices[1:],
        strict=False,
    ):
        assert left_index.intersection(right_index).empty


@pytest.mark.parametrize("fold_count", [True, 0, -1, 1.5, "3"])
def test_build_expanding_rolling_splits_rejects_invalid_fold_count(
    fold_count: object,
) -> None:
    """모호하거나 학습 기간을 만들 수 없는 fold 수를 실행 전에 거절합니다."""
    with pytest.raises(RollingValidationError):
        build_expanding_rolling_splits(
            _build_daily_samples(),
            fold_count=fold_count,  # type: ignore[arg-type]
        )


def test_build_expanding_rolling_splits_rejects_insufficient_history() -> None:
    """고정 Validation 창을 만들 과거 기간이 부족하면 명확히 실패합니다."""
    with pytest.raises(RollingValidationError, match="첫 학습 기간"):
        build_expanding_rolling_splits(
            _build_daily_samples(),
            fold_count=6,
        )


def test_validate_temporal_split_boundaries_rejects_reversed_range() -> None:
    """경계가 같거나 역전된 분할은 표본 배정 전에 거절합니다."""
    malformed = TemporalSplit(
        start_at=pd.Timestamp("2024-01-01"),
        train_end_at=pd.Timestamp("2024-02-01"),
        validation_end_at=pd.Timestamp("2024-02-01"),
        end_at=pd.Timestamp("2024-04-01"),
        train_fraction=0.5,
        validation_fraction=0.0,
        test_fraction=0.5,
    )

    with pytest.raises(RollingValidationError, match="start < train_end"):
        validate_temporal_split_boundaries(malformed)


def test_prepare_xgboost_aft_experiment_uses_supplied_split(
    uci_e2e_purchase_events: pd.DataFrame,
) -> None:
    """외부 cutoff를 받으면 내부 기본 분할로 다시 덮어쓰지 않습니다."""
    events = uci_e2e_purchase_events
    labels = build_same_product_repurchase_labels(
        events,
        observation_end_at=pd.Timestamp(events["ordered_at"].max()),
    )
    samples = build_historical_interval_features(labels)
    custom_split = build_expanding_rolling_splits(samples, fold_count=2)[0]

    prepared = prepare_xgboost_aft_experiment(
        labels,
        horizon_days=14,
        split=custom_split,
    )

    assert prepared.split == custom_split
    assert prepared.training["anchor_at"].le(custom_split.train_end_at).all()
    assert prepared.validation["anchor_at"].gt(custom_split.train_end_at).all()
    assert prepared.validation["anchor_at"].le(custom_split.validation_end_at).all()
    assert prepared.training.index.intersection(prepared.validation.index).empty


def test_rolling_validation_censors_event_after_fold_cutoff() -> None:
    """Fold 종료 뒤의 재구매는 미래 사건이므로 현재 fold에서는 검열합니다."""
    rows = pd.DataFrame(
        {
            "anchor_at": [pd.Timestamp("2024-01-10")],
            "next_same_product_at": [pd.Timestamp("2024-02-20")],
            "event_observed": [True],
            "target_duration_days": [41.0],
        }
    )
    split = TemporalSplit(
        start_at=pd.Timestamp("2024-01-01"),
        train_end_at=pd.Timestamp("2024-01-05"),
        validation_end_at=pd.Timestamp("2024-01-31"),
        end_at=pd.Timestamp("2024-04-01"),
        train_fraction=0.05,
        validation_fraction=0.26,
        test_fraction=0.69,
    )

    assigned = assign_temporal_splits(rows, split)
    validation = assigned.loc[assigned["split"].eq("validation")]
    observed = add_split_survival_observation(validation)

    assert observed["survival_event_observed"].tolist() == [False]
    assert observed["survival_observed_duration_days"].tolist() == [21.0]
