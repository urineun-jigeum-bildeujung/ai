"""Test를 열지 않고 과거 여러 시점에서 시간 강건성을 검증합니다.

최종 시간 분할의 Validation 길이를 고정한 채 학습 종료 시점만 과거로
이동합니다. 학습 구간은 점점 늘어나고 Validation 창은 서로 겹치지 않으며,
원래 Test 구간은 어떤 fold의 학습·평가에도 포함하지 않습니다.
"""

from __future__ import annotations

from numbers import Integral

import pandas as pd

from .samples import RepurchaseSampleBuildError, TemporalSplit, make_temporal_split


class RollingValidationError(ValueError):
    """Rolling cutoff 설정이나 시간 경계가 계약을 위반할 때 발생합니다."""


def validate_temporal_split_boundaries(split: TemporalSplit) -> None:
    """시간 분할 경계가 시작부터 종료까지 엄격히 증가하는지 확인합니다."""
    if not (
        split.start_at < split.train_end_at < split.validation_end_at < split.end_at
    ):
        raise RollingValidationError(
            "시간 분할은 start < train_end < validation_end < end 순서여야 합니다."
        )


def build_expanding_rolling_splits(
    samples: pd.DataFrame,
    *,
    fold_count: int = 3,
    base_split: TemporalSplit | None = None,
) -> tuple[TemporalSplit, ...]:
    """확장 학습 구간과 고정 길이 Validation 창을 생성합니다.

    기본 70%·15%·15% 시간 분할과 fold 세 개를 사용하면 학습 종료 시점은
    40%·55%·70%, Validation 종료 시점은 55%·70%·85%가 됩니다. 마지막
    15%는 원래 Test이므로 어떤 fold에도 포함하지 않습니다.
    """
    if isinstance(fold_count, bool) or not isinstance(fold_count, Integral):
        raise RollingValidationError("fold_count는 boolean이 아닌 정수여야 합니다.")
    if fold_count <= 0:
        raise RollingValidationError("fold_count는 1 이상이어야 합니다.")

    try:
        selected_base_split = (
            make_temporal_split(samples) if base_split is None else base_split
        )
    except RepurchaseSampleBuildError as error:
        raise RollingValidationError(str(error)) from error
    validate_temporal_split_boundaries(selected_base_split)

    validation_span = (
        selected_base_split.validation_end_at - selected_base_split.train_end_at
    )
    if validation_span <= pd.Timedelta(0):
        raise RollingValidationError("Validation 기간은 0일보다 길어야 합니다.")

    first_train_end_at = selected_base_split.train_end_at - validation_span * (
        fold_count - 1
    )
    if first_train_end_at <= selected_base_split.start_at:
        raise RollingValidationError("요청한 fold 수에 비해 첫 학습 기간이 부족합니다.")

    observation_span = selected_base_split.end_at - selected_base_split.start_at
    rolling_splits: list[TemporalSplit] = []
    for fold_index in range(fold_count):
        train_end_at = first_train_end_at + validation_span * fold_index
        validation_end_at = train_end_at + validation_span
        train_fraction = float(
            (train_end_at - selected_base_split.start_at) / observation_span
        )
        validation_fraction = float(validation_span / observation_span)
        test_fraction = 1.0 - train_fraction - validation_fraction
        rolling_split = TemporalSplit(
            start_at=selected_base_split.start_at,
            train_end_at=train_end_at,
            validation_end_at=validation_end_at,
            end_at=selected_base_split.end_at,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
        validate_temporal_split_boundaries(rolling_split)
        if rolling_split.validation_end_at > selected_base_split.validation_end_at:
            raise RollingValidationError(
                "Rolling Validation이 원래 Test 구간을 침범했습니다."
            )
        rolling_splits.append(rolling_split)

    for previous, current in zip(
        rolling_splits,
        rolling_splits[1:],
        strict=False,
    ):
        if previous.validation_end_at != current.train_end_at:
            raise RollingValidationError(
                "Rolling Validation 창은 겹치거나 비어 있으면 안 됩니다."
            )

    if rolling_splits[-1].train_end_at != selected_base_split.train_end_at:
        raise RollingValidationError(
            "마지막 rolling fold의 학습 종료 시점이 원래 분할과 다릅니다."
        )
    if rolling_splits[-1].validation_end_at != selected_base_split.validation_end_at:
        raise RollingValidationError(
            "마지막 rolling fold의 Validation 종료 시점이 원래 분할과 다릅니다."
        )
    return tuple(rolling_splits)
