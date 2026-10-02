"""서비스 최종 평가의 모집단·정답 확인 규칙을 데이터 조회 없이 고정합니다.

이 기록은 이미 구현된 시간 분할과 30일 IPCW 처리의 계약입니다.
Test 표본을 생성하거나 평가를 허가하지 않습니다.
"""

from __future__ import annotations

from copy import deepcopy

_POPULATION_POLICY = {
    "schema_version": 1,
    "train_anchor": "anchor_at <= train_end_at",
    "validation_anchor": "train_end_at < anchor_at <= validation_end_at",
    "test_anchor": "validation_end_at < anchor_at <= observation_end_at_assumption",
    "event_within_horizon": "observed_event_and_duration_le_30_days",
    "no_event_by_horizon": "observed_duration_ge_30_days_without_prior_event",
    "unknown_outcome": "censored_before_30_days_kept_with_zero_ipcw_weight",
    "evaluation_rows": "same_all_split_rows_for_both_candidates",
    "aft_training_zero_duration": "exclude_only_from_aft_training",
    "lightgbm_training_unknown_outcome": "exclude_from_lightgbm_training",
}


def service_evaluation_population_policy() -> dict[str, object]:
    """호출자가 결과 JSON을 수정해도 전역 계약이 바뀌지 않게 복사합니다."""
    return deepcopy(_POPULATION_POLICY)
