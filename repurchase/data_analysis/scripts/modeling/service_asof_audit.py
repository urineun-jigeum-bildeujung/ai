"""학습 표본 한 건의 과거 피처를 실제 앵커 시점 원천과 대조합니다.

정확성 진단용입니다. 전체 표본에 반복 호출하는 배치 경로가 아닙니다.
"""

from __future__ import annotations

import math

import pandas as pd

from .features import (
    COUNT_FEATURE_COLUMNS,
    MINIMAL_MODEL_FEATURE_COLUMNS,
    ModelFeatureError,
    select_minimal_model_features,
)
from .operational_purchase_inputs import prepare_operational_purchase_inputs_as_of
from .service_samples import ServiceSampleError, _utc, build_current_service_features


def compare_service_sample_features_as_of(
    sample: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    pets: pd.DataFrame,
    status_histories: pd.DataFrame,
    claims: pd.DataFrame,
    claim_items: pd.DataFrame,
) -> pd.DataFrame:
    """선택한 표본 1건의 기존 피처와 당시 복원 피처를 나란히 반환합니다."""
    if len(sample) != 1:
        raise ServiceSampleError(
            "시점별 피처 진단에는 표본을 정확히 1건 선택해야 합니다."
        )
    if not sample.columns.is_unique:
        raise ServiceSampleError("진단 표본에 중복된 열 이름이 있습니다.")
    required = (
        "user_id",
        "pet_id",
        "target_id",
        "order_id",
        "anchor_at",
        *MINIMAL_MODEL_FEATURE_COLUMNS,
    )
    missing = set(required) - set(sample.columns)
    if missing:
        raise ServiceSampleError(f"진단 표본에 필요한 열이 없습니다: {sorted(missing)}")
    selected = sample.iloc[0]
    try:
        selected_features = select_minimal_model_features(sample).iloc[0]
    except ModelFeatureError as error:
        raise ServiceSampleError(
            f"진단 표본의 모델 피처가 잘못됐습니다: {error}"
        ) from error
    if selected[["user_id", "pet_id", "target_id", "order_id"]].isna().any():
        raise ServiceSampleError("진단 표본의 대상 키에 결측값이 있습니다.")
    anchor = _utc(selected["anchor_at"], name="anchor_at")
    if "observation_end_at" in sample.columns and anchor > _utc(
        selected["observation_end_at"], name="observation_end_at"
    ):
        raise ServiceSampleError("앵커가 관측 종료 시각보다 늦습니다.")

    prepared = prepare_operational_purchase_inputs_as_of(
        orders,
        order_items,
        pets,
        status_histories,
        claims,
        claim_items,
        as_of_timestamp=anchor,
    )
    current = build_current_service_features(prepared, as_of_timestamp=anchor)
    matched = current.loc[
        current["user_id"].eq(selected["user_id"])
        & current["pet_id"].eq(selected["pet_id"])
        & current["target_id"].eq(selected["target_id"])
    ]
    if len(matched) != 1:
        raise ServiceSampleError("앵커 시점에 동일 대상의 피처 1건을 찾지 못했습니다.")
    restored = matched.iloc[0]
    if (
        restored["order_id"] != selected["order_id"]
        or _utc(restored["anchor_at"], name="복원 anchor_at") != anchor
    ):
        raise ServiceSampleError("복원된 최신 구매 사건이 선택한 학습 표본과 다릅니다.")

    result = []
    for column in MINIMAL_MODEL_FEATURE_COLUMNS:
        before, after = selected_features[column], restored[column]
        if pd.isna(before) or pd.isna(after):
            if column in COUNT_FEATURE_COLUMNS:
                raise ServiceSampleError(f"{column}에 결측값이 있습니다.")
            changed = not (pd.isna(before) and pd.isna(after))
        else:
            if not math.isfinite(float(before)) or not math.isfinite(float(after)):
                raise ServiceSampleError(f"{column}에 유한하지 않은 피처가 있습니다.")
            if column in COUNT_FEATURE_COLUMNS:
                if float(before) != int(before) or float(after) != int(after):
                    raise ServiceSampleError(f"{column}은 정수여야 합니다.")
                changed = int(before) != int(after)
            else:
                changed = not math.isclose(
                    float(before), float(after), rel_tol=1e-9, abs_tol=1e-9
                )
        result.append(
            {
                "feature": column,
                "final_snapshot_value": before,
                "as_of_value": after,
                "changed": changed,
            }
        )
    return pd.DataFrame(result)
