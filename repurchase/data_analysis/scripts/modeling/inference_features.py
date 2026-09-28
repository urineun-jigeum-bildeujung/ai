"""유효 구매 사건으로 현재 재구매 모델의 입력 피처를 생성합니다.

백엔드 주문 상태를 판정하는 모듈은 아닙니다. 취소·반품 수량을 반영해 확정한
구매 사건 전체를 입력받고, 학습에 사용한 과거 간격 계산을 그대로 재사용합니다.
"""

from __future__ import annotations

import pandas as pd

from ..preprocessing.labels import build_same_product_repurchase_labels
from .features import MINIMAL_MODEL_FEATURE_COLUMNS, select_minimal_model_features
from .samples import build_historical_interval_features


class InferenceFeatureError(ValueError):
    """운영용 구매 사건 또는 기준 시각이 피처 생성 계약을 위반할 때 발생합니다."""


VALID_PURCHASE_COLUMNS = ("user_id", "order_id", "product_id", "paid_at")
OUTPUT_ID_COLUMNS = ("user_id", "product_id", "order_id", "anchor_at")


def _as_utc_timestamp(value: object, *, name: str) -> pd.Timestamp:
    """시간대를 명시한 시각만 UTC로 변환해 로컬 시각 오해를 막습니다."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise InferenceFeatureError(f"{name}을 시각으로 읽을 수 없습니다.") from error
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise InferenceFeatureError(f"{name}에는 시간대가 포함돼야 합니다.")
    return timestamp.tz_convert("UTC")


def _validate_valid_purchases(events: pd.DataFrame) -> pd.DataFrame:
    """정규화된 구매 사건의 키·시각과 주문 단위 일관성을 확인합니다."""
    if not events.columns.is_unique:
        raise InferenceFeatureError("구매 사건에 중복된 열 이름이 있습니다.")
    missing = set(VALID_PURCHASE_COLUMNS) - set(events.columns)
    if missing:
        raise InferenceFeatureError(
            f"구매 사건 필수 열이 누락됐습니다: {sorted(missing)}"
        )
    unexpected = set(events.columns) - set(VALID_PURCHASE_COLUMNS)
    if unexpected:
        raise InferenceFeatureError(
            f"유효 구매 사건만 전달해야 합니다. 지원하지 않는 열: {sorted(unexpected)}"
        )
    if events.empty:
        raise InferenceFeatureError("유효 구매 사건이 없습니다.")

    rows = events.loc[:, list(VALID_PURCHASE_COLUMNS)].copy()
    key_columns = list(VALID_PURCHASE_COLUMNS[:3])
    if rows[key_columns].isna().any(axis=None):
        raise InferenceFeatureError("구매 사건 키에 결측값이 있습니다.")
    if (
        rows[key_columns]
        .apply(
            lambda column: column.map(
                lambda value: isinstance(value, str) and not value.strip()
            )
        )
        .any(axis=None)
    ):
        raise InferenceFeatureError("구매 사건 키에 빈 문자열이 있습니다.")
    if rows.duplicated(subset=key_columns).any():
        raise InferenceFeatureError("동일 사용자·주문·상품 사건이 중복됐습니다.")

    # 오프셋이 서로 다른 시각도 UTC에서 비교하되, 시간대 없는 값은 추측하지 않습니다.
    rows["paid_at"] = rows["paid_at"].map(
        lambda value: _as_utc_timestamp(value, name="paid_at")
    )
    orders = rows.groupby("order_id", observed=True, sort=False)
    if (
        orders["user_id"].nunique().gt(1).any()
        or orders["paid_at"].nunique().gt(1).any()
    ):
        raise InferenceFeatureError("한 주문의 사용자 또는 결제 시각이 서로 다릅니다.")
    if rows.duplicated(subset=["user_id", "product_id", "paid_at"]).any():
        raise InferenceFeatureError(
            "동일 사용자·상품의 동시 결제는 구매 순서를 확정할 수 없습니다."
        )
    return rows


def build_current_features_from_valid_purchases(
    events: pd.DataFrame,
    *,
    as_of_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """기준 시각까지의 마지막 사용자·상품 구매별 모델 입력을 만듭니다.

    입력에는 예측 대상만이 아니라 사용자의 유효 주문 전체가 필요합니다.
    그래야 다른 상품 주문도 ``user_prior_order_count``에 포함할 수 있습니다.
    """
    as_of = _as_utc_timestamp(as_of_timestamp, name="as_of_timestamp")
    rows = _validate_valid_purchases(events)
    eligible = rows.loc[rows["paid_at"].le(as_of)].copy()
    if eligible.empty:
        raise InferenceFeatureError("기준 시각까지 확인된 유효 구매가 없습니다.")

    # 라벨은 여기서 예측 정답으로 쓰지 않습니다. 이미 확인된 구매만 정렬해
    # 학습 당시와 동일한 과거 간격 계산 함수에 전달하기 위한 중간 표현입니다.
    label_events = eligible.rename(columns={"paid_at": "ordered_at"})
    labels = build_same_product_repurchase_labels(label_events, as_of)
    historical = build_historical_interval_features(labels)
    latest = (
        historical.sort_values(
            ["user_id", "product_id", "anchor_at", "order_id"],
            kind="stable",
        )
        .drop_duplicates(subset=["user_id", "product_id"], keep="last")
        .reset_index(drop=True)
    )

    # 반환 전에 피처 계약을 다시 검사해 운영 입력이 학습 입력과 달라지지 않게 합니다.
    features = select_minimal_model_features(latest)
    result = latest.loc[:, list(OUTPUT_ID_COLUMNS)].copy()
    result["as_of_timestamp"] = as_of
    result["elapsed_days"] = (as_of - result["anchor_at"]).dt.total_seconds() / 86_400
    for column in MINIMAL_MODEL_FEATURE_COLUMNS:
        result[column] = features[column]
    return result
