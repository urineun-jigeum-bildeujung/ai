"""유효 구매 사건 구간을 관측 종료 시점의 서비스 라벨로 연결합니다."""

from __future__ import annotations

import pandas as pd

from .operational_event_intervals import (
    OperationalEventIntervals,
    select_operational_events_as_of,
)
from .operational_orders import OperationalOrderError, _require_columns, _require_keys
from .operational_purchase_inputs import OperationalPurchaseInputs
from .service_samples import build_service_repurchase_samples

SERVICE_LABEL_COLUMNS = (
    "user_id",
    "pet_id",
    "target_id",
    "order_id",
    "anchor_at",
    "observation_end_at",
    "next_order_id",
    "next_same_target_at",
    "event_observed",
    "is_right_censored",
    "duration_days",
    "has_zero_day_followup",
)


def prepare_active_service_purchase_inputs(
    event_intervals: OperationalEventIntervals,
    orders: pd.DataFrame,
    *,
    as_of_timestamp: pd.Timestamp,
) -> OperationalPurchaseInputs:
    """학습 라벨과 운영 피처가 공유하는 기준 시점의 유효 구매 사건입니다."""
    _require_columns(orders, ("order_id", "user_id", "paid_at"), "orders")
    _require_keys(orders, ("order_id", "user_id"), "orders")
    if orders["order_id"].duplicated().any():
        raise OperationalOrderError("orders.order_id가 중복됐습니다.")
    active = select_operational_events_as_of(
        event_intervals, as_of_timestamp=as_of_timestamp
    )
    paid = orders.loc[:, ["order_id", "user_id", "paid_at"]].copy()

    all_events = active.user_orders.loc[:, ["user_id", "order_id"]].merge(
        paid,
        on=["user_id", "order_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if all_events["_merge"].ne("both").any():
        raise OperationalOrderError(
            "유효 사용자 주문에 연결되지 않는 결제 주문이 있습니다."
        )
    all_events = all_events.drop(columns="_merge")

    pet_events = active.pet_targets.loc[
        :, ["user_id", "pet_id", "product_group_id_snapshot", "order_id"]
    ].rename(columns={"product_group_id_snapshot": "target_id"})
    pet_events = pet_events.merge(
        paid,
        on=["user_id", "order_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if pet_events["_merge"].ne("both").any():
        raise OperationalOrderError(
            "유효 대상 사건에 연결되지 않는 결제 주문이 있습니다."
        )
    pet_events = pet_events.drop(columns="_merge")
    linked = pet_events.loc[:, ["user_id", "order_id"]].merge(
        all_events.loc[:, ["user_id", "order_id"]],
        on=["user_id", "order_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if linked["_merge"].ne("both").any():
        raise OperationalOrderError(
            "반려동물 대상 사건에 대응하는 사용자 주문이 없습니다."
        )
    pet_events["is_replenishable_snapshot"] = True
    return OperationalPurchaseInputs(
        valid_items=pd.DataFrame(),
        all_purchase_events=all_events,
        pet_history_items=pd.DataFrame(),
        pet_purchase_events=pet_events,
        excluded_late_birth_item_count=0,
    )


def rebuild_service_labels_from_event_intervals(
    event_intervals: OperationalEventIntervals,
    orders: pd.DataFrame,
    *,
    observation_end_at: pd.Timestamp,
) -> pd.DataFrame:
    """종료 시점에 유효한 구매만 정답 사건으로 삼아 라벨만 반환합니다.

    피처의 앵커 시점 복원과 달리, 라벨은 관측 종료까지 확인한 결과입니다.
    종료 전에 환불된 구매는 소급해 앵커와 다음 사건에서 제외합니다.
    기존 라벨 생성기의 최종 시점 피처는 학습에 섞이지 않도록 반환하지 않습니다.
    """
    prepared = prepare_active_service_purchase_inputs(
        event_intervals, orders, as_of_timestamp=observation_end_at
    )
    samples = build_service_repurchase_samples(
        prepared, observation_end_at=observation_end_at
    )
    return samples.loc[:, list(SERVICE_LABEL_COLUMNS)].copy()
