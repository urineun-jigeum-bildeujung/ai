"""최종 취소·반품 스냅샷이 과거 학습 이력에서 지운 구매를 셉니다."""

from __future__ import annotations

import pandas as pd

from .operational_asof import _utc
from .operational_orders import OperationalOrderError, _require_columns
from .pet_history import select_pet_history_items


def audit_removed_pet_history_at_anchors(
    samples: pd.DataFrame,
    current_pet_events: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    pets: pd.DataFrame,
    claims: pd.DataFrame,
    claim_items: pd.DataFrame,
) -> dict[str, int | float]:
    """완료 전에는 존재했지만 최종 이력에 없는 구매의 anchor 수를 반환합니다.

    이는 모델 피처/성능의 실제 변화가 아니라, 시점별 재계산이 필요한 표본의
    하한 후보입니다. 부분반품이나 사용자 전체 주문 수 변화는 별도 분석 대상입니다.
    """
    for frame, columns, name in (
        (samples, ("user_id", "pet_id", "target_id", "anchor_at"), "samples"),
        (
            current_pet_events,
            ("user_id", "pet_id", "target_id", "order_id"),
            "current_pet_events",
        ),
        (orders, ("order_id", "user_id", "paid_at", "ordered_at"), "orders"),
        (
            order_items,
            (
                "order_item_id",
                "order_id",
                "product_group_id_snapshot",
                "pet_id",
                "is_replenishable_snapshot",
                "quantity",
                "cancelled_quantity",
                "returned_quantity",
            ),
            "order_items",
        ),
        (claims, ("claim_id", "claim_status", "completed_at"), "claims"),
        (claim_items, ("claim_id", "order_item_id"), "claim_items"),
    ):
        _require_columns(frame, columns, name)
    if samples.index.has_duplicates:
        raise OperationalOrderError("samples의 행 인덱스가 중복됐습니다.")

    finished = claims.loc[claims["claim_status"].eq("COMPLETED")]
    removed = order_items.loc[
        order_items["pet_id"].notna()
        & order_items["is_replenishable_snapshot"].eq(True)
        & (order_items["cancelled_quantity"] + order_items["returned_quantity"]).eq(
            order_items["quantity"]
        )
    ]
    candidates = claim_items.merge(
        finished[["claim_id", "completed_at"]], on="claim_id", how="inner"
    ).merge(removed, on="order_item_id", how="inner")
    candidates = select_pet_history_items(candidates, orders, pets)
    if candidates.empty:
        return {
            "sample_count": len(samples),
            "potentially_affected_anchor_count": 0,
            "potentially_affected_anchor_rate": 0.0,
        }
    candidates = candidates.merge(
        orders[["order_id", "user_id", "paid_at"]],
        on="order_id",
        how="left",
        validate="many_to_one",
    ).rename(
        columns={
            "order_id": "claimed_order_id",
            "product_group_id_snapshot": "target_id",
        }
    )
    for column in ("paid_at", "completed_at"):
        candidates[column] = candidates[column].map(
            lambda value, column=column: _utc(value, column=column)
        )
    if candidates["completed_at"].le(candidates["paid_at"]).any():
        raise OperationalOrderError("클레임 완료가 결제 시각보다 빠릅니다.")

    anchors = samples.loc[:, ["user_id", "pet_id", "target_id", "anchor_at"]].copy()
    anchors["sample_key"] = anchors.index
    anchors["anchor_at"] = anchors["anchor_at"].map(
        lambda value: _utc(value, column="anchor_at")
    )
    matches = anchors.merge(
        candidates[
            [
                "user_id",
                "pet_id",
                "target_id",
                "claimed_order_id",
                "paid_at",
                "completed_at",
            ]
        ],
        on=["user_id", "pet_id", "target_id"],
    )
    matches = matches.loc[
        matches["anchor_at"].gt(matches["paid_at"])
        & matches["anchor_at"].lt(matches["completed_at"])
    ]
    existing = (
        current_pet_events.loc[:, ["user_id", "pet_id", "target_id", "order_id"]]
        .rename(columns={"order_id": "claimed_order_id"})
        .drop_duplicates()
    )
    matches = matches.merge(
        existing.assign(still_exists=True),
        on=["user_id", "pet_id", "target_id", "claimed_order_id"],
        how="left",
        validate="many_to_one",
    )
    affected = int(matches.loc[matches["still_exists"].isna(), "sample_key"].nunique())
    return {
        "sample_count": len(samples),
        "potentially_affected_anchor_count": affected,
        "potentially_affected_anchor_rate": affected / len(samples)
        if len(samples)
        else 0.0,
    }


def audit_removed_user_orders_at_anchors(
    samples: pd.DataFrame,
    current_all_events: pd.DataFrame,
    orders: pd.DataFrame,
    status_histories: pd.DataFrame,
) -> dict[str, int | float]:
    """최종 상태로 제외됐지만 anchor 당시에 유효했던 주문을 셉니다.

    사용자 전체 주문 수 피처의 시점 누수 후보이다. 표본별 실제 피처 차이는
    이 함수가 계산하지 않는다.
    """
    for frame, columns, name in (
        (samples, ("user_id", "anchor_at"), "samples"),
        (current_all_events, ("order_id",), "current_all_events"),
        (orders, ("order_id", "user_id", "paid_at"), "orders"),
        (
            status_histories,
            ("order_id", "to_status", "changed_at"),
            "status_histories",
        ),
    ):
        _require_columns(frame, columns, name)
    if samples.index.has_duplicates:
        raise OperationalOrderError("samples의 행 인덱스가 중복됐습니다.")

    missing = orders.loc[
        orders["paid_at"].notna()
        & ~orders["order_id"].isin(current_all_events["order_id"]),
        ["order_id", "user_id", "paid_at"],
    ].copy()
    if missing.empty:
        return {
            "sample_count": len(samples),
            "potentially_affected_anchor_count": 0,
            "potentially_affected_anchor_rate": 0.0,
        }
    terminal = status_histories.loc[
        status_histories["to_status"].isin({"CANCELLED", "REFUNDED"}),
        ["order_id", "changed_at"],
    ].copy()
    terminal["changed_at"] = terminal["changed_at"].map(
        lambda value: _utc(value, column="changed_at")
    )
    ended = terminal.groupby("order_id", sort=False)["changed_at"].min()
    missing["ended_at"] = missing["order_id"].map(ended)
    if missing["ended_at"].isna().any():
        raise OperationalOrderError(
            "최종 구매 이력에 없는 결제 주문의 종료 상태 이력이 없습니다."
        )
    missing["paid_at"] = missing["paid_at"].map(
        lambda value: _utc(value, column="paid_at")
    )
    if missing["ended_at"].le(missing["paid_at"]).any():
        raise OperationalOrderError("주문 종료가 결제 시각보다 빠릅니다.")

    anchors = samples.loc[:, ["user_id", "anchor_at"]].copy()
    anchors["sample_key"] = anchors.index
    anchors["anchor_at"] = anchors["anchor_at"].map(
        lambda value: _utc(value, column="anchor_at")
    )
    matches = anchors.merge(missing, on="user_id")
    matches = matches.loc[
        matches["anchor_at"].gt(matches["paid_at"])
        & matches["anchor_at"].lt(matches["ended_at"])
    ]
    affected = int(matches["sample_key"].nunique())
    return {
        "sample_count": len(samples),
        "potentially_affected_anchor_count": affected,
        "potentially_affected_anchor_rate": affected / len(samples)
        if len(samples)
        else 0.0,
    }
