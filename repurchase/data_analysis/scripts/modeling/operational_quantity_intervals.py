"""완료된 취소·반품 이력으로 주문상품의 시점별 잔여 수량을 계산합니다."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .operational_asof import _utc
from .operational_orders import (
    OperationalOrderError,
    _require_columns,
    _require_keys,
    _require_nonnegative_integer,
)


def build_order_item_quantity_intervals(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    claims: pd.DataFrame,
    claim_items: pd.DataFrame,
) -> pd.DataFrame:
    """잔여 수량이 일정한 `[valid_from, valid_until)` 구간을 반환합니다.

    현재 상품 상태를 과거에 소급하지 않습니다. 주문 상태는 별도 구간과 결합해야
    실제 유효 구매를 판정할 수 있습니다.
    """
    _require_columns(orders, ("order_id", "paid_at"), "orders")
    _require_columns(
        order_items,
        (
            "order_item_id",
            "order_id",
            "quantity",
            "cancelled_quantity",
            "returned_quantity",
        ),
        "order_items",
    )
    _require_columns(
        claims,
        ("claim_id", "order_id", "claim_type", "claim_status", "completed_at"),
        "claims",
    )
    _require_columns(
        claim_items,
        ("claim_item_id", "claim_id", "order_item_id", "quantity"),
        "claim_items",
    )
    for frame, columns, name in (
        (orders, ("order_id",), "orders"),
        (order_items, ("order_item_id", "order_id"), "order_items"),
        (claims, ("claim_id", "order_id"), "claims"),
        (claim_items, ("claim_item_id", "claim_id", "order_item_id"), "claim_items"),
    ):
        _require_keys(frame, columns, name)
        if frame[columns[0]].duplicated().any():
            raise OperationalOrderError(f"{name}의 ID가 중복됐습니다.")
    for column in ("quantity", "cancelled_quantity", "returned_quantity"):
        _require_nonnegative_integer(order_items, column)
    if not claim_items.empty:
        _require_nonnegative_integer(claim_items, "quantity")
    if order_items["quantity"].le(0).any() or claim_items["quantity"].le(0).any():
        raise OperationalOrderError("구매·클레임 상품 수량은 양수여야 합니다.")
    if not order_items["order_id"].isin(orders["order_id"]).all():
        raise OperationalOrderError("주문상품에 존재하지 않는 주문이 있습니다.")
    if not claims["order_id"].isin(orders["order_id"]).all():
        raise OperationalOrderError("클레임에 존재하지 않는 주문이 있습니다.")
    if not claim_items["claim_id"].isin(claims["claim_id"]).all():
        raise OperationalOrderError("클레임 상품에 존재하지 않는 클레임이 있습니다.")
    if not claim_items["order_item_id"].isin(order_items["order_item_id"]).all():
        raise OperationalOrderError("클레임 상품에 존재하지 않는 주문상품이 있습니다.")
    if not claims["claim_type"].isin({"CANCEL", "RETURN", "EXCHANGE"}).all():
        raise OperationalOrderError("지원하지 않는 클레임 유형이 있습니다.")
    if (
        not claims["claim_status"]
        .isin({"REQUESTED", "COLLECTING", "INSPECTING", "COMPLETED", "REJECTED"})
        .all()
    ):
        raise OperationalOrderError("지원하지 않는 클레임 상태가 있습니다.")
    if (
        claims["claim_type"].eq("EXCHANGE") & claims["claim_status"].eq("COMPLETED")
    ).any():
        raise OperationalOrderError("완료된 교환의 구매 수량 복원 규칙이 없습니다.")

    paid_at = {
        order_id: None if pd.isna(value) else _utc(value, column="paid_at")
        for order_id, value in orders[["order_id", "paid_at"]].itertuples(
            index=False, name=None
        )
    }
    item_order = dict(
        order_items[["order_item_id", "order_id"]].itertuples(index=False, name=None)
    )
    claim_order = dict(
        claims[["claim_id", "order_id"]].itertuples(index=False, name=None)
    )
    completed = claims.loc[claims["claim_status"].eq("COMPLETED")]
    if completed["completed_at"].isna().any():
        raise OperationalOrderError("완료 클레임에 완료 시각이 없습니다.")
    claim_details = {
        claim_id: (claim_type, _utc(completed_at, column="completed_at"))
        for claim_id, claim_type, completed_at in completed[
            ["claim_id", "claim_type", "completed_at"]
        ].itertuples(index=False, name=None)
    }
    changes: dict[object, dict[pd.Timestamp, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    final_totals: dict[tuple[object, str], int] = defaultdict(int)
    for _, claim_id, item_id, raw_quantity in claim_items[
        ["claim_item_id", "claim_id", "order_item_id", "quantity"]
    ].itertuples(index=False, name=None):
        if claim_order[claim_id] != item_order[item_id]:
            raise OperationalOrderError("클레임과 주문상품의 주문 ID가 다릅니다.")
        if claim_id not in claim_details:
            continue  # 미완료·거절 클레임은 과거 수량을 바꾸지 않습니다.
        claim_type, completed_at = claim_details[claim_id]
        final_totals[(item_id, claim_type)] += int(raw_quantity)
        changes[item_id][completed_at] += int(raw_quantity)

    intervals: list[dict[str, object]] = []
    for item_id, order_id, quantity, cancelled, returned in order_items[
        [
            "order_item_id",
            "order_id",
            "quantity",
            "cancelled_quantity",
            "returned_quantity",
        ]
    ].itertuples(index=False, name=None):
        if final_totals[(item_id, "CANCEL")] != int(cancelled) or final_totals[
            (item_id, "RETURN")
        ] != int(returned):
            raise OperationalOrderError(
                "현재 주문상품 수량과 완료 클레임 이력이 다릅니다."
            )
        start = paid_at[order_id]
        if start is None:
            if changes[item_id]:
                raise OperationalOrderError("미결제 주문상품에 완료 클레임이 있습니다.")
            continue
        remaining = int(quantity)
        for changed_at, delta in sorted(changes[item_id].items()):
            if changed_at < start:
                raise OperationalOrderError("클레임 완료가 결제 시각보다 빠릅니다.")
            if changed_at > start:
                intervals.append(
                    {
                        "order_item_id": item_id,
                        "order_id": order_id,
                        "valid_from": start,
                        "valid_until": changed_at,
                        "remaining_quantity": remaining,
                        "is_valid_quantity": remaining > 0,
                    }
                )
            remaining -= delta
            if remaining < 0:
                raise OperationalOrderError(
                    "누적 클레임 수량이 원래 수량을 초과합니다."
                )
            start = changed_at
        intervals.append(
            {
                "order_item_id": item_id,
                "order_id": order_id,
                "valid_from": start,
                "valid_until": pd.NaT,
                "remaining_quantity": remaining,
                "is_valid_quantity": remaining > 0,
            }
        )
    columns = (
        "order_item_id",
        "order_id",
        "valid_from",
        "valid_until",
        "remaining_quantity",
        "is_valid_quantity",
    )
    return pd.DataFrame(intervals, columns=columns)
