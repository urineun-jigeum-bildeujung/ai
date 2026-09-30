"""상태 변경·완료 클레임 이력으로 기준 시점의 유효 구매를 재구성합니다.

현재 주문/상품 행은 키와 원래 수량의 원천으로만 사용합니다. 과거의 주문 상태,
취소·반품 수량은 각 이력의 발생 시각에서 다시 계산합니다.
"""

from __future__ import annotations

import pandas as pd

from .operational_orders import (
    ALL_ORDER_STATUSES,
    VALID_ORDER_STATUSES,
    OperationalOrderError,
    _require_columns,
    _require_keys,
    _require_nonnegative_integer,
    build_valid_order_items,
)

HISTORY_COLUMNS = ("history_id", "order_id", "to_status", "changed_at")
CLAIM_COLUMNS = (
    "claim_id", "order_id", "claim_type", "claim_status", "completed_at"
)
CLAIM_ITEM_COLUMNS = ("claim_item_id", "claim_id", "order_item_id", "quantity")


def _utc(value: object, *, column: str) -> pd.Timestamp:
    """시각이 실제로 시간대를 포함할 때만 UTC로 변환합니다."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise OperationalOrderError(f"{column}이 잘못된 시각입니다.") from error
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise OperationalOrderError(f"{column}에는 시간대가 필요합니다.")
    return timestamp.tz_convert("UTC")


def build_valid_order_items_as_of(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    status_histories: pd.DataFrame,
    claims: pd.DataFrame,
    claim_items: pd.DataFrame,
    *,
    as_of_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """기준 시각까지 완료된 변경만 반영한 구매 상품을 반환합니다.

    주문별 마지막 상태 변경과 완료 클레임 수량을 사용합니다. 현재 스냅샷의
    취소·반품 수량은 이력의 완전성을 검사하는 데만 사용하고 과거 값으로 쓰지 않습니다.
    """
    cutoff = _utc(as_of_timestamp, column="as_of_timestamp")
    _require_columns(orders, ("order_id", "order_status", "paid_at"), "orders")
    _require_columns(
        order_items,
        ("order_item_id", "order_id", "quantity", "cancelled_quantity", "returned_quantity"),
        "order_items",
    )
    for frame, columns, name in (
        (status_histories, HISTORY_COLUMNS, "status_histories"),
        (claims, CLAIM_COLUMNS, "claims"),
        (claim_items, CLAIM_ITEM_COLUMNS, "claim_items"),
    ):
        _require_columns(frame, columns, name)
        _require_keys(frame, columns[:2], name)
        if frame[columns[0]].duplicated().any():
            raise OperationalOrderError(f"{name}의 ID가 중복됐습니다.")

    if not status_histories["order_id"].isin(orders["order_id"]).all():
        raise OperationalOrderError("상태 이력에 존재하지 않는 주문이 있습니다.")
    without_history = orders.loc[
        ~orders["order_id"].isin(status_histories["order_id"])
    ]
    # 아직 결제되지 않은 PENDING 주문은 구매 이력에 들어가지 않습니다.
    # 결제된 주문에 이력이 없으면 과거 상태를 복원할 수 없으므로 거절합니다.
    if (
        without_history["order_status"].ne("PENDING")
        | without_history["paid_at"].notna()
    ).any():
        raise OperationalOrderError("결제된 주문에 상태 이력이 없습니다.")
    if not claims["order_id"].isin(orders["order_id"]).all():
        raise OperationalOrderError("클레임에 존재하지 않는 주문이 있습니다.")
    if not claim_items["claim_id"].isin(claims["claim_id"]).all():
        raise OperationalOrderError("클레임 상품에 존재하지 않는 클레임이 있습니다.")
    if not claim_items["order_item_id"].isin(order_items["order_item_id"]).all():
        raise OperationalOrderError("클레임 상품에 존재하지 않는 주문상품이 있습니다.")
    if not status_histories["to_status"].isin(ALL_ORDER_STATUSES).all():
        raise OperationalOrderError("상태 이력에 지원하지 않는 주문 상태가 있습니다.")
    if not claims["claim_type"].isin({"CANCEL", "RETURN"}).all():
        raise OperationalOrderError("지원하지 않는 클레임 유형이 있습니다.")
    if not claims["claim_status"].isin(
        {"REQUESTED", "COLLECTING", "INSPECTING", "COMPLETED", "REJECTED"}
    ).all():
        raise OperationalOrderError("지원하지 않는 클레임 상태가 있습니다.")
    if not claim_items.empty:
        _require_nonnegative_integer(claim_items, "quantity")
    if claim_items["quantity"].le(0).any():
        raise OperationalOrderError("클레임 상품 수량은 양수여야 합니다.")

    histories = status_histories.copy()
    histories["changed_at"] = histories["changed_at"].map(
        lambda value: _utc(value, column="changed_at")
    )
    histories = histories.sort_values(["changed_at", "history_id"], kind="stable")
    latest = histories.drop_duplicates("order_id", keep="last")
    current_status = latest.set_index("order_id")["to_status"]
    orders_with_history = orders.loc[
        orders["order_id"].isin(current_status.index)
    ]
    if any(
        current_status.loc[order_id] != status
        for order_id, status in zip(
            orders_with_history["order_id"],
            orders_with_history["order_status"],
            strict=True,
        )
    ):
        raise OperationalOrderError("현재 주문 상태와 마지막 상태 이력이 다릅니다.")

    past = histories.loc[histories["changed_at"].le(cutoff)]
    past_status = past.drop_duplicates("order_id", keep="last").set_index("order_id")[
        "to_status"
    ]
    as_of_orders = orders.loc[orders["order_id"].isin(past_status.index)].copy()
    as_of_orders["order_status"] = as_of_orders["order_id"].map(past_status)
    paid_at_as_of = as_of_orders.loc[
        as_of_orders["order_status"].isin(VALID_ORDER_STATUSES), "paid_at"
    ].map(lambda value: _utc(value, column="paid_at"))
    if paid_at_as_of.gt(cutoff).any():
        raise OperationalOrderError(
            "기준 시각보다 늦은 결제가 이전 상태 이력에 포함됐습니다."
        )

    claim_rows = claims.copy()
    completed = claim_rows["claim_status"].eq("COMPLETED")
    if claim_rows.loc[completed, "completed_at"].isna().any():
        raise OperationalOrderError("완료 클레임에 완료 시각이 없습니다.")
    claim_rows.loc[completed, "completed_at"] = claim_rows.loc[
        completed, "completed_at"
    ].map(lambda value: _utc(value, column="completed_at"))
    joined_claims = claim_items.merge(
        claim_rows, on="claim_id", how="left", validate="many_to_one"
    )
    item_orders = order_items.loc[:, ["order_item_id", "order_id"]]
    joined_claims = joined_claims.merge(
        item_orders, on="order_item_id", how="left", validate="many_to_one",
        suffixes=("_claim", "_item"),
    )
    if joined_claims["order_id_claim"].ne(joined_claims["order_id_item"]).any():
        raise OperationalOrderError("클레임과 주문상품의 주문 ID가 다릅니다.")

    def quantities(rows: pd.DataFrame) -> dict[tuple[object, str], int]:
        result: dict[tuple[object, str], int] = {}
        for item_id, claim_type, quantity in zip(
            rows["order_item_id"], rows["claim_type"], rows["quantity"], strict=True
        ):
            key = (item_id, claim_type)
            result[key] = result.get(key, 0) + int(quantity)
        return result

    all_completed = joined_claims.loc[joined_claims["claim_status"].eq("COMPLETED")]
    final_quantities = quantities(all_completed)
    for item_id, cancelled, returned in zip(
        order_items["order_item_id"],
        order_items["cancelled_quantity"],
        order_items["returned_quantity"],
        strict=True,
    ):
        if (
            final_quantities.get((item_id, "CANCEL"), 0) != int(cancelled)
            or final_quantities.get((item_id, "RETURN"), 0) != int(returned)
        ):
            raise OperationalOrderError("현재 주문상품 수량과 완료 클레임 이력이 다릅니다.")

    known_quantities = quantities(all_completed.loc[all_completed["completed_at"].le(cutoff)])
    as_of_items = order_items.loc[
        order_items["order_id"].isin(as_of_orders["order_id"])
    ].copy()
    as_of_items["cancelled_quantity"] = as_of_items["order_item_id"].map(
        lambda item_id: known_quantities.get((item_id, "CANCEL"), 0)
    )
    as_of_items["returned_quantity"] = as_of_items["order_item_id"].map(
        lambda item_id: known_quantities.get((item_id, "RETURN"), 0)
    )
    net = [
        int(quantity) - int(cancelled) - int(returned)
        for quantity, cancelled, returned in zip(
            as_of_items["quantity"],
            as_of_items["cancelled_quantity"],
            as_of_items["returned_quantity"],
            strict=True,
        )
    ]
    as_of_items["item_status"] = [
        "RETURNED" if remaining == 0 and returned > 0 else
        "CANCELLED" if remaining == 0 else
        "PARTIAL_RETURN" if returned > 0 else "PAID"
        for remaining, returned in zip(net, as_of_items["returned_quantity"], strict=True)
    ]
    return build_valid_order_items(as_of_orders, as_of_items)
