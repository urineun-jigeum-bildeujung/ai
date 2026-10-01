"""주문 상태 이력을 반열린 시간 구간으로 펼칩니다.

상태가 유효하더라도 결제 시각과 주문상품 잔여 수량까지 확인해야 구매입니다.
"""

from __future__ import annotations

import pandas as pd

from .operational_asof import _utc
from .operational_orders import (
    ALL_ORDER_STATUSES,
    VALID_ORDER_STATUSES,
    OperationalOrderError,
    _require_columns,
    _require_keys,
)


def build_order_status_intervals(histories: pd.DataFrame) -> pd.DataFrame:
    """각 상태의 `[valid_from, valid_until)` 구간을 반환합니다.

    동일한 변경 시각에는 가장 큰 `history_id`의 상태가 그 시각부터 적용됩니다.
    마지막 구간의 `valid_until`은 결측이며, 이후 이력까지 유효하다는 뜻입니다.
    """
    required = ("history_id", "order_id", "to_status", "changed_at")
    _require_columns(histories, required, "status_histories")
    _require_keys(histories, required, "status_histories")
    if histories["history_id"].duplicated().any():
        raise OperationalOrderError("상태 이력 ID가 중복됐습니다.")
    if not histories["to_status"].isin(ALL_ORDER_STATUSES).all():
        raise OperationalOrderError("상태 이력에 지원하지 않는 주문 상태가 있습니다.")

    timeline = histories.loc[:, list(required)].copy()
    timeline["changed_at"] = timeline["changed_at"].map(
        lambda value: _utc(value, column="changed_at")
    )
    timeline = timeline.sort_values(
        ["order_id", "changed_at", "history_id"], kind="stable"
    ).drop_duplicates(["order_id", "changed_at"], keep="last")
    timeline["valid_from"] = timeline["changed_at"]
    timeline["valid_until"] = timeline.groupby("order_id", sort=False)[
        "changed_at"
    ].shift(-1)
    timeline["is_valid_status"] = timeline["to_status"].isin(VALID_ORDER_STATUSES)
    return timeline.loc[
        :, ["order_id", "valid_from", "valid_until", "to_status", "is_valid_status"]
    ].reset_index(drop=True)
