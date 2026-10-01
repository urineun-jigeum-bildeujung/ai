"""학습 앵커마다 당시 유효했던 전체 사용자 주문 수를 다시 계산합니다."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .operational_asof import _utc
from .operational_orders import OperationalOrderError, _require_columns, _require_keys
from .operational_validity_intervals import _end, _require_nonoverlapping


def recount_user_prior_orders_as_of(
    samples: pd.DataFrame,
    user_order_intervals: pd.DataFrame,
    orders: pd.DataFrame,
) -> pd.DataFrame:
    """각 표본의 앵커 직전 결제된 고유 주문 중 그때 유효한 주문만 셉니다.

    반려동물 미지정·비반복 상품 주문도 사용자 전체 주문 수에 포함합니다.
    반환 `sample_row`는 입력 표본의 0부터 시작하는 위치이며 원본은 수정하지 않습니다.
    """
    _require_columns(samples, ("user_id", "anchor_at"), "samples")
    _require_columns(
        user_order_intervals,
        ("user_id", "order_id", "valid_from", "valid_until"),
        "user_order_intervals",
    )
    _require_columns(orders, ("user_id", "order_id", "paid_at"), "orders")
    for frame, keys, name in (
        (samples, ("user_id",), "samples"),
        (user_order_intervals, ("user_id", "order_id"), "user_order_intervals"),
        (orders, ("user_id", "order_id"), "orders"),
    ):
        _require_keys(frame, keys, name)
    if orders["order_id"].duplicated().any():
        raise OperationalOrderError("orders.order_id가 중복됐습니다.")

    anchors = samples.loc[:, ["user_id", "anchor_at"]].copy().reset_index(drop=True)
    anchors["sample_row"] = anchors.index
    anchors["anchor_at"] = pd.to_datetime(
        anchors["anchor_at"].map(lambda value: _utc(value, column="anchor_at")),
        utc=True,
    )
    history = user_order_intervals.loc[
        :, ["user_id", "order_id", "valid_from", "valid_until"]
    ].copy()
    history["valid_from"] = pd.to_datetime(
        history["valid_from"].map(lambda value: _utc(value, column="valid_from")),
        utc=True,
    )
    history["valid_until"] = pd.to_datetime(
        history["valid_until"].map(
            lambda value: (
                pd.NaT if pd.isna(value) else _utc(value, column="valid_until")
            )
        ),
        utc=True,
    )
    if (
        history["valid_until"].notna()
        & history["valid_until"].le(history["valid_from"])
    ).any():
        raise OperationalOrderError(
            "사용자 주문 구간의 종료 시각이 시작보다 늦지 않습니다."
        )
    paid = orders.loc[:, ["user_id", "order_id", "paid_at"]].copy()
    history = history.merge(
        paid,
        on=["user_id", "order_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if history["_merge"].ne("both").any():
        raise OperationalOrderError("사용자 주문 구간에 연결되지 않는 주문이 있습니다.")
    history["paid_at"] = pd.to_datetime(
        history["paid_at"].map(lambda value: _utc(value, column="paid_at")),
        utc=True,
    )
    history = history.drop(columns="_merge").rename(
        columns={"order_id": "historical_order_id"}
    )
    if history["valid_from"].lt(history["paid_at"]).any():
        raise OperationalOrderError("유효 주문 구간이 결제 시각보다 먼저 시작합니다.")
    spans: dict[
        tuple[object, object], list[tuple[pd.Timestamp, pd.Timestamp | None]]
    ] = {}
    for user_id, order_id, start, raw_end in history[
        ["user_id", "historical_order_id", "valid_from", "valid_until"]
    ].itertuples(index=False, name=None):
        spans.setdefault((user_id, order_id), []).append((start, _end(raw_end)))
    _require_nonoverlapping(spans, name="사용자 주문")

    # 구간 시작·종료의 누적 개수로 계산해 모든 앵커×주문 구간을 결합하지 않습니다.
    counts = np.zeros(len(anchors), dtype="int64")
    history_by_user = {
        user_id: group for user_id, group in history.groupby("user_id", sort=False)
    }
    for user_id, group in anchors.groupby("user_id", sort=False):
        user_history = history_by_user.get(user_id)
        if user_history is None:
            continue
        anchor_times = group["anchor_at"].to_numpy(dtype="datetime64[ns]")
        starts = np.sort(user_history["valid_from"].to_numpy(dtype="datetime64[ns]"))
        ends = np.sort(
            user_history.loc[
                user_history["valid_until"].notna(), "valid_until"
            ].to_numpy(dtype="datetime64[ns]")
        )
        # 결제와 같은 시각의 주문은 이전 주문이 아닙니다.
        same_time_starts = np.sort(
            user_history.loc[
                user_history["valid_from"].eq(user_history["paid_at"]),
                "valid_from",
            ].to_numpy(dtype="datetime64[ns]")
        )
        counts[group.index.to_numpy()] = (
            np.searchsorted(starts, anchor_times, side="right")
            - np.searchsorted(ends, anchor_times, side="right")
            - (
                np.searchsorted(same_time_starts, anchor_times, side="right")
                - np.searchsorted(same_time_starts, anchor_times, side="left")
            )
        )
    result = anchors.loc[:, ["sample_row"]].copy()
    result["as_of_user_prior_order_count"] = counts
    return result
