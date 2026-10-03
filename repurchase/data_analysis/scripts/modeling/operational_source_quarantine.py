"""상태 이력으로 복원할 수 없는 주문을 원천 변경 없이 격리합니다."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .operational_asof import _utc


@dataclass(frozen=True)
class QuarantineResult:
    orders: pd.DataFrame
    order_items: pd.DataFrame
    status_histories: pd.DataFrame
    claims: pd.DataFrame
    claim_items: pd.DataFrame
    missing_history_order_count: int
    missing_history_paid_order_count: int
    missing_history_order_item_count: int
    status_mismatch_order_count: int
    status_mismatch_order_item_count: int


def quarantine_unrestorable_orders(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    status_histories: pd.DataFrame,
    claims: pd.DataFrame,
    claim_items: pd.DataFrame,
    *,
    as_of_at: pd.Timestamp | None = None,
) -> QuarantineResult:
    """이력이 없거나 마지막 상태가 다른 주문과 연결 행을 함께 제외합니다.

    미결제 PENDING은 구매 사건이 아니므로 격리 건수에 넣지 않고 기존 검증에
    맡깁니다. 컷을 지정하면 이후 상태 변경을 현재 상태 대조에 사용하지 않되,
    반환하는 이력·클레임은 자르지 않아 시점별 수량 검증을 보존합니다.
    원천 프레임은 수정하지 않으며 제외 사유별 건수를 반환합니다.
    """
    cutoff = None if as_of_at is None else _utc(as_of_at, column="as_of_at")
    ordered_histories = status_histories.assign(
        _changed_at_utc=status_histories["changed_at"].map(
            lambda value: _utc(value, column="changed_at")
        )
    ).sort_values(["_changed_at_utc", "history_id"], kind="stable")
    relevant_histories = (
        ordered_histories
        if cutoff is None
        else ordered_histories.loc[ordered_histories["_changed_at_utc"].le(cutoff)]
    )
    without_history = orders.loc[
        ~orders["order_id"].isin(relevant_histories["order_id"])
    ]
    if cutoff is not None:
        # 컷 이후 결제된 주문은 이번 학습 모집단이 아닙니다.
        paid_at = without_history["paid_at"].map(
            lambda value: None if pd.isna(value) else _utc(value, column="paid_at")
        )
        without_history = without_history.loc[
            paid_at.map(lambda value: value is not None and value <= cutoff)
        ]
    missing = without_history.loc[
        without_history["order_status"].ne("PENDING")
        | without_history["paid_at"].notna()
    ]
    latest = relevant_histories.drop_duplicates("order_id", keep="last")
    current = orders.merge(
        latest[["order_id", "to_status", "_changed_at_utc"]],
        on="order_id",
        how="inner",
        validate="one_to_one",
    )
    if cutoff is not None:
        # 컷 이후 전이가 있는 주문의 현재 상태를 과거 상태와 대조하지 않습니다.
        later_ids = ordered_histories.loc[
            ordered_histories["_changed_at_utc"].gt(cutoff), "order_id"
        ]
        current = current.loc[~current["order_id"].isin(later_ids)]
        current_paid_at = current["paid_at"].map(
            lambda value: None if pd.isna(value) else _utc(value, column="paid_at")
        )
        current = current.loc[
            current_paid_at.map(lambda value: value is not None and value <= cutoff)
        ]
    mismatched = current.loc[current["order_status"].ne(current["to_status"])]
    excluded_ids = pd.concat(
        [missing["order_id"], mismatched["order_id"]], ignore_index=True
    )
    missing_item_count = int(order_items["order_id"].isin(missing["order_id"]).sum())
    mismatch_item_count = int(
        order_items["order_id"].isin(mismatched["order_id"]).sum()
    )
    removed_claims = claims.loc[claims["order_id"].isin(excluded_ids)]
    return QuarantineResult(
        orders=orders.loc[~orders["order_id"].isin(excluded_ids)].copy(),
        order_items=order_items.loc[~order_items["order_id"].isin(excluded_ids)].copy(),
        status_histories=status_histories.loc[
            ~status_histories["order_id"].isin(excluded_ids)
        ].copy(),
        claims=claims.loc[~claims["order_id"].isin(excluded_ids)].copy(),
        claim_items=claim_items.loc[
            ~claim_items["claim_id"].isin(removed_claims["claim_id"])
        ].copy(),
        missing_history_order_count=len(missing),
        missing_history_paid_order_count=int(missing["paid_at"].notna().sum()),
        missing_history_order_item_count=missing_item_count,
        status_mismatch_order_count=len(mismatched),
        status_mismatch_order_item_count=mismatch_item_count,
    )
