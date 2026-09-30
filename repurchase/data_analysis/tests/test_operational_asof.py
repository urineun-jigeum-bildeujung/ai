"""완료된 취소·반품만 기준 시점의 유효 구매에 반영하는지 검사합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.operational_asof import build_valid_order_items_as_of
from scripts.modeling.operational_orders import OperationalOrderError
from scripts.modeling.operational_purchase_inputs import (
    prepare_operational_purchase_inputs,
    prepare_operational_purchase_inputs_as_of,
)
from scripts.modeling.service_samples import (
    build_current_service_features,
    build_service_repurchase_samples,
)


def _sources() -> tuple[pd.DataFrame, ...]:
    orders = pd.DataFrame({
        "order_id": ["o1"], "user_id": ["u1"],
        "ordered_at": ["2026-01-01T00:00:00Z"],
        "paid_at": ["2026-01-01T00:00:00Z"],
        "order_status": ["CANCELLED"], "purchase_type": ["ONE_TIME"],
    })
    items = pd.DataFrame({
        "order_item_id": ["i1"], "order_id": ["o1"],
        "product_id": ["p1"], "product_group_id_snapshot": ["g1"],
        "category_code_snapshot": ["FOOD"],
        "is_replenishable_snapshot": [True], "pet_id": ["pet1"],
        "quantity": [2], "item_status": ["CANCELLED"],
        "cancelled_quantity": [2], "returned_quantity": [0],
    })
    histories = pd.DataFrame({
        "history_id": [1, 2], "order_id": ["o1", "o1"],
        "to_status": ["PAID", "CANCELLED"],
        "changed_at": ["2026-01-01T00:00:00Z", "2026-01-10T00:00:00Z"],
    })
    claims = pd.DataFrame({
        "claim_id": ["c1"], "order_id": ["o1"],
        "claim_type": ["CANCEL"], "claim_status": ["COMPLETED"],
        "completed_at": ["2026-01-10T00:00:00Z"],
    })
    claim_items = pd.DataFrame({
        "claim_item_id": ["ci1"], "claim_id": ["c1"],
        "order_item_id": ["i1"], "quantity": [2],
    })
    return orders, items, histories, claims, claim_items


def test_later_cancellation_does_not_remove_earlier_purchase() -> None:
    sources = _sources()
    originals = tuple(frame.copy(deep=True) for frame in sources)
    before = build_valid_order_items_as_of(
        *sources, as_of_timestamp=pd.Timestamp("2026-01-05T00:00:00Z")
    )
    after = build_valid_order_items_as_of(
        *sources, as_of_timestamp=pd.Timestamp("2026-01-10T00:00:00Z")
    )
    assert before["order_item_id"].tolist() == ["i1"]
    assert before["net_quantity"].tolist() == [2]
    assert after.empty
    for source, original in zip(sources, originals, strict=True):
        pd.testing.assert_frame_equal(source, original)


def test_missing_completed_claim_is_rejected() -> None:
    orders, items, histories, claims, claim_items = _sources()
    claims.loc[0, "completed_at"] = None
    with pytest.raises(OperationalOrderError, match="완료 시각"):
        build_valid_order_items_as_of(
            orders, items, histories, claims, claim_items,
            as_of_timestamp=pd.Timestamp("2026-01-05T00:00:00Z"),
        )


def test_incomplete_claim_history_is_rejected() -> None:
    orders, items, histories, claims, claim_items = _sources()
    claim_items.loc[0, "quantity"] = 1
    with pytest.raises(OperationalOrderError, match="수량.*이력"):
        build_valid_order_items_as_of(
            orders, items, histories, claims, claim_items,
            as_of_timestamp=pd.Timestamp("2026-01-05T00:00:00Z"),
        )


def test_naive_cutoff_is_rejected() -> None:
    with pytest.raises(OperationalOrderError, match="시간대"):
        build_valid_order_items_as_of(
            *_sources(), as_of_timestamp=pd.Timestamp("2026-01-05")
        )


def test_unpaid_pending_order_without_history_is_ignored() -> None:
    orders, items, histories, claims, claim_items = _sources()
    orders.loc[len(orders)] = [
        "o2", "u2", "2026-01-02T00:00:00Z", None, "PENDING", "ONE_TIME"
    ]
    result = build_valid_order_items_as_of(
        orders, items, histories, claims, claim_items,
        as_of_timestamp=pd.Timestamp("2026-01-05T00:00:00Z"),
    )
    assert result["order_item_id"].tolist() == ["i1"]


def test_paid_order_without_history_is_rejected() -> None:
    orders, items, histories, claims, claim_items = _sources()
    histories = histories.iloc[0:0]
    with pytest.raises(OperationalOrderError, match="상태 이력"):
        build_valid_order_items_as_of(
            orders, items, histories, claims, claim_items,
            as_of_timestamp=pd.Timestamp("2026-01-05T00:00:00Z"),
        )


def test_no_claims_preserves_paid_item() -> None:
    orders, items, histories, claims, claim_items = _sources()
    orders.loc[0, "order_status"] = "PAID"
    items.loc[0, "item_status"] = "PAID"
    items.loc[0, "cancelled_quantity"] = 0
    histories = histories.iloc[:1]
    claims = claims.iloc[0:0]
    claim_items = claim_items.iloc[0:0]
    result = build_valid_order_items_as_of(
        orders, items, histories, claims, claim_items,
        as_of_timestamp=pd.Timestamp("2026-01-05T00:00:00Z"),
    )
    assert result["net_quantity"].tolist() == [2]


def test_as_of_inputs_feed_user_and_pet_events() -> None:
    orders, items, histories, claims, claim_items = _sources()
    pets = pd.DataFrame({"pet_id": ["pet1"], "birth_date": ["2025-12-01"]})
    before = prepare_operational_purchase_inputs_as_of(
        orders, items, pets, histories, claims, claim_items,
        as_of_timestamp=pd.Timestamp("2026-01-05T00:00:00Z"),
    )
    after = prepare_operational_purchase_inputs_as_of(
        orders, items, pets, histories, claims, claim_items,
        as_of_timestamp=pd.Timestamp("2026-01-10T00:00:00Z"),
    )
    assert len(before.all_purchase_events) == 1
    assert len(before.pet_purchase_events) == 1
    assert before.pet_purchase_events["net_unit_count"].tolist() == [2]
    assert after.all_purchase_events.empty
    assert after.pet_purchase_events.empty


def test_paid_status_before_actual_payment_is_rejected() -> None:
    orders, items, histories, claims, claim_items = _sources()
    orders.loc[0, "paid_at"] = "2026-01-06T00:00:00Z"
    with pytest.raises(OperationalOrderError, match="늦은 결제"):
        build_valid_order_items_as_of(
            orders, items, histories, claims, claim_items,
            as_of_timestamp=pd.Timestamp("2026-01-05T00:00:00Z"),
        )


def test_later_cancellation_changes_prior_features_at_earlier_anchor() -> None:
    orders, items, histories, claims, claim_items = _sources()
    orders.loc[len(orders)] = [
        "o2", "u1", "2026-01-05T00:00:00Z", "2026-01-05T00:00:00Z",
        "PAID", "ONE_TIME",
    ]
    items.loc[len(items)] = [
        "i2", "o2", "p1", "g1", "FOOD", True, "pet1", 1, "PAID", 0, 0,
    ]
    histories.loc[len(histories)] = [
        3, "o2", "PAID", "2026-01-05T00:00:00Z",
    ]
    pets = pd.DataFrame({"pet_id": ["pet1"], "birth_date": ["2025-12-01"]})
    anchor = pd.Timestamp("2026-01-05T00:00:00Z")

    final = prepare_operational_purchase_inputs(orders, items, pets)
    as_of = prepare_operational_purchase_inputs_as_of(
        orders, items, pets, histories, claims, claim_items,
        as_of_timestamp=anchor,
    )
    final_row = build_service_repurchase_samples(
        final, observation_end_at=pd.Timestamp("2026-01-11T00:00:00Z")
    ).set_index("order_id").loc["o2"]
    as_of_row = build_current_service_features(
        as_of, as_of_timestamp=anchor
    ).set_index("order_id").loc["o2"]

    assert final_row["history_interval_count"] == 0
    assert final_row["user_prior_order_count"] == 0
    assert as_of_row["history_interval_count"] == 1
    assert as_of_row["user_prior_order_count"] == 1
