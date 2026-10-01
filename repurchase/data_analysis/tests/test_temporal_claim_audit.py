"""최종 스냅샷이 지운 과거 구매의 anchor 수를 검증합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.operational_orders import OperationalOrderError
from scripts.modeling.temporal_claim_audit import (
    audit_removed_pet_history_at_anchors,
    audit_removed_user_orders_at_anchors,
)


def _sources() -> tuple[pd.DataFrame, ...]:
    samples = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "pet_id": ["pet1", "pet1"],
            "target_id": ["g1", "g1"],
            "anchor_at": [
                pd.Timestamp("2026-01-05T00:00:00Z"),
                pd.Timestamp("2026-01-15T00:00:00Z"),
            ],
        }
    )
    current_events = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "pet_id": ["pet1", "pet1"],
            "target_id": ["g1", "g1"],
            "order_id": ["o2", "o3"],
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3"],
            "user_id": ["u1"] * 3,
            "paid_at": [
                "2026-01-01T00:00:00Z",
                "2026-01-05T00:00:00Z",
                "2026-01-15T00:00:00Z",
            ],
            "ordered_at": [
                "2026-01-01T00:00:00Z",
                "2026-01-05T00:00:00Z",
                "2026-01-15T00:00:00Z",
            ],
        }
    )
    items = pd.DataFrame(
        {
            "order_item_id": ["i1"],
            "order_id": ["o1"],
            "product_group_id_snapshot": ["g1"],
            "pet_id": ["pet1"],
            "is_replenishable_snapshot": [True],
            "quantity": [1],
            "cancelled_quantity": [1],
            "returned_quantity": [0],
        }
    )
    pets = pd.DataFrame({"pet_id": ["pet1"], "birth_date": ["2026-01-01"]})
    claims = pd.DataFrame(
        {
            "claim_id": ["c1"],
            "claim_status": ["COMPLETED"],
            "completed_at": ["2026-01-10T00:00:00Z"],
        }
    )
    claim_items = pd.DataFrame({"claim_id": ["c1"], "order_item_id": ["i1"]})
    return samples, current_events, orders, items, pets, claims, claim_items


def test_only_anchor_before_claim_completion_is_counted() -> None:
    result = audit_removed_pet_history_at_anchors(*_sources())
    assert result == {
        "sample_count": 2,
        "potentially_affected_anchor_count": 1,
        "potentially_affected_anchor_rate": 0.5,
    }


def test_late_birth_claim_item_is_not_a_valid_prior_event() -> None:
    sources = list(_sources())
    sources[4].loc[0, "birth_date"] = "2026-01-02"
    result = audit_removed_pet_history_at_anchors(*sources)
    assert result["potentially_affected_anchor_count"] == 0


def test_existing_event_is_not_counted_as_removed() -> None:
    sources = list(_sources())
    sources[1].loc[len(sources[1])] = ["u1", "pet1", "g1", "o1"]
    result = audit_removed_pet_history_at_anchors(*sources)
    assert result["potentially_affected_anchor_count"] == 0


def test_removed_order_affects_only_anchors_before_terminal_status() -> None:
    samples, events, orders, *_ = _sources()
    histories = pd.DataFrame(
        {
            "order_id": ["o1"],
            "to_status": ["CANCELLED"],
            "changed_at": ["2026-01-10T00:00:00Z"],
        }
    )
    result = audit_removed_user_orders_at_anchors(samples, events, orders, histories)
    assert result == {
        "sample_count": 2,
        "potentially_affected_anchor_count": 1,
        "potentially_affected_anchor_rate": 0.5,
    }


def test_missing_order_termination_is_rejected() -> None:
    samples, events, orders, *_ = _sources()
    histories = pd.DataFrame(columns=["order_id", "to_status", "changed_at"])
    with pytest.raises(OperationalOrderError, match="종료 상태 이력"):
        audit_removed_user_orders_at_anchors(samples, events, orders, histories)


def test_same_time_purchase_is_not_prior_history() -> None:
    sources = list(_sources())
    sources[0].loc[0, "anchor_at"] = pd.Timestamp("2026-01-01T00:00:00Z")
    pet_result = audit_removed_pet_history_at_anchors(*sources)
    assert pet_result["potentially_affected_anchor_count"] == 0

    histories = pd.DataFrame(
        {
            "order_id": ["o1"],
            "to_status": ["CANCELLED"],
            "changed_at": ["2026-01-10T00:00:00Z"],
        }
    )
    user_result = audit_removed_user_orders_at_anchors(
        sources[0], sources[1], sources[2], histories
    )
    assert user_result["potentially_affected_anchor_count"] == 0
