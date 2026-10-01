"""주문 상태 구간의 반열린 경계와 재활성화를 검증합니다."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.modeling.operational_orders import OperationalOrderError
from scripts.modeling.operational_status_intervals import build_order_status_intervals


def _status_at(intervals: pd.DataFrame, when: str) -> str:
    cutoff = pd.Timestamp(when)
    active = intervals.loc[
        intervals["valid_from"].le(cutoff)
        & (intervals["valid_until"].isna() | intervals["valid_until"].gt(cutoff))
    ]
    assert len(active) == 1
    return str(active.iloc[0]["to_status"])


def test_status_intervals_use_latest_same_time_change_and_can_reactivate() -> None:
    histories = pd.DataFrame(
        {
            "history_id": [1, 2, 3, 4, 5],
            "order_id": ["o1"] * 5,
            "to_status": ["PENDING", "PAID", "CANCELLED", "REFUNDED", "PAID"],
            "changed_at": [
                "2026-01-01T00:00:00Z",
                "2026-01-02T00:00:00Z",
                "2026-01-10T00:00:00Z",
                "2026-01-10T00:00:00Z",
                "2026-01-12T00:00:00Z",
            ],
        }
    )
    original = histories.copy(deep=True)

    result = build_order_status_intervals(histories)

    assert result["to_status"].tolist() == ["PENDING", "PAID", "REFUNDED", "PAID"]
    assert result["is_valid_status"].tolist() == [False, True, False, True]
    assert _status_at(result, "2026-01-09T23:59:59Z") == "PAID"
    assert _status_at(result, "2026-01-10T00:00:00Z") == "REFUNDED"
    assert _status_at(result, "2026-01-12T00:00:00Z") == "PAID"
    assert pd.isna(result.iloc[-1]["valid_until"])
    pd.testing.assert_frame_equal(histories, original)


def test_duplicate_history_id_is_rejected() -> None:
    histories = pd.DataFrame(
        {
            "history_id": [1, 1],
            "order_id": ["o1", "o1"],
            "to_status": ["PENDING", "PAID"],
            "changed_at": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
        }
    )
    with pytest.raises(OperationalOrderError, match="ID가 중복"):
        build_order_status_intervals(histories)


def test_next_boundary_is_calculated_within_each_order() -> None:
    histories = pd.DataFrame(
        {
            "history_id": [1, 2, 3, 4],
            "order_id": ["o1", "o2", "o1", "o2"],
            "to_status": ["PENDING", "PENDING", "PAID", "PAID"],
            "changed_at": [
                "2026-01-01T00:00:00Z",
                "2026-01-02T00:00:00Z",
                "2026-01-03T00:00:00Z",
                "2026-01-04T00:00:00Z",
            ],
        }
    )

    intervals = build_order_status_intervals(histories)
    first_o1 = intervals.loc[intervals["order_id"].eq("o1")].iloc[0]
    first_o2 = intervals.loc[intervals["order_id"].eq("o2")].iloc[0]

    assert first_o1["valid_until"] == pd.Timestamp("2026-01-03T00:00:00Z")
    assert first_o2["valid_until"] == pd.Timestamp("2026-01-04T00:00:00Z")


def test_timezone_free_history_is_rejected() -> None:
    histories = pd.DataFrame(
        {
            "history_id": [1],
            "order_id": ["o1"],
            "to_status": ["PAID"],
            "changed_at": ["2026-01-02T00:00:00"],
        }
    )
    with pytest.raises(OperationalOrderError, match="시간대"):
        build_order_status_intervals(histories)
