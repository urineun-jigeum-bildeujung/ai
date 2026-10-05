"""시연용 SHADOW 조회가 목계정과 최신 배치에 한정되는지 검사합니다."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.modeling.demo_shadow_reader import (
    DemoShadowAccessError,
    read_demo_shadow_predictions,
)


class FakeConnection:
    def __init__(self, rows: list[tuple[object, ...]], dbname: str = "repurchase_db"):
        self.info = SimpleNamespace(dbname=dbname)
        self.rows = rows
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def execute(self, query: str, params: tuple[object, ...]) -> FakeConnection:
        self.calls.append((query, params))
        return self

    def fetchall(self) -> list[tuple[object, ...]]:
        return self.rows


def _read(connection: FakeConnection, **overrides: object):
    kwargs: dict[str, object] = {
        "viewer_user_id": "mock-42",
        "requested_user_id": "mock-42",
        "allowed_mock_user_ids": {"mock-42"},
        "expected_artifact_id": "fixed-aft",
    }
    kwargs.update(overrides)
    return read_demo_shadow_predictions(connection, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        {"viewer_user_id": "someone-else"},
        {"requested_user_id": "someone-else"},
        {"allowed_mock_user_ids": set()},
        {
            "viewer_user_id": "mock-4",
            "requested_user_id": "mock-4",
            "allowed_mock_user_ids": "mock-42",
        },
        {"viewer_user_id": ""},
        {"expected_artifact_id": ""},
        {"max_results": 0},
        {"max_results": 501},
    ],
)
def test_demo_reader_rejects_unauthorized_or_invalid_request_before_db(
    overrides: dict[str, object],
) -> None:
    connection = FakeConnection([])
    with pytest.raises(DemoShadowAccessError):
        _read(connection, **overrides)
    assert connection.calls == []


def test_demo_reader_scopes_latest_shadow_snapshot_and_marks_mock_source() -> None:
    connection = FakeConnection(
        [
            (
                "mock-42",
                7,
                "PRODUCT_GROUP",
                "11",
                30,
                "READY",
                0.35,
                "demo-shadow-20261005T0300KST-ec25eb1bcd8f-30d-v1",
                "fixed-aft",
                pd.Timestamp("2026-10-04T18:00:00Z"),
            )
        ]
    )
    predictions = _read(connection)

    assert len(predictions) == 1
    assert predictions[0].source_kind == "MOCK_DATA_DEMO"
    assert predictions[0].conditional_repurchase_probability == 0.35
    assert predictions[0].as_of_timestamp == "2026-10-04T18:00:00+00:00"
    query, params = connection.calls[0]
    assert "publication_status = 'SHADOW'" in query
    assert "LIMIT 1" in query
    assert "WHERE r.user_id = %s AND r.window_days = 30" in query
    assert params == ("fixed-aft", "mock-42", 501)


def test_demo_reader_rejects_wrong_database_and_silent_truncation() -> None:
    with pytest.raises(DemoShadowAccessError, match="repurchase_db"):
        _read(FakeConnection([], dbname="order_db"))

    connection = FakeConnection([tuple()] * 2)
    with pytest.raises(DemoShadowAccessError, match="조회 상한"):
        _read(connection, max_results=1)
