"""클라우드 원천 리더의 DB 경계와 읽기 전용 스냅샷 계약을 검사합니다."""

from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from psycopg.pq import TransactionStatus

from scripts.modeling.cloud_source_reader import read_order_source, read_pet_source


class _Cursor:
    def __init__(self, rows: dict[str, tuple[list[str], list[tuple[object, ...]]]]):
        self.rows = rows
        self.description: list[SimpleNamespace] = []
        self._values: list[tuple[object, ...]] = []

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, query: str) -> None:
        table = query.split("FROM public.", 1)[1].split()[0]
        columns, self._values = self.rows[table]
        self.description = [SimpleNamespace(name=column) for column in columns]

    def fetchall(self) -> list[tuple[object, ...]]:
        return self._values


class _Connection:
    def __init__(
        self, dbname: str, rows: dict[str, tuple[list[str], list[tuple[object, ...]]]]
    ):
        self.info = SimpleNamespace(
            dbname=dbname, transaction_status=TransactionStatus.IDLE
        )
        self.autocommit = True
        self.rows = rows
        self.commands: list[str] = []

    def transaction(self) -> nullcontext[None]:
        return nullcontext()

    def execute(self, query: str) -> _Connection:
        self.commands.append(query)
        return self

    def fetchone(self) -> tuple[datetime]:
        return (datetime(2026, 9, 30, tzinfo=UTC),)

    def cursor(self) -> _Cursor:
        return _Cursor(self.rows)


def test_order_reader_uses_read_only_snapshot_and_preserves_empty_claims() -> None:
    rows = {
        "orders": (
            [
                "order_id",
                "user_id",
                "ordered_at",
                "paid_at",
                "order_status",
                "purchase_type",
            ],
            [(1, 2, None, None, "PAID", "ONE_TIME")],
        ),
        "order_items": (
            [
                "order_item_id",
                "order_id",
                "product_id",
                "product_group_id_snapshot",
                "category_code_snapshot",
                "is_replenishable_snapshot",
                "pet_id",
                "quantity",
                "item_status",
                "cancelled_quantity",
                "returned_quantity",
            ],
            [(3, 1, 4, 5, "FOOD", True, None, 2, "PAID", 0, 0)],
        ),
        "order_status_histories": (
            ["history_id", "order_id", "from_status", "to_status", "changed_at"],
            [],
        ),
        "order_claims": (
            [
                "claim_id",
                "order_id",
                "claim_type",
                "claim_status",
                "requested_at",
                "completed_at",
            ],
            [],
        ),
        "order_claim_items": (
            ["claim_item_id", "claim_id", "order_item_id", "quantity"],
            [],
        ),
    }
    connection = _Connection("order_db", rows)

    result = read_order_source(connection)  # type: ignore[arg-type]

    assert result.extracted_at.isoformat() == "2026-09-30T00:00:00+00:00"
    assert len(result.orders) == 1
    assert result.order_items.loc[0, "quantity"] == 2
    assert result.claim_items.empty
    assert str(result.claim_items["quantity"].dtype) == "int64"
    assert connection.commands == [
        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY",
        "SELECT transaction_timestamp()",
    ]


def test_pet_reader_uses_member_db_and_keeps_birth_date() -> None:
    rows = {"pet": (["pet_id", "user_id", "birth_date"], [(1, 2, None)])}
    connection = _Connection("member_db", rows)

    result = read_pet_source(connection)  # type: ignore[arg-type]

    assert result.pets.loc[0, "user_id"] == 2
    assert connection.commands[0].endswith("READ ONLY")


def test_wrong_database_or_existing_transaction_is_rejected() -> None:
    connection = _Connection("product_db", {})
    with pytest.raises(ValueError, match="order_db"):
        read_order_source(connection)  # type: ignore[arg-type]
    connection.info.dbname = "order_db"
    connection.info.transaction_status = TransactionStatus.INTRANS
    with pytest.raises(ValueError, match="autocommit"):
        read_order_source(connection)  # type: ignore[arg-type]
