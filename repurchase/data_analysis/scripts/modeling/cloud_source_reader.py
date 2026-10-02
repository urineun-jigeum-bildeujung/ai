"""공용 PostgreSQL 원천을 변경 없이 재구매 파이프라인 입력으로 읽습니다.

연결과 인증 정보는 호출자가 전달합니다. 서로 다른 DB의 스냅샷 시각은
독립적이므로 하나의 원자적 스냅샷이라고 주장하지 않습니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import psycopg
from psycopg.pq import TransactionStatus


@dataclass(frozen=True)
class OrderSourceSnapshot:
    extracted_at: pd.Timestamp
    orders: pd.DataFrame
    order_items: pd.DataFrame
    status_histories: pd.DataFrame
    claims: pd.DataFrame
    claim_items: pd.DataFrame


@dataclass(frozen=True)
class PetSourceSnapshot:
    extracted_at: pd.Timestamp
    pets: pd.DataFrame


_ORDER_QUERIES = {
    "orders": """SELECT id AS order_id, member_id AS user_id, ordered_at,
                         paid_at, order_status, purchase_type
                  FROM public.orders ORDER BY id""",
    "order_items": """SELECT id AS order_item_id, order_id, product_id,
                              product_group_id_snapshot, category_code_snapshot,
                              is_replenishable_snapshot, pet_id, quantity,
                              item_status, cancelled_quantity, returned_quantity
                       FROM public.order_items ORDER BY id""",
    "status_histories": """SELECT id AS history_id, order_id, from_status,
                                   to_status, changed_at
                            FROM public.order_status_histories ORDER BY id""",
    "claims": """SELECT id AS claim_id, order_id, claim_type, claim_status,
                          requested_at, completed_at
                   FROM public.order_claims ORDER BY id""",
    "claim_items": """SELECT id AS claim_item_id, claim_id, order_item_id,
                               quantity
                        FROM public.order_claim_items ORDER BY id""",
}
_PET_QUERY = """SELECT id AS pet_id, member_id AS user_id, birth_date
                FROM public.pet ORDER BY id"""


def _require_fresh_connection(connection: psycopg.Connection[Any], dbname: str) -> None:
    """기존 트랜잭션이나 잘못된 DB에서 읽기 시작하지 않도록 막습니다."""
    if (
        connection.info.dbname != dbname
        or not connection.autocommit
        or connection.info.transaction_status != TransactionStatus.IDLE
    ):
        raise ValueError(f"독립된 autocommit {dbname} 연결이 필요합니다.")


def _read_rows(connection: psycopg.Connection[Any], query: str) -> pd.DataFrame:
    """커서의 컬럼명을 유지하며 빈 테이블도 같은 구조로 반환합니다."""
    with connection.cursor() as cursor:
        cursor.execute(query)
        columns = [column.name for column in cursor.description or ()]
        return pd.DataFrame.from_records(cursor.fetchall(), columns=columns)


def _read_timestamp(connection: psycopg.Connection[Any]) -> pd.Timestamp:
    result = connection.execute("SELECT transaction_timestamp()").fetchone()
    if result is None:
        raise ValueError("원천 스냅샷 시각을 읽지 못했습니다.")
    return pd.Timestamp(result[0]).tz_convert("UTC")


def read_order_source(connection: psycopg.Connection[Any]) -> OrderSourceSnapshot:
    """주문·상태·클레임을 한 DB의 동일한 읽기 전용 스냅샷에서 읽습니다."""
    _require_fresh_connection(connection, "order_db")
    with connection.transaction():
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        extracted_at = _read_timestamp(connection)
        rows = {
            name: _read_rows(connection, query)
            for name, query in _ORDER_QUERIES.items()
        }
    for column in ("quantity", "cancelled_quantity", "returned_quantity"):
        rows["order_items"][column] = rows["order_items"][column].astype("int64")
    rows["claim_items"]["quantity"] = rows["claim_items"]["quantity"].astype("int64")
    return OrderSourceSnapshot(extracted_at=extracted_at, **rows)


def read_pet_source(connection: psycopg.Connection[Any]) -> PetSourceSnapshot:
    """반려동물 생일을 회원 DB의 읽기 전용 스냅샷에서 읽습니다."""
    _require_fresh_connection(connection, "member_db")
    with connection.transaction():
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        extracted_at = _read_timestamp(connection)
        pets = _read_rows(connection, _PET_QUERY)
    return PetSourceSnapshot(extracted_at=extracted_at, pets=pets)
