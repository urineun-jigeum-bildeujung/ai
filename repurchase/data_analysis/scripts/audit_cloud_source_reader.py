"""실제 공용 DB 원천의 재구매 입력 가능 여부를 읽기 전용으로 점검합니다.

DSN은 환경변수로만 받으며 출력에는 행 데이터나 접속 정보를 포함하지 않습니다.
두 DB의 원자적 스냅샷은 제공되지 않으므로 각 추출 시각을 따로 기록합니다.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import NoReturn

import pandas as pd
import psycopg

from scripts.modeling.cloud_source_reader import (
    OrderSourceSnapshot,
    PetSourceSnapshot,
    read_order_source,
    read_pet_source,
)
from scripts.modeling.operational_orders import OperationalOrderError
from scripts.modeling.operational_purchase_inputs import (
    prepare_operational_purchase_inputs_as_of,
)

ORDER_DSN_ENV = "REPURCHASE_ORDER_DATABASE_DSN"
MEMBER_DSN_ENV = "REPURCHASE_MEMBER_DATABASE_DSN"
# 연결 수립 제한과 별개로, 공용 DB의 장시간 조회·유휴 트랜잭션을 차단합니다.
_SESSION_OPTIONS = (
    "-c statement_timeout=300000 -c idle_in_transaction_session_timeout=60000"
)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ValueError(message)


@dataclass(frozen=True)
class SourceAuditSummary:
    as_of_timestamp: str
    order_extracted_at: str
    member_extracted_at: str
    order_count: int
    order_item_count: int
    status_history_count: int
    claim_count: int
    claim_item_count: int
    pet_count: int
    valid_order_item_count: int
    all_purchase_event_count: int
    pet_purchase_event_count: int
    excluded_late_birth_item_count: int


def _utc_timestamp(value: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError("--as-of에는 시간대가 있는 ISO 시각이 필요합니다.") from error
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("--as-of에는 시간대가 있는 ISO 시각이 필요합니다.")
    return timestamp.tz_convert("UTC")


def _require_pet_ownership(valid_items: pd.DataFrame, pets: pd.DataFrame) -> None:
    """유효 구매의 반려동물이 주문 회원 소유인지 공용 DB 키로 확인합니다."""
    required_items = {"user_id", "pet_id"}
    required_pets = {"pet_id", "user_id"}
    if not required_items.issubset(valid_items) or not required_pets.issubset(pets):
        raise OperationalOrderError("반려동물 소유 관계 검증 열이 누락됐습니다.")
    specified = valid_items.loc[valid_items["pet_id"].notna(), ["user_id", "pet_id"]]
    if specified.empty:
        return
    owners = pets.loc[:, ["pet_id", "user_id"]].rename(
        columns={"user_id": "pet_owner_id"}
    )
    joined = specified.merge(owners, on="pet_id", how="left", validate="many_to_one")
    if (
        joined["user_id"].isna().any()
        or joined["pet_owner_id"].isna().any()
        or joined["user_id"].ne(joined["pet_owner_id"]).any()
    ):
        raise OperationalOrderError("주문 회원과 반려동물 소유자가 일치하지 않습니다.")


def audit_snapshots(
    orders: OrderSourceSnapshot,
    pets: PetSourceSnapshot,
    *,
    as_of_timestamp: pd.Timestamp,
) -> SourceAuditSummary:
    """명시한 관측 컷에서 과거 상태와 반려동물 이력을 검증합니다."""
    cutoff = _utc_timestamp(as_of_timestamp.isoformat())
    if cutoff > orders.extracted_at or cutoff > pets.extracted_at:
        raise ValueError("관측 컷이 원천 DB 추출 시각보다 늦습니다.")
    prepared = prepare_operational_purchase_inputs_as_of(
        orders.orders,
        orders.order_items,
        pets.pets,
        orders.status_histories,
        orders.claims,
        orders.claim_items,
        as_of_timestamp=cutoff,
    )
    _require_pet_ownership(prepared.valid_items, pets.pets)
    return SourceAuditSummary(
        as_of_timestamp=cutoff.isoformat(),
        order_extracted_at=orders.extracted_at.isoformat(),
        member_extracted_at=pets.extracted_at.isoformat(),
        order_count=len(orders.orders),
        order_item_count=len(orders.order_items),
        status_history_count=len(orders.status_histories),
        claim_count=len(orders.claims),
        claim_item_count=len(orders.claim_items),
        pet_count=len(pets.pets),
        valid_order_item_count=len(prepared.valid_items),
        all_purchase_event_count=len(prepared.all_purchase_events),
        pet_purchase_event_count=len(prepared.pet_purchase_events),
        excluded_late_birth_item_count=prepared.excluded_late_birth_item_count,
    )


def read_cloud_snapshots(
    *, prompt_password: bool = False
) -> tuple[OrderSourceSnapshot, PetSourceSnapshot]:
    """두 원천 DB를 독립된 읽기 전용 스냅샷으로 읽고 연결을 닫습니다."""
    order_dsn = os.environ.get(ORDER_DSN_ENV)
    member_dsn = os.environ.get(MEMBER_DSN_ENV)
    if not order_dsn or not member_dsn:
        raise ValueError(f"{ORDER_DSN_ENV}와 {MEMBER_DSN_ENV}가 모두 필요합니다.")
    order_password = getpass.getpass("order_db 비밀번호: ") if prompt_password else None
    with psycopg.connect(
        order_dsn,
        autocommit=True,
        connect_timeout=10,
        options=_SESSION_OPTIONS,
        **({"password": order_password} if order_password is not None else {}),
    ) as connection:
        orders = read_order_source(connection)
    member_password = (
        getpass.getpass("member_db 비밀번호: ") if prompt_password else None
    )
    with psycopg.connect(
        member_dsn,
        autocommit=True,
        connect_timeout=10,
        options=_SESSION_OPTIONS,
        **({"password": member_password} if member_password is not None else {}),
    ) as connection:
        pets = read_pet_source(connection)
    return orders, pets


def run_audit(
    as_of_timestamp: pd.Timestamp, *, prompt_password: bool = False
) -> SourceAuditSummary:
    """DB 연결을 열어 점검하며 비밀번호·원천 행은 디스크에 쓰지 않습니다."""
    orders, pets = read_cloud_snapshots(prompt_password=prompt_password)
    return audit_snapshots(orders, pets, as_of_timestamp=as_of_timestamp)


def main(argv: list[str] | None = None) -> int:
    parser = _ArgumentParser(prog="audit-cloud-source-reader")
    parser.add_argument("--as-of", required=True, help="시간대 포함 관측 컷 ISO 시각")
    parser.add_argument(
        "--prompt-password",
        action="store_true",
        help="로컬 접속 시 각 DB 비밀번호를 화면에 표시하지 않고 입력",
    )
    try:
        args = parser.parse_args(argv)
        summary = run_audit(
            _utc_timestamp(args.as_of), prompt_password=args.prompt_password
        )
    except (ValueError, OperationalOrderError) as error:
        print(
            json.dumps(
                {
                    "event": "repurchase_source_audit_rejected",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    except psycopg.Error as error:
        # 드라이버 예외 메시지에는 연결 주소 등이 들어갈 수 있어 그대로 출력하지 않습니다.
        print(
            json.dumps(
                {
                    "event": "repurchase_source_audit_failed",
                    "error_type": type(error).__name__,
                }
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {"event": "repurchase_source_audit_passed", **asdict(summary)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
