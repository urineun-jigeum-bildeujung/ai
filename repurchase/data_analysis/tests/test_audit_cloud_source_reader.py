"""실제 DB 접속 없이 읽기 전용 점검 명령의 경계를 확인합니다."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.audit_cloud_source_reader import (
    _require_pet_ownership,
    audit_snapshots,
    main,
    run_audit,
)
from scripts.modeling.cloud_source_reader import OrderSourceSnapshot, PetSourceSnapshot
from scripts.modeling.operational_orders import OperationalOrderError


def _snapshot_pair() -> tuple[OrderSourceSnapshot, PetSourceSnapshot]:
    frame = pd.DataFrame({"id": [1]})
    order = OrderSourceSnapshot(
        extracted_at=pd.Timestamp("2026-10-02T00:00:00Z"),
        orders=frame,
        order_items=frame,
        status_histories=frame,
        claims=frame,
        claim_items=frame,
    )
    pet = PetSourceSnapshot(
        extracted_at=pd.Timestamp("2026-10-02T00:00:01Z"),
        pets=pd.DataFrame({"pet_id": ["pet1"], "user_id": ["u1"]}),
    )
    return order, pet


def test_audit_counts_and_passes_explicit_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order, pet = _snapshot_pair()
    captured: dict[str, object] = {}

    def prepare(*args: object, **kwargs: object) -> SimpleNamespace:
        captured["args"] = args
        captured["cutoff"] = kwargs["as_of_timestamp"]
        return SimpleNamespace(
            valid_items=pd.DataFrame({"user_id": ["u1"], "pet_id": ["pet1"]}),
            all_purchase_events=pd.DataFrame(index=range(2)),
            pet_purchase_events=pd.DataFrame(index=range(1)),
            excluded_late_birth_item_count=1,
        )

    monkeypatch.setattr(
        "scripts.audit_cloud_source_reader.prepare_operational_purchase_inputs_as_of",
        prepare,
    )
    result = audit_snapshots(
        order, pet, as_of_timestamp=pd.Timestamp("2026-09-29T15:44:00+09:00")
    )

    assert result.order_count == 1
    assert result.all_purchase_event_count == 2
    assert result.excluded_late_birth_item_count == 1
    assert captured["cutoff"] == pd.Timestamp("2026-09-29T06:44:00Z")
    assert len(captured["args"]) == 6  # type: ignore[arg-type]


def test_pet_ownership_mismatch_is_rejected() -> None:
    """클라우드 원천의 다른 회원 반려동물 연결을 조용히 채택하지 않습니다."""
    items = pd.DataFrame({"user_id": ["u1", "u1"], "pet_id": ["pet-other", None]})
    pets = pd.DataFrame({"pet_id": ["pet-other"], "user_id": ["u2"]})
    with pytest.raises(OperationalOrderError, match="소유자"):
        _require_pet_ownership(items, pets)


def test_audit_rejects_future_cutoff_before_preparation() -> None:
    order, pet = _snapshot_pair()
    with pytest.raises(ValueError, match="추출 시각"):
        audit_snapshots(
            order, pet, as_of_timestamp=pd.Timestamp("2026-10-02T00:00:02Z")
        )


def test_missing_dsn_and_naive_cutoff_do_not_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REPURCHASE_ORDER_DATABASE_DSN", raising=False)
    monkeypatch.delenv("REPURCHASE_MEMBER_DATABASE_DSN", raising=False)
    with pytest.raises(ValueError, match="REPURCHASE_ORDER_DATABASE_DSN"):
        run_audit(pd.Timestamp("2026-09-29T00:00:00Z"))
    assert main(["--as-of", "2026-09-29T00:00:00"]) == 2


def test_both_source_connections_limit_query_and_idle_transaction_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """접속 제한뿐 아니라 각 원천 조회 세션의 실행·유휴 시간을 제한합니다."""
    order, pet = _snapshot_pair()
    connections: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setenv("REPURCHASE_ORDER_DATABASE_DSN", "dbname=order_db")
    monkeypatch.setenv("REPURCHASE_MEMBER_DATABASE_DSN", "dbname=member_db")

    def connect(dsn: str, **kwargs: object) -> nullcontext[str]:
        connections.append((dsn, kwargs))
        return nullcontext(dsn)

    monkeypatch.setattr("scripts.audit_cloud_source_reader.psycopg.connect", connect)
    monkeypatch.setattr(
        "scripts.audit_cloud_source_reader.read_order_source", lambda _: order
    )
    monkeypatch.setattr(
        "scripts.audit_cloud_source_reader.read_pet_source", lambda _: pet
    )
    monkeypatch.setattr(
        "scripts.audit_cloud_source_reader.audit_snapshots",
        lambda *_args, **_kwargs: "ok",
    )

    assert run_audit(pd.Timestamp("2026-09-29T00:00:00Z")) == "ok"
    assert [dsn for dsn, _ in connections] == ["dbname=order_db", "dbname=member_db"]
    assert all(kwargs["connect_timeout"] == 10 for _, kwargs in connections)
    assert all(
        kwargs["options"]
        == "-c statement_timeout=300000 -c idle_in_transaction_session_timeout=60000"
        for _, kwargs in connections
    )
