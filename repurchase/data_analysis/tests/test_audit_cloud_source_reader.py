"""실제 DB 접속 없이 읽기 전용 점검 명령의 경계를 확인합니다."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.audit_cloud_source_reader import (
    audit_snapshots,
    main,
    run_audit,
)
from scripts.modeling.cloud_source_reader import OrderSourceSnapshot, PetSourceSnapshot


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
        extracted_at=pd.Timestamp("2026-10-02T00:00:01Z"), pets=frame
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
            valid_items=pd.DataFrame(index=range(1)),
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
