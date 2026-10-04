"""Synthetic snapshots must be reproducible, isolated, and source-contract compatible."""

from datetime import datetime
from pathlib import Path

import pytest

from scripts.generate_independent_synthetic_sources import generate_pair
from scripts.modeling.operational_event_intervals import (
    build_operational_event_intervals,
)
from scripts.modeling.operational_quantity_intervals import (
    build_order_item_quantity_intervals,
)
from scripts.modeling.operational_source_quarantine import (
    quarantine_unrestorable_orders,
)
from scripts.modeling.operational_status_intervals import build_order_status_intervals
from scripts.modeling.operational_validity_intervals import (
    build_valid_purchase_item_intervals,
)
from scripts.run_service_model_comparison import _read_sources


def test_pair_has_distinct_ids_and_runs_through_source_pipeline(tmp_path: Path) -> None:
    results = generate_pair(
        tmp_path,
        start=datetime.fromisoformat("2026-10-04T00:00:00+00:00"),
        user_count=6,
    )
    seen: set[str] = set()
    for result in results:
        directory = Path(result["directory"])
        names = ("orders", "order_items", "histories", "claims", "claim_items", "pets")
        sources = _read_sources({name: directory / f"{name}.csv" for name in names})
        ids = set(sources["orders"]["order_id"])
        assert not seen.intersection(ids)
        seen.update(ids)
        quarantine = quarantine_unrestorable_orders(
            *(sources[name] for name in names[:-1])
        )
        assert quarantine.missing_history_order_count == 0
        assert quarantine.status_mismatch_order_count == 0
        status = build_order_status_intervals(quarantine.status_histories)
        quantity = build_order_item_quantity_intervals(
            quarantine.orders,
            quarantine.order_items,
            quarantine.claims,
            quarantine.claim_items,
        )
        valid = build_valid_purchase_item_intervals(status, quantity)
        events = build_operational_event_intervals(
            valid, quarantine.orders, quarantine.order_items, sources["pets"]
        )
        assert len(events.pet_targets) > 0


def test_generator_never_overwrites_existing_snapshot(tmp_path: Path) -> None:
    start = datetime.fromisoformat("2026-10-04T00:00:00+00:00")
    generate_pair(tmp_path, start=start, user_count=2)
    with pytest.raises(ValueError, match="이미 존재"):
        generate_pair(tmp_path, start=start, user_count=2)


def test_generator_rejects_window_too_short_for_landmark_calibration(
    tmp_path: Path,
) -> None:
    start = datetime.fromisoformat("2026-10-04T00:00:00+00:00")
    with pytest.raises(ValueError, match="450일 이상"):
        generate_pair(tmp_path, start=start, duration_days=180)
