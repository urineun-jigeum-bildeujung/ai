"""최종 Test 실행권과 기록을 실제 Test 데이터 없이 검증합니다."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import run_frozen_service_test as runner


@pytest.fixture
def frozen_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    validation = tmp_path / "validation.json"
    validation.write_text("{}", encoding="utf-8")
    paths = {name: tmp_path / f"{name}.csv" for name in runner.SOURCE_NAMES}
    for path in paths.values():
        path.write_text("source", encoding="utf-8")
    manifest = {
        "validation_end_at": "2026-06-01T00:00:00Z",
        "observation_end_at_assumption": "2026-09-01T00:00:00Z",
        "source_sha256": {
            name: runner._file_sha256(path) for name, path in paths.items()
        },
        "evaluation": {
            "horizon_days": 30,
            "bootstrap_replicates": 2,
            "bootstrap_random_seed": 42,
        },
    }
    (frozen / "pretest-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.setattr(
        runner,
        "validate_frozen_inputs",
        lambda *args: {
            "artifact_ids": {"xgboost_aft": "aft-id", "lightgbm": "lgbm-id"}
        },
    )
    return frozen, validation, paths


def test_existing_execution_rejected_before_preflight_or_source_read(
    frozen_setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen, validation, paths = frozen_setup
    (frozen.parent / "frozen-final-test").mkdir()
    monkeypatch.setattr(
        runner,
        "validate_frozen_inputs",
        lambda *args: pytest.fail("기존 실행에서 사전검증도 다시 하면 안 됩니다."),
    )
    monkeypatch.setattr(
        runner,
        "_read_sources",
        lambda *args: pytest.fail("Test를 열면 안 됩니다."),
    )

    with pytest.raises(ValueError, match="이미 있습니다"):
        runner.evaluate_frozen_test(frozen, validation, paths)


def test_failed_execution_retains_one_time_marker(
    frozen_setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen, validation, paths = frozen_setup
    monkeypatch.setattr(
        runner,
        "_read_sources",
        lambda *args: (_ for _ in ()).throw(RuntimeError("read")),
    )

    with pytest.raises(RuntimeError, match="read"):
        runner.evaluate_frozen_test(frozen, validation, paths)
    output = frozen.parent / "frozen-final-test"
    state = json.loads((output / "execution-state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["error_type"] == "RuntimeError"
    with pytest.raises(ValueError, match="이미 있습니다"):
        runner.evaluate_frozen_test(frozen, validation, paths)


def test_success_records_test_result_and_does_not_select_model(
    frozen_setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen, validation, paths = frozen_setup
    observed = {}
    monkeypatch.setattr(runner, "_read_sources", lambda _: {"mock": pd.DataFrame()})
    monkeypatch.setattr(runner, "_reference_probability", lambda *args, **kwargs: 0.2)

    def fake_test_rows(*args, **kwargs):
        observed["validation_end_at"] = kwargs["validation_end_at"]
        observed["observation_end_at"] = kwargs["observation_end_at"]
        return pd.DataFrame({"user_id": ["1", "2"]})

    monkeypatch.setattr(runner, "_test_rows", fake_test_rows)
    monkeypatch.setattr(
        runner,
        "add_split_ipcw_weights",
        lambda rows, **kwargs: rows.assign(ipcw_outcome_known=[True, False]),
    )
    monkeypatch.setattr(
        runner,
        "load_model_artifact",
        lambda path: SimpleNamespace(
            artifact_id="aft-id" if path.name == "xgboost_aft" else "lgbm-id"
        ),
    )
    monkeypatch.setattr(
        runner,
        "predict_artifact_probability",
        lambda artifact, rows: pd.Series([0.2, 0.3], index=rows.index),
    )
    monkeypatch.setattr(
        runner,
        "_evaluate_candidate",
        lambda rows, probability, **kwargs: (
            {"validation_sample_count": 2, "ipcw_c_index": 0.6},
            pd.DataFrame({"model": [kwargs["model_name"]], "bin": [1]}),
        ),
    )
    monkeypatch.setattr(
        runner,
        "bootstrap_ipcw_brier_pair_difference_by_user",
        lambda rows, **kwargs: SimpleNamespace(
            summary={"point_brier_improvement": 0.0},
            trials=pd.DataFrame({"replicate": [1], "improvement": [0.0]}),
        ),
    )

    result_path = runner.evaluate_frozen_test(frozen, validation, paths)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    state = json.loads(
        (result_path.parent / "execution-state.json").read_text(encoding="utf-8")
    )
    assert result["test_evaluated"] is True
    assert result["sample_count"] == 2
    assert result["outcome_known_count"] == 1
    assert [value["model"] for value in result["calibration_bins"]] == [
        "xgboost_aft",
        "lightgbm",
    ]
    assert all("test_sample_count" in row for row in result["model_metrics"])
    assert "selected_model" not in result
    assert observed["validation_end_at"] < observed["observation_end_at"]
    assert state["status"] == "completed"
    assert state["result_sha256"] == runner._file_sha256(result_path)


def test_test_population_excludes_validation_anchors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test 경계를 독립적으로 적용하고 원본의 미래 결제 주문을 먼저 제외합니다."""
    validation_end = pd.Timestamp("2026-06-01T00:00:00Z")
    observation_end = pd.Timestamp("2026-09-01T00:00:00Z")
    seen = {}

    def eligible(sources, *, validation_end_at):
        seen["cutoff"] = validation_end_at
        return {name: name for name in runner.SOURCE_NAMES}

    monkeypatch.setattr(runner, "_eligible_training_sources", eligible)
    monkeypatch.setattr(
        runner,
        "quarantine_unrestorable_orders",
        lambda *args, **kwargs: SimpleNamespace(
            orders="orders",
            order_items="items",
            status_histories="histories",
            claims="claims",
            claim_items="claim_items",
        ),
    )
    monkeypatch.setattr(runner, "build_order_status_intervals", lambda _: "status")
    monkeypatch.setattr(
        runner, "build_order_item_quantity_intervals", lambda *args: "quantity"
    )
    monkeypatch.setattr(
        runner, "build_valid_purchase_item_intervals", lambda *args: "valid"
    )
    monkeypatch.setattr(
        runner, "build_operational_event_intervals", lambda *args: "events"
    )
    monkeypatch.setattr(
        runner,
        "build_temporal_service_training_samples",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "anchor_at": pd.to_datetime(
                    ["2026-06-01T00:00:00Z", "2026-06-02T00:00:00Z"]
                ),
                "event_observed": [False, False],
                "duration_days": [None, None],
            }
        ),
    )

    rows = runner._test_rows(
        {"pets": "pets"},
        validation_end_at=validation_end,
        observation_end_at=observation_end,
    )

    assert seen["cutoff"] == observation_end
    assert len(rows) == 1
    assert rows.iloc[0]["anchor_at"] > validation_end
    assert rows.iloc[0]["split"] == "test"
    assert rows.iloc[0]["split_end_at"] == observation_end
