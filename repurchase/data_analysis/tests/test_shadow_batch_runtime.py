"""AFT 내부 검증의 비노출 출력과 명시적 쓰기 게이트를 검증합니다."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from scripts.modeling.artifacts import LoadedModelArtifact, ModelArtifactError
from scripts.modeling.prediction_publications import select_latest_published_predictions
from scripts.modeling.shadow_batch import prepare_shadow_publication
from scripts.run_repurchase_batch import (
    CONTRACT_REJECTION_EXIT_CODE,
    UNEXPECTED_FAILURE_EXIT_CODE,
    main,
)
from scripts.shadow_batch_runtime import (
    ShadowBatchContractError,
    ShadowDatabaseError,
    _check_cut,
    _summary,
    run_live_shadow_batch,
)
from scripts.shadow_batch_runtime import (
    _artifact as load_shadow_artifact,
)


def _artifact() -> LoadedModelArtifact:
    return LoadedModelArtifact(
        family="xgboost_aft",
        model=object(),  # 외부 모델 호출은 이 테스트에서 분리합니다.
        feature_columns=("history_interval_count",),
        feature_generation_version=2,
        horizon_days=30,
        artifact_id="fixed-aft-v2",
    )  # type: ignore[arg-type]


def test_shadow_preparation_preserves_storage_contract_without_exposure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """예측 대상 키·확률은 저장 계약에 맞고 최신 서비스 뷰에서는 빠집니다."""
    from scripts.modeling import shadow_batch

    class FakeQuarantine:
        orders = pd.DataFrame({"order_id": [1, 2]})
        order_items = status_histories = claims = claim_items = pd.DataFrame()
        missing_history_order_count = 1
        missing_history_paid_order_count = 1
        missing_history_order_item_count = 2
        status_mismatch_order_count = 0
        status_mismatch_order_item_count = 0

    monkeypatch.setattr(
        shadow_batch, "quarantine_unrestorable_orders", lambda *a, **k: FakeQuarantine()
    )
    monkeypatch.setattr(
        shadow_batch, "build_order_status_intervals", lambda *a: pd.DataFrame()
    )
    monkeypatch.setattr(
        shadow_batch, "build_order_item_quantity_intervals", lambda *a: pd.DataFrame()
    )
    monkeypatch.setattr(
        shadow_batch, "build_valid_purchase_item_intervals", lambda *a: pd.DataFrame()
    )
    monkeypatch.setattr(
        shadow_batch, "build_operational_event_intervals", lambda *a: object()
    )
    monkeypatch.setattr(
        shadow_batch,
        "predict_temporal_service_current_probability",
        lambda *a, **k: pd.DataFrame(
            {
                "user_id": [42, 42],
                "pet_id": [12.0, float("nan")],
                "target_id": [9, 10],
                "conditional_repurchase_probability": [0.4, 0.3],
            }
        ),
    )
    sources = {
        name: pd.DataFrame()
        for name in ("order_items", "histories", "claims", "claim_items", "pets")
    }
    sources["orders"] = pd.DataFrame({"order_id": [1, 2, 3]})
    sources["order_items"] = pd.DataFrame({"order_item_id": [1, 2, 3, 4]})

    prepared = prepare_shadow_publication(
        sources,
        _artifact(),
        as_of_timestamp=pd.Timestamp("2026-01-02T00:00:00Z"),
        created_at=pd.Timestamp("2026-01-03T00:00:00Z"),
        window_days=30,
        publication_id="shadow-run-1",
    )

    assert prepared.source_order_count == 3
    assert prepared.source_order_item_count == 4
    assert prepared.quarantined_order_count == 1
    assert prepared.quarantined_missing_history_order_count == 1
    assert prepared.quarantined_missing_history_paid_order_count == 1
    assert prepared.quarantined_missing_history_order_item_count == 2
    assert prepared.quarantined_status_mismatch_order_count == 0
    assert prepared.target_count == 2
    summary = _summary(prepared, mode="shadow-check", inserted=None)
    assert summary.quarantined_missing_history_order_item_count == 2
    assert summary.prediction_count == 2
    assert prepared.results.loc[0, "pet_id"] == 12
    assert pd.isna(prepared.results.loc[1, "pet_id"])
    assert str(prepared.results["pet_id"].dtype) == "Int64"
    assert prepared.results.loc[0, "user_id"] == "42"
    assert prepared.batch.loc[0, "publication_status"] == "SHADOW"
    assert select_latest_published_predictions(prepared.batch, prepared.results).empty


def test_shadow_rejects_wrong_model_and_future_cut() -> None:
    with pytest.raises(ShadowBatchContractError, match="사전 고정한 AFT"):
        load_shadow_artifact(Path("unused"), "another-model")
    with pytest.raises(ModelArtifactError, match="버전 2 AFT"):
        prepare_shadow_publication(
            {},
            replace(_artifact(), family="lightgbm"),
            as_of_timestamp=pd.Timestamp("2026-01-02T00:00:00Z"),
            created_at=pd.Timestamp("2026-01-03T00:00:00Z"),
            window_days=30,
            publication_id="shadow-run-1",
        )
    with pytest.raises(ShadowBatchContractError, match="추출 시각보다 늦습니다"):
        _check_cut(
            pd.Timestamp("2026-01-03T00:00:00Z"),
            "2026-01-02T00:00:00Z",
            "2026-01-02T00:00:01Z",
        )


def test_live_shadow_requires_explicit_write_before_any_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REPURCHASE_ORDER_DATABASE_DSN", raising=False)
    with pytest.raises(ShadowBatchContractError, match="allow-shadow-write"):
        run_live_shadow_batch(
            artifact_directory=Path("unused"),
            expected_artifact_id="fixed-aft-v2",
            as_of_timestamp="2026-01-02T00:00:00Z",
            created_at="2026-01-03T00:00:00Z",
            window_days=30,
            publication_id="shadow-run-1",
            allow_shadow_write=False,
        )


def test_live_shadow_requires_all_database_secrets_before_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "REPURCHASE_ORDER_DATABASE_DSN",
        "REPURCHASE_MEMBER_DATABASE_DSN",
        "REPURCHASE_RESULT_DATABASE_DSN",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ShadowBatchContractError, match="REPURCHASE_ORDER_DATABASE_DSN"):
        run_live_shadow_batch(
            artifact_directory=Path("unused"),
            expected_artifact_id="fixed-aft-v2",
            as_of_timestamp="2026-01-02T00:00:00Z",
            created_at="2026-01-03T00:00:00Z",
            window_days=30,
            publication_id="shadow-run-1",
            allow_shadow_write=True,
        )


def test_shadow_run_cli_rejects_missing_write_consent(capsys: object) -> None:
    exit_code = main(
        [
            "shadow-run",
            "--model-directory",
            "unused",
            "--artifact-id",
            "fixed-aft-v2",
            "--as-of",
            "2026-01-02T00:00:00Z",
            "--created-at",
            "2026-01-03T00:00:00Z",
            "--publication-id",
            "shadow-run-1",
        ]
    )
    payload = json.loads(capsys.readouterr().err)
    assert exit_code == CONTRACT_REJECTION_EXIT_CODE
    assert payload["error_type"] == "ShadowBatchContractError"


def test_shadow_run_cli_classifies_database_failure_as_execution_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: object
) -> None:
    from scripts import run_repurchase_batch

    def fail(**_kwargs: object) -> None:
        raise ShadowDatabaseError("order_db", "08006")

    monkeypatch.setattr(run_repurchase_batch, "run_live_shadow_batch", fail)
    exit_code = main(
        [
            "shadow-run",
            "--model-directory",
            "unused",
            "--artifact-id",
            "fixed-aft-v2",
            "--as-of",
            "2026-01-02T00:00:00Z",
            "--created-at",
            "2026-01-03T00:00:00Z",
            "--publication-id",
            "shadow-run-1",
            "--allow-shadow-write",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == UNEXPECTED_FAILURE_EXIT_CODE
    assert captured.out == ""
    assert len(captured.err.splitlines()) == 1
    assert payload == {
        "database": "order_db",
        "error_type": "ShadowDatabaseError",
        "event": "repurchase_batch_failed",
        "message": "데이터베이스 연결 또는 쓰기에 실패했습니다.",
        "sqlstate": "08006",
        "status": "FAILED",
    }
