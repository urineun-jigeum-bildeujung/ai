"""최종 Test 이전의 후보 고정은 입력 계약을 먼저 검증합니다."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import freeze_service_models as freezer


def test_build_manifest_fixes_evaluation_contract() -> None:
    """실험 JSON이 평가 지표를 임의로 바꾸지 못하게 독립 계약을 고정합니다."""
    comparison = {
        key: {"source_sha256": {}, "code_sha256": {}}.get(key, "value")
        for key in (
            "source_sha256",
            "code_sha256",
            "observation_end_at_assumption",
            "train_end_at",
            "validation_end_at",
            "train_fraction",
            "validation_fraction",
            "feature_generation_version",
            "runtime_versions",
            "model_configuration",
            "evaluation_population_policy",
        )
    }
    manifest = freezer.build_pretest_manifest(comparison)

    assert manifest["aft_configuration"] == freezer.FROZEN_AFT
    assert manifest["evaluation"]["horizon_days"] == 30
    assert manifest["evaluation"]["bootstrap_replicates"] == 1000
    assert manifest["schema_version"] == 1


def test_freeze_rejects_validation_failure_before_reading_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test 평가가 들어간 결과는 원천 파일을 열기 전에 거절합니다."""
    result = tmp_path / "validation.json"
    result.write_text(json.dumps({"test_evaluated": True}), encoding="utf-8")
    monkeypatch.setattr(
        freezer,
        "_read_sources",
        lambda paths: pytest.fail("거절된 비교 결과로 원천을 읽으면 안 됩니다."),
    )

    with pytest.raises(ValueError, match="Test 평가가 없는"):
        freezer.freeze_models(result, {}, tmp_path / "frozen")

    assert not (tmp_path / "frozen").exists()


def test_freeze_never_overwrites_existing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """같은 출력 경로 재사용으로 고정된 모델 파일을 덮어쓰지 않습니다."""
    result = tmp_path / "validation.json"
    result.write_text('{"test_evaluated": false}', encoding="utf-8")
    output = tmp_path / "frozen"
    output.mkdir()
    monkeypatch.setattr(freezer, "build_pretest_manifest", lambda comparison: {})
    monkeypatch.setattr(freezer, "validate_manifest", lambda *args: None)

    with pytest.raises(ValueError, match="덮어쓸 수 없습니다"):
        freezer.freeze_models(result, {}, output)


def test_refit_source_filter_keeps_late_claims_for_eligible_orders() -> None:
    """미래 주문은 빼되 과거 주문의 이후 완료 클레임은 수량 검증용으로 보존합니다."""
    sources = {
        "orders": pd.DataFrame(
            {
                "order_id": ["1", "2"],
                "paid_at": ["2026-01-01T00:00:00Z", "2026-08-01T00:00:00Z"],
            }
        ),
        "order_items": pd.DataFrame(
            {"order_item_id": ["11", "22"], "order_id": ["1", "2"]}
        ),
        "histories": pd.DataFrame(
            {"history_id": ["101", "202"], "order_id": ["1", "2"]}
        ),
        "claims": pd.DataFrame(
            {
                "claim_id": ["111", "222"],
                "order_id": ["1", "2"],
                "completed_at": ["2026-08-02T00:00:00Z"] * 2,
            }
        ),
        "claim_items": pd.DataFrame(
            {"claim_item_id": ["1111", "2222"], "claim_id": ["111", "222"]}
        ),
        "pets": pd.DataFrame({"pet_id": ["5"]}),
    }

    filtered = freezer._eligible_training_sources(
        sources, validation_end_at=pd.Timestamp("2026-06-01T00:00:00Z")
    )

    assert filtered["orders"]["order_id"].tolist() == ["1"]
    assert filtered["order_items"]["order_item_id"].tolist() == ["11"]
    assert filtered["histories"]["history_id"].tolist() == ["101"]
    assert filtered["claims"]["claim_id"].tolist() == ["111"]
    assert filtered["claim_items"]["claim_item_id"].tolist() == ["1111"]
    assert sources["orders"]["order_id"].tolist() == ["1", "2"]
