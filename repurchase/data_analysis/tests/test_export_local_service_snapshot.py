"""로컬 추출본 완료 표시와 손상 감지를 검사합니다."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import export_local_service_snapshot as exporter


def test_verify_snapshot_checks_six_files_without_exposing_rows(tmp_path: Path) -> None:
    files = {}
    for name in ("orders", "order_items", "histories", "claims", "claim_items", "pets"):
        path = tmp_path / f"{name}.csv"
        pd.DataFrame({"sample_id": [1]}).to_csv(path, index=False)
        files[name] = {
            "filename": path.name,
            "rows": 1,
            "sha256": exporter._sha256(path),
        }
    manifest = {
        "order_extracted_at": "2026-10-03T00:00:00+00:00",
        "member_extracted_at": "2026-10-03T00:00:01+00:00",
        "files": files,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert exporter.verify_snapshot(tmp_path)["files"]["orders"]["rows"] == 1
    (tmp_path / "orders.csv").write_text("sample_id\n2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="해시"):
        exporter.verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_incomplete_export(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="완전하지"):
        exporter.verify_snapshot(tmp_path)


def test_nullable_pet_id_is_serialized_as_integer() -> None:
    items = pd.DataFrame({"pet_id": [12.0, float("nan")]})
    normalized = exporter._serialize_nullable_pet_id(items)
    assert normalized["pet_id"].astype("string").tolist()[0] == "12"
    assert pd.isna(normalized["pet_id"].iloc[1])


def test_nullable_pet_id_rejects_precision_risk() -> None:
    items = pd.DataFrame({"pet_id": [float(2**53 + 2)]})
    with pytest.raises(ValueError, match="정밀도"):
        exporter._serialize_nullable_pet_id(items)
