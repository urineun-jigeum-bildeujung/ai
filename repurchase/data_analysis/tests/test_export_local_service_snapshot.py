"""로컬 추출본 완료 표시와 손상 감지를 검사합니다."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import export_local_service_snapshot as exporter


def test_verify_snapshot_checks_six_files_without_exposing_rows(tmp_path: Path) -> None:
    """여섯 원천의 건수·해시가 맞을 때만 완료 추출본으로 인정합니다."""
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
    """필수 파일 목록이 빠진 manifest를 거부합니다."""
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="완전하지"):
        exporter.verify_snapshot(tmp_path)


def test_nullable_pet_id_is_serialized_as_integer() -> None:
    """nullable 반려동물 ID의 소수점 문자열을 제거합니다."""
    items = pd.DataFrame({"pet_id": [12.0, float("nan")]})
    normalized = exporter._serialize_nullable_pet_id(items)
    assert normalized["pet_id"].astype("string").tolist()[0] == "12"
    assert pd.isna(normalized["pet_id"].iloc[1])


def test_nullable_pet_id_rejects_precision_risk() -> None:
    """부동소수점 변환으로 정밀도 손실 위험이 있는 ID를 거부합니다."""
    items = pd.DataFrame({"pet_id": [float(2**53 + 2)]})
    with pytest.raises(ValueError, match="정밀도"):
        exporter._serialize_nullable_pet_id(items)


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", "disable"),
        ("127.1.2.3", "disable"),
        ("::1", "disable"),
        ("localhost", "disable"),
        ("10.0.0.1", "require"),
        ("db.example.test", "require"),
    ],
)
def test_sslmode_requires_tls_outside_loopback(host: str, expected: str) -> None:
    """로컬 포트포워딩과 원격 DB 연결의 TLS 정책을 구분합니다."""
    assert exporter._sslmode_for_host(host) == expected


def test_prepare_model_order_items_checks_count_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """파생 행 수가 다르면 파생 파일을 만들지 않습니다."""
    original = tmp_path / "order_items.csv"
    pd.DataFrame({"order_item_id": [1], "pet_id": [12]}).to_csv(original, index=False)
    monkeypatch.setattr(
        exporter,
        "verify_snapshot",
        lambda _: {"files": {"order_items": {"rows": 1, "sha256": "test"}}},
    )
    monkeypatch.setattr(
        exporter,
        "_serialize_nullable_pet_id",
        lambda items: pd.concat([items, items], ignore_index=True),
    )
    with pytest.raises(ValueError, match="파생 파일 건수"):
        exporter.prepare_model_order_items(tmp_path)
    assert not (tmp_path / "order_items_model.csv").exists()
    assert not (tmp_path / "order_items_model.json").exists()
