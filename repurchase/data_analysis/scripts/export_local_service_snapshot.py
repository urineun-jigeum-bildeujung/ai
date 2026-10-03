"""서비스 모델에 필요한 최소 원천만 로컬 CSV로 보존합니다.

각 DB는 독립된 읽기 전용 스냅샷이며, manifest가 생성된 디렉터리만
완료된 추출본으로 취급합니다. 비밀번호와 원천 행은 로그에 남기지 않습니다.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import ipaddress
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

from scripts.modeling.cloud_source_reader import read_order_source, read_pet_source


class SnapshotDatabaseError(Exception):
    """원천 연결·조회 실패의 DB 위치와 SQLSTATE만 보존합니다."""

    def __init__(self, database: str, sqlstate: str | None) -> None:
        self.database = database
        self.sqlstate = sqlstate
        super().__init__(database)


def _sha256(path: Path) -> str:
    """파일 내용을 스트리밍으로 읽어 SHA-256 지문을 계산합니다."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serialize_nullable_pet_id(items: pd.DataFrame) -> pd.DataFrame:
    """nullable bigint의 pandas float 직렬화로 생기는 `123.0`을 방지합니다."""
    result = items.copy()
    values = pd.to_numeric(result["pet_id"], errors="raise")
    nonnull = values.dropna().to_numpy(dtype="float64")
    if (
        not np.isfinite(nonnull).all()
        or (np.abs(nonnull) > 2**53).any()
        or not np.equal(nonnull, np.floor(nonnull)).all()
    ):
        raise ValueError("pet_id를 정밀도 손실 없이 CSV로 직렬화할 수 없습니다.")
    result["pet_id"] = values.astype("Int64")
    return result


def _sslmode_for_host(host: str) -> str:
    """로컬 포트포워딩만 평문으로 연결하고 원격 주소에는 TLS를 요구합니다."""
    if host.lower() == "localhost":
        return "disable"
    try:
        return "disable" if ipaddress.ip_address(host).is_loopback else "require"
    except ValueError:
        return "require"


def verify_snapshot(directory: Path) -> dict[str, object]:
    """완료 manifest의 여섯 CSV 건수와 해시를 다시 검사합니다."""
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest["files"]
    if set(files) != {
        "orders",
        "order_items",
        "histories",
        "claims",
        "claim_items",
        "pets",
    }:
        raise ValueError("원천 파일 목록이 완전하지 않습니다.")
    for name, metadata in files.items():
        path = directory / f"{name}.csv"
        if metadata["filename"] != path.name or _sha256(path) != metadata["sha256"]:
            raise ValueError(f"{name} 원천 파일 해시가 일치하지 않습니다.")
        rows = sum(len(chunk) for chunk in pd.read_csv(path, chunksize=20_000))
        if rows != metadata["rows"]:
            raise ValueError(f"{name} 원천 파일 건수가 일치하지 않습니다.")
    return manifest


def prepare_model_order_items(directory: Path) -> Path:
    """기존 추출본의 nullable pet_id CSV 문제를 원본 보존 파생 파일로 교정합니다."""
    manifest = verify_snapshot(directory)
    original = directory / "order_items.csv"
    derived = directory / "order_items_model.csv"
    if derived.exists():
        raise ValueError("파생 파일이 이미 있어 덮어쓰지 않습니다.")
    items = pd.read_csv(
        original,
        dtype={
            "order_item_id": "string",
            "order_id": "string",
            "product_id": "string",
            "product_group_id_snapshot": "string",
        },
    )
    normalized = _serialize_nullable_pet_id(items)
    if len(normalized) != manifest["files"]["order_items"]["rows"]:
        raise ValueError("파생 파일 건수가 원본과 다릅니다.")
    old_umask = os.umask(0o077)
    try:
        normalized.to_csv(derived, index=False)
        (directory / "order_items_model.json").write_text(
            json.dumps(
                {
                    "source_sha256": manifest["files"]["order_items"]["sha256"],
                    "derived_sha256": _sha256(derived),
                    "rows": len(normalized),
                    "normalization": "nullable integer pet_id rendered without decimal suffix",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    finally:
        os.umask(old_umask)
    return derived


def export_snapshot(*, host: str, port: int, user: str, output_root: Path) -> Path:
    """비밀번호를 대화형으로 받고 여섯 원천 파일과 검증용 manifest를 씁니다."""
    order_password = getpass.getpass("order_db 비밀번호: ")
    try:
        with psycopg.connect(
            host=host,
            port=port,
            user=user,
            password=order_password,
            dbname="order_db",
            sslmode=_sslmode_for_host(host),
            autocommit=True,
            connect_timeout=10,
            options="-c statement_timeout=300000 -c idle_in_transaction_session_timeout=60000",
        ) as connection:
            orders = read_order_source(connection)
    except psycopg.Error as error:
        raise SnapshotDatabaseError("order_db", error.sqlstate) from None
    del order_password

    member_password = getpass.getpass("member_db 비밀번호: ")
    try:
        with psycopg.connect(
            host=host,
            port=port,
            user=user,
            password=member_password,
            dbname="member_db",
            sslmode=_sslmode_for_host(host),
            autocommit=True,
            connect_timeout=10,
            options="-c statement_timeout=300000 -c idle_in_transaction_session_timeout=60000",
        ) as connection:
            pets = read_pet_source(connection)
    except psycopg.Error as error:
        raise SnapshotDatabaseError("member_db", error.sqlstate) from None
    del member_password

    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if output_root.is_symlink() or output_root.stat().st_mode & 0o077:
        raise ValueError(
            "출력 상위 폴더는 심볼릭 링크가 아니고 본인만 접근 가능해야 합니다."
        )
    directory = output_root / datetime.now(UTC).strftime("service-%Y%m%dT%H%M%SZ")
    directory.mkdir(mode=0o700, exist_ok=False)
    frames: dict[str, pd.DataFrame] = {
        "orders": orders.orders,
        "order_items": _serialize_nullable_pet_id(orders.order_items),
        "histories": orders.status_histories,
        "claims": orders.claims,
        "claim_items": orders.claim_items,
        "pets": pets.pets,
    }
    old_umask = os.umask(0o077)
    try:
        files: dict[str, dict[str, object]] = {}
        for name, frame in frames.items():
            path = directory / f"{name}.csv"
            frame.to_csv(path, index=False)
            files[name] = {
                "filename": path.name,
                "rows": len(frame),
                "sha256": _sha256(path),
            }
        manifest = {
            "order_extracted_at": orders.extracted_at.isoformat(),
            "member_extracted_at": pets.extracted_at.isoformat(),
            "cross_database_atomic_snapshot": False,
            "files": files,
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        os.umask(old_umask)
    verify_snapshot(directory)
    return directory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-dir", type=Path)
    parser.add_argument("--prepare-model-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15432)
    parser.add_argument("--user", default="ai_dev")
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/raw/local_service_snapshots")
    )
    args = parser.parse_args()
    try:
        if args.verify_dir is not None and args.prepare_model_dir is not None:
            raise ValueError("검증과 파생 파일 준비는 별도로 실행해야 합니다.")
        if args.prepare_model_dir is not None:
            path = prepare_model_order_items(args.prepare_model_dir)
            print(
                json.dumps(
                    {"event": "local_snapshot_model_input_prepared", "path": str(path)}
                )
            )
            return 0
        if args.verify_dir is not None:
            manifest = verify_snapshot(args.verify_dir)
            print(
                json.dumps(
                    {
                        "event": "local_snapshot_verified",
                        "directory": str(args.verify_dir),
                        "row_counts": {
                            name: info["rows"]
                            for name, info in manifest["files"].items()
                        },
                        "order_extracted_at": manifest["order_extracted_at"],
                        "member_extracted_at": manifest["member_extracted_at"],
                    }
                )
            )
            return 0
        directory = export_snapshot(
            host=args.host, port=args.port, user=args.user, output_root=args.output_root
        )
    except SnapshotDatabaseError as error:
        print(
            json.dumps(
                {
                    "event": "local_snapshot_failed",
                    "database": error.database,
                    "sqlstate": error.sqlstate,
                }
            )
        )
        return 1
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        psycopg.Error,
    ) as error:
        # 연결 예외에는 주소·사용자명이 섞일 수 있으므로 세부 내용을 출력하지 않습니다.
        print(
            json.dumps(
                {"event": "local_snapshot_failed", "error_type": type(error).__name__}
            )
        )
        return 1
    print(json.dumps({"event": "local_snapshot_exported", "directory": str(directory)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
