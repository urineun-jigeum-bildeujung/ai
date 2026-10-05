"""AFT 내부 검증 배치를 로컬 스냅샷 또는 공용 DB 원천에서 실행합니다."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd
import psycopg
from psycopg.pq import TransactionStatus

from scripts.export_local_service_snapshot import verify_snapshot
from scripts.modeling.artifacts import LoadedModelArtifact, load_model_artifact
from scripts.modeling.cloud_source_reader import read_order_source, read_pet_source
from scripts.modeling.prediction_storage import publish_prediction_publication
from scripts.modeling.shadow_batch import ShadowPreparation, prepare_shadow_publication


class ShadowBatchContractError(ValueError):
    """모델·스냅샷·실행 인자가 내부 검증 계약과 다를 때 발생합니다."""


class ShadowDatabaseError(RuntimeError):
    """자격 증명을 출력하지 않는 DB 단계 실패입니다."""

    def __init__(self, database: str, sqlstate: str | None) -> None:
        self.database = database
        self.sqlstate = sqlstate
        super().__init__(f"{database} 단계 실패 (SQLSTATE={sqlstate or 'unknown'})")


# 최종 Test 보고서에 사전 고정한 AFT 후보. 내부 검증 교체는 코드 리뷰를 거칩니다.
FROZEN_SHADOW_AFT_ARTIFACT_ID = (
    "ec25eb1bcd8f24ce36a2397c1544cb087cb970af532365de7f8ec900aeb4b7fb"
)
KST = timezone(timedelta(hours=9))


@dataclass(frozen=True)
class ShadowBatchSummary:
    mode: str
    publication_id: str
    artifact_id: str
    as_of_timestamp: str
    source_order_count: int
    source_order_item_count: int
    quarantined_order_count: int
    quarantined_missing_history_order_count: int
    quarantined_missing_history_paid_order_count: int
    quarantined_missing_history_order_item_count: int
    quarantined_status_mismatch_order_count: int
    quarantined_status_mismatch_order_item_count: int
    prediction_count: int
    inserted: bool | None


@dataclass(frozen=True)
class DailyShadowSlot:
    """시연용 하루 한 번 배치의 논리적 실행 슬롯입니다."""

    as_of_timestamp: str
    created_at: str
    publication_id: str


def daily_shadow_slot(
    *,
    now: pd.Timestamp,
    scheduled_time_kst: str,
    run_date_kst: str | None = None,
) -> DailyShadowSlot:
    """같은 KST 날짜·예정 시각의 재시도가 같은 컷과 ID를 사용하게 합니다.

    created_at은 실제 컨테이너 시작 시각이 아닌 논리적 예정 실행 시각입니다.
    지연된 날짜의 재실행에는 run_date_kst를 명시합니다.
    """
    current = _timestamp(str(now), name="now")
    if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", scheduled_time_kst):
        raise ShadowBatchContractError("scheduled_time_kst는 HH:MM 형식이어야 합니다.")
    if run_date_kst is None:
        run_day = current.tz_convert(KST).date()
    else:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", run_date_kst):
            raise ShadowBatchContractError(
                "run_date_kst는 YYYY-MM-DD 형식이어야 합니다."
            )
        try:
            run_day = date.fromisoformat(run_date_kst)
        except ValueError as error:
            raise ShadowBatchContractError(
                "run_date_kst는 유효한 날짜여야 합니다."
            ) from error
    hour, minute = map(int, scheduled_time_kst.split(":"))
    slot = pd.Timestamp(datetime.combine(run_day, time(hour, minute, tzinfo=KST)))
    slot_utc = slot.tz_convert("UTC")
    if slot_utc > current:
        raise ShadowBatchContractError("예정 실행 시각이 현재보다 늦습니다.")
    timestamp = slot_utc.isoformat()
    return DailyShadowSlot(
        as_of_timestamp=timestamp,
        created_at=timestamp,
        publication_id=(
            f"demo-shadow-{run_day:%Y%m%d}T{hour:02d}{minute:02d}KST-"
            f"{FROZEN_SHADOW_AFT_ARTIFACT_ID[:12]}-30d-v1"
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value: str, *, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ShadowBatchContractError(f"{name}은 유효한 시각이어야 합니다.") from error
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ShadowBatchContractError(f"{name}에는 시간대가 필요합니다.")
    return timestamp.tz_convert("UTC")


def _check_cut(as_of: pd.Timestamp, order_at: object, member_at: object) -> None:
    if as_of > min(
        _timestamp(str(order_at), name="order_extracted_at"),
        _timestamp(str(member_at), name="member_extracted_at"),
    ):
        raise ShadowBatchContractError("예측 기준 시각이 원천 추출 시각보다 늦습니다.")


def _artifact(directory: Path, expected_artifact_id: str) -> LoadedModelArtifact:
    if expected_artifact_id != FROZEN_SHADOW_AFT_ARTIFACT_ID:
        raise ShadowBatchContractError("사전 고정한 AFT artifact_id가 필요합니다.")
    artifact = load_model_artifact(directory)
    if artifact.artifact_id != expected_artifact_id:
        raise ShadowBatchContractError("고정 모델 artifact_id가 일치하지 않습니다.")
    return artifact


def _read_local_snapshot(
    directory: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    manifest = verify_snapshot(directory)
    derived_metadata_path = directory / "order_items_model.json"
    derived_path = directory / "order_items_model.csv"
    metadata = json.loads(derived_metadata_path.read_text(encoding="utf-8"))
    if (
        metadata["source_sha256"] != manifest["files"]["order_items"]["sha256"]
        or metadata["derived_sha256"] != _sha256(derived_path)
        or metadata["rows"] != manifest["files"]["order_items"]["rows"]
    ):
        raise ShadowBatchContractError("주문상품 파생 CSV가 원천 스냅샷과 다릅니다.")
    id_columns = {
        "orders": ("order_id", "user_id"),
        "order_items": (
            "order_item_id",
            "order_id",
            "product_id",
            "product_group_id_snapshot",
            "pet_id",
        ),
        "pets": ("pet_id",),
        "histories": ("history_id", "order_id"),
        "claims": ("claim_id", "order_id"),
        "claim_items": ("claim_item_id", "claim_id", "order_item_id"),
    }
    sources = {
        name: pd.read_csv(
            derived_path if name == "order_items" else directory / f"{name}.csv",
            dtype={column: "string" for column in columns},
            true_values=["t", "true", "True"],
            false_values=["f", "false", "False"],
        )
        for name, columns in id_columns.items()
    }
    return sources, manifest


def _prepare(
    sources: dict[str, pd.DataFrame],
    *,
    artifact_directory: Path,
    expected_artifact_id: str,
    as_of: pd.Timestamp,
    created_at: pd.Timestamp,
    window_days: int,
    publication_id: str,
) -> ShadowPreparation:
    if window_days <= 0:
        raise ShadowBatchContractError("window_days는 양의 정수여야 합니다.")
    artifact = _artifact(artifact_directory, expected_artifact_id)
    return prepare_shadow_publication(
        sources,
        artifact,
        as_of_timestamp=as_of,
        created_at=created_at,
        window_days=window_days,
        publication_id=publication_id,
    )


def _summary(
    prepared: ShadowPreparation, *, mode: str, inserted: bool | None
) -> ShadowBatchSummary:
    batch = prepared.batch.iloc[0]
    return ShadowBatchSummary(
        mode=mode,
        publication_id=str(batch["publication_id"]),
        artifact_id=str(batch["artifact_id"]),
        as_of_timestamp=batch["as_of_timestamp"].isoformat(),
        source_order_count=prepared.source_order_count,
        source_order_item_count=prepared.source_order_item_count,
        quarantined_order_count=prepared.quarantined_order_count,
        quarantined_missing_history_order_count=prepared.quarantined_missing_history_order_count,
        quarantined_missing_history_paid_order_count=prepared.quarantined_missing_history_paid_order_count,
        quarantined_missing_history_order_item_count=prepared.quarantined_missing_history_order_item_count,
        quarantined_status_mismatch_order_count=prepared.quarantined_status_mismatch_order_count,
        quarantined_status_mismatch_order_item_count=prepared.quarantined_status_mismatch_order_item_count,
        prediction_count=prepared.target_count,
        inserted=inserted,
    )


def check_local_shadow_batch(
    *,
    snapshot_directory: Path,
    artifact_directory: Path,
    expected_artifact_id: str,
    as_of_timestamp: str,
    window_days: int,
) -> ShadowBatchSummary:
    """완료 스냅샷으로 실제 AFT 추론을 검증하되 DB에는 연결하지 않습니다."""
    as_of = _timestamp(as_of_timestamp, name="as_of_timestamp")
    try:
        sources, manifest = _read_local_snapshot(snapshot_directory)
    except (OSError, UnicodeError, KeyError, ValueError, json.JSONDecodeError) as error:
        raise ShadowBatchContractError(
            "완료된 로컬 스냅샷을 읽지 못했습니다."
        ) from error
    _check_cut(as_of, manifest["order_extracted_at"], manifest["member_extracted_at"])
    prepared = _prepare(
        sources,
        artifact_directory=artifact_directory,
        expected_artifact_id=expected_artifact_id,
        as_of=as_of,
        created_at=as_of,
        window_days=window_days,
        publication_id="local-shadow-check",
    )
    return _summary(prepared, mode="shadow-check", inserted=None)


def _dsn(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ShadowBatchContractError(f"{name} 환경변수가 필요합니다.")
    return value


def _read_live_sources() -> tuple[dict[str, pd.DataFrame], pd.Timestamp, pd.Timestamp]:
    try:
        with psycopg.connect(
            _dsn("REPURCHASE_ORDER_DATABASE_DSN"), autocommit=True
        ) as connection:
            order = read_order_source(connection)
    except psycopg.Error as error:
        raise ShadowDatabaseError("order_db", error.sqlstate) from error
    try:
        with psycopg.connect(
            _dsn("REPURCHASE_MEMBER_DATABASE_DSN"), autocommit=True
        ) as connection:
            member = read_pet_source(connection)
    except psycopg.Error as error:
        raise ShadowDatabaseError("member_db", error.sqlstate) from error
    return (
        {
            "orders": order.orders,
            "order_items": order.order_items,
            "histories": order.status_histories,
            "claims": order.claims,
            "claim_items": order.claim_items,
            "pets": member.pets,
        },
        order.extracted_at,
        member.extracted_at,
    )


def run_live_shadow_batch(
    *,
    artifact_directory: Path,
    expected_artifact_id: str,
    as_of_timestamp: str,
    created_at: str,
    window_days: int,
    publication_id: str,
    allow_shadow_write: bool,
) -> ShadowBatchSummary:
    """명시적 쓰기 허용·고정 컷·고정 실행 ID가 있을 때만 SHADOW로 적재합니다."""
    if not allow_shadow_write:
        raise ShadowBatchContractError("--allow-shadow-write가 필요합니다.")
    as_of = _timestamp(as_of_timestamp, name="as_of_timestamp")
    created = _timestamp(created_at, name="created_at")
    # 원천 조회를 시작하기 전에 누락된 Secret과 모델 지문을 먼저 확인합니다.
    _dsn("REPURCHASE_ORDER_DATABASE_DSN")
    _dsn("REPURCHASE_MEMBER_DATABASE_DSN")
    _dsn("REPURCHASE_RESULT_DATABASE_DSN")
    _artifact(artifact_directory, expected_artifact_id)
    sources, order_at, member_at = _read_live_sources()
    _check_cut(as_of, order_at, member_at)
    prepared = _prepare(
        sources,
        artifact_directory=artifact_directory,
        expected_artifact_id=expected_artifact_id,
        as_of=as_of,
        created_at=created,
        window_days=window_days,
        publication_id=publication_id,
    )
    try:
        with psycopg.connect(
            _dsn("REPURCHASE_RESULT_DATABASE_DSN"), autocommit=True
        ) as connection:
            if (
                connection.info.dbname != "repurchase_db"
                or not connection.autocommit
                or connection.info.transaction_status != TransactionStatus.IDLE
            ):
                raise ShadowBatchContractError(
                    "독립된 repurchase_db 쓰기 연결이 필요합니다."
                )
            written = publish_prediction_publication(
                connection, prepared.batch, prepared.results
            )
    except psycopg.Error as error:
        raise ShadowDatabaseError("repurchase_db", error.sqlstate) from error
    return _summary(prepared, mode="shadow-run", inserted=written.inserted)
