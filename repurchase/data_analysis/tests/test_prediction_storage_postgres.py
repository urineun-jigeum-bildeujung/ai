"""일회용 로컬 PostgreSQL에서 적재의 원자성과 멱등성을 확인합니다."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

from scripts.modeling.prediction_publications import PredictionPublicationError
from scripts.modeling.prediction_storage import publish_prediction_publication


@pytest.fixture
def local_connection() -> psycopg.Connection[object]:
    """명시적으로 지정된 로컬 테스트 DB만 사용합니다. 운영 DB는 거절합니다."""
    dsn = os.environ.get("REPURCHASE_TEST_DATABASE_DSN")
    if not dsn:
        pytest.skip("REPURCHASE_TEST_DATABASE_DSN 미설정")
    params = conninfo_to_dict(dsn)
    if (
        params.get("host") not in {"127.0.0.1", "localhost", "::1"}
        or params.get("dbname") != "repurchase_writer_test"
    ):
        pytest.fail("로컬 repurchase_writer_test DB 주소만 허용합니다.")
    connection = psycopg.connect(dsn, autocommit=True)
    info = connection.info
    if (
        info.host not in {"127.0.0.1", "localhost", "::1"}
        or info.dbname != "repurchase_writer_test"
    ):
        connection.close()
        pytest.fail("로컬 repurchase_writer_test DB에서만 실행할 수 있습니다.")
    if (
        connection.execute(
            "SELECT 1 FROM pg_namespace WHERE nspname = 'repurchase'"
        ).fetchone()
        is not None
    ):
        connection.close()
        pytest.fail("기존 repurchase 스키마가 있는 테스트 DB는 사용하지 않습니다.")
    migration_dir = Path(__file__).resolve().parents[2] / "database"
    try:
        for migration in ("001_prediction_storage.sql", "002_shadow_publication.sql"):
            connection.execute((migration_dir / migration).read_text(encoding="utf-8"))
        yield connection
    finally:
        connection.execute("DROP SCHEMA IF EXISTS repurchase CASCADE")
        connection.close()


def _candidate(
    *, publication_id: str = "pub-1", probability: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    batch = pd.DataFrame(
        [
            {
                "publication_id": publication_id,
                "idempotency_key": "cut-2026-01-01-model-v1",
                "as_of_timestamp": "2026-01-01T00:00:00+00:00",
                "created_at": "2026-01-02T00:00:00+00:00",
                "publication_status": "PUBLISHED",
                "expected_result_count": 1,
                "artifact_id": "model-v1",
                "feature_generation_version": 1,
            }
        ]
    )
    result = pd.DataFrame(
        [
            {
                "publication_id": publication_id,
                "user_id": "42",
                "pet_id": None,
                "target_scope": "PRODUCT_GROUP",
                "target_id": "11",
                "window_days": 30,
                "prediction_status": "READY",
                "conditional_repurchase_probability": probability,
            }
        ]
    )
    return batch, result


def test_publish_retry_conflict_and_rollback(
    local_connection: psycopg.Connection[object],
) -> None:
    """성공 1건, 동일 재실행 0건, 충돌 오류, DB 오류 롤백을 확인합니다."""
    batch, result = _candidate()
    first = publish_prediction_publication(local_connection, batch, result)
    assert first.inserted is True
    assert first.publication_id == "pub-1"
    assert local_connection.execute(
        "SELECT count(*) FROM repurchase.latest_predictions"
    ).fetchone() == (1,)

    retry = publish_prediction_publication(local_connection, batch, result)
    assert retry.inserted is False
    assert local_connection.execute(
        "SELECT count(*) FROM repurchase.repurchase_predictions"
    ).fetchone() == (1,)

    changed_batch, changed_result = _candidate(probability=0.8)
    with pytest.raises(PredictionPublicationError, match="서로 다른 발행 내용"):
        publish_prediction_publication(local_connection, changed_batch, changed_result)

    # Python 계약에는 아직 pet_id 타입 제약이 없어 DB가 COPY 단계에서 거절합니다.
    # 같은 멱등키가 이미 있으면 비교 단계에서 멈추므로 새 키를 사용합니다.
    invalid_batch, invalid_result = _candidate(publication_id="pub-invalid")
    invalid_batch.loc[0, "idempotency_key"] = "invalid-pet-id"
    invalid_result.loc[0, "pet_id"] = "not-a-bigint"
    with pytest.raises(psycopg.Error):
        publish_prediction_publication(local_connection, invalid_batch, invalid_result)
    assert local_connection.execute(
        "SELECT count(*) FROM repurchase.prediction_batches"
    ).fetchone() == (1,)
    assert local_connection.execute(
        "SELECT count(*) FROM repurchase.latest_predictions"
    ).fetchone() == (1,)


def test_incomplete_batch_is_rejected_before_writing(
    local_connection: psycopg.Connection[object],
) -> None:
    batch, result = _candidate(publication_id="pub-incomplete")
    batch.loc[0, "idempotency_key"] = "incomplete"
    batch.loc[0, "expected_result_count"] = 2
    with pytest.raises(PredictionPublicationError, match="실제 결과 수"):
        publish_prediction_publication(local_connection, batch, result)
    assert local_connection.execute(
        "SELECT count(*) FROM repurchase.prediction_batches"
    ).fetchone() == (0,)


def test_shadow_batch_is_complete_immutable_and_invisible(
    local_connection: psycopg.Connection[object],
) -> None:
    batch, result = _candidate()
    batch.loc[0, "publication_status"] = "SHADOW"

    assert publish_prediction_publication(local_connection, batch, result).inserted
    assert local_connection.execute(
        "SELECT publication_status FROM repurchase.prediction_batches"
    ).fetchone() == ("SHADOW",)
    assert local_connection.execute(
        "SELECT count(*) FROM repurchase.repurchase_predictions"
    ).fetchone() == (1,)
    assert local_connection.execute(
        "SELECT count(*) FROM repurchase.latest_predictions"
    ).fetchone() == (0,)
    assert not publish_prediction_publication(local_connection, batch, result).inserted
    with pytest.raises(psycopg.Error, match="STAGING이 아닌 배치"):
        local_connection.execute(
            "UPDATE repurchase.repurchase_predictions "
            "SET conditional_repurchase_probability = 0.8"
        )


def test_incomplete_shadow_batch_is_rejected_before_writing(
    local_connection: psycopg.Connection[object],
) -> None:
    batch, result = _candidate()
    batch.loc[0, "publication_status"] = "SHADOW"
    batch.loc[0, "expected_result_count"] = 2
    with pytest.raises(PredictionPublicationError, match="실제 결과 수"):
        publish_prediction_publication(local_connection, batch, result)
    assert local_connection.execute(
        "SELECT count(*) FROM repurchase.prediction_batches"
    ).fetchone() == (0,)
