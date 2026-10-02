"""검증된 재구매 예측 배치를 PostgreSQL에 원자적으로 발행합니다.

연결 생성과 Secret 관리는 호출자 책임입니다. 공용 DB에 자동으로 연결하거나
마이그레이션을 적용하지 않습니다. 연결은 autocommit 상태여야 이 함수의
transaction() 블록이 실제 COMMIT/ROLLBACK 경계가 됩니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import psycopg
from psycopg.pq import TransactionStatus

from scripts.modeling.prediction_publications import (
    BATCH_COLUMNS,
    RESULT_COLUMNS,
    PredictionPublicationError,
    merge_idempotent_publication,
    validate_prediction_publications,
)


@dataclass(frozen=True)
class PublicationWriteResult:
    publication_id: str
    inserted: bool


_SELECT_BATCH = """
SELECT publication_id, idempotency_key, as_of_timestamp, created_at,
       publication_status, expected_result_count, artifact_id,
       feature_generation_version
FROM repurchase.prediction_batches
WHERE idempotency_key = %s
FOR UPDATE
"""

_SELECT_RESULTS = """
SELECT publication_id, user_id, pet_id, target_scope, target_id, window_days,
       prediction_status, conditional_repurchase_probability
FROM repurchase.repurchase_predictions
WHERE publication_id = %s
"""

_INSERT_BATCH = """
INSERT INTO repurchase.prediction_batches
    (publication_id, idempotency_key, as_of_timestamp, created_at,
     publication_status, expected_result_count, artifact_id,
     feature_generation_version)
VALUES (%s, %s, %s, %s, 'STAGING', %s, %s, %s)
"""

_COPY_RESULTS = """
COPY repurchase.repurchase_predictions
    (publication_id, user_id, pet_id, target_scope, target_id, window_days,
     prediction_status, conditional_repurchase_probability)
FROM STDIN
"""


def _db_value(value: Any) -> Any:
    """pandas/NumPy 스칼라를 psycopg가 다룰 수 있는 Python 값으로 바꿉니다."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.generic):
        return value.item()
    return value


def publish_prediction_publication(
    connection: psycopg.Connection[Any],
    candidate_batch: pd.DataFrame,
    candidate_results: pd.DataFrame,
) -> PublicationWriteResult:
    """한 배치를 발행하거나 동일 내용 재실행을 중복 없이 인정합니다.

    동일 멱등키의 다른 내용, 미완성 배치, DB 오류는 예외로 전달됩니다.
    모든 새 행과 PUBLISHED 전환은 하나의 트랜잭션으로 롤백됩니다.
    """
    batches, results = validate_prediction_publications(
        candidate_batch, candidate_results
    )
    if len(batches) != 1 or batches.iloc[0]["publication_status"] != "PUBLISHED":
        raise PredictionPublicationError("완결된 PUBLISHED 배치 한 건이 필요합니다.")
    if (
        not connection.autocommit
        or connection.info.transaction_status != TransactionStatus.IDLE
    ):
        raise PredictionPublicationError(
            "독립된 autocommit 연결에서만 예측 결과를 발행할 수 있습니다."
        )

    batch = batches.iloc[0]
    publication_id = str(batch["publication_id"])
    idempotency_key = str(batch["idempotency_key"])

    with connection.transaction():
        # 같은 키의 병렬 실행은 INSERT 전에 직렬화합니다. 해시 충돌은 대기만 늘립니다.
        connection.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (idempotency_key,),
        )
        with connection.cursor() as cursor:
            cursor.execute(_SELECT_BATCH, (idempotency_key,))
            existing = cursor.fetchone()
            if existing is not None:
                existing_batch = pd.DataFrame([existing], columns=list(BATCH_COLUMNS))
                cursor.execute(_SELECT_RESULTS, (existing[0],))
                existing_results = pd.DataFrame(
                    cursor.fetchall(), columns=list(RESULT_COLUMNS)
                )
                merge_idempotent_publication(
                    existing_batch, existing_results, batches, results
                )
                return PublicationWriteResult(publication_id, inserted=False)

            cursor.execute(
                _INSERT_BATCH,
                tuple(
                    _db_value(batch[column])
                    for column in BATCH_COLUMNS
                    if column != "publication_status"
                ),
            )
            with cursor.copy(_COPY_RESULTS) as copy:
                for row in results.itertuples(index=False, name=None):
                    copy.write_row(tuple(_db_value(value) for value in row))
            cursor.execute(
                """UPDATE repurchase.prediction_batches
                   SET publication_status = 'PUBLISHED'
                   WHERE publication_id = %s""",
                (publication_id,),
            )
    return PublicationWriteResult(publication_id, inserted=True)
