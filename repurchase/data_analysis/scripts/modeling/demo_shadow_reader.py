"""인증된 목계정만 최신 시연용 SHADOW 스냅샷을 읽는 DB 어댑터입니다.

호출자는 인증을 완료한 viewer_user_id와 서버가 관리하는 목계정 허용 목록을
전달해야 합니다. 이 모듈은 HTTP 인증을 구현하거나 일반 서비스 조회 뷰를 바꾸지
않습니다. DB에는 읽기 전용 계정을 사용하는 것이 별도 요구 사항입니다.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import pandas as pd
import psycopg


class DemoShadowAccessError(ValueError):
    """목계정 전용 조회의 접근·응답 계약이 충족되지 않았습니다."""


@dataclass(frozen=True)
class DemoShadowPrediction:
    """화면에 넘기기 전까지 시연 출처를 잃지 않는 결과 한 건입니다."""

    user_id: str
    pet_id: int | None
    target_scope: str
    target_id: str
    window_days: int
    prediction_status: str
    conditional_repurchase_probability: float | None
    publication_id: str
    artifact_id: str
    as_of_timestamp: str
    source_kind: str = "MOCK_DATA_DEMO"


_LATEST_DEMO_SQL = """
WITH latest_demo AS (
    SELECT publication_id, artifact_id, as_of_timestamp
    FROM repurchase.prediction_batches
    WHERE publication_status = 'SHADOW'
      AND publication_id ~ '^demo-shadow-[0-9]{8}T[0-9]{4}KST-[0-9a-f]{12}-30d-v1$'
      AND artifact_id = %s
      AND feature_generation_version = 2
    ORDER BY as_of_timestamp DESC, created_at DESC,
             publication_id COLLATE "C" DESC
    LIMIT 1
)
SELECT r.user_id, r.pet_id, r.target_scope, r.target_id, r.window_days,
       r.prediction_status, r.conditional_repurchase_probability,
       b.publication_id, b.artifact_id, b.as_of_timestamp
FROM latest_demo AS b
JOIN repurchase.repurchase_predictions AS r USING (publication_id)
WHERE r.user_id = %s AND r.window_days = 30
ORDER BY r.target_scope, r.target_id, r.pet_id NULLS FIRST
LIMIT %s
"""


def read_demo_shadow_predictions(
    connection: psycopg.Connection[object],
    *,
    viewer_user_id: str,
    requested_user_id: str,
    allowed_mock_user_ids: Collection[str],
    expected_artifact_id: str,
    max_results: int = 500,
) -> list[DemoShadowPrediction]:
    """최신 일별 SHADOW 배치에서 승인된 목계정의 30일 결과만 조회합니다.

    viewer_user_id는 서버 인증 계층에서 얻은 값이어야 하며, 요청 본문 값을 그대로
    넘겨서는 안 됩니다. 최신 배치에 대상이 없으면 과거 배치에서 보충하지 않습니다.
    """
    if (
        not isinstance(viewer_user_id, str)
        or not viewer_user_id.strip()
        or not isinstance(requested_user_id, str)
        or viewer_user_id != requested_user_id
        or isinstance(allowed_mock_user_ids, (str, bytes))
        or requested_user_id not in allowed_mock_user_ids
    ):
        raise DemoShadowAccessError(
            "승인된 목계정 본인만 시연 결과를 조회할 수 있습니다."
        )
    if (
        isinstance(max_results, bool)
        or not isinstance(max_results, int)
        or not 1 <= max_results <= 500
    ):
        raise DemoShadowAccessError("max_results는 1~500의 정수여야 합니다.")
    if not isinstance(expected_artifact_id, str) or not expected_artifact_id.strip():
        raise DemoShadowAccessError("고정 모델 ID가 필요합니다.")
    if connection.info.dbname != "repurchase_db":
        raise DemoShadowAccessError("repurchase_db 읽기 연결이 필요합니다.")

    rows = connection.execute(
        _LATEST_DEMO_SQL, (expected_artifact_id, requested_user_id, max_results + 1)
    ).fetchall()
    if len(rows) > max_results:
        raise DemoShadowAccessError("시연 결과가 조회 상한을 넘었습니다.")
    return [
        DemoShadowPrediction(
            user_id=str(row[0]),
            pet_id=row[1],
            target_scope=str(row[2]),
            target_id=str(row[3]),
            window_days=row[4],
            prediction_status=str(row[5]),
            conditional_repurchase_probability=row[6],
            publication_id=str(row[7]),
            artifact_id=str(row[8]),
            as_of_timestamp=pd.Timestamp(row[9]).tz_convert("UTC").isoformat(),
        )
        for row in rows
    ]
