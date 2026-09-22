"""재구매 예측 결과의 원자적 발행과 최신 결과 선택 계약을 검증합니다.

특정 DB나 클라우드 SDK를 호출하지 않습니다. 저장소가 정해진 뒤에도 유지해야 할
배치 완결성·멱등성·최신 데이터 컷 우선 규칙을 순수 DataFrame 연산으로 정의합니다.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import pandas as pd


class PredictionPublicationError(ValueError):
    """예측 결과 발행 데이터가 공개 가능한 계약을 위반할 때 발생합니다."""


BATCH_COLUMNS = (
    "publication_id",
    "idempotency_key",
    "as_of_timestamp",
    "created_at",
    "publication_status",
    "expected_result_count",
    "artifact_id",
    "feature_generation_version",
)
RESULT_COLUMNS = (
    "publication_id",
    "user_id",
    "pet_id",
    "target_scope",
    "target_id",
    "window_days",
    "prediction_status",
    "conditional_repurchase_probability",
)
PUBLICATION_STATUSES = frozenset({"STAGING", "PUBLISHED", "FAILED"})
PREDICTION_STATUSES = frozenset({"READY", "INSUFFICIENT_DATA", "SUPPRESSED"})
TARGET_SCOPES = frozenset({"PRODUCT_GROUP", "CATEGORY"})
RESULT_KEY_COLUMNS = (
    "user_id",
    "pet_id",
    "target_scope",
    "target_id",
    "window_days",
)


def _require_exact_columns(
    rows: pd.DataFrame, expected: tuple[str, ...], name: str
) -> None:
    """필수 열 누락과 아직 합의되지 않은 출력 열의 조용한 유입을 거절합니다."""
    if not rows.columns.is_unique:
        raise PredictionPublicationError(f"{name}에 중복된 열 이름이 있습니다.")
    missing = set(expected) - set(rows.columns)
    unexpected = set(rows.columns) - set(expected)
    if missing:
        raise PredictionPublicationError(
            f"{name} 필수 열이 누락됐습니다: {sorted(missing)}"
        )
    if unexpected:
        raise PredictionPublicationError(
            f"{name}에 지원하지 않는 열이 있습니다: {sorted(unexpected)}"
        )


def _require_nonempty_string(rows: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """발행·사용자·모델 식별키가 결측 또는 공백 문자열인지 확인합니다."""
    for column in columns:
        values = rows[column]
        if (
            values.isna().any()
            or values.map(
                lambda value: not isinstance(value, str) or not value.strip()
            ).any()
        ):
            raise PredictionPublicationError(
                f"{column}은 비어 있지 않은 문자열이어야 합니다."
            )


def _as_utc_timestamp(value: object, *, name: str) -> pd.Timestamp:
    """발행 기준 시각의 시간대를 보존해 UTC로 비교할 수 있게 합니다."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise PredictionPublicationError(
            f"{name}을 시각으로 읽을 수 없습니다."
        ) from error
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise PredictionPublicationError(f"{name}에는 시간대가 포함돼야 합니다.")
    return timestamp.tz_convert("UTC")


def _normalize_batches(batches: pd.DataFrame) -> pd.DataFrame:
    """배치 메타데이터의 키·상태·시간·예상 행 수를 검사하고 복사합니다."""
    _require_exact_columns(batches, BATCH_COLUMNS, "prediction_batches")
    rows = batches.loc[:, list(BATCH_COLUMNS)].copy().reset_index(drop=True)
    _require_nonempty_string(
        rows,
        ("publication_id", "idempotency_key", "artifact_id"),
    )
    if rows["publication_id"].duplicated().any():
        raise PredictionPublicationError("publication_id가 중복됐습니다.")
    if rows["idempotency_key"].duplicated().any():
        raise PredictionPublicationError("idempotency_key가 중복됐습니다.")
    if not rows["publication_status"].isin(PUBLICATION_STATUSES).all():
        raise PredictionPublicationError("지원하지 않는 publication_status가 있습니다.")
    for column in ("expected_result_count", "feature_generation_version"):
        if (
            rows[column]
            .map(
                lambda value: isinstance(value, bool) or not isinstance(value, Integral)
            )
            .any()
        ):
            raise PredictionPublicationError(f"{column}은 정수여야 합니다.")
    if rows["expected_result_count"].lt(0).any():
        raise PredictionPublicationError("expected_result_count는 0 이상이어야 합니다.")
    if rows["feature_generation_version"].le(0).any():
        raise PredictionPublicationError(
            "feature_generation_version은 1 이상이어야 합니다."
        )
    for column in ("as_of_timestamp", "created_at"):
        rows[column] = [_as_utc_timestamp(value, name=column) for value in rows[column]]
    return rows


def _normalize_results(results: pd.DataFrame, batches: pd.DataFrame) -> pd.DataFrame:
    """개별 결과의 조회 키·상태·확률 범위와 배치 참조를 검사합니다."""
    _require_exact_columns(results, RESULT_COLUMNS, "repurchase_predictions")
    rows = results.loc[:, list(RESULT_COLUMNS)].copy().reset_index(drop=True)
    _require_nonempty_string(
        rows,
        ("publication_id", "user_id", "target_scope", "target_id"),
    )
    if not rows["publication_id"].isin(batches["publication_id"]).all():
        raise PredictionPublicationError(
            "존재하지 않는 publication_id의 결과가 있습니다."
        )
    if not rows["target_scope"].isin(TARGET_SCOPES).all():
        raise PredictionPublicationError("지원하지 않는 target_scope가 있습니다.")
    if not rows["prediction_status"].isin(PREDICTION_STATUSES).all():
        raise PredictionPublicationError("지원하지 않는 prediction_status가 있습니다.")
    if (
        rows["window_days"]
        .map(lambda value: isinstance(value, bool) or not isinstance(value, Integral))
        .any()
        or rows["window_days"].le(0).any()
    ):
        raise PredictionPublicationError("window_days는 0보다 큰 정수여야 합니다.")

    duplicate_columns = ["publication_id", *RESULT_KEY_COLUMNS]
    if rows.duplicated(subset=duplicate_columns).any():
        raise PredictionPublicationError(
            "한 발행 배치 안에 중복된 예측 조회 키가 있습니다."
        )

    ready = rows["prediction_status"].eq("READY")
    probability = rows["conditional_repurchase_probability"]
    invalid_ready = ready & probability.map(
        lambda value: (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(float(value))
            or not 0 <= float(value) <= 1
        )
    )
    if invalid_ready.any():
        raise PredictionPublicationError(
            "READY 결과에는 0~1의 유한한 확률이 필요합니다."
        )
    if probability.loc[~ready].notna().any():
        raise PredictionPublicationError(
            "사용 불가 상태의 예측 확률은 null이어야 합니다."
        )
    return rows


def validate_prediction_publications(
    batches: pd.DataFrame, results: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """결과 발행 계약을 검사하고 정규화한 안전한 복사본을 반환합니다.

    작성 중이거나 실패한 배치는 일부 결과를 가질 수 있습니다. 반면 PUBLISHED
    배치는 예상한 결과 수가 모두 존재해야 조회 가능한 완결 배치로 인정합니다.
    """
    normalized_batches = _normalize_batches(batches)
    normalized_results = _normalize_results(results, normalized_batches)
    actual_counts = normalized_results.groupby("publication_id", observed=True).size()
    for batch in normalized_batches.itertuples(index=False):
        actual = int(actual_counts.get(batch.publication_id, 0))
        if batch.publication_status == "PUBLISHED" and actual != int(
            batch.expected_result_count
        ):
            raise PredictionPublicationError(
                "PUBLISHED 배치의 실제 결과 수가 expected_result_count와 다릅니다."
            )
    return normalized_batches, normalized_results


def select_latest_published_predictions(
    batches: pd.DataFrame, results: pd.DataFrame
) -> pd.DataFrame:
    """완결 발행된 결과 중 조회 키별 최신 데이터 컷 한 행을 선택합니다."""
    normalized_batches, normalized_results = validate_prediction_publications(
        batches, results
    )
    published = normalized_batches.loc[
        normalized_batches["publication_status"].eq("PUBLISHED"),
        [
            "publication_id",
            "as_of_timestamp",
            "created_at",
            "artifact_id",
            "feature_generation_version",
        ],
    ]
    visible = normalized_results.merge(
        published,
        on="publication_id",
        how="inner",
        validate="many_to_one",
        sort=False,
    )
    if visible.empty:
        return visible

    # created_at이 늦더라도 오래된 데이터 컷이 최신 결과를 덮지 못하게
    # as_of_timestamp를 첫 번째 기준으로 두고 같은 컷의 재발행만 created_at으로 정렬합니다.
    ordered = visible.sort_values(
        [*RESULT_KEY_COLUMNS, "as_of_timestamp", "created_at", "publication_id"],
        kind="stable",
        na_position="first",
    )
    return ordered.drop_duplicates(
        subset=list(RESULT_KEY_COLUMNS), keep="last"
    ).reset_index(drop=True)


def merge_idempotent_publication(
    batches: pd.DataFrame,
    results: pd.DataFrame,
    candidate_batch: pd.DataFrame,
    candidate_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """같은 멱등키의 동일 재실행은 무시하고 상충하는 재실행은 거절합니다.

    실제 저장소에서는 이 규칙과 함께 트랜잭션 또는 원자적 발행 기능이 필요합니다.
    이 함수는 저장 방식과 무관한 계약 동작만 검증합니다.
    """
    existing_batches, existing_results = validate_prediction_publications(
        batches, results
    )
    new_batches, new_results = validate_prediction_publications(
        candidate_batch, candidate_results
    )
    if len(new_batches) != 1:
        raise PredictionPublicationError(
            "한 번에 하나의 발행 배치만 병합할 수 있습니다."
        )
    if existing_batches.empty:
        return new_batches.copy(), new_results.copy()

    candidate = new_batches.iloc[0]
    same_key = existing_batches["idempotency_key"].eq(candidate["idempotency_key"])
    if same_key.any():
        current_batch = existing_batches.loc[same_key].reset_index(drop=True)
        current_publication_id = current_batch.loc[0, "publication_id"]
        current_results = (
            existing_results.loc[
                existing_results["publication_id"].eq(current_publication_id)
            ]
            .sort_values(list(RESULT_KEY_COLUMNS), kind="stable", na_position="first")
            .reset_index(drop=True)
        )
        comparable_new_results = new_results.sort_values(
            list(RESULT_KEY_COLUMNS), kind="stable", na_position="first"
        ).reset_index(drop=True)
        try:
            # 저장소 조회 순서와 정수 폭은 달라도 같은 키·값이면 동일 payload입니다.
            pd.testing.assert_frame_equal(
                current_batch,
                new_batches,
                check_dtype=False,
            )
            pd.testing.assert_frame_equal(
                current_results,
                comparable_new_results,
                check_dtype=False,
            )
        except AssertionError as error:
            raise PredictionPublicationError(
                "같은 idempotency_key에 서로 다른 발행 내용이 전달됐습니다."
            ) from error
        return existing_batches.copy(), existing_results.copy()

    merged_batches = pd.concat(
        [existing_batches, new_batches], ignore_index=True, sort=False
    )
    merged_results = pd.concat(
        [existing_results, new_results], ignore_index=True, sort=False
    )
    return validate_prediction_publications(merged_batches, merged_results)
