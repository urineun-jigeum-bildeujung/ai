"""재구매 예측 결과의 완결 발행·멱등 재실행·최신 선택을 검증합니다."""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from scripts.modeling.prediction_publications import (
    PredictionPublicationError,
    merge_idempotent_publication,
    select_latest_published_predictions,
    validate_prediction_publications,
)


def _as_rows(payload: dict[str, object], name: str) -> pd.DataFrame:
    """JSON의 객체 배열을 계약 검증용 DataFrame으로 변환합니다."""
    value = payload.get(name)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise AssertionError(f"고정 예제의 {name}은 객체 배열이어야 합니다.")
    return pd.DataFrame(cast(list[dict[str, object]], value))


def test_only_complete_published_batches_are_exposed_as_latest(
    prediction_publication_contract: dict[str, object],
) -> None:
    """실패·작성 중 결과와 늦게 저장된 오래된 데이터 컷을 노출하지 않습니다."""
    batches = _as_rows(prediction_publication_contract, "batches")
    results = _as_rows(prediction_publication_contract, "results")

    latest = select_latest_published_predictions(batches, results)

    assert len(latest) == 2
    assert set(latest["publication_id"]) == {"pub-new"}
    assert dict(
        zip(
            latest["user_id"],
            latest["conditional_repurchase_probability"],
            strict=True,
        )
    ) == {"user-1": 0.7, "user-2": 0.3}


def test_published_batch_must_contain_every_expected_result(
    prediction_publication_contract: dict[str, object],
) -> None:
    """일부 결과만 적재된 배치를 완료로 표시하면 실행 전에 거절합니다."""
    batches = _as_rows(prediction_publication_contract, "batches")
    results = _as_rows(prediction_publication_contract, "results")
    batches.loc[
        batches["publication_id"].eq("pub-newer-failed"), "publication_status"
    ] = "PUBLISHED"

    with pytest.raises(PredictionPublicationError, match="실제 결과 수"):
        validate_prediction_publications(batches, results)


def test_same_idempotency_key_and_payload_do_not_create_duplicates(
    prediction_publication_contract: dict[str, object],
) -> None:
    """동일한 성공 배치를 재시도해도 행 수와 결과가 변하지 않습니다."""
    batches = _as_rows(prediction_publication_contract, "batches")
    results = _as_rows(prediction_publication_contract, "results")
    candidate_batch = batches.loc[batches["publication_id"].eq("pub-new")].copy()
    candidate_results = results.loc[results["publication_id"].eq("pub-new")].copy()
    existing_batches = candidate_batch.iloc[0:0].copy()
    existing_results = candidate_results.iloc[0:0].copy()

    stored_batches, stored_results = merge_idempotent_publication(
        existing_batches,
        existing_results,
        candidate_batch,
        candidate_results,
    )
    reordered_batch = candidate_batch.astype(
        {"expected_result_count": "int32", "feature_generation_version": "int32"}
    )
    reordered_results = candidate_results.iloc[::-1].reset_index(drop=True)
    reordered_results["window_days"] = reordered_results["window_days"].astype("int32")
    rerun_batches, rerun_results = merge_idempotent_publication(
        stored_batches,
        stored_results,
        reordered_batch,
        reordered_results,
    )

    assert len(rerun_batches) == 1
    assert len(rerun_results) == 2
    pd.testing.assert_frame_equal(rerun_batches, stored_batches)
    pd.testing.assert_frame_equal(rerun_results, stored_results)


def test_same_idempotency_key_with_changed_payload_fails(
    prediction_publication_contract: dict[str, object],
) -> None:
    """같은 실행을 가장한 다른 확률이 기존 정상 결과를 덮지 못하게 합니다."""
    batches = _as_rows(prediction_publication_contract, "batches")
    results = _as_rows(prediction_publication_contract, "results")
    candidate_batch = batches.loc[batches["publication_id"].eq("pub-new")].copy()
    candidate_results = results.loc[results["publication_id"].eq("pub-new")].copy()
    changed = candidate_results.copy()
    changed.loc[
        changed["user_id"].eq("user-1"), "conditional_repurchase_probability"
    ] = 0.8

    with pytest.raises(PredictionPublicationError, match="서로 다른 발행 내용"):
        merge_idempotent_publication(
            candidate_batch,
            candidate_results,
            candidate_batch,
            changed,
        )


def test_unavailable_prediction_cannot_contain_probability(
    prediction_publication_contract: dict[str, object],
) -> None:
    """이력 부족 상태에 임의 확률을 채워 READY처럼 보이게 하지 않습니다."""
    batches = _as_rows(prediction_publication_contract, "batches")
    results = _as_rows(prediction_publication_contract, "results")
    results.loc[
        results["prediction_status"].eq("INSUFFICIENT_DATA"),
        "conditional_repurchase_probability",
    ] = 0.5

    with pytest.raises(PredictionPublicationError, match="null"):
        validate_prediction_publications(batches, results)


def test_unapproved_output_field_fails_explicitly(
    prediction_publication_contract: dict[str, object],
) -> None:
    """검증하지 않은 구매일·신뢰도 필드가 계약에 조용히 섞이지 않게 합니다."""
    batches = _as_rows(prediction_publication_contract, "batches")
    results = _as_rows(prediction_publication_contract, "results")
    results["confidence_score"] = 0.9

    with pytest.raises(PredictionPublicationError, match="지원하지 않는 열"):
        validate_prediction_publications(batches, results)
