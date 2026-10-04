"""서비스 시점 복원 원천으로 AFT 내부 검증용 비노출 배치를 구성합니다."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from .artifacts import LoadedModelArtifact, ModelArtifactError
from .current_prediction import predict_temporal_service_current_probability
from .operational_event_intervals import build_operational_event_intervals
from .operational_quantity_intervals import build_order_item_quantity_intervals
from .operational_source_quarantine import quarantine_unrestorable_orders
from .operational_status_intervals import build_order_status_intervals
from .operational_validity_intervals import build_valid_purchase_item_intervals
from .prediction_publications import validate_prediction_publications


@dataclass(frozen=True)
class ShadowPreparation:
    batch: pd.DataFrame
    results: pd.DataFrame
    source_order_count: int
    quarantined_order_count: int
    target_count: int


def _utc(value: object, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}은 유효한 시각이어야 합니다.") from error
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{name}에는 시간대가 필요합니다.")
    return timestamp.tz_convert("UTC")


def prepare_shadow_publication(
    sources: Mapping[str, pd.DataFrame],
    artifact: LoadedModelArtifact,
    *,
    as_of_timestamp: pd.Timestamp,
    created_at: pd.Timestamp,
    window_days: int,
    publication_id: str,
) -> ShadowPreparation:
    """원천을 수정하지 않고 시점 피처·AFT 확률·SHADOW 저장 행을 생성합니다."""
    as_of = _utc(as_of_timestamp, "as_of_timestamp")
    created = _utc(created_at, "created_at")
    if created < as_of:
        raise ValueError("created_at은 as_of_timestamp보다 이르면 안 됩니다.")
    if not publication_id or not publication_id.strip():
        raise ValueError("publication_id가 비어 있습니다.")
    if artifact.family != "xgboost_aft" or artifact.feature_generation_version != 2:
        raise ModelArtifactError("내부 검증에는 시점 피처 버전 2 AFT가 필요합니다.")

    orders = sources["orders"]
    quarantine = quarantine_unrestorable_orders(
        orders,
        sources["order_items"],
        sources["histories"],
        sources["claims"],
        sources["claim_items"],
        as_of_at=as_of,
    )
    status = build_order_status_intervals(quarantine.status_histories)
    quantity = build_order_item_quantity_intervals(
        quarantine.orders,
        quarantine.order_items,
        quarantine.claims,
        quarantine.claim_items,
    )
    valid = build_valid_purchase_item_intervals(status, quantity)
    events = build_operational_event_intervals(
        valid, quarantine.orders, quarantine.order_items, sources["pets"]
    )
    predictions = predict_temporal_service_current_probability(
        artifact,
        events,
        quarantine.orders,
        as_of_timestamp=as_of,
        window_days=window_days,
    )
    if predictions.empty:
        raise ValueError("내부 검증 대상이 0건이라 결과를 적재하지 않습니다.")
    results = pd.DataFrame(
        {
            "publication_id": publication_id,
            "user_id": predictions["user_id"].astype("string"),
            # 미지정 pet이 섞이면 pandas가 정수 ID를 12.0 같은 float로 올립니다.
            # nullable 정수형으로 복원하되 소수 ID는 조용히 잘라내지 않습니다.
            "pet_id": predictions["pet_id"].astype("Int64"),
            "target_scope": "PRODUCT_GROUP",
            "target_id": predictions["target_id"].astype("string"),
            "window_days": window_days,
            "prediction_status": "READY",
            "conditional_repurchase_probability": predictions[
                "conditional_repurchase_probability"
            ],
        },
        columns=[
            "publication_id",
            "user_id",
            "pet_id",
            "target_scope",
            "target_id",
            "window_days",
            "prediction_status",
            "conditional_repurchase_probability",
        ],
    )
    batch = pd.DataFrame(
        [
            {
                "publication_id": publication_id,
                "idempotency_key": publication_id,
                "as_of_timestamp": as_of,
                "created_at": created,
                "publication_status": "SHADOW",
                "expected_result_count": len(results),
                "artifact_id": artifact.artifact_id,
                "feature_generation_version": artifact.feature_generation_version,
            }
        ]
    )
    batch, results = validate_prediction_publications(batch, results)
    return ShadowPreparation(
        batch=batch,
        results=results,
        source_order_count=len(orders),
        quarantined_order_count=len(orders) - len(quarantine.orders),
        target_count=len(results),
    )
