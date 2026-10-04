"""완료된 로컬 스냅샷에 주문 1건을 메모리에서만 추가해 SHADOW 반응을 검증합니다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.modeling.artifacts import load_model_artifact
from scripts.modeling.shadow_batch import prepare_shadow_publication
from scripts.shadow_batch_runtime import (
    FROZEN_SHADOW_AFT_ARTIFACT_ID,
    _check_cut,
    _read_local_snapshot,
    _timestamp,
)


def _next_id(frame: pd.DataFrame, column: str) -> str:
    return str(frame[column].astype("int64").max() + 1)


def _candidate(
    sources: dict[str, pd.DataFrame], baseline: pd.DataFrame, *, as_of: pd.Timestamp
) -> tuple[pd.Series, pd.Series]:
    orders = sources["orders"]
    items = sources["order_items"]
    eligible = items.merge(
        orders[["order_id", "user_id", "order_status", "ordered_at", "paid_at"]],
        on="order_id",
        validate="many_to_one",
    )
    eligible = eligible.loc[
        eligible["pet_id"].notna()
        & eligible["item_status"].eq("PAID")
        & eligible["order_status"].eq("CONFIRMED")
        & eligible["is_replenishable_snapshot"].eq(True)
        & pd.to_datetime(eligible["ordered_at"], utc=True).le(as_of)
        & pd.to_datetime(eligible["paid_at"], utc=True).le(as_of)
    ]
    repeated = (
        eligible.groupby(["user_id", "pet_id", "product_group_id_snapshot"], sort=True)[
            "order_id"
        ]
        .nunique()
        .loc[lambda counts: counts.ge(2)]
        .sort_values(ascending=False, kind="stable")
    )
    visible_keys = set(
        zip(
            baseline["user_id"].astype(str),
            baseline["pet_id"].astype(str),
            baseline["target_id"].astype(str),
            strict=True,
        )
    )
    for user_id, pet_id, group_id in repeated.index:
        if (str(user_id), str(pet_id), str(group_id)) in visible_keys:
            item = eligible.loc[
                eligible["user_id"].eq(user_id)
                & eligible["pet_id"].eq(pet_id)
                & eligible["product_group_id_snapshot"].eq(group_id)
            ].iloc[0]
            order = orders.loc[orders["order_id"].eq(item["order_id"])].iloc[0]
            return order, item
    raise ValueError("변경 검증에 쓸 기존 사용자·상품군 이력을 찾지 못했습니다.")


def verify_delta(
    snapshot_directory: Path,
    model_directory: Path,
    *,
    as_of_timestamp: str,
) -> dict[str, object]:
    sources, manifest = _read_local_snapshot(snapshot_directory)
    as_of = _timestamp(as_of_timestamp, name="as_of_timestamp")
    _check_cut(as_of, manifest["order_extracted_at"], manifest["member_extracted_at"])
    artifact = load_model_artifact(model_directory)
    if artifact.artifact_id != FROZEN_SHADOW_AFT_ARTIFACT_ID:
        raise ValueError("사전 고정 AFT 아티팩트가 아닙니다.")

    def prepare(input_sources: dict[str, pd.DataFrame], publication_id: str):
        return prepare_shadow_publication(
            input_sources,
            artifact,
            as_of_timestamp=as_of,
            created_at=as_of,
            window_days=30,
            publication_id=publication_id,
        )

    before = prepare(sources, "local-source-delta-before")
    old_order, old_item = _candidate(sources, before.results, as_of=as_of)
    new_order_id = _next_id(sources["orders"], "order_id")
    new_item_id = _next_id(sources["order_items"], "order_item_id")
    first_history_id = int(_next_id(sources["histories"], "history_id"))
    ordered_at = as_of - pd.Timedelta(hours=3)
    paid_at = as_of - pd.Timedelta(hours=2)

    new_order = old_order.copy()
    new_order["order_id"] = new_order_id
    new_order["ordered_at"] = ordered_at.isoformat()
    new_order["paid_at"] = paid_at.isoformat()
    new_order["order_status"] = "PAID"
    new_item = old_item[sources["order_items"].columns].copy()
    new_item["order_item_id"] = new_item_id
    new_item["order_id"] = new_order_id
    new_item["quantity"] = 1
    new_item["cancelled_quantity"] = 0
    new_item["returned_quantity"] = 0
    new_item["item_status"] = "PAID"
    history = pd.DataFrame(
        [
            {
                "history_id": str(first_history_id),
                "order_id": new_order_id,
                "from_status": None,
                "to_status": "PENDING",
                "changed_at": ordered_at.isoformat(),
            },
            {
                "history_id": str(first_history_id + 1),
                "order_id": new_order_id,
                "from_status": "PENDING",
                "to_status": "PAID",
                "changed_at": paid_at.isoformat(),
            },
        ],
        columns=sources["histories"].columns,
    )
    changed = dict(sources)
    changed["orders"] = pd.concat(
        [sources["orders"], new_order.to_frame().T], ignore_index=True
    )
    changed["order_items"] = pd.concat(
        [sources["order_items"], new_item.to_frame().T], ignore_index=True
    )
    changed["order_items"] = changed["order_items"].astype(
        {
            column: sources["order_items"][column].dtype
            for column in (
                "quantity",
                "cancelled_quantity",
                "returned_quantity",
                "is_replenishable_snapshot",
            )
        }
    )
    changed["histories"] = pd.concat([sources["histories"], history], ignore_index=True)
    after = prepare(changed, "local-source-delta-after")

    def probability(results: pd.DataFrame) -> float:
        target = results.loc[
            results["user_id"].eq(str(old_order["user_id"]))
            & results["pet_id"].astype(str).eq(str(old_item["pet_id"]))
            & results["target_id"].eq(str(old_item["product_group_id_snapshot"])),
            "conditional_repurchase_probability",
        ]
        if len(target) != 1:
            raise ValueError("대상 예측 결과가 정확히 1건이어야 합니다.")
        return float(target.iloc[0])

    before_probability = probability(before.results)
    after_probability = probability(after.results)
    if before_probability == after_probability:
        raise ValueError("유효 구매 1건 추가 후 대상 예측 확률이 바뀌지 않았습니다.")
    if after.source_order_count != before.source_order_count + 1:
        raise ValueError("변경 주문 수가 정확히 1건이 아닙니다.")
    if after.source_order_item_count != before.source_order_item_count + 1:
        raise ValueError("변경 주문상품 수가 정확히 1건이 아닙니다.")
    if after.quarantined_order_count != before.quarantined_order_count:
        raise ValueError("새 유효 주문이 격리되었습니다.")
    return {
        "event": "repurchase_shadow_source_delta_verified",
        "mode": "local_memory_only",
        "source_order_count_before": before.source_order_count,
        "source_order_count_after": after.source_order_count,
        "source_order_item_count_before": before.source_order_item_count,
        "source_order_item_count_after": after.source_order_item_count,
        "prediction_count_before": before.target_count,
        "prediction_count_after": after.target_count,
        "target_probability_before": before_probability,
        "target_probability_after": after_probability,
        "target_probability_changed": True,
        "database_writes": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-directory", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_delta(
                args.snapshot_directory,
                args.model_directory,
                as_of_timestamp=args.as_of,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
