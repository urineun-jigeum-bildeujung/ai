"""추출한 운영 CSV에서 반려동물 생일 불일치와 구매 보존 건수를 점검합니다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.modeling.operational_purchase_inputs import (
    prepare_operational_purchase_inputs,
)


def _read_sources(
    orders_path: Path, items_path: Path, pets_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """BE 원천의 컬럼 이름과 자료형을 기존 운영 입력 계약에 맞춥니다."""
    orders = pd.read_csv(orders_path, dtype={"id": "string", "member_id": "string"})
    items = pd.read_csv(
        items_path,
        dtype={
            "id": "string",
            "order_id": "string",
            "product_id": "string",
            "product_group_id_snapshot": "string",
            "pet_id": "string",
        },
    )
    pets = pd.read_csv(pets_path, dtype={"pet_id": "string"})
    orders = orders.rename(columns={"id": "order_id", "member_id": "user_id"})
    items = items.rename(columns={"id": "order_item_id"})
    if "id" in pets and "pet_id" not in pets:
        pets = pets.rename(columns={"id": "pet_id"})
    # SQL 추출이 id 또는 *_id 별칭을 사용해도 조인 키의 자료형을 일치시킵니다.
    for frame, columns in (
        (orders, ("order_id", "user_id")),
        (
            items,
            (
                "order_item_id",
                "order_id",
                "product_id",
                "product_group_id_snapshot",
                "pet_id",
            ),
        ),
        (pets, ("pet_id",)),
    ):
        for column in columns:
            frame[column] = frame[column].astype("string")
    # CSV에는 PostgreSQL boolean이 t/f로 저장될 수도 있습니다.
    replenishable = items["is_replenishable_snapshot"].astype("string").str.lower()
    if not replenishable.isin(("true", "false", "t", "f")).all():
        raise ValueError("is_replenishable_snapshot에 알 수 없는 불리언 값이 있습니다.")
    items["is_replenishable_snapshot"] = replenishable.isin(("true", "t"))
    return orders, items, pets


def audit_sources(
    orders_path: Path, items_path: Path, pets_path: Path
) -> dict[str, int]:
    """원본 파일은 읽기만 하고, 개인 이력 제외 전후 건수만 반환합니다."""
    orders, items, pets = _read_sources(orders_path, items_path, pets_path)
    prepared = prepare_operational_purchase_inputs(orders, items, pets)
    return {
        "source_order_count": len(orders),
        "source_order_item_count": len(items),
        "valid_order_item_count": len(prepared.valid_items),
        "valid_purchase_order_count": prepared.all_purchase_events[
            "order_id"
        ].nunique(),
        "pet_history_item_count": len(prepared.pet_history_items),
        "excluded_late_birth_item_count": prepared.excluded_late_birth_item_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=Path, required=True)
    parser.add_argument("--order-items", type=Path, required=True)
    parser.add_argument("--pets", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_sources(args.orders, args.order_items, args.pets),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
