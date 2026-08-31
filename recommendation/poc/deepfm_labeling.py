# -*- coding: utf-8 -*-
"""
DeepFM 학습 라벨 생성 (합성 order_items 활용).

- positive: order_items 스키마의 "유효 구매" 조건을 만족하는 (pet_id, product_id)
- negative: 같은 종(species)의 상품 중 구매하지 않은 것에서 샘플링
- 데이터 누수 방지: 구매 시점(purchased_at) 이전에 작성된 리뷰만 aspect score 계산에 사용
  (이 모듈에서는 개념만 반영, 실제 리뷰 written_at 필터링은 review 집계 단계에서 처리 필요)
"""

import random


def build_positive_samples(order_items: list, orders_by_id: dict) -> list:
    """
    order_items + orders 정보를 조합해 유효 구매만 positive 샘플로 추출.

    order_items 항목 예시 (스키마 기준):
    {
        "order_id": "order_001",
        "product_id": "prod_001",
        "pet_id": "pet_001",
        "quantity": 2,
        "cancelled_quantity": 0,
        "returned_quantity": 0,
        "item_status": "PAID",
    }
    orders_by_id: {"order_001": {"order_status": "PAID", "purchased_at": "2026-05-01"}}
    """
    positives = []
    for item in order_items:
        valid_qty = item["quantity"] - item["cancelled_quantity"] - item["returned_quantity"]
        order = orders_by_id.get(item["order_id"], {})
        order_status_ok = order.get("order_status") in ("PAID", "PARTIAL_REFUND")

        if valid_qty >= 1 and order_status_ok and item.get("pet_id"):
            positives.append({
                "pet_id": item["pet_id"],
                "product_id": item["product_id"],
                "purchased_at": order.get("paid_at") or order.get("ordered_at"),
                "label": 1,
            })
    return positives


def sample_negatives(positives: list, pets_by_id: dict, products: list, n_per_positive: int = 3, seed: int = 42) -> list:
    """
    positive 샘플 1개당 같은 종(species)의 미구매 상품에서 n_per_positive개를 negative로 샘플링.
    """
    random.seed(seed)
    negatives = []

    # pet별로 이미 구매한 product_id 집합 (같은 pet이 산 건 negative 후보에서 제외)
    purchased_by_pet = {}
    for p in positives:
        purchased_by_pet.setdefault(p["pet_id"], set()).add(p["product_id"])

    for pos in positives:
        pet = pets_by_id.get(pos["pet_id"])
        if not pet:
            continue

        candidates = [
            prod for prod in products
            if pet["species"] in prod["target_species"]
            and prod["product_id"] not in purchased_by_pet.get(pos["pet_id"], set())
        ]

        sampled = random.sample(candidates, min(n_per_positive, len(candidates)))
        for prod in sampled:
            negatives.append({
                "pet_id": pos["pet_id"],
                "product_id": prod["product_id"],
                "purchased_at": pos["purchased_at"],  # 학습 시점 기준을 positive와 동일하게 맞춤
                "label": 0,
            })

    return negatives


def build_training_pairs(order_items: list, orders_by_id: dict, pets_by_id: dict, products: list) -> list:
    """
    최종 학습용 (pet_id, product_id, label) 쌍 리스트 생성.
    """
    positives = build_positive_samples(order_items, orders_by_id)
    negatives = sample_negatives(positives, pets_by_id, products)
    return positives + negatives