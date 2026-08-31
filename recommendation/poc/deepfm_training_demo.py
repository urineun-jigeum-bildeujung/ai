# -*- coding: utf-8 -*-
"""
합성 order_items(더미) -> positive/negative 라벨 생성 -> feature와 결합 -> 최종 학습 샘플 데모.
실제 백엔드 합성 order_items가 오면 DUMMY_ORDER_ITEMS, DUMMY_ORDERS 부분만 교체하면 됨.
"""

import json
from collections import defaultdict

from dummy_data import PET_PROFILES, PRODUCTS, REVIEWS
from sentiment import analyze_sentiment
from aspect_tagging import tag_aspects
from deepfm_features import build_interaction_features
from deepfm_labeling import build_training_pairs

# -----------------------------
# 더미 orders / order_items (실제로는 백엔드 합성데이터로 교체)
# -----------------------------
DUMMY_ORDERS = {
    "order_001": {"order_status": "PAID", "paid_at": "2026-06-01", "ordered_at": "2026-06-01"},
    "order_002": {"order_status": "PAID", "paid_at": "2026-06-10", "ordered_at": "2026-06-10"},
    "order_003": {"order_status": "PARTIAL_REFUND", "paid_at": "2026-07-01", "ordered_at": "2026-07-01"},
}

DUMMY_ORDER_ITEMS = [
    {
        "order_id": "order_001", "product_id": "prod_001", "pet_id": "pet_001",
        "quantity": 1, "cancelled_quantity": 0, "returned_quantity": 0, "item_status": "PAID",
    },
    {
        "order_id": "order_002", "product_id": "prod_003", "pet_id": "pet_002",
        "quantity": 2, "cancelled_quantity": 0, "returned_quantity": 0, "item_status": "PAID",
    },
    {
        "order_id": "order_003", "product_id": "prod_004", "pet_id": "pet_003",
        "quantity": 3, "cancelled_quantity": 0, "returned_quantity": 1, "item_status": "PARTIAL_RETURN",
    },
]

PETS_BY_ID = {p["pet_id"]: p for p in PET_PROFILES}


def build_product_review_summary():
    grouped = defaultdict(list)
    for review in REVIEWS:
        sentiment = analyze_sentiment(review["review_text"])
        tags = tag_aspects(review["review_text"])
        grouped[review["product_id"]].append({"sentiment_label": sentiment["sentiment_label"], "keyword_tags": tags})

    summary = {}
    for product_id, feats in grouped.items():
        positive_tags = [tag for f in feats if f["sentiment_label"] == "POSITIVE" for tag in f["keyword_tags"]]
        negative_tags = [tag for f in feats for tag in f["keyword_tags"] if "후기 있음" in tag or "낮음" in tag]
        summary[product_id] = {
            "positive_tags": positive_tags,
            "negative_tags": negative_tags,
            "total_reviews": len(feats),
        }
    return summary


def run_demo():
    print("=" * 60)
    print("STEP 1. order_items -> positive/negative 라벨 생성")
    print("=" * 60)
    training_pairs = build_training_pairs(DUMMY_ORDER_ITEMS, DUMMY_ORDERS, PETS_BY_ID, PRODUCTS)
    for pair in training_pairs:
        print(json.dumps(pair, ensure_ascii=False))

    print(f"\n총 샘플 수: {len(training_pairs)} (positive: {sum(1 for p in training_pairs if p['label'] == 1)}, "
          f"negative: {sum(1 for p in training_pairs if p['label'] == 0)})")

    print()
    print("=" * 60)
    print("STEP 2. 라벨 + feature 결합 -> 최종 학습 샘플")
    print("=" * 60)
    product_summary = build_product_review_summary()
    products_by_id = {p["product_id"]: p for p in PRODUCTS}

    for pair in training_pairs[:3]:  # 데모용으로 3개만 출력
        pet = PETS_BY_ID[pair["pet_id"]]
        product = products_by_id[pair["product_id"]]
        summary = product_summary.get(product["product_id"], {"positive_tags": [], "negative_tags": [], "total_reviews": 1})

        sample = build_interaction_features(pet, product, summary)
        sample["label"] = pair["label"]  # 앞서 만든 라벨로 채움

        print(f"\n--- pet:{pair['pet_id']} x product:{pair['product_id']} (label={pair['label']}) ---")
        print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_demo()