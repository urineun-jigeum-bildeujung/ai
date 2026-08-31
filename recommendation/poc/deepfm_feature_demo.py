# -*- coding: utf-8 -*-
"""
DeepFM feature 설계 데모.
POC의 review_features 집계 결과를 그대로 활용해서
pet x product 조합의 feature 벡터가 실제로 만들어지는지 확인.
"""

import json
from collections import defaultdict

from dummy_data import PET_PROFILES, PRODUCTS, REVIEWS
from sentiment import analyze_sentiment
from aspect_tagging import tag_aspects
from deepfm_features import build_interaction_features


def build_review_features():
    review_features = []
    for review in REVIEWS:
        sentiment = analyze_sentiment(review["review_text"])
        tags = tag_aspects(review["review_text"])
        review_features.append({
            "product_id": review["product_id"],
            "sentiment_label": sentiment["sentiment_label"],
            "keyword_tags": tags,
        })
    return review_features


def summarize_by_product(review_features):
    grouped = defaultdict(list)
    for rf in review_features:
        grouped[rf["product_id"]].append(rf)

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
    review_features = build_review_features()
    product_summary = summarize_by_product(review_features)

    print("=" * 60)
    print("DeepFM feature 벡터 데모 (pet_001 x 각 상품)")
    print("=" * 60)

    pet = PET_PROFILES[0]  # pet_001
    for product in PRODUCTS:
        summary = product_summary.get(product["product_id"], {"positive_tags": [], "negative_tags": [], "total_reviews": 1})
        features = build_interaction_features(pet, product, summary)
        print(f"\n--- {product['product_id']} ({product['product_name']}) ---")
        print(json.dumps(features, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_demo()