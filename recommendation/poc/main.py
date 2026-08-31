# -*- coding: utf-8 -*-
"""
전체 파이프라인 실행 (POC).

더미 리뷰 -> 감성분석(룰베이스) -> aspect 태깅 -> 상품별 집계
-> 반려동물별 추천/역추천(룰베이스) -> recommendation_items 형태 출력
"""

import json
from collections import defaultdict

from dummy_data import PET_PROFILES, PRODUCTS, REVIEWS
from sentiment import analyze_sentiment
from aspect_tagging import tag_aspects
from recommend import recommend_products


def build_review_features():
    """
    reviews -> review_features 스키마 형태로 변환.
    (sentiment_label, sentiment_score, keyword_tags)
    """
    review_features = []
    for review in REVIEWS:
        sentiment = analyze_sentiment(review["review_text"])
        tags = tag_aspects(review["review_text"])

        review_features.append({
            "review_id": review["review_id"],
            "product_id": review["product_id"],
            "rating": review["rating"],
            "sentiment_label": sentiment["sentiment_label"],
            "sentiment_score": sentiment["sentiment_score"],
            "keyword_tags": tags,
        })
    return review_features


def summarize_by_product(review_features: list) -> dict:
    """
    상품별로 review_features 를 집계해서 recommend.py 에 넘길 요약 정보 생성.
    """
    grouped = defaultdict(list)
    for rf in review_features:
        grouped[rf["product_id"]].append(rf)

    summary = {}
    for product_id, feats in grouped.items():
        total = len(feats)
        positive_feats = [f for f in feats if f["sentiment_label"] == "POSITIVE"]
        avg_sentiment_score = round(sum(f["sentiment_score"] for f in feats) / total, 2)
        positive_ratio = round(len(positive_feats) / total, 2)

        positive_tags = sorted({tag for f in positive_feats for tag in f["keyword_tags"] if "없음" in tag or ("후기" not in tag)})
        negative_tags = sorted({tag for f in feats for tag in f["keyword_tags"] if "후기 있음" in tag})

        summary[product_id] = {
            "avg_sentiment_score": avg_sentiment_score,
            "positive_ratio": positive_ratio,
            "positive_tags": positive_tags,
            "negative_tags": negative_tags,
        }
    return summary


def run_pipeline():
    print("=" * 60)
    print("STEP 1. 리뷰 감성분석 + aspect 태깅 (review_features 생성)")
    print("=" * 60)
    review_features = build_review_features()
    for rf in review_features:
        print(json.dumps(rf, ensure_ascii=False))

    print()
    print("=" * 60)
    print("STEP 2. 상품별 리뷰 집계")
    print("=" * 60)
    product_summary = summarize_by_product(review_features)
    for pid, s in product_summary.items():
        print(f"{pid}: {json.dumps(s, ensure_ascii=False)}")

    print()
    print("=" * 60)
    print("STEP 3. 반려동물별 추천/역추천 (recommendation_items)")
    print("=" * 60)
    for pet in PET_PROFILES:
        print(f"\n--- pet_id: {pet['pet_id']} ({pet['species']}, {pet['breed']}, 알러지: {pet['allergy_codes']}) ---")
        items = recommend_products(pet, PRODUCTS, product_summary)
        for item in items:
            print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    run_pipeline()