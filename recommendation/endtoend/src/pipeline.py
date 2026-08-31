# -*- coding: utf-8 -*-
"""
전체 추천 파이프라인 통합.

reviews -> KcELECTRA 감성분석 -> aspect 태깅 -> 상품별 review_features 집계
-> pet_id 입력 -> 알러지 필터링(역추천) -> DeepFM 스코어링 -> recommendation_items 출력

실행: python3 src/pipeline.py
사전 조건: train/finetune_kcelectra.py, train/train_deepfm.py 를 먼저 실행해서
           models/kcelectra/, models/deepfm/ 에 학습 결과물이 있어야 한다.
"""

import sys
import os
import json
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data", "dummy"))
sys.path.append(os.path.join(os.path.dirname(__file__), "sentiment"))
sys.path.append(os.path.join(os.path.dirname(__file__), "aspect"))
sys.path.append(os.path.join(os.path.dirname(__file__), "features"))
sys.path.append(os.path.join(os.path.dirname(__file__), "recommend"))

from dummy_data import PET_PROFILES, PRODUCTS
from dummy_reviews import DUMMY_REVIEWS
from kcelectra_infer import analyze_sentiment
from tagging import tag_aspects, ASPECTS
from deepfm_features import build_interaction_features
from allergy_filter import check_allergy_conflict
from deepfm_model import load_deepfm

DEEPFM_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "deepfm")

# tagging.py의 ASPECTS에 정의된 부정 태그 문구를 그대로 참조 (문자열 패턴 추측 금지)
NEGATIVE_TAGS = {info["tag_negative"] for info in ASPECTS.values()}


def build_review_features():
    """reviews -> review_features (KcELECTRA 감성분석 + aspect 태깅)."""
    review_features = []
    for review in DUMMY_REVIEWS:
        sentiment = analyze_sentiment(review["review_text"])
        tags = tag_aspects(review["review_text"])
        review_features.append({
            "review_id": review["review_id"],
            "product_id": review["product_id"],
            "sentiment_label": sentiment["sentiment_label"],
            "sentiment_score": sentiment["sentiment_score"],
            "keyword_tags": tags,
        })
    return review_features


def summarize_by_product(review_features: list) -> dict:
    """상품별로 review_features 집계 -> DeepFM feature 및 추천 사유 생성에 사용."""
    grouped = defaultdict(list)
    for rf in review_features:
        grouped[rf["product_id"]].append(rf)

    summary = {}
    for product_id, feats in grouped.items():
        positive_feats = [f for f in feats if f["sentiment_label"] == "POSITIVE"]
        # 긍정 리뷰의 태그 중에서도, tagging.py가 정의한 부정 태그 목록(NEGATIVE_TAGS)에
        # 없는 것만 추천 사유로 사용 (문자열 패턴 추측이 아니라 정의를 직접 참조)
        positive_tags = [
            tag for f in positive_feats for tag in f["keyword_tags"]
            if tag not in NEGATIVE_TAGS
        ]
        negative_tags = [tag for f in feats for tag in f["keyword_tags"] if tag in NEGATIVE_TAGS]
        summary[product_id] = {
            "positive_tags": positive_tags,
            "negative_tags": negative_tags,
            "total_reviews": len(feats),
        }
    return summary


def recommend_for_pet(pet: dict, products: list, review_summary: dict, encoder, model) -> list:
    """
    pet_id 하나에 대해 전체 상품 후보군을 순회하며
    recommendation_items 스키마 형태의 결과 리스트를 반환한다.
    """
    results = []

    for product in products:
        # 1) 종(species) 불일치 -> 후보 자체에서 제외
        if pet["species"] not in product["target_species"]:
            continue

        # 2) 알러지 매칭 -> 역추천(EXCLUDE)
        allergy_result = check_allergy_conflict(pet["allergy_codes"], product["allergen_flags"])
        if allergy_result["has_conflict"]:
            matched = allergy_result["matched_allergen"]
            results.append({
                "product_id": product["product_id"],
                "product_name": product["product_name"],
                "recommend_type": "EXCLUDE",
                "score": 0.0,
                "reason_keywords": [f"알러지 성분 포함: {', '.join(matched)}"],
                "reason_text": f"{', '.join(matched)} 성분이 포함되어 있어 등록하신 알러지 정보와 맞지 않아 제외되었습니다.",
                "matched_allergen": matched,
            })
            continue

        # 3) DeepFM 스코어링
        summary = review_summary.get(product["product_id"], {
            "positive_tags": [], "negative_tags": [], "total_reviews": 1,
        })
        features = build_interaction_features(pet, product, summary)
        encoded = encoder.encode(features)
        batch = encoder.collate([encoded])

        import torch
        with torch.no_grad():
            score = model(batch).item()

        reason_keywords = list(dict.fromkeys(summary["positive_tags"]))  # 중복 제거, 순서 유지
        if reason_keywords:
            reason_text = "리뷰에서 " + ", ".join(reason_keywords) + " 등의 반응이 있어 추천합니다."
        else:
            reason_text = "등록하신 반려동물 정보를 기준으로 추천합니다."

        results.append({
            "product_id": product["product_id"],
            "product_name": product["product_name"],
            "recommend_type": "RECOMMEND",
            "score": round(score, 4),
            "reason_keywords": reason_keywords,
            "reason_text": reason_text,
            "matched_allergen": [],
        })

    # 4) 점수 기준 정렬 + rank 부여 (RECOMMEND만 랭킹)
    recommend_items = sorted(
        [r for r in results if r["recommend_type"] == "RECOMMEND"],
        key=lambda x: x["score"],
        reverse=True,
    )
    exclude_items = [r for r in results if r["recommend_type"] == "EXCLUDE"]

    for idx, item in enumerate(recommend_items, start=1):
        item["rank"] = idx
    for item in exclude_items:
        item["rank"] = None

    return recommend_items + exclude_items


def run_pipeline():
    print("=" * 60)
    print("STEP 1. 리뷰 -> KcELECTRA 감성분석 + aspect 태깅")
    print("=" * 60)
    review_features = build_review_features()
    print(f"{len(review_features)}건 처리 완료")

    print()
    print("=" * 60)
    print("STEP 2. 상품별 review_features 집계")
    print("=" * 60)
    review_summary = summarize_by_product(review_features)
    for pid, s in review_summary.items():
        print(f"{pid}: positive_tags={s['positive_tags']}, total_reviews={s['total_reviews']}")

    print()
    print("=" * 60)
    print("STEP 3. DeepFM 모델 로드")
    print("=" * 60)
    encoder, model = load_deepfm(DEEPFM_MODEL_DIR)
    print("모델 로드 완료")

    print()
    print("=" * 60)
    print("STEP 4. 반려동물별 추천/역추천 (recommendation_items)")
    print("=" * 60)
    for pet in PET_PROFILES:
        print(f"\n--- pet_id: {pet['pet_id']} ({pet['species']}, {pet['breed']}, 알러지: {pet['allergy_codes']}) ---")
        items = recommend_for_pet(pet, PRODUCTS, review_summary, encoder, model)
        for item in items:
            print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    run_pipeline()