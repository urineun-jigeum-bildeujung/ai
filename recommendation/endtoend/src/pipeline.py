# -*- coding: utf-8 -*-
"""
전체 추천 파이프라인 통합 (리뷰 작성자 프로필 유사도 반영 버전).

reviews -> KcELECTRA 감성분석 -> aspect 태깅 -> 상품별 review_features 집계
-> pet_id 입력 -> 알러지 필터링(역추천) -> DeepFM 스코어링 -> recommendation_items 출력

[이번 변경 사항 - 콜드스타트 로직: 사용자 프로필 + 리뷰 작성자 프로필 + 리뷰 keywords]
- DeepFM feature와 추천 사유 모두 "추천 대상과 프로필이 비슷한 리뷰 작성자"의
  반응에 가중치를 둬서 계산 (reviewer_profile_similarity.py 사용)
- 리뷰 작성자의 pet_profile(reviewer_pet)은 data_access.reviews_repository를 통해 가져온다.
  reviews.pet_id 스키마가 실제로 반영되었으므로, 실제 DB 연동 시에는
  reviews_repository.py의 USE_DUMMY_DATA 환경변수만 false로 바꾸면 되고
  이 파일(pipeline.py)은 수정할 필요가 없다.

실행: python3 src/pipeline.py
사전 조건: train/finetune_kcelectra.py, train/train_deepfm.py 를 먼저 실행해서
           models/kcelectra/, models/deepfm/ 에 학습 결과물이 있어야 한다.
"""

import sys
import os
import json
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # data_access 등 src/ 하위 모듈 import용
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data", "dummy"))
sys.path.append(os.path.join(os.path.dirname(__file__), "sentiment"))
sys.path.append(os.path.join(os.path.dirname(__file__), "aspect"))
sys.path.append(os.path.join(os.path.dirname(__file__), "features"))
sys.path.append(os.path.join(os.path.dirname(__file__), "recommend"))

from dummy_data import PET_PROFILES, PRODUCTS
from src.data_access.reviews_repository import load_reviews_with_reviewer_pet
from kcelectra_infer import analyze_sentiment
from tagging import tag_aspects, ASPECTS
from deepfm_features import build_interaction_features
from allergy_filter import check_allergy_conflict
from deepfm_model import load_deepfm
from reviewer_profile_similarity import compute_weighted_aspect_scores

DEEPFM_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "deepfm")

# tagging.py의 ASPECTS에 정의된 부정 태그 문구를 그대로 참조 (문자열 패턴 추측 금지)
NEGATIVE_TAGS = {info["tag_negative"] for info in ASPECTS.values()}


def build_review_features_with_authors():
    """
    reviews -> review_features (KcELECTRA 감성분석 + aspect 태깅) + reviewer_pet 포함.
    상품별로 묶어서 반환: {product_id: [{"reviewer_pet":, "sentiment_label":, "keyword_tags":}, ...]}

    데이터 출처(더미 파일 vs 실제 PostgreSQL)는 data_access.reviews_repository가 담당하며,
    USE_DUMMY_DATA 환경변수로 전환된다. 이 함수는 출처와 무관하게 동일하게 동작한다.
    """
    grouped = defaultdict(list)
    for review in load_reviews_with_reviewer_pet():
        sentiment = analyze_sentiment(review["review_text"])
        tags = tag_aspects(review["review_text"])
        grouped[review["product_id"]].append({
            "reviewer_pet": review["reviewer_pet"],
            "sentiment_label": sentiment["sentiment_label"],
            "keyword_tags": tags,
        })
    return grouped


def summarize_by_product(product_reviews_by_id: dict) -> dict:
    """
    [현재 파이프라인에서는 미사용]
    이전 방식(상품 전체 리뷰 단순 평균, pet과 무관)의 잔재.
    이제 DeepFM feature와 추천 사유 모두 compute_weighted_aspect_scores()
    (pet마다 다른 유사도 가중 결과)를 사용하므로 run_pipeline()에서 호출하지 않는다.
    deepfm_features.build_product_aspect_features()가 이 형식(positive_tags/negative_tags)도
    폴백으로 계속 지원하므로, 필요 시(예: pet 정보 없이 상품 전체 통계만 보고 싶을 때) 재사용 가능.
    """
    summary = {}
    for product_id, feats in product_reviews_by_id.items():
        positive_feats = [f for f in feats if f["sentiment_label"] == "POSITIVE"]
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


def build_reason_from_weighted_scores(weighted_result: dict, top_n: int = 5) -> tuple:
    """
    compute_weighted_aspect_scores() 결과에서 긍정적인(값이 양수인) 태그를 골라
    reason_keywords, reason_text를 생성한다.
    (나와 비슷한 프로필의 리뷰어 반응일수록 이미 가중치가 높게 반영되어 있음)
    """
    weighted_scores = weighted_result["weighted_aspect_scores"]
    # 값이 양수(긍정 방향)인 것만, 점수 높은 순으로 정렬
    positive_ranked = sorted(
        [(tag, score) for tag, score in weighted_scores.items() if score > 0],
        key=lambda x: x[1],
        reverse=True,
    )
    reason_keywords = [tag for tag, _ in positive_ranked[:top_n]]

    if reason_keywords:
        reason_text = (
            "나와 비슷한 반려동물을 키우는 분들의 리뷰에서 "
            + ", ".join(reason_keywords) + " 등의 반응이 있어 추천합니다."
        )
    elif weighted_result["used_review_count"] > 0:
        reason_text = "등록하신 반려동물 정보를 기준으로 추천합니다."
    else:
        reason_text = "등록하신 반려동물 정보를 기준으로 추천합니다. (참고할 만한 비슷한 프로필의 리뷰가 아직 없어요)"

    return reason_keywords, reason_text


def recommend_for_pet(pet: dict, products: list,
                       product_reviews_by_id: dict, encoder, model) -> list:
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

        # 3) 리뷰 작성자 프로필 유사도 가중 aspect score 계산
        #    (콜드스타트 로직: 사용자 프로필 + 리뷰 작성자 프로필 + keywords 를
        #     DeepFM feature와 추천 사유 양쪽에 동일하게 반영 -- 이전에는 추천 사유에만 반영했었음)
        product_reviews = product_reviews_by_id.get(product["product_id"], [])
        weighted_result = compute_weighted_aspect_scores(pet, product_reviews)
        summary = {"weighted_aspect_scores": weighted_result["weighted_aspect_scores"]}

        # 4) DeepFM 스코어링 (유사도 가중 feature 사용)
        features = build_interaction_features(pet, product, summary)
        encoded = encoder.encode(features)
        batch = encoder.collate([encoded])

        import torch
        with torch.no_grad():
            score = model(batch).item()

        # 5) 추천 사유 생성 (3번에서 이미 계산한 weighted_result 재사용)
        reason_keywords, reason_text = build_reason_from_weighted_scores(weighted_result)

        results.append({
            "product_id": product["product_id"],
            "product_name": product["product_name"],
            "recommend_type": "RECOMMEND",
            "score": round(score, 4),
            "reason_keywords": reason_keywords,
            "reason_text": reason_text,
            "matched_allergen": [],
            # 참고용 -- 이 추천 사유/점수가 실제로 몇 건의 "유사 프로필 리뷰"를 근거로 했는지
            "reviewer_similarity_meta": {
                "used_review_count": weighted_result["used_review_count"],
                "total_similarity_weight": weighted_result["total_weight"],
            },
        })

    # 6) 점수 기준 정렬 + rank 부여 (RECOMMEND만 랭킹)
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
    print("STEP 1. 리뷰 -> KcELECTRA 감성분석 + aspect 태깅 + reviewer_pet 결합")
    print("=" * 60)
    product_reviews_by_id = build_review_features_with_authors()
    total_reviews = sum(len(v) for v in product_reviews_by_id.values())
    print(f"{total_reviews}건 처리 완료")

    print()
    print("=" * 60)
    print("STEP 2. DeepFM 모델 로드")
    print("=" * 60)
    encoder, model = load_deepfm(DEEPFM_MODEL_DIR)
    print("모델 로드 완료")

    print()
    print("=" * 60)
    print("STEP 3. 반려동물별 추천/역추천 (리뷰 작성자 프로필 유사도를 feature+사유 양쪽에 반영)")
    print("=" * 60)
    for pet in PET_PROFILES:
        print(f"\n--- pet_id: {pet['pet_id']} ({pet['species']}, {pet['breed']}, 알러지: {pet['allergy_codes']}) ---")
        items = recommend_for_pet(pet, PRODUCTS, product_reviews_by_id, encoder, model)
        for item in items:
            print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    run_pipeline()