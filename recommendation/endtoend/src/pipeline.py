# -*- coding: utf-8 -*-
"""
전체 추천 파이프라인 통합 (정형 aspect 평점 기반 버전).

reviews(정형 aspect 평점 5개 + 전체 별점) -> 알러지 필터링(역추천)
-> 리뷰 작성자 프로필 유사도 반영 -> DeepFM 스코어링 -> recommendation_items 출력

[이번 변경 사항 - KcELECTRA/tagging.py 제거]
- 리뷰 작성 화면에서 사용자가 5개 aspect(기호성/소화·배변/피부·모질/체중·활력/알러지반응)를
  1~3점으로 직접 선택하는 정형 입력 방식으로 확정됨에 따라, 텍스트에서 aspect를
  추출하던 tagging.py가 더 이상 필요 없어짐.
- 전체적인 리뷰 긍/부정 판단(KcELECTRA)도, 이미 사용자가 남기는 1~5점 별점(rating)이
  같은 역할을 하고 있어 추천 로직에서 제외하기로 확정.
- 가격·가성비는 추천 근거에서 아예 제외하기로 확정되어, 5개 aspect만 다룬다.

실행: python3 src/pipeline.py
사전 조건: train/train_deepfm.py 를 먼저 실행해서 models/deepfm/ 에 학습 결과물이 있어야 한다.
          (KcELECTRA 모델은 더 이상 필요 없음)
"""

import sys
import os
import json
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # data_access 등 src/ 하위 모듈 import용
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data", "dummy"))
sys.path.append(os.path.join(os.path.dirname(__file__), "aspect"))
sys.path.append(os.path.join(os.path.dirname(__file__), "features"))
sys.path.append(os.path.join(os.path.dirname(__file__), "recommend"))

from dummy_data import PET_PROFILES, PRODUCTS
from src.data_access.reviews_repository import load_reviews_with_reviewer_pet
from rating_converter import ASPECT_FIELD_TO_CODE, convert_rating_to_score
from deepfm_features import build_interaction_features
from allergy_filter import check_allergy_conflict
from deepfm_model import load_deepfm
from reviewer_profile_similarity import compute_weighted_aspect_scores

DEEPFM_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "deepfm")

ASPECT_KO_NAMES = {
    "palatability": "기호성",
    "digestion": "소화·배변",
    "skin_coat": "피부·모질",
    "vitality_weight": "체중·활력",
    "allergic_reaction": "알러지 반응",
}


def build_reviews_with_ratings():
    """
    reviews -> {reviewer_pet, ratings} 형태로 변환, 상품별로 묶어서 반환.
    ratings의 각 값은 -1~1로 변환된 aspect score이며, 사용자가 평가하지 않은 항목은 None
    (rating_converter.convert_rating_to_score는 None을 0.0/중립으로 바꾸므로,
     "평가 안 함"과 "중립 평가"를 구분하기 위해 원본이 None이면 여기서도 None으로 유지한다).
    """
    grouped = defaultdict(list)
    for review in load_reviews_with_reviewer_pet():
        ratings = {}
        for field_name, aspect_code in ASPECT_FIELD_TO_CODE.items():
            raw_rating = review.get(field_name)  # 1~3 또는 None (평가 안 함)
            ratings[aspect_code] = convert_rating_to_score(raw_rating) if raw_rating is not None else None
        grouped[review["product_id"]].append({
            "reviewer_pet": review["reviewer_pet"],
            "ratings": ratings,
        })
    return grouped


def build_reason_from_weighted_scores(weighted_result: dict, top_n: int = 5) -> tuple:
    """
    compute_weighted_aspect_scores() 결과에서 긍정적인(값이 양수인) aspect를 골라
    reason_keywords, reason_text를 생성한다.
    """
    weighted_scores = weighted_result["weighted_aspect_scores"]
    positive_ranked = sorted(
        [(code, score) for code, score in weighted_scores.items() if score > 0],
        key=lambda x: x[1],
        reverse=True,
    )
    reason_keywords = [ASPECT_KO_NAMES.get(code, code) for code, _ in positive_ranked[:top_n]]

    if reason_keywords:
        reason_text = (
            "나와 비슷한 반려동물을 키우는 분들이 남긴 "
            + ", ".join(reason_keywords) + " 평가가 좋아 추천합니다."
        )
    elif weighted_result["used_review_count"] > 0:
        reason_text = "등록하신 반려동물 정보를 기준으로 추천합니다."
    else:
        reason_text = "등록하신 반려동물 정보를 기준으로 추천합니다. (참고할 만한 비슷한 프로필의 리뷰가 아직 없어요)"

    return reason_keywords, reason_text


def recommend_for_pet(pet: dict, products: list, reviews_by_product: dict, encoder, model) -> list:
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
        product_reviews = reviews_by_product.get(product["product_id"], [])
        weighted_result = compute_weighted_aspect_scores(pet, product_reviews)
        summary = {"weighted_aspect_scores_by_code": weighted_result["weighted_aspect_scores"]}

        # 4) DeepFM 스코어링
        features = build_interaction_features(pet, product, summary)
        encoded = encoder.encode(features)
        batch = encoder.collate([encoded])

        import torch
        with torch.no_grad():
            score = model(batch).item()

        # 5) 추천 사유 생성 (3번에서 계산한 weighted_result 재사용)
        reason_keywords, reason_text = build_reason_from_weighted_scores(weighted_result)

        results.append({
            "product_id": product["product_id"],
            "product_name": product["product_name"],
            "recommend_type": "RECOMMEND",
            "score": round(score, 4),
            "reason_keywords": reason_keywords,
            "reason_text": reason_text,
            "matched_allergen": [],
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
    print("STEP 1. 리뷰 -> 정형 aspect 평점 변환 + reviewer_pet 결합")
    print("=" * 60)
    reviews_by_product = build_reviews_with_ratings()
    total_reviews = sum(len(v) for v in reviews_by_product.values())
    print(f"{total_reviews}건 처리 완료")

    print()
    print("=" * 60)
    print("STEP 2. DeepFM 모델 로드")
    print("=" * 60)
    encoder, model = load_deepfm(DEEPFM_MODEL_DIR)
    print("모델 로드 완료")

    print()
    print("=" * 60)
    print("STEP 3. 반려동물별 추천/역추천")
    print("=" * 60)
    for pet in PET_PROFILES:
        print(f"\n--- pet_id: {pet['pet_id']} ({pet['species']}, {pet['breed']}, 알러지: {pet['allergy_codes']}) ---")
        items = recommend_for_pet(pet, PRODUCTS, reviews_by_product, encoder, model)
        for item in items:
            print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    run_pipeline()