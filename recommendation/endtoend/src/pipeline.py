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

[이번 변경 사항 - 구매 이력 기반 유사도 추가]
- 확정된 추천 로직 중 "구매 이력 있음" 시나리오(사용자 프로필 + 리뷰 작성자 프로필 +
  keywords + 기존 구매 상품과의 유사도)의 마지막 요소를 추가.
- FR-AI-2-02(대체상품 추천)가 쓰는 product_embeddings(pgvector)를 재사용해서,
  사용자의 기존 구매 상품 평균 임베딩과 후보 상품 임베딩 간 코사인 유사도를 계산.
- 구매 이력이 없는 사용자(콜드스타트)는 0.0(중립)으로 자동 처리되어,
  같은 DeepFM 구조로 콜드스타트/기존 유저를 모두 다룬다.

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
from purchase_history_similarity import build_purchase_history_feature

# USE_DUMMY_DATA=false 인 경우, 구매 이력/상품 임베딩은 order_embedding_repository를
# 거쳐 실제 DB(orders/order_items/product_embeddings)에서 조회한다.
USE_DUMMY_DATA = os.environ.get("USE_DUMMY_DATA", "true").lower() != "false"

DEEPFM_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "deepfm")

# -----------------------------
# 더미 구매 이력 + 상품 임베딩 (실제 orders/order_items/product_embeddings 테이블로 교체 예정)
# -----------------------------
DUMMY_ORDERS = {
    "order_001": {"user_id": "user_001", "order_status": "PAID"},
}
DUMMY_ORDER_ITEMS = [
    {"order_id": "order_001", "product_id": "prod_003", "quantity": 1, "cancelled_quantity": 0, "returned_quantity": 0},
]
# 실제로는 product_embeddings 테이블(pgvector)에서 조회. 여기서는 구조 검증용 저차원 더미 벡터 사용.
DUMMY_PRODUCT_EMBEDDINGS = {
    "prod_001": [0.9, 0.1, 0.0, 0.2],
    "prod_002": [0.85, 0.15, 0.05, 0.1],  # prod_001과 유사 (둘 다 사료류)
    "prod_003": [0.1, 0.9, 0.3, 0.0],
    "prod_004": [0.8, 0.2, 0.0, 0.15],
    "prod_005": [0.05, 0.1, 0.9, 0.2],
}

ASPECT_KO_NAMES = {
    "palatability": "기호성",
    "digestion": "소화·배변",
    "skin_coat": "피부·모질",
    "vitality_weight": "체중·활력",
    "allergic_reaction": "알러지 반응",
}

# 알레르기 충돌/성분 미분석 상품 처리 정책 (기획팀 확정, 2026-09-29)
# - 완전 제외 대신 점수 페널티로 후순위 배치
# - PENALIZED(알레르기 충돌 확정): 70% 감점
# - PENDING(성분 미분석, 판단 보류): 30% 감점 (마일드하게 후순위)
ALLERGY_PENALTY_MULTIPLIER = 0.3
PENDING_PENALTY_MULTIPLIER = 0.7

def score_to_100(score_0_1: float) -> int:
    """
    DeepFM sigmoid 출력(0~1)을 프론트엔드 확정 스펙(0~100 정수)으로 변환.
    프론트에 전달한 답변: "score는 0~100(정수)으로 드리겠습니다. 마이너스 값은 없습니다."
    """
    return round(max(0.0, min(1.0, score_0_1)) * 100)


def build_reviews_with_ratings(reviews: list = None):
    """
    reviews -> {reviewer_pet, ratings} 형태로 변환, 상품별로 묶어서 반환.
    ratings의 각 값은 -1~1로 변환된 aspect score이며, 사용자가 평가하지 않은 항목은 None.

    reviews를 명시적으로 넘기면 그 리스트를 쓰고(API 서버가 요청 body로 받은 리뷰),
    넘기지 않으면 기존처럼 load_reviews_with_reviewer_pet()(더미/DB)을 사용한다
    (로컬 run_pipeline() 스크립트 실행용 하위호환).
    """
    if reviews is None:
        reviews = load_reviews_with_reviewer_pet()

    grouped = defaultdict(list)
    for review in reviews:
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


def recommend_for_pet(
    pet: dict, products: list, reviews_by_product: dict, encoder, model,
    purchase_history_embeddings: list = None,
) -> list:
    """
    pet_id 하나에 대해 전체 상품 후보군을 순회하며
    recommendation_items 스키마 형태의 결과 리스트를 반환한다.

    [2026-09-29 변경] 알레르기 충돌 상품은 더 이상 후보에서 완전히 제외(EXCLUDE)되지 않는다.
    대신 점수에 페널티를 곱해 순위만 뒤로 미루는 방식으로 변경 (기획팀 확정).
    - allergen_flags가 None(미분석)인 상품: allergy_status="PENDING", 30% 감점
    - allergen_flags와 pet.allergy_codes가 겹치는 상품: allergy_status="PENALIZED", 70% 감점
    - 그 외: allergy_status="SAFE", 감점 없음
    장바구니 팝업 역추천(/recommend/exclusions)은 폐기되었고, 이 로직으로 통합됨.
    """
    results = []

    for product in products:
        # 1) 종(species) 불일치 -> 후보 자체에서 제외 (이것만 유일한 하드 필터로 유지)
        if pet["species"] not in product["target_species"]:
            continue

        # 2) 알레르기 판정: SAFE / PENALIZED / PENDING (완전 제외 없음)
        allergen_flags = product.get("allergen_flags")
        if allergen_flags is None:
            allergy_status = "PENDING"
            matched_allergen = []
        else:
            allergy_result = check_allergy_conflict(pet["allergy_codes"], allergen_flags)
            if allergy_result["has_conflict"]:
                allergy_status = "PENALIZED"
                matched_allergen = allergy_result["matched_allergen"]
            else:
                allergy_status = "SAFE"
                matched_allergen = []

        # 3) 리뷰 작성자 프로필 유사도 가중 aspect score 계산
        product_reviews = reviews_by_product.get(product["product_id"], [])
        weighted_result = compute_weighted_aspect_scores(pet, product_reviews)
        summary = {"weighted_aspect_scores_by_code": weighted_result["weighted_aspect_scores"]}

        # 3.5) 구매 이력 기반 유사도 계산
        candidate_embedding = product.get("embedding")
        if purchase_history_embeddings and candidate_embedding is not None:
            from purchase_history_similarity import compute_purchase_history_similarity
            purchase_sim = compute_purchase_history_similarity(candidate_embedding, purchase_history_embeddings)
        elif USE_DUMMY_DATA and purchase_history_embeddings is None:
            purchase_sim = build_purchase_history_feature(
                user_id=pet["user_id"],
                candidate_product_id=product["product_id"],
                order_items=DUMMY_ORDER_ITEMS,
                orders=DUMMY_ORDERS,
                product_embeddings=DUMMY_PRODUCT_EMBEDDINGS,
            )
        else:
            purchase_sim = 0.0

        # 4) DeepFM 스코어링
        features = build_interaction_features(pet, product, summary, purchase_history_similarity=purchase_sim)
        encoded = encoder.encode(features)
        batch = encoder.collate([encoded])

        import torch
        with torch.no_grad():
            raw_score = model(batch).item()

        # 4.5) 알레르기 판정에 따른 점수 페널티 적용
        if allergy_status == "PENALIZED":
            score = raw_score * ALLERGY_PENALTY_MULTIPLIER
        elif allergy_status == "PENDING":
            score = raw_score * PENDING_PENALTY_MULTIPLIER
        else:
            score = raw_score

        # 5) 추천 사유 생성
        reason_keywords, reason_text = build_reason_from_weighted_scores(weighted_result)
        if allergy_status == "PENALIZED":
            reason_keywords = [f"알러지 성분 포함: {', '.join(matched_allergen)}"] + reason_keywords
            reason_text = f"{', '.join(matched_allergen)} 성분이 포함되어 있어 등록하신 알러지 정보와 맞지 않을 수 있어요. " + reason_text
        elif allergy_status == "PENDING":
            reason_text = "성분 정보 확인 중인 상품이에요. " + reason_text

        results.append({
            "product_id": product["product_id"],
            "product_name": product["product_name"],
            "recommend_type": "RECOMMEND",
            "score": round(score, 4),
            "score_100": score_to_100(score),
            "reason_keywords": reason_keywords,
            "reason_text": reason_text,
            "allergy_status": allergy_status,
            "matched_allergen": matched_allergen,
            "reviewer_similarity_meta": {
                "used_review_count": weighted_result["used_review_count"],
                "total_similarity_weight": weighted_result["total_weight"],
            },
            "purchase_history_similarity": purchase_sim,
        })

    # 6) 점수 기준 정렬 + rank 부여 (페널티 반영된 score 기준이라 자동으로 후순위 배치됨)
    recommend_items = sorted(results, key=lambda x: x["score"], reverse=True)
    for idx, item in enumerate(recommend_items, start=1):
        item["rank"] = idx

    return recommend_items


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