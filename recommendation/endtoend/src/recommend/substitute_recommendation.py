# -*- coding: utf-8 -*-
"""
FR-AI-2-02 대체상품 추천.

상품 상세페이지에서 성분·용도가 유사한 상품을 추천하는 기능.
기준 상품 하나를 입력받아, 같은 카테고리 내에서 임베딩+원료 유사도로 후보를 추리고
안전 필터(알레르기·연령) 적용 후, 리뷰 매칭 캐스케이드로 최종 재정렬한다.

파이프라인:
① 기준 상품과 동일 category_code 내에서 코사인 유사도 상위 N=50 후보 추출
② 유사도 스코어 = 0.7*cosine_similarity + 0.3*jaccard(ingredients)
③ 안전 필터(알레르기·연령) 적용해 후보 제외
④ 리뷰 매칭 캐스케이드로 산출한 aspect score로 최종 재정렬
⑤ 이유 문장 생성 (FR-AI-2-01의 build_reason_from_weighted_scores 재사용)
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "aspect"))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # src.data_access import용
from purchase_history_similarity import cosine_similarity
from allergy_filter import check_allergy_conflict
from reviewer_profile_similarity import compute_weighted_aspect_scores

TOP_N_CANDIDATES = 50
SIMILARITY_WEIGHTS = {"cosine": 0.7, "jaccard": 0.3}


def jaccard_similarity(set_a: list, set_b: list) -> float:
    """원료(ingredients) 리스트 간 자카드 유사도 (교집합/합집합)."""
    a, b = set(set_a or []), set(set_b or [])
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def compute_candidate_score(
    base_embedding: list, base_ingredients: list,
    candidate_embedding: list, candidate_ingredients: list,
) -> float:
    """0.7*cosine + 0.3*jaccard 결합 스코어."""
    cosine = cosine_similarity(base_embedding, candidate_embedding)
    jaccard = jaccard_similarity(base_ingredients, candidate_ingredients)
    return round(
        SIMILARITY_WEIGHTS["cosine"] * cosine + SIMILARITY_WEIGHTS["jaccard"] * jaccard,
        4,
    )


def _check_age_conflict(pet_age_group: str, product_target_age_group) -> bool:
    """
    연령 안전 필터. 상품이 특정 생애주기를 명시적으로 타겟하는데
    반려동물의 생애주기와 아예 다르면(GROWTH 전용 상품을 SENIOR에게 등) 안전하지 않다고 본다.
    타겟이 지정 안 된 상품(전 연령)은 항상 통과.
    """
    if not product_target_age_group:
        return False  # 충돌 없음
    return pet_age_group != product_target_age_group


def build_review_summary_by_product(pet: dict, product_ids: list) -> dict:
    """
    find_substitute_products()의 review_summary_by_product 파라미터를 채우는 헬퍼.

    pipeline.py의 build_reviews_with_ratings() + compute_weighted_aspect_scores()와
    동일한 방식으로, 대체상품 후보 목록에 대해 "이 pet과 유사한 프로필의 리뷰어들"
    기준 aspect score를 미리 계산해둔다.

    pet이 None이면(비로그인 등 안전 필터 생략 상황과 동일 조건) 빈 딕셔너리를 반환하며,
    이 경우 find_substitute_products()는 리뷰 보정 없이 순수 유사도 스코어로만 정렬한다.
    """
    if pet is None:
        return {}

    from rating_converter import ASPECT_FIELD_TO_CODE, convert_rating_to_score
    from src.data_access.reviews_repository import load_reviews_with_reviewer_pet
    from collections import defaultdict

    reviews_by_product = defaultdict(list)
    for review in load_reviews_with_reviewer_pet():
        if review["product_id"] not in product_ids:
            continue
        ratings = {}
        for field_name, aspect_code in ASPECT_FIELD_TO_CODE.items():
            raw_rating = review.get(field_name)
            ratings[aspect_code] = convert_rating_to_score(raw_rating) if raw_rating is not None else None
        reviews_by_product[review["product_id"]].append({
            "reviewer_pet": review["reviewer_pet"],
            "ratings": ratings,
        })

    summary = {}
    for product_id in product_ids:
        product_reviews = reviews_by_product.get(product_id, [])
        weighted_result = compute_weighted_aspect_scores(pet, product_reviews)
        summary[product_id] = {"weighted_aspect_scores": weighted_result["weighted_aspect_scores"]}

    return summary


def find_substitute_products(
    base_product: dict,
    candidate_products: list,
    product_embeddings: dict,
    pet: dict = None,
    pet_age_group: str = None,
    review_summary_by_product: dict = None,
    top_k: int = 10,
) -> list:
    """
    base_product: 기준 상품 (상품 상세페이지에서 보고 있는 상품)
    candidate_products: 같은 category_code 후보군 전체 (아직 필터링 전)
    product_embeddings: {product_id: embedding_vector}
    pet: 안전 필터에 사용할 반려동물 프로필 (없으면 알레르기/연령 필터 생략 -- 비로그인 등)
    pet_age_group: 미리 계산된 pet의 age_group (GROWTH/ADULT/SENIOR)
    review_summary_by_product: {product_id: {"weighted_aspect_scores": {...}}} 형태
                                (reviewer_profile_similarity.compute_weighted_aspect_scores 결과)
    top_k: 최종 반환할 대체상품 개수

    반환: recommendation_items 스키마와 유사한 형태의 리스트 (RECOMMEND만, rank 부여됨)
    """
    base_embedding = product_embeddings.get(base_product["product_id"])
    if base_embedding is None:
        return []  # 기준 상품 임베딩이 없으면 추천 자체가 불가능

    # ① 동일 category_code 내에서 후보 추출 (자기 자신 제외)
    same_category = [
        p for p in candidate_products
        if p["category_code"] == base_product["category_code"]
        and p["product_id"] != base_product["product_id"]
    ]

    # ② 유사도 스코어 계산 후 상위 N=50 추출
    scored_candidates = []
    for product in same_category:
        candidate_embedding = product_embeddings.get(product["product_id"])
        if candidate_embedding is None:
            continue
        score = compute_candidate_score(
            base_embedding, base_product.get("ingredients", []),
            candidate_embedding, product.get("ingredients", []),
        )
        scored_candidates.append((product, score))

    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = scored_candidates[:TOP_N_CANDIDATES]

    # ③ 안전 필터 (알레르기·연령) -- pet 정보가 있을 때만 적용
    safe_candidates = []
    for product, score in top_candidates:
        if pet is not None:
            allergy_result = check_allergy_conflict(pet.get("allergy_codes", []), product.get("allergen_flags", []))
            if allergy_result["has_conflict"]:
                continue
            if pet_age_group and _check_age_conflict(pet_age_group, product.get("target_age_group")):
                continue
        safe_candidates.append((product, score))

    # ④ 리뷰 매칭 캐스케이드로 재정렬 (aspect score 평균을 유사도 스코어에 보정치로 반영)
    review_summary_by_product = review_summary_by_product or {}
    final_scored = []
    for product, similarity_score in safe_candidates:
        summary = review_summary_by_product.get(product["product_id"], {})
        aspect_scores = summary.get("weighted_aspect_scores", {})
        review_adjustment = (sum(aspect_scores.values()) / len(aspect_scores)) if aspect_scores else 0.0
        # 유사도 스코어(0~1)에 리뷰 보정치(-1~1을 -0.1~0.1 범위로 축소)를 더해 미세 조정
        final_score = round(similarity_score + review_adjustment * 0.1, 4)
        final_scored.append((product, final_score, aspect_scores))

    final_scored.sort(key=lambda x: x[1], reverse=True)
    top_results = final_scored[:top_k]

    # ⑤ 결과 조립
    results = []
    for rank, (product, score, aspect_scores) in enumerate(top_results, start=1):
        positive_aspects = sorted(
            [(code, s) for code, s in aspect_scores.items() if s > 0],
            key=lambda x: x[1], reverse=True,
        )
        reason_keywords = [code for code, _ in positive_aspects[:3]]
        if reason_keywords:
            reason_text = f"{base_product['product_name']}과(와) 성분·용도가 유사하고, " + ", ".join(reason_keywords) + " 평가가 좋은 상품입니다."
        else:
            reason_text = f"{base_product['product_name']}과(와) 성분·용도가 유사한 상품입니다."

        results.append({
            "product_id": product["product_id"],
            "product_name": product["product_name"],
            "recommend_type": "RECOMMEND",
            "score": score,
            "rank": rank,
            "reason_keywords": reason_keywords,
            "reason_text": reason_text,
        })

    return results