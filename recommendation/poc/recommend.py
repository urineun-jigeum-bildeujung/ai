# -*- coding: utf-8 -*-
"""
룰베이스 추천 스코어링 (DeepFM 자리의 임시 대체).
POC 목적: pet_profile + product_master + review_features(집계) 를 조합해
recommendation_items 스키마 형태(recommend_type, score, reason_keywords, reason_text, matched_allergen)
로 결과가 끊기지 않고 나오는지 검증하는 것.

시나리오 반영:
- 알러지 성분 포함 상품 -> EXCLUDE (역추천) + 사유
- 그 외 -> 종/생애주기/체구 적합도 + 긍정 리뷰 신호로 스코어링 -> RECOMMEND
"""

from datetime import date
from allergy_filter import check_allergy_conflict


def calc_age_group(birth_date_str: str) -> str:
    """생년월일 기준 대략적인 생애주기 구분 (POC 단순화 버전)."""
    birth = date.fromisoformat(birth_date_str)
    today = date.today()
    months = (today.year - birth.year) * 12 + (today.month - birth.month)
    if months < 12:
        return "GROWTH"
    elif months < 84:
        return "ADULT"
    else:
        return "SENIOR"


def calc_breed_size(weight: float) -> str:
    """체중 기준 대략적인 체구 구분 (POC 단순화 버전)."""
    if weight < 10:
        return "SMALL"
    elif weight < 25:
        return "MEDIUM"
    else:
        return "LARGE"


def recommend_products(pet: dict, products: list, product_review_summary: dict) -> list:
    """
    pet: pet_profile 딕셔너리
    products: product_master 리스트
    product_review_summary: {product_id: {"avg_sentiment_score": float,
                                            "positive_ratio": float,
                                            "positive_tags": [...],
                                            "negative_tags": [...]}}

    반환: recommendation_items 스키마 형태의 딕셔너리 리스트 (rank 포함)
    """
    pet_age_group = calc_age_group(pet["birth_date"])
    pet_breed_size = calc_breed_size(pet["weight"])

    results = []

    for product in products:
        # 1) 종(species) 안 맞으면 애초에 추천 후보에서 제외 (추천/역추천 대상 자체가 아님)
        if pet["species"] not in product["target_species"]:
            continue

        # 2) 알러지 매칭 -> 역추천
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

        # 3) 스코어링 (룰베이스)
        summary = product_review_summary.get(product["product_id"], {
            "avg_sentiment_score": 0.5,
            "positive_ratio": 0.5,
            "positive_tags": [],
            "negative_tags": [],
        })

        score = 0.0
        reason_keywords = []

        # 3-1) 리뷰 감성 기반 점수
        score += summary["avg_sentiment_score"] * 0.4
        score += summary["positive_ratio"] * 0.3

        # 3-2) 생애주기 적합도
        if product.get("target_age_group") and product["target_age_group"] == pet_age_group:
            score += 0.15
            reason_keywords.append(f"{pet_age_group} 생애주기에 적합")

        # 3-3) 체구 적합도
        if product.get("target_breed_size") and product["target_breed_size"] == pet_breed_size:
            score += 0.15
            reason_keywords.append(f"{pet_breed_size} 체구에 적합")

        # 3-4) 긍정 aspect 태그 -> 추천 사유에 반영
        if summary["positive_tags"]:
            reason_keywords.extend(summary["positive_tags"])

        score = round(min(score, 1.0), 2)

        # reason_text 생성 (1차: 키워드 템플릿, 2차 LLM 고도화는 이후 단계)
        if reason_keywords:
            reason_text = "리뷰에서 " + ", ".join(reason_keywords) + " 등의 반응이 있어 추천합니다."
        else:
            reason_text = "등록하신 반려동물 정보를 기준으로 추천합니다."

        results.append({
            "product_id": product["product_id"],
            "product_name": product["product_name"],
            "recommend_type": "RECOMMEND",
            "score": score,
            "reason_keywords": reason_keywords,
            "reason_text": reason_text,
            "matched_allergen": [],
        })

    # 4) 점수 기준 정렬 + rank 부여 (RECOMMEND만 랭킹, EXCLUDE는 랭킹 없음)
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