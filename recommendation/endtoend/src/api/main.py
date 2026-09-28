# -*- coding: utf-8 -*-
"""
추천 파트 추론 API (FastAPI).

프론트엔드가 실제로 그리는 화면 기준으로 확정된 3개 라우터:
- GET /recommend/home        : 홈/카테고리 추천 목록
- GET /recommend/substitute  : 상품 상세 "대체 상품" 추천
- GET /recommend/exclusions  : 장바구니 담을 때 역추천(알러지 등) 팝업

+ GET /health : 배포/오케스트레이션용 헬스체크

BE/FE와 확정한 사항 (답변 완료):
- score는 0~100 정수, 음수 없음 (DeepFM sigmoid 출력이 0~1이라 항상 양수).
- match_level은 AI 파트가 내려주지 않음 -- 프론트가 score 기준 자체 문구 처리.
- "근거 세 줄"(성분 함량 분석)은 영양 파트(/products/{id}/nutrition-analysis) 영역이고,
  추천 파트는 알러지 관련 필드(reason_code/severity/matched_terms)만 제공.
- 알러지 체크(장바구니 팝업)는 이 서버가 자체적으로 pet_profile.allergy_codes와
  product_master.allergen_flags를 비교해 처리 (영양 파트 /v1/feeds/analyze와는 별도 경로).

실행 (로컬):
    USE_DUMMY_DATA=true uvicorn src.api.main:app --reload --port 8000
배포:
    USE_DUMMY_DATA=false DATABASE_URL=postgresql://... uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # repo root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # src/
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "aspect"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "features"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "recommend"))

from fastapi import FastAPI, HTTPException, Query
from typing import List, Optional

from src.data_access.pet_repository import get_pet_by_id
from src.data_access.product_repository import list_products, get_products_by_ids, get_product_by_id
from src.recommend.allergy_filter import check_allergy_conflict
from src.recommend.substitute_recommendation import find_substitute_products, build_review_summary_by_product
from pipeline import (
    build_reviews_with_ratings,
    recommend_for_pet,
    score_to_100,
)
from deepfm_model import load_deepfm

app = FastAPI(title="골라주개냥 추천 API", version="1.0.0")

DEEPFM_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "models", "deepfm")

CATEGORY_QUERY_TO_CODE = {
    "food": "FOOD",
    "supplement": "SUPPLEMENT",
    "treat": "TREAT",
}

# 모델/리뷰 캐시. 서버 프로세스가 살아있는 동안 재사용 (매 요청마다 모델을 다시 로드하지 않도록).
_state = {"encoder": None, "model": None, "reviews_by_product": None}


def _get_model_and_reviews():
    if _state["encoder"] is None or _state["model"] is None:
        _state["encoder"], _state["model"] = load_deepfm(DEEPFM_MODEL_DIR)
    if _state["reviews_by_product"] is None:
        _state["reviews_by_product"] = build_reviews_with_ratings()
    return _state["encoder"], _state["model"], _state["reviews_by_product"]


def _get_pet_or_404(pet_id: str) -> dict:
    pet = get_pet_by_id(pet_id)
    if pet is None:
        raise HTTPException(status_code=404, detail=f"pet_id={pet_id}를 찾을 수 없습니다.")
    return pet


def _pet_name(pet: dict) -> str:
    # pet_profile.name 컬럼이 확정되면 그 값을 그대로 쓰고, 없으면 안전한 기본 표시명으로 대체.
    return pet.get("name") or f"{pet.get('breed', '반려동물')}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _item_from_recommendation(rec_item: dict, product: dict) -> dict:
    """recommend_for_pet()/find_substitute_products() 결과 + product_master 조회 결과를 합쳐
    프론트엔드 확정 스펙의 item 하나를 만든다."""
    return {
        "product_id": rec_item["product_id"],
        "rank": rec_item.get("rank"),
        "score": rec_item.get("score_100", score_to_100(rec_item.get("score", 0.0))),
        "reason_text": rec_item.get("reason_text"),
        "product_name": product.get("product_name") if product else rec_item.get("product_name"),
        "thumbnail_url": product.get("thumbnail_url") if product else None,
        "category": product.get("category_code") if product else None,
        "price": product.get("price") if product else None,
        "original_price": product.get("original_price") if product else None,
        "unit_price": product.get("unit_price") if product else None,
        "unit_label": product.get("unit_label") if product else None,
        "rating": product.get("rating") if product else None,
        "review_count": product.get("review_count") if product else None,
        "sales_count": product.get("sales_count") if product else None,
        "status": product.get("status") if product else None,
        "created_at": product.get("created_at") if product else None,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend/home")
def recommend_home(
    pet_id: str = Query(...),
    category: Optional[str] = Query(None, description="food|supplement|treat"),
    sort: str = Query("recommend"),
    size: int = Query(9, ge=1, le=50),
):
    pet = _get_pet_or_404(pet_id)
    category_code = CATEGORY_QUERY_TO_CODE.get((category or "").lower())
    if category and category_code is None:
        raise HTTPException(status_code=400, detail=f"알 수 없는 category={category}")

    encoder, model, reviews_by_product = _get_model_and_reviews()
    products = list_products(category_code=category_code)
    # status=soldout 상품은 홈 추천에서 노출하지 않음 (품절 상품을 추천하는 것은 의미 없음)
    products = [p for p in products if p.get("status") != "soldout"]

    all_items = recommend_for_pet(pet, products, reviews_by_product, encoder, model)
    recommend_only = [item for item in all_items if item["recommend_type"] == "RECOMMEND"]

    # sort=recommend(기본)는 recommend_for_pet이 이미 score 내림차순으로 rank를 매겨 반환.
    # 그 외 정렬 옵션은 현재 프론트 확정 스펙에 없어 기본 정렬을 그대로 사용.
    top_items = recommend_only[:size]

    product_map = {p["product_id"]: p for p in products}
    items = [_item_from_recommendation(rec, product_map.get(rec["product_id"])) for rec in top_items]

    return {
        "pet_id": pet_id,
        "pet_name": _pet_name(pet),
        "generated_at": _now_iso(),
        "items": items,
    }


@app.get("/recommend/substitute")
def recommend_substitute(
    pet_id: str = Query(...),
    product_id: str = Query(...),
    size: int = Query(4, ge=1, le=20),
):
    pet = _get_pet_or_404(pet_id)
    base_product = get_product_by_id(product_id)
    if base_product is None:
        raise HTTPException(status_code=404, detail=f"product_id={product_id}를 찾을 수 없습니다.")

    from src.data_access.order_embedding_repository import get_product_embeddings

    candidates = list_products(category_code=base_product.get("category_code"))
    candidates = [p for p in candidates if p.get("status") != "soldout"]
    candidate_ids = [p["product_id"] for p in candidates] + [product_id]
    embeddings = get_product_embeddings(candidate_ids)

    pet_age_group = None
    try:
        from reviewer_profile_similarity import _calc_age_group

        pet_age_group = _calc_age_group(pet["birth_date"])
    except Exception:
        pass

    review_summary = build_review_summary_by_product(pet, [p["product_id"] for p in candidates])
    substitutes = find_substitute_products(
        base_product=base_product,
        candidate_products=candidates,
        product_embeddings=embeddings,
        pet=pet,
        pet_age_group=pet_age_group,
        review_summary_by_product=review_summary,
        top_k=size,
    )

    product_map = {p["product_id"]: p for p in candidates}
    items = []
    for rec in substitutes:
        item = _item_from_recommendation(rec, product_map.get(rec["product_id"]))
        item["replaced_product_id"] = product_id
        items.append(item)

    return {
        "pet_id": pet_id,
        "pet_name": _pet_name(pet),
        "generated_at": _now_iso(),
        "items": items,
    }


def _severity_and_alternatives(pet: dict, product: dict, matched_allergen: list) -> dict:
    """
    알러지 매칭 결과를 BLOCK/WARN severity로 구분하고, 대체 상품 목록을 함께 구성한다.
    - BLOCK: 등록된 알러지 성분과 직접 겹치는 원료가 있는 경우 (안전 문제, 항상 확정 매칭)
    - 현재 규칙 기반 매칭은 예/아니오만 존재하므로 매칭이 있으면 전부 BLOCK으로 처리하고,
      WARN(주의 수준) 세분화는 영양 파트 '함량 기준' 판단이 추가되면 협의 예정.
    """
    same_category = [
        p for p in list_products(category_code=product.get("category_code"))
        if p["product_id"] != product["product_id"] and p.get("status") != "soldout"
    ]
    from src.data_access.order_embedding_repository import get_product_embeddings

    embeddings = get_product_embeddings([product["product_id"]] + [p["product_id"] for p in same_category])
    alternatives = []
    if product["product_id"] in embeddings:
        subs = find_substitute_products(
            base_product=product,
            candidate_products=same_category,
            product_embeddings=embeddings,
            pet=pet,
            top_k=3,
        )
        alternatives = [
            {"product_id": s["product_id"], "product_name": s["product_name"], "score": s["score_100"]}
            for s in subs
        ]

    return {
        "severity": "BLOCK",
        "alternatives": alternatives,
    }


@app.get("/recommend/exclusions")
def recommend_exclusions(
    pet_id: str = Query(...),
    cart_items: str = Query(..., description="콤마로 구분된 product_id 목록, 예: 1024,1025"),
):
    pet = _get_pet_or_404(pet_id)
    product_ids = [pid.strip() for pid in cart_items.split(",") if pid.strip()]
    products = get_products_by_ids(product_ids)

    excluded = []
    for product_id in product_ids:
        product = products.get(product_id)
        if product is None:
            continue

        allergy_result = check_allergy_conflict(pet.get("allergy_codes", []), product.get("allergen_flags", []))
        if not allergy_result["has_conflict"]:
            continue

        matched = allergy_result["matched_allergen"]
        extra = _severity_and_alternatives(pet, product, matched)
        excluded.append({
            "product_id": product_id,
            "product_name": product.get("product_name"),
            "thumbnail_url": product.get("thumbnail_url"),
            "reason_code": "ALLERGY_MATCH",
            "reason_text": f"{', '.join(matched)} 성분이 포함되어 있어 등록하신 알러지 정보와 맞지 않아요.",
            "severity": extra["severity"],
            "matched_terms": matched,
            "alternatives": extra["alternatives"],
        })

    return {
        "pet_id": pet_id,
        "pet_name": _pet_name(pet),
        "checked_at": _now_iso(),
        "excluded": excluded,
    }