# -*- coding: utf-8 -*-
"""
추천 파트 추론 API (FastAPI).

[중요] DB 접근 주체 변경 (2026-09-29 확정)
이 서버는 더 이상 PostgreSQL에 직접 연결하지 않는다. pet/product/review/구매이력
데이터는 전부 백엔드가 요청 body에 담아 보내주고, 이 서버는 그 데이터를 받아
DeepFM 스코어링 + 추천 로직만 수행하는 순수 계산 서버로 동작한다.
(data_access/*.py는 로컬 더미 데이터 테스트/개발용으로만 남겨둔다.)

프론트엔드가 실제로 그리는 화면 기준으로 확정된 3개 라우터 (GET -> POST로 변경,
쿼리 파라미터 대신 요청 body에 전체 데이터를 담아 받음):
- POST /recommend/home        : 홈/카테고리 추천 목록
- POST /recommend/substitute  : 상품 상세 "대체 상품" 추천
- POST /recommend/exclusions  : 장바구니 담을 때 역추천(알러지 등) 팝업

+ GET /health : 배포/오케스트레이션용 헬스체크

BE/FE와 확정한 사항:
- score는 0~100 정수, 음수 없음 (DeepFM sigmoid 출력이 0~1이라 항상 양수).
- match_level은 AI 파트가 내려주지 않음 -- 프론트가 score 기준 자체 문구 처리.
- "근거 세 줄"(성분 함량 분석)은 영양 파트(/products/{id}/nutrition-analysis) 영역이고,
  추천 파트는 알러지 관련 필드(reason_code/severity/matched_terms)만 제공.
- 알러지 체크(장바구니 팝업)는 이 서버가 자체적으로 pet.allergy_codes와
  product.allergen_flags를 비교해 처리 (영양 파트 /v1/feeds/analyze와는 별도 경로).

실행 (로컬):
    uvicorn src.api.main:app --reload --port 8000
    (DB 관련 환경변수 불필요. USE_DUMMY_DATA도 이 서버 자체 동작에는 더 이상 영향 없음
     -- 요청 body에 데이터가 없으면 그냥 빈 결과가 나올 뿐이다.)
"""

import os
import sys
from datetime import datetime, timezone
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # repo root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # src/
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "aspect"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "features"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "recommend"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.recommend.allergy_filter import check_allergy_conflict
from src.recommend.substitute_recommendation import find_substitute_products, build_review_summary_by_product
from pipeline import build_reviews_with_ratings, recommend_for_pet, score_to_100
from deepfm_model import load_deepfm

app = FastAPI(title="골라주개냥 추천 API", version="2.0.0")

DEEPFM_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "models", "deepfm")

CATEGORY_QUERY_TO_CODE = {
    "food": "FOOD",
    "supplement": "SUPPLEMENT",
    "treat": "TREAT",
}

# 모델만 캐시 (리뷰/DB 캐시는 더 이상 필요 없음 -- 매 요청 body로 받으므로).
_state = {"encoder": None, "model": None}


def _get_model():
    if _state["encoder"] is None or _state["model"] is None:
        _state["encoder"], _state["model"] = load_deepfm(DEEPFM_MODEL_DIR)
    return _state["encoder"], _state["model"]


def _pet_name(pet: dict) -> str:
    return pet.get("name") or pet.get("breed") or "반려동물"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _item_from_recommendation(rec_item: dict, product: dict) -> dict:
    return {
        "product_id": rec_item["product_id"],
        "rank": rec_item.get("rank"),
        "score": rec_item.get("score_100", score_to_100(rec_item.get("score", 0.0))),
        "reason_text": rec_item.get("reason_text"),
        "allergy_status": rec_item.get("allergy_status", "SAFE"),
        "matched_allergen": rec_item.get("matched_allergen", []),
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


# ----------------------------------------------------------------------
# 요청 body 스키마
# ----------------------------------------------------------------------

class ReviewerPetIn(BaseModel):
    species: str
    birth_date: str
    weight: float
    allergy_codes: List[str] = []


class ReviewIn(BaseModel):
    review_id: str
    product_id: str
    rating: Optional[int] = None
    palatability_rating: Optional[int] = None
    digestion_rating: Optional[int] = None
    skin_coat_rating: Optional[int] = None
    vitality_weight_rating: Optional[int] = None
    allergic_reaction_rating: Optional[int] = None
    reviewer_pet: ReviewerPetIn


class PetIn(BaseModel):
    pet_id: str
    user_id: str
    species: str
    breed: Optional[str] = None
    name: Optional[str] = None
    birth_date: str
    sex: Optional[str] = None
    neutered: Optional[bool] = None
    weight: float
    bcs: Optional[str] = None
    allergy_codes: List[str] = []
    concerns: List[str] = []


class ProductIn(BaseModel):
    product_id: str
    product_name: str
    brand_name: Optional[str] = None
    category_code: str
    subcategory_code: Optional[str] = None
    ingredients: List[str] = []
    allergen_flags: List[str] = None    # None=미분석, []=분석완료·성분없음
    target_species: List[str] = []
    target_breed_size: Optional[str] = None
    target_age_group: Optional[str] = None
    price: Optional[float] = None
    original_price: Optional[float] = None
    unit_price: Optional[float] = None
    unit_label: Optional[str] = None
    thumbnail_url: Optional[str] = None
    rating: Optional[float] = None
    review_count: Optional[int] = 0
    sales_count: Optional[int] = 0
    status: Optional[str] = None
    created_at: Optional[str] = None
    embedding: Optional[List[float]] = Field(
        default=None,
        description="product_embeddings(pgvector) 벡터. 없으면 유사도 계산에서 자동 제외됨.",
    )


class RecommendHomeRequest(BaseModel):
    pet: PetIn
    products: List[ProductIn]
    reviews: List[ReviewIn] = []
    purchase_history_embeddings: List[List[float]] = Field(
        default=[],
        description="이 pet(사용자)의 과거 구매 상품 임베딩 목록. 구매 이력 없으면 빈 리스트.",
    )
    category: Optional[str] = Field(None, description="food|supplement|treat")
    sort: str = "recommend"
    size: int = Field(9, ge=1, le=50)


class RecommendSubstituteRequest(BaseModel):
    pet: PetIn
    base_product: ProductIn
    candidate_products: List[ProductIn]
    reviews: List[ReviewIn] = []
    size: int = Field(4, ge=1, le=20)


class RecommendExclusionsRequest(BaseModel):
    pet: PetIn
    cart_items: List[ProductIn]
    alternative_candidates: List[ProductIn] = Field(
        default=[],
        description="장바구니 상품이 알러지로 제외될 때 대체품을 뽑을 후보 풀 (같은 카테고리 상품들).",
    )
    reviews: List[ReviewIn] = []
    size_per_item: int = Field(3, ge=1, le=10)


def _reviews_to_dicts(reviews: List[ReviewIn]) -> list:
    return [r.model_dump() for r in reviews]


def _products_to_dicts(products: List[ProductIn]) -> list:
    return [p.model_dump() for p in products]


# ----------------------------------------------------------------------
# 라우터
# ----------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend/home")
def recommend_home(req: RecommendHomeRequest):
    pet = req.pet.model_dump()
    category_code = CATEGORY_QUERY_TO_CODE.get((req.category or "").lower())
    if req.category and category_code is None:
        raise HTTPException(status_code=400, detail=f"알 수 없는 category={req.category}")

    encoder, model = _get_model()
    reviews_by_product = build_reviews_with_ratings(_reviews_to_dicts(req.reviews))

    products = _products_to_dicts(req.products)
    if category_code:
        products = [p for p in products if p.get("category_code") == category_code]
    products = [p for p in products if p.get("status") != "soldout"]

    all_items = recommend_for_pet(
        pet, products, reviews_by_product, encoder, model,
        purchase_history_embeddings=req.purchase_history_embeddings,
    )
    top_items = all_items[: req.size]   # 이미 score 기준 정렬 + 페널티 반영됨

    product_map = {p["product_id"]: p for p in products}
    items = [_item_from_recommendation(rec, product_map.get(rec["product_id"])) for rec in top_items]

    return {
        "pet_id": pet["pet_id"],
        "pet_name": _pet_name(pet),
        "generated_at": _now_iso(),
        "items": items,
    }


@app.post("/recommend/substitute")
def recommend_substitute(req: RecommendSubstituteRequest):
    pet = req.pet.model_dump()
    base_product = req.base_product.model_dump()
    candidates = _products_to_dicts(req.candidate_products)
    candidates = [p for p in candidates if p.get("status") != "soldout"]

    embeddings = {}
    if base_product.get("embedding") is not None:
        embeddings[base_product["product_id"]] = base_product["embedding"]
    for p in candidates:
        if p.get("embedding") is not None:
            embeddings[p["product_id"]] = p["embedding"]

    pet_age_group = None
    try:
        from reviewer_profile_similarity import _calc_age_group
        pet_age_group = _calc_age_group(pet["birth_date"])
    except Exception:
        pass

    reviews = _reviews_to_dicts(req.reviews)
    review_summary = build_review_summary_by_product(
        pet, [p["product_id"] for p in candidates], reviews=reviews,
    )
    substitutes = find_substitute_products(
        base_product=base_product,
        candidate_products=candidates,
        product_embeddings=embeddings,
        pet=pet,
        pet_age_group=pet_age_group,
        review_summary_by_product=review_summary,
        top_k=req.size,
    )

    product_map = {p["product_id"]: p for p in candidates}
    items = []
    for rec in substitutes:
        item = _item_from_recommendation(rec, product_map.get(rec["product_id"]))
        item["replaced_product_id"] = base_product["product_id"]
        items.append(item)

    return {
        "pet_id": pet["pet_id"],
        "pet_name": _pet_name(pet),
        "generated_at": _now_iso(),
        "items": items,
    }


def _severity_and_alternatives(
    pet: dict, product: dict, alternative_candidates: list, reviews: list, size_per_item: int,
) -> dict:
    """
    알러지 매칭 결과를 BLOCK/WARN severity로 구분하고, 대체 상품 목록을 함께 구성한다.
    - BLOCK: 등록된 알러지 성분과 직접 겹치는 원료가 있는 경우 (안전 문제, 항상 확정 매칭)
    - 현재 규칙 기반 매칭은 예/아니오만 존재하므로 매칭이 있으면 전부 BLOCK으로 처리하고,
      WARN(주의 수준) 세분화는 영양 파트 '함량 기준' 판단이 추가되면 협의 예정.
    """
    same_category = [
        p for p in alternative_candidates
        if p["product_id"] != product["product_id"]
        and p.get("category_code") == product.get("category_code")
        and p.get("status") != "soldout"
    ]

    embeddings = {}
    if product.get("embedding") is not None:
        embeddings[product["product_id"]] = product["embedding"]
    for p in same_category:
        if p.get("embedding") is not None:
            embeddings[p["product_id"]] = p["embedding"]

    alternatives = []
    if product["product_id"] in embeddings:
        review_summary = build_review_summary_by_product(
            pet, [p["product_id"] for p in same_category], reviews=reviews,
        )
        subs = find_substitute_products(
            base_product=product,
            candidate_products=same_category,
            product_embeddings=embeddings,
            pet=pet,
            review_summary_by_product=review_summary,
            top_k=size_per_item,
        )
        alternatives = [
            {"product_id": s["product_id"], "product_name": s["product_name"], "score": s["score_100"]}
            for s in subs
        ]

    return {"severity": "BLOCK", "alternatives": alternatives}


@app.post("/recommend/exclusions")
def recommend_exclusions(req: RecommendExclusionsRequest):
    pet = req.pet.model_dump()
    cart_items = _products_to_dicts(req.cart_items)
    alternative_candidates = _products_to_dicts(req.alternative_candidates)
    reviews = _reviews_to_dicts(req.reviews)

    excluded = []
    for product in cart_items:
        allergy_result = check_allergy_conflict(pet.get("allergy_codes", []), product.get("allergen_flags", []))
        if not allergy_result["has_conflict"]:
            continue

        matched = allergy_result["matched_allergen"]
        extra = _severity_and_alternatives(pet, product, alternative_candidates, reviews, req.size_per_item)
        excluded.append({
            "product_id": product["product_id"],
            "product_name": product.get("product_name"),
            "thumbnail_url": product.get("thumbnail_url"),
            "reason_code": "ALLERGY_MATCH",
            "reason_text": f"{', '.join(matched)} 성분이 포함되어 있어 등록하신 알러지 정보와 맞지 않아요.",
            "severity": extra["severity"],
            "matched_terms": matched,
            "alternatives": extra["alternatives"],
        })

    return {
        "pet_id": pet["pet_id"],
        "pet_name": _pet_name(pet),
        "checked_at": _now_iso(),
        "excluded": excluded,
    }