# -*- coding: utf-8 -*-
"""
추천 파트 추론 API (FastAPI).

[중요] DB 접근 주체 변경 (2026-09-30 재확정)
2026-09-29에는 "백엔드가 pet/product/review 데이터를 요청 body에 담아 보내주고
이 서버는 순수 계산만 한다"는 방향으로 기록돼 있었으나, 실제로는 그 중계 레이어가
만들어지지 않았다. 최종적으로는 이 추천 서버가 PostgreSQL(member_db/product_db/
review_db/order_db)에 직접 연결해서 필요한 데이터를 스스로 조회하는 구조로 간다
(2026-09-30 확정). 프론트는 pet_id 등 식별자만 보내면 된다.

라우터 (2026-09-30: /recommend/exclusions 제거):
- POST /recommend/home        : 홈/카테고리 추천 목록
- POST /recommend/substitute  : 상품 상세 "대체 상품" 추천
- GET  /health                : 배포/오케스트레이션용 헬스체크

[2026-09-29] 알러지 충돌 상품을 별도로 "장바구니 팝업 역추천"(/recommend/exclusions)으로
빼서 보여주는 방식은 폐기됐다. 대신 recommend_for_pet() 내부에서 점수 페널티(70%
감점)로 순위만 뒤로 미루는 방식으로 통합됐다 (pipeline.py의 recommend_for_pet
docstring 참고). 그래서 exclusions 전용 엔드포인트는 더 이상 필요 없다.

BE/FE와 확정한 사항:
- score는 0~100 정수, 음수 없음.
- match_level은 AI 파트가 내려주지 않음 -- 프론트가 score 기준 자체 문구 처리.
- 알러지 충돌 상품은 제외되지 않고 allergy_status(SAFE/PENALIZED/PENDING) +
  matched_allergen 필드와 함께 낮은 순위로 내려가서 그대로 응답에 포함된다.

id 타입 주의 (프론트 연동 시 중요): pet_id/product_id는 실제 DB에서 bigint라
정수(int)로 주고받는다. (예전 버전은 str이었음.)

실행 (로컬):
    kubectl --context petflow-dev -n database port-forward svc/petflow-db-rw 15432:5432
    USE_DUMMY_DATA=false uvicorn src.api.main:app --reload --port 8000

API 문서: 서버 실행 후 http://localhost:8000/docs (Swagger UI, 프론트에 그대로 공유 가능)
"""

import os
import sys
from datetime import datetime, timezone
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # repo root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # src/
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "aspect"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "features"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "recommend"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.recommend.substitute_recommendation import find_substitute_products, build_review_summary_by_product
from pipeline import build_reviews_with_ratings, recommend_for_pet, score_to_100
from deepfm_model import load_deepfm

from src.data_access.pet_repository import get_pet_by_id
from src.data_access.product_repository import list_products, get_product_by_id
from src.data_access.reviews_repository import load_reviews_with_reviewer_pet
from src.data_access.order_embedding_repository import get_purchased_product_ids_for_user, get_product_embeddings

app = FastAPI(title="골라주개냥 추천 API", version="3.0.0")

DEEPFM_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "models", "deepfm")

CATEGORY_QUERY_TO_CODE = {
    "food": "FOOD",
    "supplement": "SUPPLEMENT",
    "treat": "TREAT",
}

# 모델만 캐시. products/reviews는 매 요청마다 DB에서 새로 조회한다
# (TODO: 리뷰 57000+건을 매 요청 전체 로드하는 건 비효율적 -- 캐싱/증분 로드는 추후 개선).
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
# 요청 body 스키마 (2026-09-30: pet_id/product_id 식별자만 받도록 축소)
# ----------------------------------------------------------------------

class RecommendHomeRequest(BaseModel):
    pet_id: int
    category: Optional[str] = Field(None, description="food|supplement|treat")
    sort: str = "recommend"
    size: int = Field(9, ge=1, le=50)


class RecommendSubstituteRequest(BaseModel):
    pet_id: int
    base_product_id: int
    size: int = Field(4, ge=1, le=20)


# ----------------------------------------------------------------------
# 라우터
# ----------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend/home")
def recommend_home(req: RecommendHomeRequest):
    pet = get_pet_by_id(req.pet_id)
    if pet is None:
        raise HTTPException(status_code=404, detail=f"pet_id={req.pet_id}를 찾을 수 없습니다.")

    category_code = CATEGORY_QUERY_TO_CODE.get((req.category or "").lower())
    if req.category and category_code is None:
        raise HTTPException(status_code=400, detail=f"알 수 없는 category={req.category}")

    encoder, model = _get_model()

    raw_reviews = load_reviews_with_reviewer_pet()
    reviews_by_product = build_reviews_with_ratings(raw_reviews)

    products = list_products(category_code=category_code)
    products = [p for p in products if p.get("status") == "ON_SALE"]

    purchased_ids = get_purchased_product_ids_for_user(pet["user_id"])
    purchase_history_embeddings = (
        list(get_product_embeddings(purchased_ids).values()) if purchased_ids else []
    )

    all_items = recommend_for_pet(
        pet, products, reviews_by_product, encoder, model,
        purchase_history_embeddings=purchase_history_embeddings,
    )
    top_items = all_items[: req.size]  # 이미 score 기준 정렬 + 페널티 반영됨 (알러지 충돌 상품도 포함, 순위만 밀림)

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
    pet = get_pet_by_id(req.pet_id)
    if pet is None:
        raise HTTPException(status_code=404, detail=f"pet_id={req.pet_id}를 찾을 수 없습니다.")

    base_product = get_product_by_id(req.base_product_id)
    if base_product is None:
        raise HTTPException(status_code=404, detail=f"product_id={req.base_product_id}를 찾을 수 없습니다.")

    candidates = [
        p for p in list_products(category_code=base_product["category_code"])
        if p["product_id"] != base_product["product_id"] and p.get("status") == "ON_SALE"
    ]
    candidate_ids = [p["product_id"] for p in candidates]

    embeddings = get_product_embeddings([base_product["product_id"]] + candidate_ids)

    pet_age_group = None
    try:
        from reviewer_profile_similarity import _calc_age_group
        pet_age_group = _calc_age_group(pet["birth_date"])
    except Exception:
        pass

    raw_reviews = load_reviews_with_reviewer_pet()
    review_summary = build_review_summary_by_product(
        pet, candidate_ids, reviews=raw_reviews,
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