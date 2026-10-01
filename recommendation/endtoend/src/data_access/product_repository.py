# -*- coding: utf-8 -*-
"""
product_master(실제 테이블명: products) 조회 데이터 접근 계층.

DB팀 확인: 상품 데이터는 product_db의 "products" 테이블에 있음
(테이블명이 product_master가 아니라 products).
product_embeddings는 다른 파트가 아직 만드는 중이라, order_embedding_repository.py에서
그 테이블이 없어도 에러 없이 빈 결과로 처리한다.

추천 로직에 필요한 필드(category_code, ingredients, allergen_flags, target_species,
target_breed_size, target_age_group, price)뿐 아니라, 프론트엔드가 /recommend/* 응답에
그대로 함께 내려달라고 요청한 표시용 필드(product_name, thumbnail_url, original_price,
unit_price, unit_label, rating, review_count, sales_count, status, created_at)도
같은 조회에서 채운다.

USE_DUMMY_DATA 환경변수로 더미 데이터 / 실제 DB 전환 (reviews_repository.py와 동일 패턴).

실제 스키마 주의 (DB팀 ERD 확인 완료):
- products에는 deleted_at 컬럼이 없다. soft delete는 is_active로 한다.
- brand_name은 컬럼이 아니라 brand_id로 brands 테이블 조인.
- ingredients/allergen_flags/target_species는 products 컬럼이 아니라 각각
  product_ingredients / product_allergens / product_target_species 별도 테이블.
- unit_price/unit_label은 실제 스키마에 없어서 normalized_quantity_value/unit로 계산한다.
- rating은 실제 컬럼명이 avg_rating.
- [알려진 한계] product_allergens에 해당 상품 행이 없으면 무조건 빈 리스트로 나온다.
  "성분 미분석(PENDING)"과 "성분 없음 확인"을 DB에서 구분할 방법이 없어서,
  현재는 "조인 결과 없음 = 알러지 성분 없음(SAFE)"으로 단순화해서 처리하기로 결정함
  (2026-09-30). 추후 백엔드팀이 분석 상태 컬럼을 추가하면 재검토.
"""

import os
import sys

USE_DUMMY_DATA = os.environ.get("USE_DUMMY_DATA", "true").lower() != "false"

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "data", "dummy"))
sys.path.append(os.path.dirname(__file__))

from db import get_connection

PRODUCT_DB_ENV = "PRODUCT_DATABASE_URL"

# 더미 데이터에는 아직 표시용 필드(썸네일/가격 등)가 없으므로, 더미 모드에서는
# product_id를 시드로 결정적인 더미 값을 만들어 응답 스펙을 미리 검증할 수 있게 한다.
_DUMMY_DISPLAY_DEFAULTS = {
    "thumbnail_url": "https://cdn.example.com/products/{product_id}.jpg",
    "original_price": None,
    "unit_price": None,
    "unit_label": "1개",
    "rating": 4.5,
    "review_count": 0,
    "sales_count": 0,
    "status": "normal",
    "created_at": "2025-01-01T00:00:00+09:00",
}


def _apply_dummy_display_fields(product: dict) -> dict:
    """더미 product_master(dummy_data.PRODUCTS) 원소에 표시용 필드를 채워 넣은 사본을 반환."""
    enriched = dict(product)
    enriched.setdefault("original_price", None)
    enriched.setdefault("unit_price", None)
    enriched["thumbnail_url"] = _DUMMY_DISPLAY_DEFAULTS["thumbnail_url"].format(product_id=product["product_id"])
    enriched["unit_label"] = _DUMMY_DISPLAY_DEFAULTS["unit_label"]
    enriched["rating"] = _DUMMY_DISPLAY_DEFAULTS["rating"]
    enriched["review_count"] = _DUMMY_DISPLAY_DEFAULTS["review_count"]
    enriched["sales_count"] = _DUMMY_DISPLAY_DEFAULTS["sales_count"]
    enriched["status"] = _DUMMY_DISPLAY_DEFAULTS["status"]
    enriched["created_at"] = _DUMMY_DISPLAY_DEFAULTS["created_at"]
    return enriched


_PRODUCT_QUERY = """
    SELECT
        p.id,
        p.product_name,
        br.name,
        p.category_code,
        p.subcategory_code,
        p.price,
        p.original_price,
        p.thumbnail_url,
        p.avg_rating,
        p.review_count,
        p.sales_count,
        p.status,
        p.created_at,
        p.normalized_quantity_value,
        p.normalized_quantity_unit,
        COALESCE(
            (SELECT array_agg(pi.ingredient_code ORDER BY pi.sort_order)
             FROM product_ingredients pi WHERE pi.product_id = p.id),
            '{}'
        ) AS ingredients,
        COALESCE(
            (SELECT array_agg(pa.allergen_code)
             FROM product_allergens pa WHERE pa.product_id = p.id),
            '{}'
        ) AS allergen_flags,
        COALESCE(
            (SELECT array_agg(pts.species)
             FROM product_target_species pts WHERE pts.product_id = p.id),
            '{}'
        ) AS target_species,
        p.target_breed_size,
        p.target_age_group
    FROM products p
    LEFT JOIN brands br ON br.id = p.brand_id
"""


def _row_to_product(row) -> dict:
    (product_id, product_name, brand_name, category_code, subcategory_code,
     price, original_price, thumbnail_url, rating, review_count, sales_count,
     status, created_at, norm_qty_value, norm_qty_unit,
     ingredients, allergen_flags, target_species,
     target_breed_size, target_age_group) = row

    norm_qty_value_f = float(norm_qty_value) if norm_qty_value is not None else None
    unit_price = (
        float(price) / norm_qty_value_f
        if price is not None and norm_qty_value_f else None
    )
    unit_label = (
        f"{norm_qty_value_f:g}{norm_qty_unit}"
        if norm_qty_value_f is not None and norm_qty_unit else None
    )

    return {
        "product_id": product_id,
        "product_name": product_name,
        "brand_name": brand_name,
        "category_code": category_code,
        "subcategory_code": subcategory_code,
        "ingredients": list(ingredients) if ingredients else [],
        # product_allergens에 행이 없으면 무조건 빈 리스트 (PENDING 구분 없음 - 위 docstring 참고)
        "allergen_flags": list(allergen_flags) if allergen_flags else [],
        "target_species": list(target_species) if target_species else [],
        "target_breed_size": target_breed_size,
        "target_age_group": target_age_group,
        "price": float(price) if price is not None else None,
        "original_price": float(original_price) if original_price is not None else None,
        "unit_price": unit_price,
        "unit_label": unit_label,
        "thumbnail_url": thumbnail_url,
        "rating": float(rating) if rating is not None else None,
        "review_count": review_count or 0,
        "sales_count": sales_count or 0,
        "status": status,
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else created_at,
    }


def _fetch_products_from_db(category_code: str = None, product_ids: list = None) -> list:
    conn = get_connection(PRODUCT_DB_ENV)
    try:
        with conn.cursor() as cur:
            query = _PRODUCT_QUERY + " WHERE p.is_active = true"
            params = []
            if category_code:
                query += " AND p.category_code = %s"
                params.append(category_code)
            if product_ids:
                query += " AND p.id = ANY(%s)"
                params.append(list(product_ids))
            cur.execute(query, tuple(params))
            return [_row_to_product(row) for row in cur.fetchall()]
    finally:
        conn.close()


def list_products(category_code: str = None) -> list:
    """
    추천 후보군 조회 (pipeline.py의 recommend_for_pet에 넘길 전체/카테고리별 상품 목록).
    """
    if USE_DUMMY_DATA:
        from dummy_data import PRODUCTS
        products = PRODUCTS
        if category_code:
            products = [p for p in products if p["category_code"] == category_code]
        return [_apply_dummy_display_fields(p) for p in products]
    return _fetch_products_from_db(category_code=category_code)


def get_products_by_ids(product_ids: list) -> dict:
    """
    {product_id: product_dict} 형태로 반환. /recommend/exclusions 처럼
    장바구니에 담긴 상품 몇 개만 조회할 때 사용.
    """
    if not product_ids:
        return {}
    if USE_DUMMY_DATA:
        from dummy_data import PRODUCTS
        matched = [p for p in PRODUCTS if p["product_id"] in product_ids]
        return {p["product_id"]: _apply_dummy_display_fields(p) for p in matched}
    products = _fetch_products_from_db(product_ids=product_ids)
    return {p["product_id"]: p for p in products}


def get_product_by_id(product_id: str) -> dict:
    result = get_products_by_ids([product_id])
    return result.get(product_id)