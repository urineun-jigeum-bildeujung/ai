# -*- coding: utf-8 -*-
"""
product_embeddings 생성 배치 스크립트 (1회성/필요시 재실행용, 서빙 API와 무관).

product_db.products에서 상품 데이터를 읽어와 텍스트 표현을 만들고,
sentence-transformers로 임베딩 벡터를 계산해 JSON으로 저장한다.

사전 준비:
    pip install sentence-transformers psycopg2-binary python-dotenv
    (requirements.txt에 이미 포함되어 있으면 생략)

실행:
    kubectl --context petflow-dev -n database port-forward svc/petflow-db-rw 15432:5432
    python3 scripts/generate_product_embeddings.py
"""

import os
import json
import time

from dotenv import load_dotenv
load_dotenv()

import psycopg2
from sentence_transformers import SentenceTransformer

PRODUCT_DATABASE_URL = os.environ.get("PRODUCT_DATABASE_URL")
OUTPUT_PATH = "product_embeddings.json"

# 다국어(한국어 포함) 지원하는 경량 모델. 상품 수가 적어서 더 큰 모델 써도 무방.
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def fetch_products(conn):
    query = """
        SELECT
            p.id AS product_id,
            p.product_name,
            b.name AS brand_name,
            p.category_code,
            p.subcategory_code,
            p.avg_rating,
            p.status,
            COALESCE(
                (SELECT array_agg(pa.allergen_code)
                 FROM product_allergens pa
                 WHERE pa.product_id = p.id),
                ARRAY[]::varchar[]
            ) AS allergen_flags,
            COALESCE(
                (SELECT array_agg(pi.ingredient_code ORDER BY pi.sort_order)
                 FROM product_ingredients pi
                 WHERE pi.product_id = p.id),
                ARRAY[]::varchar[]
            ) AS ingredients,
            COALESCE(
                (SELECT array_agg(pts.species)
                 FROM product_target_species pts
                 WHERE pts.product_id = p.id),
                ARRAY[]::varchar[]
            ) AS target_species
        FROM products p
        LEFT JOIN brands b ON b.id = p.brand_id
    """
    with conn.cursor() as cur:
        cur.execute(query)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    return [dict(zip(columns, row)) for row in rows]

def build_text(product: dict) -> str:
    """상품 하나를 임베딩할 텍스트로 변환. 성분/이름/카테고리를 합쳐서 의미 유사도가 잘 반영되게 함."""
    parts = [
        product["product_name"],
        product["brand_name"],
        product["category_code"],
        " ".join(product["ingredients"]),
    ]
    return " ".join(p for p in parts if p)


def main():
    if not PRODUCT_DATABASE_URL:
        raise RuntimeError("PRODUCT_DATABASE_URL 환경변수가 필요합니다 (.env 확인).")

    print("상품 목록 조회 중...")
    conn = psycopg2.connect(PRODUCT_DATABASE_URL)
    try:
        products = fetch_products(conn)
    finally:
        conn.close()
    print(f"{len(products)}개 상품 조회 완료")

    print(f"모델 로딩 중... ({MODEL_NAME})")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"모델 로딩 완료 ({time.time() - t0:.1f}초)")

    texts = [build_text(p) for p in products]

    print("임베딩 계산 중...")
    t0 = time.time()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    print(f"임베딩 계산 완료 ({time.time() - t0:.1f}초)")

    result = [
        {"product_id": p["product_id"], "embedding": emb.tolist()}
        for p, emb in zip(products, embeddings)
    ]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    print(f"완료: {OUTPUT_PATH} ({len(result)}건, 벡터 차원={len(result[0]['embedding']) if result else 0})")


if __name__ == "__main__":
    main()