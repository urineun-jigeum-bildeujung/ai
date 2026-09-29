# -*- coding: utf-8 -*-
"""
구매 이력(orders/order_items, order_db) + 상품 임베딩(product_embeddings, product_db, pgvector) 조회.

주의: product_embeddings 테이블은 다른 파트에서 아직 만드는 중이라 실제 DB에 없을 수 있다.
      이 모듈은 그 테이블이 없어도(UndefinedTable 등) 에러로 죽지 않고 빈 딕셔너리를 반환한다.
      -> purchase_history_similarity는 자동으로 0.0(중립), find_substitute_products는
         빈 리스트로 안전하게 처리됨 (임베딩이 준비되면 별도 코드 수정 없이 자동으로 채워짐).
"""

import os
import sys

USE_DUMMY_DATA = os.environ.get("USE_DUMMY_DATA", "true").lower() != "false"

sys.path.append(os.path.dirname(__file__))
from db import get_connection

ORDER_DB_ENV = "ORDER_DATABASE_URL"
PRODUCT_DB_ENV = "PRODUCT_DATABASE_URL"

_VALID_ORDER_STATUSES = ("PAID", "PARTIAL_REFUND")


def _parse_vector(raw) -> list:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    text = str(raw).strip().strip("[]")
    if not text:
        return []
    return [float(x) for x in text.split(",")]


def _fetch_purchased_product_ids_from_db(user_id: str) -> list:
    conn = get_connection(ORDER_DB_ENV)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT oi.product_id
                FROM order_items oi
                JOIN orders o ON o.order_id = oi.order_id
                WHERE o.user_id = %s
                  AND o.order_status = ANY(%s)
                  AND (oi.quantity - oi.cancelled_quantity - oi.returned_quantity) >= 1
                """,
                (user_id, list(_VALID_ORDER_STATUSES)),
            )
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def get_purchased_product_ids_for_user(user_id: str) -> list:
    if USE_DUMMY_DATA:
        from src.recommend.purchase_history_similarity import get_purchased_product_ids
        from pipeline import DUMMY_ORDERS, DUMMY_ORDER_ITEMS

        return get_purchased_product_ids(user_id, DUMMY_ORDER_ITEMS, DUMMY_ORDERS)
    return _fetch_purchased_product_ids_from_db(user_id)


def _fetch_embeddings_from_db(product_ids: list) -> dict:
    if not product_ids:
        return {}
    conn = get_connection(PRODUCT_DB_ENV)
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    "SELECT product_id, embedding FROM product_embeddings WHERE product_id = ANY(%s)",
                    (list(product_ids),),
                )
            except Exception as e:
                # product_embeddings 테이블이 아직 없는 경우 (다른 파트 작업 진행 중) -- 빈 결과로 안전 처리
                conn.rollback()
                print(f"[order_embedding_repository] product_embeddings 조회 실패, 빈 결과로 처리: {e}")
                return {}

            result = {}
            for product_id, embedding in cur.fetchall():
                vec = _parse_vector(embedding)
                if vec is not None:
                    result[product_id] = vec
            return result
    finally:
        conn.close()


def get_product_embeddings(product_ids: list = None) -> dict:
    if USE_DUMMY_DATA:
        from pipeline import DUMMY_PRODUCT_EMBEDDINGS

        if product_ids is None:
            return dict(DUMMY_PRODUCT_EMBEDDINGS)
        return {pid: DUMMY_PRODUCT_EMBEDDINGS[pid] for pid in product_ids if pid in DUMMY_PRODUCT_EMBEDDINGS}

    if product_ids is None:
        raise NotImplementedError("실제 DB 모드에서는 전체 임베딩 조회 대신 product_ids를 지정하세요.")
    return _fetch_embeddings_from_db(product_ids)