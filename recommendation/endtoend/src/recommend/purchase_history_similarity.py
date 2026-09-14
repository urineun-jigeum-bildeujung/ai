# -*- coding: utf-8 -*-
"""
구매 이력 기반 유사도 계산.

콜드스타트 이후 단계: 사용자 프로필 + 리뷰 작성자 프로필 + keywords + 기존 구매 상품과의 유사도
-> 이 모듈은 마지막 요소(기존 구매 상품과의 유사도)를 담당한다.

FR-AI-2-02(대체상품 추천)에서 이미 사용하는 product_embeddings(pgvector)를 재사용해서,
"이 사용자가 지금까지 구매한 상품들과 후보 상품이 얼마나 비슷한지"를 계산한다.

구매 이력이 없는 사용자(콜드스타트)는 계산할 대상 자체가 없으므로 0.0(중립)으로 처리되며,
같은 DeepFM 구조로 콜드스타트/기존 유저를 모두 처리할 수 있게 설계했다.
"""

import numpy as np


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_purchase_history_similarity(
    candidate_embedding: list,
    purchased_embeddings: list,
) -> float:
    """
    candidate_embedding: 추천 후보 상품의 임베딩 벡터
    purchased_embeddings: 사용자가 기존에 구매한 상품들의 임베딩 벡터 리스트
                           (구매 이력 없으면 빈 리스트)

    반환: -1~1 사이 유사도 점수. 구매 이력이 없으면 0.0(중립).

    방식: 후보 상품과 "기존 구매 상품들의 평균 임베딩" 간 코사인 유사도.
    (구매 상품 하나하나와 개별 비교 후 평균 내는 방식도 가능하지만,
     평균 임베딩과 비교하는 방식이 계산량이 적고 "전반적인 구매 취향"을
     더 안정적으로 반영한다.)
    """
    if not purchased_embeddings:
        return 0.0

    avg_embedding = np.mean(np.array(purchased_embeddings, dtype=float), axis=0)
    return round(cosine_similarity(candidate_embedding, avg_embedding.tolist()), 4)


def get_purchased_product_ids(user_id: str, order_items: list, orders: dict) -> list:
    """
    order_items에서 이 사용자가 유효하게 구매한 상품 ID 목록을 추출.
    유효 구매 기준(라벨링 가이드라인과 동일): 수량-취소-반품>=1, 주문상태 PAID/PARTIAL_REFUND.
    """
    product_ids = []
    for item in order_items:
        order = orders.get(item["order_id"])
        if not order or order.get("user_id") != user_id:
            continue
        if order["order_status"] not in ("PAID", "PARTIAL_REFUND"):
            continue

        effective_qty = (
            item.get("quantity", 0)
            - item.get("cancelled_quantity", 0)
            - item.get("returned_quantity", 0)
        )
        if effective_qty >= 1:
            product_ids.append(item["product_id"])

    return product_ids


def build_purchase_history_feature(
    user_id: str,
    candidate_product_id: str,
    order_items: list,
    orders: dict,
    product_embeddings: dict,
) -> float:
    """
    파이프라인에서 바로 호출하는 진입점.

    user_id: 추천 대상 사용자
    candidate_product_id: 추천 후보 상품
    order_items, orders: 구매 이력 원천 데이터
    product_embeddings: {product_id: embedding_vector} 딕셔너리
                         (실제로는 product_embeddings 테이블 조회 결과)

    반환: purchase_history_similarity dense feature 값 (-1~1, 구매 이력 없으면 0.0)
    """
    purchased_ids = get_purchased_product_ids(user_id, order_items, orders)
    if not purchased_ids:
        return 0.0

    candidate_embedding = product_embeddings.get(candidate_product_id)
    if candidate_embedding is None:
        return 0.0  # 후보 상품 임베딩이 아직 없는 경우 (신규 상품 등) 안전하게 중립 처리

    purchased_embeddings = [
        product_embeddings[pid] for pid in purchased_ids if pid in product_embeddings
    ]
    if not purchased_embeddings:
        return 0.0

    return compute_purchase_history_similarity(candidate_embedding, purchased_embeddings)