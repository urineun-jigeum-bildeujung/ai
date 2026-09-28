# -*- coding: utf-8 -*-
"""
리뷰 작성자의 pet_profile을 가져오는 데이터 접근 계층.

[중요] review/review_question은 review_db에, pet_profile(pet 테이블)은 member_db에
서로 다른 DB로 나뉘어 있어서 SQL JOIN이 불가능하다 (PostgreSQL은 크로스 DB JOIN 미지원).
그래서 1) review_db에서 review+review_question을 먼저 가져오고,
       2) 등장한 pet_id들을 모아 member_db에서 한 번에 조회한 뒤,
       3) 파이썬에서 pet_id 기준으로 합친다.
"""

import os
import sys

USE_DUMMY_DATA = os.environ.get("USE_DUMMY_DATA", "true").lower() != "false"

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "data", "dummy"))
sys.path.append(os.path.dirname(__file__))

from db import get_connection

REVIEW_DB_ENV = "REVIEW_DATABASE_URL"

QUESTION_TYPE_TO_FIELD = {
    "PALATABILITY": "palatability_rating",
    "DIGESTION": "digestion_rating",
    "SKIN_COAT": "skin_coat_rating",
    "WEIGHT_VITALITY": "vitality_weight_rating",
    "ALLERGY": "allergic_reaction_rating",
}


def _fetch_reviews_from_db(product_id: str = None) -> list:
    """
    review_db에서 review + review_question(1:N)만 먼저 가져온다 (pet_profile 조인 없음).
    review_question이 리뷰당 여러 행이므로 review_id 기준으로 묶는다.
    """
    conn = get_connection(REVIEW_DB_ENV)
    try:
        with conn.cursor() as cur:
            query = """
                SELECT r.id, r.product_id, r.pet_id, r.star_rate,
                       rq.review_question_type, rq.review_answer
                FROM review r
                JOIN review_question rq ON rq.review_id = r.id
                WHERE r.pet_id IS NOT NULL
                  AND r.deleted_at IS NULL
                  AND rq.review_question_type != 'FEEDING_CONVENIENCE'
            """
            params = ()
            if product_id:
                query += " AND r.product_id = %s"
                params = (product_id,)

            cur.execute(query, params)
            rows = cur.fetchall()

            reviews_by_id = {}
            for (review_id, pid, pet_id, star_rate, question_type, answer) in rows:
                if review_id not in reviews_by_id:
                    reviews_by_id[review_id] = {
                        "review_id": review_id,
                        "product_id": pid,
                        "pet_id": pet_id,
                        "rating": star_rate,
                        "palatability_rating": None,
                        "digestion_rating": None,
                        "skin_coat_rating": None,
                        "vitality_weight_rating": None,
                        "allergic_reaction_rating": None,
                    }
                field_name = QUESTION_TYPE_TO_FIELD.get(question_type)
                if field_name:
                    reviews_by_id[review_id][field_name] = answer

            return list(reviews_by_id.values())
    finally:
        conn.close()


def load_reviews_with_reviewer_pet(product_id: str = None) -> list:
    """
    pipeline.py / train_deepfm.py / substitute_recommendation.py에서 사용하는 단일 진입점.
    반환 형태는 항상 동일:
    [{"review_id":, "product_id":, "rating":, "reviewer_pet": {...}, ...*_rating}, ...]
    """
    if USE_DUMMY_DATA:
        from dummy_reviews import DUMMY_REVIEWS
        if product_id:
            return [r for r in DUMMY_REVIEWS if r["product_id"] == product_id]
        return DUMMY_REVIEWS

    from src.data_access.pet_repository import get_pets_by_ids

    reviews = _fetch_reviews_from_db(product_id)
    pet_ids = list({r["pet_id"] for r in reviews if r.get("pet_id")})
    pets_by_id = get_pets_by_ids(pet_ids)

    result = []
    for review in reviews:
        reviewer_pet = pets_by_id.get(review["pet_id"])
        if reviewer_pet is None:
            continue  # member_db에서 못 찾은 pet_id는 안전하게 스킵 (탈퇴 회원 등)
        merged = dict(review)
        merged["reviewer_pet"] = reviewer_pet
        result.append(merged)
    return result