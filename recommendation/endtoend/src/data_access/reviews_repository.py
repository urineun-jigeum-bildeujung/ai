# -*- coding: utf-8 -*-
"""
리뷰 작성자의 pet_profile을 가져오는 데이터 접근 계층.

reviews.pet_id 스키마가 반영되어, 실제로는 다음과 같은 조인이 필요하다:
    reviews.pet_id -> pet_profile.pet_id

지금은 실제 리뷰 데이터가 아직 없어서(스키마만 반영됨), 더미 데이터 기반으로
동작하되, pipeline.py 입장에서는 "어디서 데이터를 가져오는지" 신경 쓸 필요 없이
동일한 함수 시그니처로 사용할 수 있도록 설계했다.

실제 리뷰 데이터가 준비되면 아래 두 가지만 하면 된다:
1. .env 등에 DATABASE_URL 설정
2. USE_DUMMY_DATA = False 로 변경 (또는 환경변수로 제어)
그 외 pipeline.py, train_deepfm.py 등 호출부는 전혀 수정할 필요가 없다.
"""

import os
import sys

# 환경변수로 더미/실제 DB 전환 (기본값: 더미 -- 실데이터 준비 전까지는 이게 안전)
USE_DUMMY_DATA = os.environ.get("USE_DUMMY_DATA", "true").lower() != "false"

# review_question.review_question_type -> dummy_reviews.py / rating_converter.py에서
# 쓰는 필드명(*_rating) 매핑. FEEDING_CONVENIENCE는 추천 근거 제외가 확정되어 매핑에서 뺐다.
QUESTION_TYPE_TO_FIELD = {
    "PALATABILITY": "palatability_rating",
    "DIGESTION": "digestion_rating",
    "SKIN_COAT": "skin_coat_rating",
    "WEIGHT_VITALITY": "vitality_weight_rating",
    "ALLERGY": "allergic_reaction_rating",
}

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "data", "dummy"))


def _get_db_connection():
    """
    PostgreSQL 연결을 생성한다.
    실제 연결 정보(호스트/포트/DB명/계정)가 정해지면 환경변수(DATABASE_URL)로 주입.
    지금은 USE_DUMMY_DATA=True인 동안에는 호출되지 않는 함수라 미완성 상태로 둬도 안전하다.
    """
    import psycopg2  # 실제 연결 시점에만 import (더미 모드에서는 설치 안 되어 있어도 무방)

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL 환경변수가 설정되지 않았습니다. "
            "실제 DB를 사용하려면 USE_DUMMY_DATA=false와 함께 DATABASE_URL을 설정해야 합니다."
        )
    return psycopg2.connect(database_url)


def _fetch_reviewer_pet_from_db(pet_id: str) -> dict:
    """
    실제 PostgreSQL에서 pet_id로 pet_profile을 조회.
    반환 형태는 더미 데이터의 reviewer_pet과 동일한 스키마로 맞춘다:
    {"species": str, "birth_date": str, "weight": float, "allergy_codes": list}
    """
    conn = _get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT species, birth_date, weight, allergy_codes
                FROM pet_profile
                WHERE pet_id = %s
                """,
                (pet_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            species, birth_date, weight, allergy_codes = row
            return {
                "species": species,
                "birth_date": birth_date.isoformat() if hasattr(birth_date, "isoformat") else birth_date,
                "weight": float(weight),
                "allergy_codes": allergy_codes or [],
            }
    finally:
        conn.close()


def _fetch_reviews_from_db(product_id: str = None) -> list:
    """
    실제 PostgreSQL에서 review + review_question(1:N)을 pet_profile과 조인해서 가져온다.
    product_id를 지정하면 해당 상품 리뷰만, 없으면 전체를 가져온다.
    반환 형태는 더미 리뷰 데이터(dummy_reviews.DUMMY_REVIEWS)와 동일한 스키마로 맞춘다.

    [최종 스키마 확정 사항]
    - 5개 aspect 평점은 review 테이블의 컬럼이 아니라, review_question 테이블에
      리뷰 하나당 여러 행(질문-답변 방식)으로 저장됨.
    - review_question.review_question_type 값: PALATABILITY, DIGESTION,
      FEEDING_CONVENIENCE, SKIN_COAT, WEIGHT_VITALITY, ALLERGY
      (FEEDING_CONVENIENCE는 추천 근거에서 제외 확정 -- 우리 5개 aspect에 미포함)
    - review_question.review_answer 값은 NEGATIVE/NEUTRAL/POSITIVE 문자열로 저장되나,
      API 응답 시 1(NEGATIVE)/2(NEUTRAL)/3(POSITIVE) 숫자로 변환되어 내려오기로
      백엔드와 확정함 -- 이 함수는 이미 숫자로 변환된 값을 받는다고 가정한다.
    - review.pet_id로 pet_profile과 직접 조인 가능. member_id는 작성자 식별용으로
      존재하나 지금 파이프라인에서는 pet_id 기준 조인만 사용.
    """
    conn = _get_db_connection()
    try:
        with conn.cursor() as cur:
            query = """
                SELECT r.id, r.product_id, r.star_rate,
                       rq.review_question_type, rq.review_answer,
                       p.species, p.birth_date, p.weight, p.allergy_codes
                FROM review r
                JOIN review_question rq ON rq.review_id = r.id
                JOIN pet_profile p ON r.pet_id = p.pet_id
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

            # review_question이 리뷰당 여러 행이므로 review_id 기준으로 묶는다
            reviews_by_id = {}
            for (review_id, pid, star_rate, question_type, answer,
                 species, birth_date, weight, allergy_codes) in rows:
                if review_id not in reviews_by_id:
                    reviews_by_id[review_id] = {
                        "review_id": review_id,
                        "product_id": pid,
                        "rating": star_rate,
                        "palatability_rating": None,
                        "digestion_rating": None,
                        "skin_coat_rating": None,
                        "vitality_weight_rating": None,
                        "allergic_reaction_rating": None,
                        "reviewer_pet": {
                            "species": species,
                            "birth_date": birth_date.isoformat() if hasattr(birth_date, "isoformat") else birth_date,
                            "weight": float(weight),
                            "allergy_codes": allergy_codes or [],
                        },
                    }
                field_name = QUESTION_TYPE_TO_FIELD.get(question_type)
                if field_name:
                    reviews_by_id[review_id][field_name] = answer

            return list(reviews_by_id.values())
    finally:
        conn.close()


def load_reviews_with_reviewer_pet(product_id: str = None) -> list:
    """
    pipeline.py / train_deepfm.py에서 사용하는 단일 진입점.
    USE_DUMMY_DATA 값에 따라 더미 데이터 또는 실제 DB에서 가져온다.
    반환 형태는 항상 동일:
    [{"review_id":, "product_id":, "rating":, "review_text":, "reviewer_pet": {...}}, ...]
    """
    if USE_DUMMY_DATA:
        from dummy_reviews import DUMMY_REVIEWS
        if product_id:
            return [r for r in DUMMY_REVIEWS if r["product_id"] == product_id]
        return DUMMY_REVIEWS
    else:
        return _fetch_reviews_from_db(product_id)