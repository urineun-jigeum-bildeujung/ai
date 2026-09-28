# -*- coding: utf-8 -*-
"""
DB별 커넥션 생성 공용 헬퍼.

우리 도메인에 필요한 테이블이 4개 DB에 나뉘어 있다 (DB팀 확인 완료):
- pet_profile 관련        -> member_db  (테이블명: pet)
- product_master 관련     -> product_db (테이블명: products)
- review/review_question  -> review_db  (테이블명 동일)
- orders/order_items      -> order_db   (테이블명 동일)

같은 Postgres 서버(같은 host:port)일 수도, 아닐 수도 있어서 안전하게
DB별로 별도 환경변수(MEMBER_DATABASE_URL 등)를 쓴다.
"""

import os


def get_connection(env_var_name: str):
    import psycopg2

    database_url = os.environ.get(env_var_name)
    if not database_url:
        raise RuntimeError(
            f"{env_var_name} 환경변수가 설정되지 않았습니다. "
            f"실제 DB를 사용하려면 USE_DUMMY_DATA=false와 함께 {env_var_name}을 설정해야 합니다."
        )
    return psycopg2.connect(database_url)