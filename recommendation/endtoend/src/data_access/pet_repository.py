# -*- coding: utf-8 -*-
"""
pet_profile 조회 데이터 접근 계층.

reviews_repository.py와 동일한 패턴: USE_DUMMY_DATA 환경변수로
더미 데이터 / 실제 DB를 전환한다. 호출부(pipeline.py, api/main.py)는
어디서 데이터를 가져오는지 신경 쓸 필요 없이 동일한 함수 시그니처로 사용.
"""

import os
import sys

USE_DUMMY_DATA = os.environ.get("USE_DUMMY_DATA", "true").lower() != "false"

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "data", "dummy"))


def _get_db_connection():
    import psycopg2

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL 환경변수가 설정되지 않았습니다. "
            "실제 DB를 사용하려면 USE_DUMMY_DATA=false와 함께 DATABASE_URL을 설정해야 합니다."
        )
    return psycopg2.connect(database_url)


def _row_to_pet(row) -> dict:
    (pet_id, user_id, species, breed, birth_date, sex, neutered,
     weight, bcs, allergy_codes, concerns) = row
    return {
        "pet_id": pet_id,
        "user_id": user_id,
        "species": species,
        "breed": breed,
        "birth_date": birth_date.isoformat() if hasattr(birth_date, "isoformat") else birth_date,
        "sex": sex,
        "neutered": bool(neutered),
        "weight": float(weight),
        "bcs": bcs,
        "allergy_codes": allergy_codes or [],
        "concerns": concerns or [],
    }


def _fetch_pet_from_db(pet_id: str) -> dict:
    conn = _get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT pet_id, user_id, species, breed, birth_date, sex, neutered,
                       weight, bcs, allergy_codes, concerns
                FROM pet_profile
                WHERE pet_id = %s
                """,
                (pet_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return _row_to_pet(row)
    finally:
        conn.close()


def get_pet_by_id(pet_id: str) -> dict:
    """
    api/main.py, pipeline.py에서 사용하는 단일 진입점.
    반환 형태는 항상 dummy_data.PET_PROFILES의 원소와 동일한 스키마.
    존재하지 않으면 None.
    """
    if USE_DUMMY_DATA:
        from dummy_data import PET_PROFILES
        for pet in PET_PROFILES:
            if pet["pet_id"] == pet_id:
                return pet
        return None
    return _fetch_pet_from_db(pet_id)


def list_all_pets() -> list:
    """더미 모드에서만 쓰는 전체 반려동물 목록 (배치/검증용)."""
    if USE_DUMMY_DATA:
        from dummy_data import PET_PROFILES
        return PET_PROFILES
    raise NotImplementedError("실제 DB 모드에서는 전체 조회 대신 get_pet_by_id(pet_id)를 사용하세요.")