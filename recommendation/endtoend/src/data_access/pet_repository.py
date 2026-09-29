# -*- coding: utf-8 -*-
"""
member_db.pet 조회 데이터 접근 계층.

order_embedding_repository.py / product_repository.py / reviews_repository.py와
동일한 패턴: db.py의 get_connection(env_var_name)을 공용으로 사용하고,
USE_DUMMY_DATA 환경변수로 더미 데이터 / 실제 DB를 전환한다.
"""

import os
import sys

USE_DUMMY_DATA = os.environ.get("USE_DUMMY_DATA", "true").lower() != "false"

MEMBER_DB_ENV = "MEMBER_DATABASE_URL"

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "data", "dummy"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))  # src.data_access import용

from src.data_access.db import get_connection


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


_PET_COLUMNS = """
    pet_id, user_id, species, breed, birth_date, sex, neutered,
    weight, bcs, allergy_codes, concerns
"""


def _fetch_pet_from_db(pet_id: str) -> dict:
    conn = get_connection(MEMBER_DB_ENV)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {_PET_COLUMNS} FROM pet WHERE pet_id = %s",
                (pet_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return _row_to_pet(row)
    finally:
        conn.close()


def _fetch_pets_from_db(pet_ids: list) -> dict:
    if not pet_ids:
        return {}
    conn = get_connection(MEMBER_DB_ENV)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {_PET_COLUMNS} FROM pet WHERE pet_id = ANY(%s)",
                (list(pet_ids),),
            )
            rows = cur.fetchall()
            pets = [_row_to_pet(row) for row in rows]
            return {p["pet_id"]: p for p in pets}
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


def get_pets_by_ids(pet_ids: list) -> dict:
    """
    {pet_id: pet_dict} 형태로 반환. reviews_repository.py가 여러 리뷰 작성자의
    pet을 한 번에 가져올 때 사용 (N+1 쿼리 방지).
    """
    if not pet_ids:
        return {}
    if USE_DUMMY_DATA:
        from dummy_data import PET_PROFILES
        matched = [p for p in PET_PROFILES if p["pet_id"] in pet_ids]
        return {p["pet_id"]: p for p in matched}
    return _fetch_pets_from_db(pet_ids)


def list_all_pets() -> list:
    """더미 모드에서만 쓰는 전체 반려동물 목록 (배치/검증용)."""
    if USE_DUMMY_DATA:
        from dummy_data import PET_PROFILES
        return PET_PROFILES
    raise NotImplementedError("실제 DB 모드에서는 전체 조회 대신 get_pet_by_id(pet_id)를 사용하세요.")