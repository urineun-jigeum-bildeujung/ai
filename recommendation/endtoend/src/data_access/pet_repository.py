# -*- coding: utf-8 -*-
"""
member_db.pet 조회 데이터 접근 계층.

order_embedding_repository.py / product_repository.py / reviews_repository.py와
동일한 패턴: db.py의 get_connection(env_var_name)을 공용으로 사용하고,
USE_DUMMY_DATA 환경변수로 더미 데이터 / 실제 DB를 전환한다.

실제 스키마 주의: pet 테이블의 PK/FK는 pet_id가 아니라 id, user 참조는 user_id가 아니라
member_id, breed는 문자열 컬럼이 아니라 breed_id(FK -> breed_master), neutered는
is_neutered. allergy_codes/concerns는 pet 테이블 컬럼이 아니라 각각 pet_allergy,
pet_concern(+concern_master) 별도 테이블이라 조인 + array_agg로 묶어서 가져온다.
"""

import os
import sys

USE_DUMMY_DATA = os.environ.get("USE_DUMMY_DATA", "true").lower() != "false"

MEMBER_DB_ENV = "MEMBER_DATABASE_URL"

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "data", "dummy"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))  # src.data_access import용

from src.data_access.db import get_connection
from src.recommend.allergy_filter import resolve_allergy_profile


def _row_to_pet(row) -> dict:
    (pet_id, user_id, species, breed, birth_date, sex, neutered,
     weight, bcs, allergy_codes, concerns, allergy_profile_status) = row
    codes = list(allergy_codes) if allergy_codes else []
    return {
        "pet_id": pet_id,
        "user_id": user_id,
        "species": species,
        "breed": breed,
        "birth_date": birth_date.isoformat() if hasattr(birth_date, "isoformat") else birth_date,
        "sex": sex,
        "neutered": bool(neutered),
        "weight": float(weight) if weight is not None else None,
        "bcs": bcs,
        "allergy_codes": codes,
        "allergy_profile_status": resolve_allergy_profile(allergy_profile_status, codes),
        "concerns": list(concerns) if concerns else [],
    }


_PET_QUERY = """
    SELECT
        p.id,
        p.member_id,
        p.species,
        b.breed_name,
        p.birth_date,
        p.sex,
        p.is_neutered,
        p.weight,
        p.bcs,
        COALESCE(
            array_agg(DISTINCT pa.allergy_code) FILTER (WHERE pa.allergy_code IS NOT NULL),
            '{}'
        ) AS allergy_codes,
        COALESCE(
            array_agg(DISTINCT cm.concern_code) FILTER (WHERE cm.concern_code IS NOT NULL),
            '{}'
        ) AS concerns,
        to_jsonb(p)->>'allergy_profile_status' AS allergy_profile_status
    FROM pet p
    LEFT JOIN breed_master b ON b.id = p.breed_id
    LEFT JOIN pet_allergy pa ON pa.pet_id = p.id
    LEFT JOIN pet_concern pc ON pc.pet_id = p.id
    LEFT JOIN concern_master cm ON cm.id = pc.concern_id
"""


def _fetch_pet_from_db(pet_id: str) -> dict:
    conn = get_connection(MEMBER_DB_ENV)
    try:
        with conn.cursor() as cur:
            cur.execute(
                _PET_QUERY + " WHERE p.id = %s GROUP BY p.id, b.breed_name",
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
                _PET_QUERY + " WHERE p.id = ANY(%s) GROUP BY p.id, b.breed_name",
                (list(pet_ids),),
            )
            rows = cur.fetchall()
            pets = [_row_to_pet(row) for row in rows]
            return {p["pet_id"]: p for p in pets}
    finally:
        conn.close()


def get_pet_by_id(pet_id: str) -> dict:
    """
    내부 배치/검증용 조회. HTTP 요청은 get_owned_pet_by_id를 사용한다.
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


def get_owned_pet_by_id(pet_id: int, member_id: int) -> dict | None:
    """추천 요청의 대상은 인증된 회원이 소유한 삭제되지 않은 Pet으로 한정한다."""
    if type(member_id) is not int or not 0 < member_id <= 9223372036854775807:
        return None
    if USE_DUMMY_DATA:
        pet = get_pet_by_id(pet_id)
        if pet and pet.get("user_id") == member_id and pet.get("deleted_at") is None:
            return pet
        return None
    conn = get_connection(MEMBER_DB_ENV)
    try:
        with conn.cursor() as cur:
            cur.execute(
                _PET_QUERY + " WHERE p.id = %s AND p.member_id = %s AND p.deleted_at IS NULL"
                " GROUP BY p.id, b.breed_name",
                (pet_id, member_id),
            )
            row = cur.fetchone()
            return _row_to_pet(row) if row is not None else None
    finally:
        conn.close()


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
