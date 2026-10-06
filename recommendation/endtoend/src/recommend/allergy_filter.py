# -*- coding: utf-8 -*-
"""
알러지 매칭(역추천) 로직.
온보딩 pet_profile.allergy_codes 와 product_master.allergen_flags 를 비교해
겹치는 게 있으면 EXCLUDE 대상으로 판단한다.
규칙 기반(확률적 추론 아님) — 안전 문제이므로 명확한 매칭/제외 로직으로 처리.

[버그 수정 이력]
pet_allergy_codes와 product_allergen_flags의 표기가 다르면(예: "chicken" vs "CHICKEN")
완전 일치 비교라 실제로는 같은 알러지인데도 충돌을 놓치는 문제가 있었다
(2026-09-30 안전성 검증 테스트에서 발견). 비교 시 대소문자를 구분하지 않도록 수정.
"""


def check_allergy_conflict(pet_allergy_codes: list, product_allergen_flags: list) -> dict:
    """
    반환값:
    {
        "has_conflict": bool,
        "matched_allergen": [겹치는 알러지 코드 리스트]  # 상품 쪽 원본 표기 기준
    }
    """
    pet_set = {str(code).strip().upper() for code in (pet_allergy_codes or []) if code}

    matched = sorted({
        flag for flag in (product_allergen_flags or [])
        if flag and str(flag).strip().upper() in pet_set
    })

    return {
        "has_conflict": len(matched) > 0,
        "matched_allergen": matched,
    }


def resolve_allergy_profile(status, codes) -> str:
    """빈 목록을 없음으로 추정하지 않고 저장된 상태와 목록의 일관성을 검사한다."""
    if not isinstance(codes, list) or any(
        not isinstance(code, str) or not code.strip() for code in codes
    ):
        return "UNKNOWN"
    # 이전 스키마의 비어 있지 않은 등록 목록만 KNOWN_LIST로 호환한다.
    if status is None:
        return "KNOWN_LIST" if codes else "UNKNOWN"
    if status == "KNOWN_NONE" and not codes:
        return status
    if status == "KNOWN_LIST" and codes:
        return status
    return "UNKNOWN"


def evaluate_recommendation_allergy(pet: dict, product_allergen_flags) -> dict:
    """추천의 기존 감점 계약을 유지하되 미확인 프로필을 SAFE로 표시하지 않는다."""
    codes = pet.get("allergy_codes")
    valid_codes = [code for code in codes if isinstance(code, str) and code.strip()] if isinstance(codes, list) else []
    flags = product_allergen_flags
    valid_flags = [flag for flag in flags if isinstance(flag, str) and flag.strip()] if isinstance(flags, list) else []
    conflict = check_allergy_conflict(valid_codes, valid_flags)
    # 상태가 모순이어도 명시적으로 확인된 충돌을 PENDING으로 약화하지 않는다.
    if conflict["has_conflict"]:
        return {"allergy_status": "PENALIZED", "matched_allergen": conflict["matched_allergen"], "pending_reason": None}
    profile = resolve_allergy_profile(pet.get("allergy_profile_status"), codes)
    reason = None
    if profile == "UNKNOWN":
        reason = "PROFILE_UNKNOWN"
    elif not isinstance(flags, list) or len(valid_flags) != len(flags):
        reason = "PRODUCT_UNKNOWN"
    return {"allergy_status": "PENDING" if reason else "SAFE", "matched_allergen": [], "pending_reason": reason}


def allergy_pending_message(result: dict) -> str:
    if result["pending_reason"] == "PROFILE_UNKNOWN":
        return "반려동물의 알레르기 정보를 확인해 주세요. "
    return "성분 정보 확인 중인 상품이에요. "
