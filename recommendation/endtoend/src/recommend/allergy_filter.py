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