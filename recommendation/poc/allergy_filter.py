# -*- coding: utf-8 -*-
"""
알러지 매칭(역추천) 로직.
온보딩 pet_profile.allergy_codes 와 product_master.allergen_flags 를 비교해
겹치는 게 있으면 EXCLUDE 대상으로 판단한다.
규칙 기반(확률적 추론 아님) — 안전 문제이므로 명확한 매칭/제외 로직으로 처리.
"""


def check_allergy_conflict(pet_allergy_codes: list, product_allergen_flags: list) -> dict:
    """
    반환값:
    {
        "has_conflict": bool,
        "matched_allergen": [겹치는 알러지 코드 리스트]
    }
    """
    pet_set = set(pet_allergy_codes or [])
    product_set = set(product_allergen_flags or [])
    matched = sorted(pet_set & product_set)

    return {
        "has_conflict": len(matched) > 0,
        "matched_allergen": matched,
    }