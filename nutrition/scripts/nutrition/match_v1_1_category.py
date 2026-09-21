"""매칭 v1.1 — 5종 카테고리 분기 (8/20~21)

- food: NIAS 2024 + 알레르기 + AAFCO + 라이프스테이지
- treat: 알레르기 + 원료 (NIAS 미적용)
- pad: product_attributes (MATERIAL/ABSORBENCY/SIZE)
- litter: product_attributes (MATERIAL/CLUMPING/DUST/SCENT)
- supplement: NIAS + 함량 + 알레르기

사용법:
    from ai.scripts.nutrition.match_v1_1_category import match_by_category
    result = match_by_category(pet, product)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


# match_v1.py가 scripts/ 루트에 있으므로 path 추가
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from match_v1 import (  # type: ignore  # noqa: E402
    load_seed,
    match_allergens,
    match_nutrients,
    generate_consumer_card,
)
from allergen_service import evaluate_safety  # type: ignore  # noqa: E402


SEED_FEED_CODES = load_seed("seed_feed_codes.json")
SEED_NUTRITION = load_seed("seed_nutrition_standard.json")


def _pet_to_persona(pet: dict) -> dict:
    """PetIn → match_v1 페르소나 dict 변환"""
    lifestage = pet.get("life_stage")
    if not lifestage:
        if pet.get("species") == "cat":
            lifestage = "kitten" if pet.get("age_years", 0) < 1 else "adult"
        else:
            lifestage = "puppy" if pet.get("age_years", 0) < 1 else "adult"
    return {
        "id": pet.get("id", "unknown"),
        "label": f"{pet.get('id', 'unknown')} ({pet.get('species', '?')}, {lifestage})",
        "species": pet.get("species", "dog"),
        "age_years": pet.get("age_years", 0),
        "weight_kg": pet.get("weight_kg", 0),
        "allergies": pet.get("allergies", []),
        "lifestage": lifestage,
    }


def match_food(pet: dict, product: dict) -> dict:
    """사료 = NIAS 2024 + 알레르기 + AAFCO + 라이프스테이지"""
    persona = _pet_to_persona(pet)
    ingredient_list = product.get("ingredient_list", [])

    # 1) 알레르기 hard filter (FR-AI-1-04)
    allergen = match_allergens(ingredient_list, SEED_FEED_CODES, persona)
    if allergen.get("excluded"):
        return {
            "category_branch": "food",
            "excluded": True,
            "exclude_reasons": [m["ingredient"] for m in allergen.get("matches", [])],
            "consumer_card": allergen.get("reason", "[제외] 알레르기 성분 포함"),
        }

    # 2) NIAS 비교 (FR-AI-1-02)
    feed = {
        "product_name": product.get("name", "(unknown)"),
        "ingredient_list": ingredient_list,
        **product.get("guaranteed_analysis", {}),
    }
    nutrients = match_nutrients(feed, persona, SEED_NUTRITION)

    # 3) Label life-stage compatibility is not an AAFCO adequacy determination.
    # A guaranteed analysis is partial and may contain label guarantees rather than
    # measured composition, so it must never yield an AAFCO PASS by itself.
    aafco_life_stage = product.get("aafco_life_stage")
    label_lifestage_match = (
        _check_aafco(persona, aafco_life_stage)
        if aafco_life_stage else None
    )
    if not aafco_life_stage:
        nutrition_comparison_status = "UNKNOWN"
        aafco_note = "AAFCO 영양적합성 문구/생애주기 정보가 없습니다."
    elif label_lifestage_match is False:
        nutrition_comparison_status = "FALSE"
        aafco_note = "상품 라벨의 생애주기 표기가 반려동물 생애주기와 맞지 않습니다."
    else:
        nutrition_comparison_status = "UNKNOWN"
        aafco_note = (
            "부분 보증성분만으로는 AAFCO complete-and-balanced 적합성을 판정할 수 없습니다."
        )

    # 4) consumer_card (FR-AI-1-06 P0)
    card = generate_consumer_card(persona, feed, allergen, nutrients)

    warnings: list[str] = []
    for r in nutrients.get("hazard", []):
        warnings.append(f"{r['nutrient']}: {r['feed_value']} (안전 최대 {r['max']} 초과) — 위험")
    for r in nutrients.get("excess", []):
        warnings.append(f"{r['nutrient']}: {r['feed_value']} (적정 최대 {r['max']} 초과) — 과다")
    for r in nutrients.get("deficient", []):
        warnings.append(f"{r['nutrient']}: {r['feed_value']} (권장 {r['min']} 미만) — 부족")

    return {
        "category_branch": "food",
        "excluded": False,
        "exclude_reasons": [],
        "nutrient_score": nutrients.get("overall_score", 0),
        "nutrients_checked": nutrients.get("nutrients_checked", 0),
        # Deprecated compatibility field.  UNKNOWN is deliberately represented as
        # None, never as a false positive boolean.
        "aafco_pass": None,
        "nutrition_comparison_status": nutrition_comparison_status,
        "aafco_match": {
            "value": nutrition_comparison_status,
            "label_life_stage": aafco_life_stage,
            "label_lifestage_match": label_lifestage_match,
            "note": aafco_note,
        },
        "aafco_life_stage": aafco_life_stage,
        "lifestage_match": label_lifestage_match,
        "deficient_count": len(nutrients.get("deficient", [])),
        "excess_count": len(nutrients.get("excess", [])),
        "hazard_count": len(nutrients.get("hazard", [])),
        "warnings": warnings,
        "consumer_card": card,
        "details": nutrients,
    }


def match_treat(pet: dict, product: dict) -> dict:
    """간식 = 알레르기 + 원료 (NIAS 미적용, 칼로리 가벼움)"""
    persona = _pet_to_persona(pet)
    ingredient_list = product.get("ingredient_list", [])

    allergen = match_allergens(ingredient_list, SEED_FEED_CODES, persona)
    if allergen.get("excluded"):
        return {
            "category_branch": "treat",
            "excluded": True,
            "exclude_reasons": [m["ingredient"] for m in allergen.get("matches", [])],
            "consumer_card": allergen.get("reason", "[제외] 알레르기 성분 포함"),
        }

    ingredient_count = len(ingredient_list)
    # 가벼운 점수: 원료 수 기반
    if ingredient_count == 0:
        score = 0
        note = "원료 정보 없음"
    elif ingredient_count <= 5:
        score = 80
        note = "원료 간단"
    elif ingredient_count <= 10:
        score = 60
        note = "원료 보통"
    else:
        score = 40
        note = "원료 많음 (첨가제 多)"

    return {
        "category_branch": "treat",
        "excluded": False,
        "exclude_reasons": [],
        "nutrient_score": score,
        "ingredient_count": ingredient_count,
        "consumer_card": (
            f"간식 = 원료 {ingredient_count}개 ({note}). "
            + ("알레르기 성분을 판정할 원료 정보가 없습니다." if ingredient_count == 0 else "등록된 알레르겐과 일치하는 원료가 없습니다.")
        ),
    }


def match_pad(pet: dict, product: dict) -> dict:
    """배변패드 = product_attributes (시드 14, 8/22 예정)

    임시: attrs 없으면 not_stated
    """
    attrs = product.get("product_attributes", {})
    return {
        "category_branch": "pad",
        "excluded": False,
        "material": attrs.get("MATERIAL", "not_stated"),
        "absorbency": attrs.get("ABSORBENCY", "not_stated"),
        "size": attrs.get("SIZE", "not_stated"),
        "consumer_card": f"배변패드 — 소재 {attrs.get('MATERIAL', 'not_stated')}, 흡수력 {attrs.get('ABSORBENCY', 'not_stated')}, 크기 {attrs.get('SIZE', 'not_stated')} (시드 14 placeholder, 8/22 보강)",
    }


def match_litter(pet: dict, product: dict) -> dict:
    """모래 = product_attributes (시드 14, 8/22 예정)"""
    attrs = product.get("product_attributes", {})
    return {
        "category_branch": "litter",
        "excluded": False,
        "material": attrs.get("MATERIAL", "not_stated"),
        "clumping": attrs.get("CLUMPING", "not_stated"),
        "dust": attrs.get("DUST", "not_stated"),
        "scent": attrs.get("SCENT", "not_stated"),
        "consumer_card": f"모래 — 소재 {attrs.get('MATERIAL', 'not_stated')}, 응고력 {attrs.get('CLUMPING', 'not_stated')}, 먼지 {attrs.get('DUST', 'not_stated')}, 향 {attrs.get('SCENT', 'not_stated')} (시드 14 placeholder, 8/22 보강)",
    }


def match_supplement(pet: dict, product: dict) -> dict:
    """영양제 = NIAS + 함량 + 알레르기"""
    persona = _pet_to_persona(pet)
    ingredient_list = product.get("ingredient_list", [])

    allergen = match_allergens(ingredient_list, SEED_FEED_CODES, persona)
    if allergen.get("excluded"):
        return {
            "category_branch": "supplement",
            "excluded": True,
            "exclude_reasons": [m["ingredient"] for m in allergen.get("matches", [])],
            "consumer_card": allergen.get("reason", "[제외] 알레르기 성분 포함"),
        }

    feed = {
        "product_name": product.get("name", "(unknown)"),
        "ingredient_list": ingredient_list,
        **product.get("guaranteed_analysis", {}),
    }
    nutrients = match_nutrients(feed, persona, SEED_NUTRITION)

    ingredient_count = len(ingredient_list)
    return {
        "category_branch": "supplement",
        "excluded": False,
        "exclude_reasons": [],
        "nutrient_score": nutrients.get("overall_score", 0),
        "nutrients_checked": nutrients.get("nutrients_checked", 0),
        "ingredient_count": ingredient_count,
        "consumer_card": f"영양제 = 원료 {ingredient_count}개. NIAS 매칭 {nutrients.get('nutrients_checked', 0)}개 영양소. (수족관/구강 영양제 단독 급여 권장, 수의사 상담 병행)",
    }


def _check_aafco(persona: dict, aafco_life_stage: str) -> bool:
    """AAFCO 적합성 — puppy/kitten → GROWTH_REPRODUCTION, adult/senior → MAINTENANCE/ALL_LIFE_STAGES"""
    lifestage = persona.get("lifestage", "adult")
    if lifestage in ("puppy", "kitten"):
        return aafco_life_stage in ("GROWTH_REPRODUCTION", "ALL_LIFE_STAGES")
    elif lifestage in ("adult", "senior"):
        return aafco_life_stage in ("MAINTENANCE", "ALL_LIFE_STAGES")
    return True


def _check_lifestage(persona: dict, aafco_life_stage: str) -> bool:
    """라이프스테이지 매칭 (FR-AI-1-05) — 단순 적합성"""
    lifestage = persona.get("lifestage", "adult")
    if lifestage == "all":
        return True
    mapping = {
        "puppy": ["GROWTH_REPRODUCTION", "ALL_LIFE_STAGES"],
        "kitten": ["GROWTH_REPRODUCTION", "ALL_LIFE_STAGES"],
        "adult": ["MAINTENANCE", "ALL_LIFE_STAGES"],
        "senior": ["MAINTENANCE", "ALL_LIFE_STAGES"],
    }
    return aafco_life_stage in mapping.get(lifestage, ["ALL_LIFE_STAGES"])


def match_by_category(pet: dict, product: dict) -> dict:
    """5종 카테고리 분기 메인 진입점"""
    category = product.get("category", "food")
    if category == "food":
        result = match_food(pet, product)
    elif category == "treat":
        result = match_treat(pet, product)
    elif category == "pad":
        result = match_pad(pet, product)
    elif category == "litter":
        result = match_litter(pet, product)
    elif category == "supplement":
        result = match_supplement(pet, product)
    else:
        result = {"category_branch": category, "excluded": False, "consumer_card": f"알 수 없는 카테고리: {category}"}
    # Public direct matcher and HTTP adapter share one fail-closed decision.
    safety = evaluate_safety(pet, product)
    result.update(safety)
    return result


if __name__ == "__main__":
    # self-test
    pet = {"id": "maru", "species": "cat", "age_years": 4, "weight_kg": 4.2, "allergies": ["chicken"]}
    food_with_chicken = {
        "name": "치킨 사료 (코코 알레르기 위험 시뮬레이션)",
        "category": "food",
        "ingredient_list": ["닭고기", "쌀", "옥수수"],
        "guaranteed_analysis": {"단백질_g": 24.0, "지방_g": 12.0, "칼슘_g": 1.0},
        "aafco_life_stage": "MAINTENANCE",
    }
    food_no_allergen = {
        "name": "연어 사료 (마루 안전)",
        "category": "food",
        "ingredient_list": ["연어", "쌀", "비트펄프"],
        "guaranteed_analysis": {"단백질_g": 26.0, "지방_g": 14.0, "칼슘_g": 1.2},
        "aafco_life_stage": "MAINTENANCE",
    }
    treat = {
        "name": "닭가슴살 간식",
        "category": "treat",
        "ingredient_list": ["닭고기", "쌀"],
    }
    pad = {
        "name": "배변패드 대형",
        "category": "pad",
        "product_attributes": {"MATERIAL": "cotton", "ABSORBENCY": "high", "SIZE": "L"},
    }
    litter = {
        "name": "벤토나이트 모래",
        "category": "litter",
        "product_attributes": {"MATERIAL": "bentonite", "CLUMPING": "good", "DUST": "low", "SCENT": "lavender"},
    }
    supplement = {
        "name": "오메가3 영양제",
        "category": "supplement",
        "ingredient_list": ["연어유", "비타민E"],
        "guaranteed_analysis": {"단백질_g": 0.0, "지방_g": 50.0},
    }
    print("=== match_by_category self-test (마루 / cat / 닭고기 알레르기) ===")
    for label, p in [("food w/ chicken", food_with_chicken), ("food no allergen", food_no_allergen), ("treat", treat), ("pad", pad), ("litter", litter), ("supplement", supplement)]:
        r = match_by_category(pet, p)
        print(f"\n[{label}] {p['name']}")
        print(f"  category={r.get('category_branch')}, excluded={r.get('excluded')}, score={r.get('nutrient_score', '-')}")
        print(f"  consumer_card:\n{r.get('consumer_card', '')[:200]}")
