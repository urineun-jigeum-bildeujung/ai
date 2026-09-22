"""
매칭 알고리즘 v1 — 표준사료 검색 메인 (규칙 기반)
- 입력: 페르소나 dict + 사료 dict (영양소 + 성분표)
- 출력: 매칭 점수 (0~100) + 부족/적정/과다/위험 카드 + 역추천 사유 + 알레르기 자동 제외

사용법:
    python3 ai/scripts/match_v1.py --persona maru --feed test_feed_a

    # 시드 3 + 시드 1 + 시드 4 자동 로드
    # 매칭 결과 JSON 출력
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

from nutrition.allergen_service import evaluate_safety


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = PROJECT_ROOT / "data" / "raw"


def load_seed(name: str) -> dict:
    p = SEED_DIR / name
    return json.loads(p.read_text(encoding="utf-8"))


def get_persona(seed_persona: dict, persona_id: str) -> dict:
    for p in seed_persona["personas"]:
        if p["id"] == persona_id:
            return p
    raise ValueError(f"persona {persona_id} not found")


def match_allergens(feed_ingredients: list[str], seed_feed_codes: dict, persona: dict) -> dict:
    """알레르기 exact-match hard filter.

    Product ingredients and dictionary synonyms are normalized then compared for
    equality only.  Substring matching (for example ``chicken`` in an unrelated
    longer token) must not create an allergy conflict.
    """
    # seed_feed_codes stays in this signature as a compatibility adapter only.
    profile_status = persona.get("allergy_profile_status") or ("KNOWN_LIST" if persona.get("allergies") else "KNOWN_NONE")
    safety = evaluate_safety({**persona, "allergy_profile_status": profile_status},
                             {"id": "legacy", "category": "treat", "ingredient_list": feed_ingredients})
    refs = safety.get("product_allergen_refs", [])
    matches = [{"ingredient": r["raw_text"], "matched_synonym": r.get("matched_text"), "allergy_id": r["allergen_code"]}
               for r in refs if r.get("allergen_code") in safety.get("exclude_reasons", [])]
    return {"excluded": safety["excluded"], "matches": matches,
            "unmapped_ingredients": safety.get("unmapped_ingredients", []),
            "reason": "알레르기 성분 포함 — 추천 안 함" if safety["excluded"] else "알레르기 성분 없음",
            "safety": safety}


def match_nutrients(feed: dict, persona: dict, seed_nutrition: dict) -> dict:
    """영양소별 NIAS 권장 범위 비교"""
    ref = seed_nutrition["reference_tables"]
    if persona["species"] == "dog":
        life_stage = persona.get("life_stage", "adult")
        if life_stage == "puppy":
            table = ref.get("growing_dog_per_dm_100g_NIAS", ref["adult_dog_per_dm_100g_NIAS"])
        else:
            table = ref["adult_dog_per_dm_100g_NIAS"]
    else:  # cat
        if persona.get("life_stage") == "kitten":
            table = ref["growing_cat_per_dm_100g_NIAS"]
        else:
            table = ref["adult_cat_per_dm_100g_NIAS"]

    results: list[dict] = []
    score_sum = 0
    score_count = 0

    for nutrient_key, nutrient_value in table.items():
        if nutrient_key.startswith("_"):
            continue
        # 사료에서 해당 영양소 값 찾기
        feed_val = feed.get(nutrient_key)
        if feed_val is None:
            continue

        # 권장량 비교
        if isinstance(nutrient_value, (int, float)):
            min_val = nutrient_value
            # PR-B-2: 1.5x fallback 제거. max_state UNDEFINED 시 상한 비교 안 함.
            max_val = table.get("_max_safe", {}).get(nutrient_key)
            max_state = "DEFINED" if max_val is not None else "UNDEFINED"

            # 부족 / 적정 / 과다 / 위험 (PR-B-2: 영문 status 통일)
            ratio = feed_val / min_val if min_val else 0
            if ratio < 0.9:
                status = "DEFICIENT"
                score = max(0, int(ratio * 100))
            elif max_state == "DEFINED" and ratio > (max_val / min_val):
                status = "HAZARD"
                score = 0
            elif max_state == "DEFINED" and ratio > 1.1:
                status = "EXCESS"
                score = 70
            else:
                # min 충족, max_state == UNDEFINED 이거나 max 미만
                status = "ADEQUATE"
                score = 100

            results.append({
                "nutrient": nutrient_key,
                "feed_value": feed_val,
                "min": min_val,
                "max": max_val,  # None if UNDEFINED
                "min_state": "DEFINED",
                "max_state": max_state,
                "status": status,
                "score": score,
                "ratio": round(ratio, 2),
            })
            score_sum += score
            score_count += 1

    overall_score = round(score_sum / score_count, 1) if score_count else 0
    return {
        "overall_score": overall_score,
        "nutrients_checked": score_count,
        # PR-B-2: status 영문 통일 (DEFICIENT/ADEQUATE/EXCESS/HAZARD)
        "deficient": [r for r in results if r["status"] == "DEFICIENT"],
        "adequate": [r for r in results if r["status"] == "ADEQUATE"],
        "excess": [r for r in results if r["status"] == "EXCESS"],
        "hazard": [r for r in results if r["status"] == "HAZARD"],
        # PR-B-2: max_state 분포 (호환용)
        "max_state": {
            "DEFINED": sum(1 for r in results if r["max_state"] == "DEFINED"),
            "UNDEFINED": sum(1 for r in results if r["max_state"] == "UNDEFINED"),
        },
        "details": results,
    }


def generate_consumer_card(persona: dict, feed: dict, allergen: dict, nutrient: dict) -> str:
    """초보 보호자용 쉬운 용어 결과 카드 (P0)"""
    lines = []
    lines.append(f"=== {persona['label']} — {feed.get('product_name', '이 사료')} ===")
    lines.append(f"매칭 점수: {nutrient['overall_score']} / 100")
    lines.append("")

    if allergen["excluded"]:
        lines.append(f"[제외] {allergen['reason']}")
        return "\n".join(lines)

    if nutrient["hazard"]:
        lines.append("[위험 — 추천 안 함]")
        for h in nutrient["hazard"]:
            lines.append(f"  - {h['nutrient']}: {h['feed_value']} (안전 최대 {h['max']} 초과)")
        return "\n".join(lines)

    if nutrient["deficient"]:
        lines.append("[부족 영양소 — 참고]")
        for d in nutrient["deficient"]:
            lines.append(f"  - {d['nutrient']}: {d['feed_value']} (권장 {d['min']} 이상, 부족 {d['ratio']*100:.0f}%)")
        lines.append("")

    if nutrient["excess"]:
        lines.append("[과다 영양소 — 참고]")
        for e in nutrient["excess"]:
            lines.append(f"  - {e['nutrient']}: {e['feed_value']} (권장 대비 {e['ratio']*100:.0f}%)")
        lines.append("")

    if nutrient["overall_score"] >= 80:
        lines.append(f"[결론] 우리 아이에게 잘 맞는 사료로 보여요 (신뢰도 {nutrient['overall_score']}%)")
    elif nutrient["overall_score"] >= 60:
        lines.append(f"[결론] 보통이에요 — 부족한 영양소를 다른 사료로 보완해 보세요 (신뢰도 {nutrient['overall_score']}%)")
    else:
        lines.append(f"[결론] 신뢰도 낮음 ({nutrient['overall_score']}%) — 수의사 선생님과 상담 권장")

    return "\n".join(lines)


def run_match(persona_id: str, feed: dict) -> dict:
    seed_persona = load_seed("seed_persona5_nutrient_map.json")
    seed_nutrition = load_seed("seed_nutrition_standard.json")
    seed_feed_codes = load_seed("seed_feed_codes.json")

    persona = get_persona(seed_persona, persona_id)
    ingredients = feed.get("ingredient_list", [])
    allergens = match_allergens(ingredients, seed_feed_codes, persona)
    nutrients = match_nutrients(feed, persona, seed_nutrition)

    card = generate_consumer_card(persona, feed, allergens, nutrients)

    return {
        "persona": persona["label"],
        "feed": feed.get("product_name", "(unknown)"),
        "overall_score": nutrients["overall_score"],
        "allergen_check": allergens,
        "nutrient_check": {
            "checked": nutrients["nutrients_checked"],
            "deficient_count": len(nutrients["deficient"]),
            "excess_count": len(nutrients["excess"]),
            "hazard_count": len(nutrients["hazard"]),
        },
        "consumer_card": card,
    }


# ===== 테스트 시드 (수동) =====
TEST_FEED_GOOD = {
    "product_name": "프리미엄 닭고기 사료 (테스트 A)",
    "ingredient_list": ["닭고기", "쌀", "비트펄프", "오메가3"],
    "단백질_g": 26.0,
    "지방_g": 14.0,
    "칼슘_g": 1.2,
    "인_g": 0.9,
    "나트륨_g": 0.30,
    "리놀레산_오메가6_g": 1.5,
    "아연_mg": 8.0,
    "비타민E_IU": 5.0,
}

TEST_FEED_BAD_FOR_COCO = {
    "product_name": "치킨 사료 (코코 알레르기 위험)",
    "ingredient_list": ["닭고기", "닭기름", "쌀", "옥수수"],
    "단백질_g": 24.0,
    "지방_g": 12.0,
    "칼슘_g": 1.0,
    "인_g": 0.8,
    "나트륨_g": 0.40,
    "리놀레산_오메가6_g": 1.0,
    "아연_mg": 6.0,
    "비타민E_IU": 3.0,
}

TEST_FEED_LOW_QUALITY = {
    "product_name": "저가 사료 (영양 부족)",
    "ingredient_list": ["옥수수", "밀", "대두"],
    "단백질_g": 14.0,
    "지방_g": 6.0,
    "칼슘_g": 0.4,
    "인_g": 0.3,
    "나트륨_g": 0.20,
    "리놀레산_오메가6_g": 0.5,
    "아연_mg": 4.0,
    "비타민E_IU": 2.0,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", default="maru")
    parser.add_argument("--feed", default="good", choices=["good", "bad_allergy", "low_quality"])
    parser.add_argument("--json", action="store_true", help="JSON 출력만")
    args = parser.parse_args()

    feed_map = {
        "good": TEST_FEED_GOOD,
        "bad_allergy": TEST_FEED_BAD_FOR_COCO,
        "low_quality": TEST_FEED_LOW_QUALITY,
    }
    feed = feed_map[args.feed]
    result = run_match(args.persona, feed)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["consumer_card"])
        print()
        print(json.dumps({
            "overall_score": result["overall_score"],
            "allergen_excluded": result["allergen_check"]["excluded"],
            "nutrient_check": result["nutrient_check"],
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
