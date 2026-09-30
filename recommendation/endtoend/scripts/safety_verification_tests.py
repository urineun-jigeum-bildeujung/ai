"""
AI 안전성 검증 실행 스크립트
- 1) 악의적/비정상 입력값(인젝션성 문자열, 과도한 길이, 특수문자) 내성 테스트
- 2) 필수 필드 누락 크래시 재현 (bcs, price)
- 3) 알레르기 필터 경계 케이스 테스트
- 4) (best-effort) 반려동물 그룹별 피처 구조 편향 점검

실행: (endtoend 프로젝트 루트에서)
    python3 scripts/safety_verification_tests.py

결과는 콘솔에 출력되고, scripts/safety_test_results.json 으로도 저장됩니다.
이 JSON을 AI 안전성 검증 보고서의 TODO 항목에 그대로 채워 넣으면 됩니다.
"""
import copy
import json
import sys
import traceback
from pathlib import Path

# 프로젝트 루트를 import 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dummy.dummy_data import PET_PROFILES, PRODUCTS
from src.features.deepfm_features import build_pet_features, build_product_features
from src.recommend.allergy_filter import check_allergy_conflict

results = {
    "injection": [],
    "missing_field_crash": [],
    "allergy_edge_cases": [],
    "group_feature_check": [],
}


def safe_call(fn, *args, **kwargs):
    try:
        out = fn(*args, **kwargs)
        return {"ok": True, "error": None, "result_repr": repr(out)[:300]}
    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(limit=3),
        }


# ---------------------------------------------------------------------------
# 1) 악의적/비정상 입력값 내성 테스트
# ---------------------------------------------------------------------------
MALICIOUS_STRINGS = {
    "sql_injection": "'; DROP TABLE products; --",
    "script_tag": "<script>alert(1)</script>",
    "very_long": "A" * 100_000,
    "null_byte": "abc\x00def",
    "unicode_rtl_override": "\u202Eevil",
    "empty_string": "",
    "emoji_spam": "\U0001F436" * 5000,
    "format_string": "%s%s%s%s%s",
    "path_traversal": "../../../etc/passwd",
}

base_product = copy.deepcopy(PRODUCTS[0])

for label, payload in MALICIOUS_STRINGS.items():
    product = copy.deepcopy(base_product)
    product["product_name"] = payload
    product["ingredients"] = [payload]
    outcome = safe_call(build_product_features, product, {})
    results["injection"].append({"case": label, "field": "product_name/ingredients", **outcome})

for label, payload in MALICIOUS_STRINGS.items():
    product = copy.deepcopy(base_product)
    product["category_code"] = payload
    outcome = safe_call(build_product_features, product, {})
    results["injection"].append({"case": label, "field": "category_code", **outcome})


# ---------------------------------------------------------------------------
# 2) 필수 필드 누락 크래시 재현 (알려진 버그: bcs, price)
# ---------------------------------------------------------------------------
pet_missing_bcs = copy.deepcopy(PET_PROFILES[0])
pet_missing_bcs["bcs"] = None
results["missing_field_crash"].append({
    "case": "pet.bcs = None",
    **safe_call(build_pet_features, pet_missing_bcs),
})

product_missing_price = copy.deepcopy(base_product)
product_missing_price["price"] = None
results["missing_field_crash"].append({
    "case": "product.price = None",
    **safe_call(build_product_features, product_missing_price, {}),
})

pet_no_bcs_key = copy.deepcopy(PET_PROFILES[0])
del pet_no_bcs_key["bcs"]
results["missing_field_crash"].append({
    "case": "pet 딕셔너리에 'bcs' 키 자체가 없음",
    **safe_call(build_pet_features, pet_no_bcs_key),
})


# ---------------------------------------------------------------------------
# 3) 알레르기 필터 경계 케이스
# ---------------------------------------------------------------------------
allergy_cases = [
    ("정상 겹침", ["CHICKEN", "BEEF"], ["CHICKEN"]),
    ("겹침 없음", ["CHICKEN"], ["BEEF"]),
    ("빈 리스트 - 반려동물 알레르기 없음", [], ["CHICKEN"]),
    ("빈 리스트 - 상품 알레르기 정보 없음", ["CHICKEN"], []),
    ("둘 다 빈 리스트", [], []),
    ("대소문자 불일치", ["chicken"], ["CHICKEN"]),
    ("중복 코드", ["CHICKEN", "CHICKEN"], ["CHICKEN"]),
    ("None 값 포함", ["CHICKEN", None], ["CHICKEN"]),
]
for label, pet_codes, product_codes in allergy_cases:
    outcome = safe_call(check_allergy_conflict, pet_codes, product_codes)
    results["allergy_edge_cases"].append({
        "case": label,
        "pet_allergy_codes": pet_codes,
        "product_allergen_flags": product_codes,
        **outcome,
    })


# ---------------------------------------------------------------------------
# 4) (best-effort) 그룹별 피처 구조 편향 점검
#    * 실제 "추천 점수"가 아닌 "피처 벡터 생성 단계"의 구조적 비대칭만 확인합니다.
#    * 학습된 DeepFM 모델 가중치가 있으면, 이 섹션을 점수 기준 편향 테스트로 확장할 수 있어요.
#      (모델 로딩 방법 알려주면 다음 버전에서 반영할게요)
# ---------------------------------------------------------------------------
for species in ["DOG", "CAT"]:
    pet = copy.deepcopy(PET_PROFILES[0])
    pet["species"] = species
    outcome = safe_call(build_pet_features, pet)
    results["group_feature_check"].append({"case": f"species={species}", **outcome})


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------
def summarize(section_name, items):
    total = len(items)
    failed = [i for i in items if not i.get("ok", True)]
    print(f"\n[{section_name}] {total}건 중 {len(failed)}건 실패/예외")
    for item in items:
        status = "OK " if item.get("ok", True) else "FAIL"
        case = item.get("case")
        extra = f" -> {item.get('error')}" if not item.get("ok", True) else ""
        print(f"  [{status}] {case}{extra}")


for section, items in results.items():
    summarize(section, items)

out_path = Path(__file__).resolve().parent / "safety_test_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n전체 결과 JSON 저장: {out_path}")