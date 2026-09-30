# -*- coding: utf-8 -*-
"""
AI 안전성 검증 - 할루시네이션(근거-데이터 일치성) 검증 스크립트

검증 대상: src/pipeline.py의 reason_text/reason_keywords 생성 로직이
실제 계산된 weighted_aspect_scores / matched_allergen과 항상 일치하는지.

학습된 DeepFM 모델(models/deepfm/)이 없어도 실행 가능하다 — reason_text 생성은
DeepFM 스코어링과 무관하게 weighted_aspect_scores / allergy 판정만으로 결정되기 때문.
(recommend_for_pet()의 5)번 "추천 사유 생성" 블록만 별도로 재현해서 검사한다.
원본 로직이 바뀌면 이 스크립트의 REPLICATED 표시 부분도 같이 갱신해야 한다.)

검사 항목:
1) reason_keywords에 들어간 aspect가 실제로 양수 weighted score를 갖는가
   (음수/0점인데 "좋다"고 적히면 안 됨)
2) reason_keywords의 모든 문구가 reason_text 안에 실제로 포함되는가
3) allergy_status가 PENALIZED일 때 matched_allergen 성분명이 reason_text에
   실제로 언급되는가 (엉뚱한 성분명이 나오면 안 됨)

실행: (endtoend 프로젝트 루트에서)
    python3 scripts/hallucination_verification_tests.py

결과는 콘솔에 출력되고 scripts/hallucination_test_results.json 으로도 저장된다.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.pipeline as pipeline
from src.recommend.reviewer_profile_similarity import compute_weighted_aspect_scores
from src.recommend.allergy_filter import check_allergy_conflict
from data.dummy.dummy_data import PET_PROFILES, PRODUCTS

KO_TO_CODE = {v: k for k, v in pipeline.ASPECT_KO_NAMES.items()}

mismatches = []
checked = 0

reviews_by_product = pipeline.build_reviews_with_ratings()

for pet in PET_PROFILES:
    for product in PRODUCTS:
        # pipeline.recommend_for_pet()과 동일하게 species 불일치는 애초에 후보가 아니므로 스킵
        if pet["species"] not in product["target_species"]:
            continue

        checked += 1
        case_id = f"{pet['pet_id']} x {product['product_id']}"

        # --- 알레르기 판정 (pipeline.recommend_for_pet() 2)번과 동일 로직) ---
        allergen_flags = product.get("allergen_flags")
        if allergen_flags is None:
            allergy_status = "PENDING"
            matched_allergen = []
        else:
            allergy_result = check_allergy_conflict(pet["allergy_codes"], allergen_flags)
            if allergy_result["has_conflict"]:
                allergy_status = "PENALIZED"
                matched_allergen = allergy_result["matched_allergen"]
            else:
                allergy_status = "SAFE"
                matched_allergen = []

        # --- aspect 가중 점수 계산 (3번과 동일) ---
        product_reviews = reviews_by_product.get(product["product_id"], [])
        weighted_result = compute_weighted_aspect_scores(pet, product_reviews)
        weighted_scores = weighted_result["weighted_aspect_scores"]

        # --- 추천 사유 생성 (5번과 동일 — REPLICATED from pipeline.recommend_for_pet) ---
        reason_keywords, reason_text = pipeline.build_reason_from_weighted_scores(weighted_result)
        if allergy_status == "PENALIZED":
            reason_keywords = [f"알러지 성분 포함: {', '.join(matched_allergen)}"] + reason_keywords
            reason_text = f"{', '.join(matched_allergen)} 성분이 포함되어 있어 등록하신 알러지 정보와 맞지 않을 수 있어요. " + reason_text
        elif allergy_status == "PENDING":
            reason_text = "성분 정보 확인 중인 상품이에요. " + reason_text

        # === 검사 1: aspect 키워드가 실제로 양수 점수인가 ===
        for kw in reason_keywords:
            if kw.startswith("알러지 성분 포함"):
                continue
            code = KO_TO_CODE.get(kw)
            if code is None:
                mismatches.append({"case": case_id, "type": "unknown_keyword", "keyword": kw})
                continue
            score = weighted_scores.get(code, 0.0)
            if not (score > 0):
                mismatches.append({
                    "case": case_id, "type": "keyword_not_positive",
                    "keyword": kw, "code": code, "actual_score": score,
                })

        # === 검사 2: reason_keywords가 reason_text에 실제로 포함되는가 ===
        for kw in reason_keywords:
            plain_kw = kw.split(": ")[-1] if kw.startswith("알러지 성분 포함") else kw
            if plain_kw not in reason_text:
                mismatches.append({
                    "case": case_id, "type": "keyword_missing_from_text",
                    "keyword": kw, "reason_text": reason_text,
                })

        # === 검사 3: PENALIZED일 때 matched_allergen이 reason_text에 언급되는가 ===
        if allergy_status == "PENALIZED":
            joined = ", ".join(matched_allergen)
            if joined not in reason_text:
                mismatches.append({
                    "case": case_id, "type": "allergy_text_mismatch",
                    "matched_allergen": matched_allergen, "reason_text": reason_text,
                })

print(f"검사 조합(종 불일치 제외): {checked}건")
print(f"불일치: {len(mismatches)}건")
for m in mismatches:
    print(" -", m)

out_path = Path(__file__).resolve().parent / "hallucination_test_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"checked": checked, "mismatches": mismatches}, f, ensure_ascii=False, indent=2)

print(f"\n전체 결과 JSON 저장: {out_path}")