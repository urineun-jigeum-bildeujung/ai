# -*- coding: utf-8 -*-
"""
생성된 리뷰에 반려동물 속성/aspect 표현을 확률적으로 삽입.

LM(파인튜닝된 언어모델)은 "자연스러운 리뷰 문체"를 담당하고,
이 모듈은 review_features 추출 대상이 되는 구체적인 정보
(품종/나이/체중/알러지반응/관심건강정보/aspect 표현)를
정해진 확률로 리뷰에 자연스럽게 끼워넣는 역할을 한다.

각 리뷰가 실제로 어떤 속성을 포함하게 됐는지 ground truth로 함께 기록해서,
나중에 review_features 추출 로직(KcELECTRA, aspect tagging 등)의 정확도를
검증하는 데 사용할 수 있게 한다.

[버그 수정 이력]
- 나이(age) 삽입 시 "살"과 "개월" 템플릿이 같은 정수값을 공유해서,
  ground_truth에 단위 구분 없이 숫자만 저장되던 문제를 수정.
  review_features.extracted_age 스키마 정의(개월 수)에 맞춰,
  "살" 템플릿이 뽑히면 ground_truth["age"]를 *12 해서 개월로 통일 기록.
"""

import random
import sys
import os

sys.path.append(os.path.dirname(__file__))
from allergen_master import ALLERGEN_MASTER
from concern_master import CONCERN_MASTER
from allergen_ko_names import ALLERGEN_KO_NAMES
from concern_ko_names import CONCERN_KO_NAMES
from breed_master import DOG_BREEDS, CAT_BREEDS


# -----------------------------
# aspect 표현 (표 2 그대로 -- 자연스러운 완결형 문장)
# tagging.py의 ASPECTS는 규칙매칭용 짧은 단서라 여기서는 별도로
# "실제 문장으로 쓸 수 있는 완전한 표현"을 직접 정의한다.
# -----------------------------
ASPECT_PHRASES = {
    "palatability": {
        "positive": ["잘 먹어요", "기호성 좋아요", "그릇을 싹 비워요", "밥 시간을 기다려요"],
        "negative": ["안 먹어요", "냄새만 맡고 안 먹어요", "며칠 먹다 안 먹어요", "입맛에 안 맞나봐요"],
    },
    "digestion": {
        "positive": ["대변 상태 좋아졌어요", "소화 잘 시켜요", "변 냄새 줄었어요", "배변량 적당해요"],
        "negative": ["설사했어요", "구토했어요", "무른 변을 봐요", "속이 안 좋아 보여요"],
    },
    "skin_coat": {
        "positive": ["털에 윤기가 나요", "피부가 좋아졌어요", "가려워하지 않아요", "털빠짐이 줄었어요"],
        "negative": ["가려워해요", "털이 푸석해요", "피부 트러블 생겼어요", "털빠짐이 심해졌어요"],
    },
    "vitality_weight": {
        "positive": ["활력이 넘쳐요", "체중 관리에 도움돼요", "적정 체중 유지돼요", "활발해졌어요"],
        "negative": ["살이 쪘어요", "살이 너무 빠졌어요", "기운이 없어 보여요", "무기력해 보여요"],
    },
    "allergic_reaction": {
        "positive": ["알러지 반응 없었어요", "예민한 아이인데 잘 맞아요", "이상 반응 없이 잘 먹어요"],
        "negative": ["알러지 반응 있었어요", "먹고 나서 두드러기가 났어요", "예민한 아이한텐 안 맞아요"],
    },
    "price_value": {
        "positive": ["가성비 좋아요", "가격 대비 만족해요", "이 정도면 합리적이에요"],
        "negative": ["가격이 비싸요", "가성비가 아쉬워요", "이 가격이면 다른 걸 사겠어요"],
    },
}

# -----------------------------
# 삽입 확률 설정
# -----------------------------
# "모든 리뷰에 다 들어가지 않고, 일정 비율로만 들어가야 한다"는 요구사항 반영.
# 실제 리뷰에서 관찰되는 대략적인 언급 빈도를 감안한 초기값 -- 생성 결과 보고 조정 가능.
INJECTION_PROB = {
    "breed_age_weight": 0.4,   # 품종/나이/체중 언급
    "allergy_reaction": 0.15,  # 알러지 반응 언급 (실제 리뷰에서도 흔하진 않음)
    "concern": 0.2,            # 관심 건강정보 언급
    "aspect_phrase": 0.6,      # aspect 표현 (기호성/소화 등) -- 실제 리뷰에서 가장 흔한 유형이라 비율 높게
}

# 나이 템플릿: "살"과 "개월"을 명확히 분리 (버그 수정 핵심)
AGE_YEAR_TEMPLATES = ["{age}살인데", "우리 애가 {age}살이라"]
AGE_MONTH_TEMPLATES = ["{age}개월인데"]

WEIGHT_TEMPLATES = ["{weight}kg 정도인데", "체중이 {weight}kg인데"]
BREED_TEMPLATES = ["{breed}인데", "저희 집 {breed}가", "{breed} 키우는데"]

ALLERGY_TEMPLATES = [
    "예전에 {allergen} 먹고 반응 있었는데 이건 괜찮았어요",
    "{allergen} 알러지가 있어서 걱정했는데 문제없었어요",
    "{allergen} 성분 때문에 예민한 편인데 이번엔 괜찮네요",
]

CONCERN_TEMPLATES = [
    "{concern} 때문에 걱정했는데 도움이 되는 것 같아요",
    "{concern} 있는 아이라 신경 쓰이는데",
    "{concern} 관리하려고 알아보다가 샀어요",
]


def _flatten_allergens() -> list:
    """(코드, 한국어명) 튜플 리스트로 반환."""
    codes = [item for items in ALLERGEN_MASTER.values() for item in items]
    return [(code, ALLERGEN_KO_NAMES.get(code, code)) for code in codes]


def _flatten_concerns() -> list:
    """(코드, 한국어명) 튜플 리스트로 반환."""
    codes = [item for items in CONCERN_MASTER.values() for item in items]
    return [(code, CONCERN_KO_NAMES.get(code, code)) for code in codes]


ALL_ALLERGENS = _flatten_allergens()
ALL_CONCERNS = _flatten_concerns()


def inject_attributes(base_text: str, species: str, sentiment: str, seed: int = None) -> dict:
    """
    base_text: LM이 생성한 원본 리뷰 텍스트
    species: "DOG" 또는 "CAT" (품종 선택에 사용)
    sentiment: "POSITIVE" 또는 "NEGATIVE" (aspect 표현 방향 결정에 사용)

    반환:
    {
        "final_text": 속성이 삽입된 최종 텍스트,
        "ground_truth": {
            "breed": str or None,
            "age": int or None,       # 항상 "개월 수"로 통일해서 기록 (스키마 정의 준수)
            "weight": float or None,
            "allergen_mentioned": str or None,
            "concern_mentioned": str or None,
            "aspect_mentioned": [(aspect_code, sentiment), ...],
        }
    }
    """
    rng = random.Random(seed)
    clauses = []
    ground_truth = {
        "breed": None, "age": None, "weight": None,
        "allergen_mentioned": None, "concern_mentioned": None,
        "aspect_mentioned": [],
    }

    breed_pool = DOG_BREEDS if species == "DOG" else CAT_BREEDS

    # 1) 품종/나이/체중
    if rng.random() < INJECTION_PROB["breed_age_weight"]:
        breed = rng.choice(breed_pool)
        age_years = rng.randint(1, 14)
        weight = round(rng.uniform(2.0, 30.0), 1)

        choice = rng.choice(["breed", "age", "weight", "breed_age"])
        if choice == "breed":
            clauses.append(rng.choice(BREED_TEMPLATES).format(breed=breed))
            ground_truth["breed"] = breed
        elif choice == "age":
            # "살"/"개월" 템플릿 중 무엇이 뽑혔는지에 따라 ground_truth를 개월 수로 통일해서 기록
            # (extracted_age 스키마 정의가 "개월 수"이므로, 문장 표현 단위와 무관하게 항상 개월로 저장)
            use_month_template = rng.random() < 0.5
            if use_month_template:
                age_months_for_text = age_years  # "개월" 문구에는 그대로 표시 (1~14개월)
                clauses.append(rng.choice(AGE_MONTH_TEMPLATES).format(age=age_months_for_text))
                ground_truth["age"] = age_months_for_text
            else:
                clauses.append(rng.choice(AGE_YEAR_TEMPLATES).format(age=age_years))
                ground_truth["age"] = age_years * 12
        elif choice == "weight":
            clauses.append(rng.choice(WEIGHT_TEMPLATES).format(weight=weight))
            ground_truth["weight"] = weight
        else:
            clauses.append(f"{age_years}살 {breed}인데")
            ground_truth["breed"] = breed
            ground_truth["age"] = age_years * 12

    # 2) 알러지 반응 언급
    if rng.random() < INJECTION_PROB["allergy_reaction"]:
        allergen_code, allergen_ko = rng.choice(ALL_ALLERGENS)
        clauses.append(rng.choice(ALLERGY_TEMPLATES).format(allergen=allergen_ko))
        ground_truth["allergen_mentioned"] = allergen_code

    # 3) 관심 건강정보 언급
    if rng.random() < INJECTION_PROB["concern"]:
        concern_code, concern_ko = rng.choice(ALL_CONCERNS)
        clauses.append(rng.choice(CONCERN_TEMPLATES).format(concern=concern_ko))
        ground_truth["concern_mentioned"] = concern_code

    # 4) aspect 표현 -- sentiment에 맞는 방향의 표현을 우선 사용
    if rng.random() < INJECTION_PROB["aspect_phrase"]:
        aspect_code = rng.choice(list(ASPECT_PHRASES.keys()))
        direction = sentiment.lower()  # "positive" or "negative"
        phrase = rng.choice(ASPECT_PHRASES[aspect_code][direction])
        clauses.append(phrase)
        ground_truth["aspect_mentioned"].append((aspect_code, direction))

    # 삽입 문구들을 base_text 앞에 자연스럽게 붙임 (문장 단위로 연결)
    if clauses:
        prefix = " ".join(clauses)
        final_text = f"{prefix} {base_text}".strip()
    else:
        final_text = base_text

    return {"final_text": final_text, "ground_truth": ground_truth}