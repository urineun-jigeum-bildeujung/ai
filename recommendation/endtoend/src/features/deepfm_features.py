# -*- coding: utf-8 -*-
"""
DeepFM 입력 feature 설계.

- 유저(반려동물)측: 온보딩 필드 (species/breed/sex/age/weight/bcs/allergy/concerns)
- 아이템(상품)측: 상품 메타 + aspect_keyword_dict 기반 리뷰 집계 feature

[버그 수정 이력]
bcs(신체조건점수, 1~5)와 age_group(GROWTH/ADULT/SENIOR)은 순서가 있는(ordinal) 값인데,
지금까지 sparse(카테고리) feature로만 다뤄서 species/breed 같은 순서 없는 값과
동일하게 취급되고 있었다. 즉 모델 입장에서 bcs=3과 bcs=4가 "가깝다"는 것을
전혀 알 수 없는 상태였다.

수정: sparse 표현은 그대로 유지하되(비선형 패턴 포착 가능성 남겨둠),
0~1로 정규화한 숫자값을 dense feature로 추가해서 순서/거리 정보를 모델에
직접 알려주도록 개선했다. (예: bcs_norm, age_group_ordinal)
"""

from datetime import date
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "data", "masters"))
from allergen_master import ALLERGEN_VOCAB  # 확정 마스터 120개 전체
from concern_master import CONCERN_VOCAB    # 확정 마스터 95개 전체

SPECIES_VOCAB = ["DOG", "CAT"]
SEX_VOCAB = ["MALE", "FEMALE"]
AGE_GROUP_VOCAB = ["GROWTH", "ADULT", "SENIOR"]  # 순서 있음 -- ordinal 인코딩 대상
BREED_SIZE_VOCAB = ["SMALL", "MEDIUM", "LARGE"]
CATEGORY_VOCAB = ["FOOD", "SUPPLEMENT", "TREAT"]

ASPECT_CODES = [
    "palatability",
    "digestion",
    "skin_coat",
    "vitality_weight",
    "allergic_reaction",
    "price_value",
]

BCS_MIN, BCS_MAX = 1, 5


def _calc_age_group(birth_date_str: str) -> str:
    birth = date.fromisoformat(birth_date_str)
    today = date.today()
    months = (today.year - birth.year) * 12 + (today.month - birth.month)
    if months < 12:
        return "GROWTH"
    elif months < 84:
        return "ADULT"
    else:
        return "SENIOR"


def _calc_age_months(birth_date_str: str) -> int:
    birth = date.fromisoformat(birth_date_str)
    today = date.today()
    return (today.year - birth.year) * 12 + (today.month - birth.month)


def _calc_breed_size(weight: float) -> str:
    if weight < 10:
        return "SMALL"
    elif weight < 25:
        return "MEDIUM"
    else:
        return "LARGE"


def _normalize(value: float, min_v: float, max_v: float) -> float:
    if max_v == min_v:
        return 0.0
    return round((value - min_v) / (max_v - min_v), 4)


def _ordinal_normalize(value: str, order: list) -> float:
    """
    순서형 카테고리 값을 0~1 사이 숫자로 변환.
    예: AGE_GROUP_VOCAB=["GROWTH","ADULT","SENIOR"] 일 때
        GROWTH -> 0.0, ADULT -> 0.5, SENIOR -> 1.0
    값이 리스트에 없으면 중간값(0.5)으로 처리 (알 수 없는 값에 대한 안전한 기본값).
    """
    if value not in order or len(order) <= 1:
        return 0.5
    idx = order.index(value)
    return round(idx / (len(order) - 1), 4)


def build_pet_features(pet: dict) -> dict:
    """
    pet_profile -> DeepFM 유저측 feature.
    sparse: 인덱스/카테고리 값 그대로 반환 (실제 모델단에서 임베딩 레이어가 처리)
    dense: 0~1 정규화된 수치 (bcs_norm, age_group_ordinal 포함 -- 순서 정보 명시)
    multi_hot: 알러지처럼 여러 값 가능한 필드 -> 0/1 벡터
    """
    age_months = _calc_age_months(pet["birth_date"])
    age_group = _calc_age_group(pet["birth_date"])
    breed_size = _calc_breed_size(pet["weight"])

    allergy_multi_hot = [1 if code in pet.get("allergy_codes", []) else 0 for code in ALLERGEN_VOCAB]
    concern_multi_hot = [1 if code in pet.get("concerns", []) else 0 for code in CONCERN_VOCAB]

    return {
        "sparse": {
            "species": pet["species"],
            "breed": pet["breed"],  # 실제로는 품종 마스터 코드로 매핑 필요
            "sex": pet.get("sex"),
            "age_group": age_group,      # 카테고리 표현은 그대로 유지
            "breed_size": breed_size,
            "bcs": pet["bcs"],            # 카테고리 표현은 그대로 유지 (1~5, ordinal sparse)
        },
        "dense": {
            "age_months_norm": _normalize(age_months, 0, 180),  # 0~15세 가정
            "weight_norm": _normalize(pet["weight"], 0, 50),    # 0~50kg 가정
            "neutered": 1.0 if pet.get("neutered") else 0.0,
            # --- 순서 정보를 명시적으로 담은 신규 dense feature ---
            "bcs_norm": _normalize(pet["bcs"], BCS_MIN, BCS_MAX),
            "age_group_ordinal": _ordinal_normalize(age_group, AGE_GROUP_VOCAB),
        },
        "multi_hot": {
            "allergy_codes": allergy_multi_hot,   # ALLERGEN_VOCAB(120) 순서와 매칭
            "concerns": concern_multi_hot,        # CONCERN_VOCAB(95) 순서와 매칭
        },
    }


def build_product_aspect_features(product_review_summary: dict) -> dict:
    """
    aspect_keyword_dict 기반 리뷰 집계 -> dense feature 6개.

    두 가지 입력 형태를 지원한다:
    1) 기존 방식(카운트 기반, pet과 무관하게 상품 전체 리뷰 단순 집계):
       {"positive_tags": [...], "negative_tags": [...], "total_reviews": int}
    2) 신규 방식(유사도 가중, pet마다 다른 값 -- reviewer_profile_similarity.compute_weighted_aspect_scores 결과):
       {"weighted_aspect_scores": {tag_string: 0~1 가중 점수, ...}, ...}
       주의: compute_weighted_aspect_scores의 출력 키는 aspect_code가 아니라
       "기호성 좋음" 같은 원본 태그 문자열이므로, 아래 aspect_tag_map으로 변환한 뒤
       (긍정 태그 가중 점수) - (부정 태그 가중 점수)를 최종 -1~1 점수로 사용한다.
    """
    aspect_tag_map = {
        "palatability": ("기호성 좋음", "기호성 낮음"),
        "digestion": ("소화 잘됨", "소화 불편 후기 있음"),
        "skin_coat": ("피부·모질 개선", "피부 트러블 후기 있음"),
        "vitality_weight": ("활력 개선", "체중·활력 저하 후기 있음"),
        "allergic_reaction": ("알러지 반응 없음(후기)", "알러지 반응 있음(후기)"),
        "price_value": ("가성비 좋음", "가격 부담 후기 있음"),
    }

    if "weighted_aspect_scores" in product_review_summary:
        weighted = product_review_summary["weighted_aspect_scores"]
        dense = {}
        for aspect_code, (pos_tag, neg_tag) in aspect_tag_map.items():
            pos_score = weighted.get(pos_tag, 0.0)  # 존재하면 +1.0 근처 (긍정 방향)
            neg_score = weighted.get(neg_tag, 0.0)  # 존재하면 -1.0 근처 (이미 음수로 나옴)
            # 주의: neg_score가 이미 음수이므로 덧셈으로 합쳐야 함 (뺄셈하면 부호가 이중 반영되는 버그 발생)
            combined = pos_score + neg_score
            dense[f"{aspect_code}_score"] = round(max(-1.0, min(1.0, combined)), 4)
        return dense

    # 기존 방식: 상품 전체 리뷰 단순 집계 (pet과 무관, 폴백용)
    positive_tags = product_review_summary.get("positive_tags", [])
    negative_tags = product_review_summary.get("negative_tags", [])
    total_reviews = product_review_summary.get("total_reviews", 1) or 1

    dense = {}
    for aspect_code, (pos_tag, neg_tag) in aspect_tag_map.items():
        pos_count = positive_tags.count(pos_tag) if pos_tag in positive_tags else 0
        neg_count = negative_tags.count(neg_tag) if neg_tag in negative_tags else 0
        score = round((pos_count - neg_count) / total_reviews, 4)
        dense[f"{aspect_code}_score"] = score

    return dense


def _calc_age_fit_score(pet_age_group: str, product_target_age_group) -> float:
    """
    반려동물의 실제 생애주기와 상품의 타겟 생애주기가 얼마나 잘 맞는지 계산.
    완전 일치=1.0, 한 단계 차이=0.5, 두 단계 이상 차이=0.0.
    상품에 타겟 생애주기가 지정되지 않은 경우("전 연령 대상")는 제한이 없다는 뜻이므로
    항상 적합하다고 보고 1.0으로 처리한다.
    """
    if not product_target_age_group:
        return 1.0
    if pet_age_group not in AGE_GROUP_VOCAB or product_target_age_group not in AGE_GROUP_VOCAB:
        return 0.5  # 알 수 없는 값에 대한 안전한 기본값 (중립)

    pet_idx = AGE_GROUP_VOCAB.index(pet_age_group)
    product_idx = AGE_GROUP_VOCAB.index(product_target_age_group)
    diff = abs(pet_idx - product_idx)
    max_diff = len(AGE_GROUP_VOCAB) - 1  # 2 (GROWTH~SENIOR)

    return round(1.0 - diff / max_diff, 4)


def build_product_features(product: dict, product_review_summary: dict) -> dict:
    """
    product_master + aspect 리뷰 집계 -> DeepFM 아이템측 feature.

    target_age_group은 sparse(카테고리)로도 유지하되, 반려동물의 실제 생애주기와
    비교한 "적합도" 자체는 pet 정보가 함께 있어야 계산 가능하므로
    build_interaction_features()에서 age_fit_score로 별도 추가한다.
    """
    aspect_dense = build_product_aspect_features(product_review_summary)

    return {
        "sparse": {
            "category_code": product["category_code"],
            "subcategory_code": product["subcategory_code"],
            "target_breed_size": product.get("target_breed_size"),
            "target_age_group": product.get("target_age_group"),
        },
        "multi_hot": {
            "target_species": [1 if s in product["target_species"] else 0 for s in SPECIES_VOCAB],
            "allergen_flags": [1 if a in product.get("allergen_flags", []) else 0 for a in ALLERGEN_VOCAB],
        },
        "dense": {
            "price_norm": _normalize(product["price"], 0, 100000),
            **aspect_dense,
        },
    }


def build_interaction_features(pet: dict, product: dict, product_review_summary: dict) -> dict:
    """
    유저측 + 아이템측 feature를 합쳐 하나의 학습 샘플 형태로 반환.
    age_fit_score는 pet과 product 정보가 둘 다 필요해서 이 단계에서 계산해
    product_features.dense에 추가한다.
    """
    pet_features = build_pet_features(pet)
    product_features = build_product_features(product, product_review_summary)

    pet_age_group = _calc_age_group(pet["birth_date"])
    age_fit_score = _calc_age_fit_score(pet_age_group, product.get("target_age_group"))
    product_features["dense"]["age_fit_score"] = age_fit_score

    return {
        "pet_id": pet["pet_id"],
        "product_id": product["product_id"],
        "pet_features": pet_features,
        "product_features": product_features,
        "label": None,
    }