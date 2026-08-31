# -*- coding: utf-8 -*-
"""
DeepFM 입력 feature 설계 (구체화 버전).

- 유저(반려동물)측: 온보딩 6개 필드 (species/breed/age/weight/bcs/allergy)
- 아이템(상품)측: 상품 메타 + aspect_keyword_dict 기반 리뷰 집계 feature

주의: 이 모듈은 "feature 벡터를 만드는 부분"만 다룬다.
실제 DeepFM 학습에 필요한 label(구매/클릭 등 정답)은 아직 없어서
학습 자체는 이 모듈 범위 밖 — 데이터 쌓이면 별도로 붙인다.
"""

from datetime import date
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "data", "masters"))
from allergen_master import ALLERGEN_VOCAB  # 확정 마스터 120개 전체
from concern_master import CONCERN_VOCAB    # 확정 마스터 95개 전체

SPECIES_VOCAB = ["DOG", "CAT"]
SEX_VOCAB = ["MALE", "FEMALE"]
AGE_GROUP_VOCAB = ["GROWTH", "ADULT", "SENIOR"]
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


def build_pet_features(pet: dict) -> dict:
    """
    pet_profile -> DeepFM 유저측 feature.
    sparse: 인덱스/카테고리 값 그대로 반환 (실제 모델단에서 임베딩 레이어가 처리)
    dense: 0~1 정규화된 수치
    multi_hot: 알러지처럼 여러 값 가능한 필드 -> 0/1 벡터
    """
    age_months = _calc_age_months(pet["birth_date"])
    age_group = _calc_age_group(pet["birth_date"])
    breed_size = _calc_breed_size(pet["weight"])

    allergy_multi_hot = [1 if code in pet["allergy_codes"] else 0 for code in ALLERGEN_VOCAB]
    concern_multi_hot = [1 if code in pet.get("concerns", []) else 0 for code in CONCERN_VOCAB]

    return {
        "sparse": {
            "species": pet["species"],
            "breed": pet["breed"],  # 실제로는 품종 마스터 코드로 매핑 필요
            "sex": pet["sex"],      # MALE/FEMALE
            "age_group": age_group,
            "breed_size": breed_size,
            "bcs": pet["bcs"],  # 1~5, ordinal sparse로 취급
        },
        "dense": {
            "age_months_norm": _normalize(age_months, 0, 180),  # 0~15세 가정
            "weight_norm": _normalize(pet["weight"], 0, 50),    # 0~50kg 가정
            "neutered": 1.0 if pet["neutered"] else 0.0,        # boolean -> dense 0/1
        },
        "multi_hot": {
            "allergy_codes": allergy_multi_hot,   # ALLERGEN_VOCAB(120) 순서와 매칭
            "concerns": concern_multi_hot,        # CONCERN_VOCAB(95) 순서와 매칭
        },
    }


def build_product_aspect_features(product_review_summary: dict) -> dict:
    """
    aspect_keyword_dict 기반 리뷰 집계 -> dense feature 6개.
    product_review_summary: {"positive_tags": [...], "negative_tags": [...], "total_reviews": int}
    각 aspect 별로 (긍정 언급 - 부정 언급) / 전체 리뷰 수 로 -1~1 사이 점수 계산.
    """
    positive_tags = product_review_summary.get("positive_tags", [])
    negative_tags = product_review_summary.get("negative_tags", [])
    total_reviews = product_review_summary.get("total_reviews", 1) or 1

    # aspect_code 별 긍/부정 태그 매핑 (aspect_tagging.py의 tag_positive/tag_negative와 매칭)
    aspect_tag_map = {
        "palatability": ("기호성 좋음", "기호성 낮음"),
        "digestion": ("소화 잘됨", "소화 불편 후기 있음"),
        "skin_coat": ("피부·모질 개선", "피부 트러블 후기 있음"),
        "vitality_weight": ("활력 개선", "체중·활력 저하 후기 있음"),
        "allergic_reaction": ("알러지 반응 없음(후기)", "알러지 반응 있음(후기)"),
        "price_value": ("가성비 좋음", "가격 부담 후기 있음"),
    }

    dense = {}
    for aspect_code, (pos_tag, neg_tag) in aspect_tag_map.items():
        pos_count = positive_tags.count(pos_tag) if pos_tag in positive_tags else 0
        neg_count = negative_tags.count(neg_tag) if neg_tag in negative_tags else 0
        score = round((pos_count - neg_count) / total_reviews, 4)
        dense[f"{aspect_code}_score"] = score

    return dense


def build_product_features(product: dict, product_review_summary: dict) -> dict:
    """
    product_master + aspect 리뷰 집계 -> DeepFM 아이템측 feature.
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
            "allergen_flags": [1 if a in product["allergen_flags"] else 0 for a in ALLERGEN_VOCAB],
        },
        "dense": {
            "price_norm": _normalize(product["price"], 0, 100000),
            **aspect_dense,
        },
    }


def build_interaction_features(pet: dict, product: dict, product_review_summary: dict) -> dict:
    """
    유저측 + 아이템측 feature를 합쳐 하나의 학습 샘플 형태로 반환.
    label(구매/클릭 등 정답)은 아직 없음 -> 별도 컬럼으로 비워둠 (추후 결정).
    """
    pet_features = build_pet_features(pet)
    product_features = build_product_features(product, product_review_summary)

    return {
        "pet_id": pet["pet_id"],
        "product_id": product["product_id"],
        "pet_features": pet_features,
        "product_features": product_features,
        "label": None,  # TODO: 구매 이력/합성 주문 데이터 확보 후 결정
    }