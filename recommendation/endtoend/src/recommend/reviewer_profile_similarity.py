# -*- coding: utf-8 -*-
"""
리뷰 작성자 프로필 유사도 기반 가중 스코어링.

콜드스타트 추천 로직: 사용자 프로필 + 리뷰 작성자 프로필 + 리뷰 keywords
-> "나(추천 대상)와 비슷한 반려동물을 가진 사람이 남긴 리뷰"에 더 높은 가중치를 줘서
   상품의 aspect(기호성/소화/피부 등) 반응을 재계산한다.

주의: reviews.pet_id가 아직 스키마에 반영되지 않은 상태라, 이 모듈은
- 더미 pet_profile을 리뷰에 가상으로 연결한 테스트 데이터로 로직을 먼저 검증하고
- reviews.pet_id가 실제로 채워지면 real_review_pet_profiles 딕셔너리 자리에
  실제 DB 조회 결과를 넣기만 하면 되도록 설계했다.
"""

from datetime import date


# -----------------------------
# 유사도 가중치 (초안 -- 실제 데이터로 튜닝 필요)
# -----------------------------
SIMILARITY_WEIGHTS = {
    "species_match": 0.4,      # 종(개/고양이) 일치 여부 -- 가장 중요, 불일치 시 전체 유사도 0
    "breed_size_sim": 0.25,    # 체구(소/중/대형) 유사도
    "age_group_sim": 0.2,      # 생애주기(유아/성견/노령) 유사도
    "allergy_overlap_sim": 0.15,  # 알러지 프로필 겹치는 정도
}

AGE_GROUP_ORDER = ["GROWTH", "ADULT", "SENIOR"]
BREED_SIZE_ORDER = ["SMALL", "MEDIUM", "LARGE"]


def _calc_age_months(birth_date_str: str) -> int:
    birth = date.fromisoformat(birth_date_str)
    today = date.today()
    return (today.year - birth.year) * 12 + (today.month - birth.month)


def _calc_age_group(birth_date_str: str) -> str:
    months = _calc_age_months(birth_date_str)
    if months < 12:
        return "GROWTH"
    elif months < 84:
        return "ADULT"
    else:
        return "SENIOR"


def _calc_breed_size(weight: float) -> str:
    if weight < 10:
        return "SMALL"
    elif weight < 25:
        return "MEDIUM"
    else:
        return "LARGE"


def _ordinal_similarity(value_a: str, value_b: str, order: list) -> float:
    """
    순서가 있는 범주형 값의 유사도.
    같으면 1.0, 한 단계 차이면 0.5, 두 단계 이상 차이면 0.0.
    (예: ADULT-SENIOR는 0.5, GROWTH-SENIOR는 0.0)
    """
    if value_a not in order or value_b not in order:
        return 0.0
    idx_a = order.index(value_a)
    idx_b = order.index(value_b)
    diff = abs(idx_a - idx_b)
    if diff == 0:
        return 1.0
    elif diff == 1:
        return 0.5
    else:
        return 0.0


def _allergy_overlap_similarity(allergy_a: list, allergy_b: list) -> float:
    """
    자카드 유사도(교집합/합집합) 방식.
    둘 다 알러지가 없으면 "같은 상태"로 보아 1.0 (둘 다 특이사항 없음이 일치).
    """
    set_a, set_b = set(allergy_a or []), set(allergy_b or [])
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    intersection = set_a & set_b
    return len(intersection) / len(union)


def compute_profile_similarity(target_pet: dict, reviewer_pet: dict) -> float:
    """
    target_pet: 추천 대상 반려동물의 pet_profile
    reviewer_pet: 리뷰 작성자의 pet_profile

    반환: 0~1 사이 유사도 점수
    """
    # species 불일치는 전체 유사도를 0으로 처리 (다른 종 리뷰는 참고 가치가 크게 떨어짐)
    if target_pet["species"] != reviewer_pet["species"]:
        return 0.0

    species_score = 1.0  # 여기 도달했다는 건 이미 일치한다는 뜻

    target_breed_size = _calc_breed_size(target_pet["weight"])
    reviewer_breed_size = _calc_breed_size(reviewer_pet["weight"])
    breed_size_score = _ordinal_similarity(target_breed_size, reviewer_breed_size, BREED_SIZE_ORDER)

    target_age_group = _calc_age_group(target_pet["birth_date"])
    reviewer_age_group = _calc_age_group(reviewer_pet["birth_date"])
    age_group_score = _ordinal_similarity(target_age_group, reviewer_age_group, AGE_GROUP_ORDER)

    allergy_score = _allergy_overlap_similarity(
        target_pet.get("allergy_codes", []),
        reviewer_pet.get("allergy_codes", []),
    )

    total = (
        SIMILARITY_WEIGHTS["species_match"] * species_score
        + SIMILARITY_WEIGHTS["breed_size_sim"] * breed_size_score
        + SIMILARITY_WEIGHTS["age_group_sim"] * age_group_score
        + SIMILARITY_WEIGHTS["allergy_overlap_sim"] * allergy_score
    )
    return round(total, 4)


def compute_weighted_aspect_scores(target_pet: dict, review_features_with_authors: list) -> dict:
    """
    target_pet: 추천 대상 반려동물의 pet_profile
    review_features_with_authors: [
        {
            "reviewer_pet": {...pet_profile...},   # 리뷰 작성자의 pet_profile
            "sentiment_label": "POSITIVE"/"NEGATIVE",
            "keyword_tags": [...],                  # aspect 태깅 결과
        }, ...
    ]

    반환: {
        "weighted_aspect_scores": {aspect_code: -1~1 사이 가중 평균 점수, ...},
        "total_weight": float,           # 전체 유사도 가중치 합 (참고용, 신뢰도 판단에 쓸 수 있음)
        "used_review_count": int,        # 유사도 > 0인, 실제로 반영된 리뷰 수
    }
    """
    from collections import defaultdict

    # aspect_tagging.py의 ASPECTS와 동일한 negative 태그 목록
    # (pipeline.py에서 이미 이 방식으로 긍/부정 태그를 구분하고 있어 동일 기준 사용)
    NEGATIVE_TAG_MARKERS = ["후기 있음", "낮음", "있음(후기)"]

    def _is_negative_tag(tag: str) -> bool:
        return any(marker in tag for marker in NEGATIVE_TAG_MARKERS)

    aspect_weighted_sum = defaultdict(float)
    aspect_weight_sum = defaultdict(float)
    total_weight = 0.0
    used_review_count = 0

    for review in review_features_with_authors:
        similarity = compute_profile_similarity(target_pet, review["reviewer_pet"])
        if similarity <= 0:
            continue  # 종이 다르면 아예 반영 안 함

        used_review_count += 1
        total_weight += similarity

        for tag in review["keyword_tags"]:
            # 태그 문자열에서 aspect 방향(긍/부정)을 판별해 +1/-1로 환산
            direction = -1.0 if _is_negative_tag(tag) else 1.0
            # aspect_code 자체는 태그 문자열만으로는 알 수 없으므로,
            # 여기서는 태그 문자열 자체를 key로 사용 (tagging.py와 연동 시 aspect_code로 교체 권장)
            aspect_weighted_sum[tag] += similarity * direction
            aspect_weight_sum[tag] += similarity

    weighted_scores = {}
    for tag, weight_sum in aspect_weight_sum.items():
        if weight_sum > 0:
            weighted_scores[tag] = round(aspect_weighted_sum[tag] / weight_sum, 4)

    return {
        "weighted_aspect_scores": weighted_scores,
        "total_weight": round(total_weight, 4),
        "used_review_count": used_review_count,
    }