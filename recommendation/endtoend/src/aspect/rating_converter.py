# -*- coding: utf-8 -*-
"""
정형 aspect 평가(1~3점, 사용자가 리뷰 작성 시 직접 선택) -> DeepFM aspect score(-1~1) 변환.

[배경]
기존에는 리뷰 텍스트에서 aspect(기호성/소화·배변/피부·모질/체중·활력/알러지반응)를
규칙 기반으로 추출(tagging.py)했으나, 실제 리뷰 작성 화면에서 사용자가
이 5개 항목을 1~3점으로 직접 선택하는 정형 입력 방식으로 확정되어
텍스트 추출이 더 이상 필요 없어졌다.
(가격·가성비는 추천 근거에서 제외되어 이 변환 대상에 포함하지 않는다.)

점수 의미: 1=부정, 2=보통(중립), 3=긍정
"""

# 정형 입력 필드명 -> DeepFM aspect_code 매핑
# 실제 백엔드 필드명이 확정되면 이 매핑만 수정하면 된다.
ASPECT_FIELD_TO_CODE = {
    "palatability_rating": "palatability",
    "digestion_rating": "digestion",
    "skin_coat_rating": "skin_coat",
    "vitality_weight_rating": "vitality_weight",
    "allergic_reaction_rating": "allergic_reaction",
}

RATING_MIN, RATING_MID, RATING_MAX = 1, 2, 3


def convert_rating_to_score(rating: int) -> float:
    """
    1~3점 정형 평가를 -1~1 aspect score로 변환.
    1 -> -1.0 (부정), 2 -> 0.0 (중립), 3 -> +1.0 (긍정)
    범위를 벗어난 값이 들어오면 가장 가까운 경계값으로 clamp 처리.
    """
    if rating is None:
        return 0.0  # 평가 안 한 경우 중립으로 처리 (긍정/부정 어느 쪽으로도 치우치지 않게)

    rating = max(RATING_MIN, min(RATING_MAX, rating))
    return round((rating - RATING_MID) / (RATING_MAX - RATING_MID), 4)


def convert_review_ratings_to_aspect_scores(review_ratings: dict) -> dict:
    """
    review_ratings: {"palatability_rating": 3, "digestion_rating": 2, ...} 형태
                    (필드명은 ASPECT_FIELD_TO_CODE 기준, 값은 1~3 또는 None)
    반환: {"palatability": 1.0, "digestion": 0.0, ...} -- aspect_code를 키로 하는 -1~1 점수
    """
    scores = {}
    for field_name, aspect_code in ASPECT_FIELD_TO_CODE.items():
        rating = review_ratings.get(field_name)
        scores[aspect_code] = convert_rating_to_score(rating)
    return scores