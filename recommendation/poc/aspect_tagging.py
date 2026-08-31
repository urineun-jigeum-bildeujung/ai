# -*- coding: utf-8 -*-
"""
aspect 키워드 태깅 (aspect_keyword_dictionary 기반 규칙 매칭).
최종 태그 = aspect_code(주제) + 해당 주제 내 긍/부정 매칭 결과 조합.
review_features.keyword_tags 필드에 들어갈 값을 생성한다.
"""

ASPECTS = {
    "palatability": {
        "ko": "기호성",
        "positive": ["잘 먹", "기호성 좋", "그릇을 싹", "밥 시간을 기다"],
        "negative": ["안 먹", "며칠 먹다 안 먹", "입맛에 안 맞"],
        "tag_positive": "기호성 좋음",
        "tag_negative": "기호성 낮음",
    },
    "digestion": {
        "ko": "소화·배변",
        "positive": ["대변 상태 좋", "소화 잘", "변 냄새 줄", "배변량 적당"],
        "negative": ["설사", "구토", "무른 변", "속이 안 좋아"],
        "tag_positive": "소화 잘됨",
        "tag_negative": "소화 불편 후기 있음",
    },
    "skin_coat": {
        "ko": "피부·모질",
        "positive": ["털에 윤기", "피부가 좋아", "가려워하지 않", "털빠짐이 줄"],
        "negative": ["가려워", "털이 푸석", "피부 트러블", "털빠짐이 심해"],
        "tag_positive": "피부·모질 개선",
        "tag_negative": "피부 트러블 후기 있음",
    },
    "vitality_weight": {
        "ko": "체중·활력",
        "positive": ["활력이 넘", "체중 관리에 도움", "적정 체중 유지", "활발해"],
        "negative": ["살이 쪘", "살이 너무 빠졌", "기운이 없", "무기력"],
        "tag_positive": "활력 개선",
        "tag_negative": "체중·활력 저하 후기 있음",
    },
    "allergic_reaction": {
        "ko": "알러지 반응",
        "positive": ["알러지 반응 없", "예민한 아이인데 잘 맞", "이상 반응 없이 잘 먹"],
        "negative": ["알러지 반응 있", "두드러기", "안 맞아요"],
        "tag_positive": "알러지 반응 없음(후기)",
        "tag_negative": "알러지 반응 있음(후기)",
    },
    "price_value": {
        "ko": "가격·가성비",
        "positive": ["가성비 좋", "가격 대비 만족", "합리적"],
        "negative": ["비싸", "가성비가 아쉬", "다른 걸 사겠"],
        "tag_positive": "가성비 좋음",
        "tag_negative": "가격 부담 후기 있음",
    },
}


def tag_aspects(review_text: str) -> list:
    """
    리뷰 텍스트에서 매칭되는 aspect별 태그 리스트 반환.
    같은 리뷰에서 여러 aspect가 동시에 태깅될 수 있음.
    """
    tags = []
    for aspect_code, info in ASPECTS.items():
        pos_hit = any(kw in review_text for kw in info["positive"])
        neg_hit = any(kw in review_text for kw in info["negative"])
        if pos_hit:
            tags.append(info["tag_positive"])
        if neg_hit:
            tags.append(info["tag_negative"])
    return tags