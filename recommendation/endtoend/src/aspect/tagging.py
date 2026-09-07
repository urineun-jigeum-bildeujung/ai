# -*- coding: utf-8 -*-
"""
aspect 키워드 태깅 (aspect_keyword_dictionary 기반 규칙 매칭).
최종 태그 = aspect_code(주제) + 해당 주제 내 긍/부정 매칭 결과 조합.
review_features.keyword_tags 필드에 들어갈 값을 생성한다.

[버그 수정 이력]
기존 방식은 positive/negative 키워드 리스트를 각각 단순 substring 매칭했는데,
"가려워하지 않아요"처럼 부정어로 반전된 문장이 다음 두 가지를 동시에 유발했다:
  1) positive 리스트의 완전한 문구("가려워하지 않아요")와 매칭 -> 긍정 태그
  2) negative 리스트의 짧은 부분 문자열("가려워")과도 매칭 -> 부정 태그
결과적으로 한 문장에서 서로 모순되는 태그(피부·모질 개선 + 피부 트러블 후기 있음)가
동시에 생성되는 문제가 있었다.

수정: negative 키워드가 매칭된 지점 바로 뒤에 "지 않", "지않", "안 ", "없" 같은
부정 표현이 곧바로 이어지면, 그건 부정적 상태를 다시 부정한 것(=사실상 긍정)으로
보고 negative 매칭에서 제외한다.
"""

import re

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

# 부정어로 반전된 표현 패턴 -- 키워드 매칭 지점 바로 뒤(최대 8자 이내)에 이 패턴이
# 이어지면 원래 뜻이 반전된 것으로 보고 해당 매칭을 무시한다.
_NEGATION_WINDOW = 8
_NEGATION_PATTERN = re.compile(r"(지\s?않|지않|안\s|없)")


def _is_negated_after(text: str, match_end_pos: int) -> bool:
    """매칭 종료 지점 뒤 일정 범위 안에 부정어 패턴이 있는지 확인."""
    window = text[match_end_pos:match_end_pos + _NEGATION_WINDOW]
    return bool(_NEGATION_PATTERN.search(window))


def _find_valid_match(text: str, keywords: list, skip_if_negated: bool) -> bool:
    """
    키워드 리스트 중 하나라도 text에 매칭되는지 확인.
    skip_if_negated=True이면, 매칭 지점 바로 뒤에 부정어가 이어질 경우 그 매칭은 무시한다
    (negative 키워드 리스트 검사 시 사용 -- "가려워"+"하지 않아요" 같은 반전 케이스 방지).
    """
    for kw in keywords:
        idx = text.find(kw)
        if idx == -1:
            continue
        if skip_if_negated and _is_negated_after(text, idx + len(kw)):
            continue  # 부정어로 반전됐으므로 이 매칭은 무효 처리
        return True
    return False


def tag_aspects(review_text: str) -> list:
    """
    리뷰 텍스트에서 매칭되는 aspect별 태그 리스트 반환.
    같은 리뷰에서 여러 aspect가 동시에 태깅될 수 있음.
    """
    tags = []
    for aspect_code, info in ASPECTS.items():
        pos_hit = _find_valid_match(review_text, info["positive"], skip_if_negated=False)
        neg_hit = _find_valid_match(review_text, info["negative"], skip_if_negated=True)
        if pos_hit:
            tags.append(info["tag_positive"])
        if neg_hit:
            tags.append(info["tag_negative"])
    return tags