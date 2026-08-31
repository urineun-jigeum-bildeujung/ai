# -*- coding: utf-8 -*-
"""
룰베이스 감성분석 (KcELECTRA 파인튜닝 모델의 임시 대체).
POC 목적: "리뷰 텍스트 -> sentiment_label" 이 파이프라인에서 끊기지 않고
review_features 스키마 형태로 나오는지 검증하는 것.
실제 정확도는 검증 범위 밖 (나중에 KcELECTRA 파인튜닝 모델로 이 모듈만 교체).

review_features 스키마 반영:
- sentiment_label: POSITIVE, NEGATIVE (2-class)
- sentiment_score: 감성 확신도(0~1)
"""

# 긍/부정 키워드는 aspect_keyword_dictionary의 예시 표현을 재사용
POSITIVE_KEYWORDS = [
    "잘 먹", "기호성 좋", "그릇을 싹", "밥 시간을 기다",
    "대변 상태 좋", "소화 잘", "변 냄새 줄", "배변량 적당",
    "털에 윤기", "피부가 좋아", "가려워하지 않", "털빠짐이 줄",
    "활력이 넘", "체중 관리에 도움", "적정 체중 유지", "활발해",
    "알러지 반응 없", "예민한 아이인데 잘 맞", "이상 반응 없이 잘 먹",
    "가성비 좋", "가격 대비 만족", "합리적",
]

NEGATIVE_KEYWORDS = [
    "안 먹", "냄새만 맡고 안 먹", "며칠 먹다 안 먹", "입맛에 안 맞",
    "설사", "구토", "무른 변", "속이 안 좋아",
    "가려워", "털이 푸석", "피부 트러블", "털빠짐이 심해",
    "살이 쪘", "살이 너무 빠졌", "기운이 없", "무기력",
    "알러지 반응 있", "두드러기", "안 맞아요",
    "비싸", "가성비가 아쉬", "다른 걸 사겠",
]


def analyze_sentiment(review_text: str) -> dict:
    """
    리뷰 텍스트를 받아 sentiment_label, sentiment_score를 반환.
    review_features 스키마 필드명과 동일하게 맞춤.
    """
    pos_hits = [kw for kw in POSITIVE_KEYWORDS if kw in review_text]
    neg_hits = [kw for kw in NEGATIVE_KEYWORDS if kw in review_text]

    pos_count = len(pos_hits)
    neg_count = len(neg_hits)
    total = pos_count + neg_count

    if total == 0:
        # 키워드 매칭 안 되는 경우 -> POC 한계, 중립적으로 negative 처리하지 않고 낮은 확신도의 positive로 처리
        # (실제 모델이라면 텍스트 임베딩 기반으로 판단하겠지만 룰베이스 한계)
        return {
            "sentiment_label": "POSITIVE",
            "sentiment_score": 0.5,
            "matched_positive_keywords": [],
            "matched_negative_keywords": [],
        }

    if pos_count >= neg_count:
        label = "POSITIVE"
        score = round(pos_count / total, 2)
    else:
        label = "NEGATIVE"
        score = round(neg_count / total, 2)

    return {
        "sentiment_label": label,
        "sentiment_score": score,
        "matched_positive_keywords": pos_hits,
        "matched_negative_keywords": neg_hits,
    }