# -*- coding: utf-8 -*-
"""
KcELECTRA 파인튜닝용 데이터 준비.

라벨링 가이드라인 반영:
- 2-class (POSITIVE/NEGATIVE)
- 별점 4,5 -> POSITIVE / 1,2,3 -> NEGATIVE
- 별점 없는 리뷰는 이 단계 이전(수집 단계)에서 이미 제외된 것으로 가정
"""

import sys
import os
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "data", "dummy"))
from dummy_reviews import DUMMY_REVIEWS  # noqa: E402

LABEL2ID = {"NEGATIVE": 0, "POSITIVE": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def rating_to_label(rating: int) -> str:
    """별점 -> 라벨 매핑 (라벨링 가이드라인 확정 규칙)."""
    return "POSITIVE" if rating in (4, 5) else "NEGATIVE"


def load_labeled_reviews():
    """
    더미 리뷰에 별점 기반 라벨을 붙여서 반환.
    실제 서비스에서는 이 자리가 백엔드 합성 리뷰(reviews 테이블)로 교체된다.
    """
    labeled = []
    for review in DUMMY_REVIEWS:
        label = rating_to_label(review["rating"])
        labeled.append({
            "review_id": review["review_id"],
            "text": review["review_text"],
            "label": label,
            "label_id": LABEL2ID[label],
        })
    return labeled


def train_val_split(labeled_reviews: list, val_ratio: float = 0.2, seed: int = 42):
    """
    학습/검증 분할.
    지금은 더미셋 규모가 작아 정식 검증셋(500건, 직접 라벨링) 역할을 대신할 수 없고,
    어디까지나 "학습 코드가 정상 동작하는지" 확인하는 용도의 분할이다.
    """
    random.seed(seed)
    shuffled = labeled_reviews.copy()
    random.shuffle(shuffled)

    val_size = max(1, int(len(shuffled) * val_ratio))
    val_set = shuffled[:val_size]
    train_set = shuffled[val_size:]
    return train_set, val_set


if __name__ == "__main__":
    labeled = load_labeled_reviews()
    train_set, val_set = train_val_split(labeled)

    print(f"전체: {len(labeled)}건")
    print(f"학습: {len(train_set)}건, 검증: {len(val_set)}건")
    pos = sum(1 for r in labeled if r["label"] == "POSITIVE")
    neg = sum(1 for r in labeled if r["label"] == "NEGATIVE")
    print(f"POSITIVE: {pos}건, NEGATIVE: {neg}건")