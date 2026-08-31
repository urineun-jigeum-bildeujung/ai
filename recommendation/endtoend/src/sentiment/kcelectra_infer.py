# -*- coding: utf-8 -*-
"""
파인튜닝된 KcELECTRA 모델로 추론하는 래퍼.

rule_based.py의 analyze_sentiment(review_text) 와 동일한 반환 형태를 유지해서,
pipeline.py에서 이 함수로 그대로 교체 가능하도록 인터페이스를 맞춘다.

주의: 이 모듈은 train/finetune_kcelectra.py 로 학습된 모델(models/kcelectra/)이
있어야 동작한다. 학습 전이거나 모델 파일이 없으면 로드 시 에러가 난다.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "kcelectra")

_tokenizer = None
_model = None
_device = None


def _lazy_load():
    """모델을 최초 호출 시 1번만 로드 (매 호출마다 로드하면 느림)."""
    global _tokenizer, _model, _device
    if _model is not None:
        return

    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(
            f"파인튜닝된 모델을 찾을 수 없습니다: {MODEL_DIR}\n"
            "먼저 train/finetune_kcelectra.py 를 실행해서 모델을 학습·저장해야 합니다."
        )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    _model.to(_device)
    _model.eval()


def analyze_sentiment(review_text: str) -> dict:
    """
    rule_based.analyze_sentiment 와 동일한 반환 스키마.
    {
        "sentiment_label": "POSITIVE" | "NEGATIVE",
        "sentiment_score": float (해당 라벨의 softmax 확률, 감성 확신도),
    }
    """
    _lazy_load()

    inputs = _tokenizer(review_text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()

    pred_id = int(torch.argmax(probs).item())
    label = _model.config.id2label[pred_id]
    score = round(float(probs[pred_id].item()), 4)

    return {
        "sentiment_label": label,
        "sentiment_score": score,
    }


if __name__ == "__main__":
    # 간단한 동작 확인용
    samples = [
        "정말 잘 먹어요. 털에 윤기도 나고 완전 만족합니다.",
        "며칠 먹다가 안 먹어요. 설사도 했어요.",
    ]
    for text in samples:
        result = analyze_sentiment(text)
        print(f"{text} -> {result}")