# -*- coding: utf-8 -*-
"""
샘플링된 영어 리뷰(sampled_reviews.jsonl)를 한국어로 번역.

--model_name 인자로 번역 모델을 바꿔가며 품질을 비교할 수 있다.
후보:
- facebook/nllb-200-distilled-600M (기본값, 가벼움, 범용 다국어)
- facebook/nllb-200-1.3B (같은 계열, 더 무겁지만 품질 개선 기대)
- Helsinki-NLP/opus-mt-tc-big-en-ko (영-한 전용, 존재 여부/품질은 실행해서 확인 필요)

주의: NLLB 계열과 MarianMT(opus-mt) 계열은 로딩 방식이 달라서,
      모델 종류에 따라 자동으로 적절한 tokenizer/model 클래스를 선택한다.

실행 (도커 컨테이너, GPU 있으면 자동 사용):
    python3 data/amazon/translate_reviews.py \
        --input_path data/amazon/sampled_reviews.jsonl \
        --output_path data/amazon/translated_reviews_nllb600m.jsonl \
        --model_name facebook/nllb-200-distilled-600M \
        --limit 20

    python3 data/amazon/translate_reviews.py \
        --input_path data/amazon/sampled_reviews.jsonl \
        --output_path data/amazon/translated_reviews_opusmt.jsonl \
        --model_name Helsinki-NLP/opus-mt-tc-big-en-ko \
        --limit 20

카테고리(category_bucket)와 별점(rating) 정보는 그대로 보존해서,
다음 단계(언어모델 파인튜닝)에서 "[카테고리][감성]" 조건부 태그를 만드는 데 사용한다.
"""

import argparse
import json

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

NLLB_SRC_LANG = "eng_Latn"
NLLB_TGT_LANG = "kor_Hang"


def is_nllb_model(model_name: str) -> bool:
    return "nllb" in model_name.lower()


def load_model(model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"모델 로딩: {model_name} (device: {device})")

    if is_nllb_model(model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=NLLB_SRC_LANG)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model, device


def translate_batch(texts: list, tokenizer, model, device, model_name: str, max_length: int = 256) -> list:
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generate_kwargs = {"max_length": max_length}
    if is_nllb_model(model_name):
        generate_kwargs["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids(NLLB_TGT_LANG)

    with torch.no_grad():
        translated = model.generate(**inputs, **generate_kwargs)

    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="data/amazon/sampled_reviews.jsonl")
    parser.add_argument("--output_path", default="data/amazon/translated_reviews.jsonl")
    parser.add_argument("--model_name", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="테스트용: 앞에서 N건만 번역")
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    if args.limit:
        items = items[: args.limit]

    print(f"번역 대상: {len(items)}건")

    tokenizer, model, device = load_model(args.model_name)

    translated_items = []
    for i in range(0, len(items), args.batch_size):
        batch = items[i : i + args.batch_size]
        texts = [item["review_text"] for item in batch]

        try:
            translations = translate_batch(texts, tokenizer, model, device, args.model_name)
        except Exception as e:
            print(f"배치 {i} 번역 실패: {e} -> 이 배치는 건너뜀")
            continue

        for item, ko_text in zip(batch, translations):
            item["review_text_ko"] = ko_text
            item["translation_model"] = args.model_name
            translated_items.append(item)

        if (i // args.batch_size) % 20 == 0:
            print(f"진행: {i + len(batch)}/{len(items)}건 완료")

    with open(args.output_path, "w", encoding="utf-8") as f:
        for item in translated_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n번역 완료: {len(translated_items)}건 -> {args.output_path}")

    # 샘플 몇 개 출력 (품질 육안 확인용)
    print("\n--- 샘플 확인 ---")
    for item in translated_items[:5]:
        print(f"[{item['category_bucket']}, 별점 {item['rating']}]")
        print(f"  EN: {item['review_text'][:80]}")
        print(f"  KO: {item['review_text_ko']}")
        print()


if __name__ == "__main__":
    main()