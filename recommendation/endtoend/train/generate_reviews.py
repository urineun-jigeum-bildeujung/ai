# -*- coding: utf-8 -*-
"""
파인튜닝된 리뷰 생성 모델로 최종 데이터 생성.

목표 비율:
- 카테고리: FOOD 60% : SUPPLEMENT 20% : TREAT 20%
- 감성: POSITIVE 60% : NEGATIVE 40%
- 전체: 학습용 10,000건 + 검증용 500건 = 10,500건
  (학습/검증은 서로 겹치지 않게 생성 단계에서부터 분리)

실행 (GPU 필요):
    docker run --rm --gpus all -v $(pwd):/app endtoend-gpu \
        python3 train/generate_reviews.py \
            --model_dir models/review_generator \
            --output_dir data/amazon/generated
"""

import argparse
import json
import os
import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data", "masters"))
from entity_injection import inject_attributes

BASE_MODEL_NAME = "EleutherAI/polyglot-ko-1.3b"

CATEGORY_RATIO = {"FOOD": 0.6, "SUPPLEMENT": 0.2, "TREAT": 0.2}
SENTIMENT_RATIO = {"POSITIVE": 0.6, "NEGATIVE": 0.4}

TRAIN_TARGET = 10000
VAL_TARGET = 500


def load_model(model_dir: str):
    print(f"베이스 모델 로딩: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"LoRA 어댑터 로딩: {model_dir}")
    model = PeftModel.from_pretrained(base_model, model_dir)
    model.eval()
    return tokenizer, model


def build_generation_plan(total: int) -> list:
    """
    목표 건수를 카테고리 x 감성 조합별로 몇 건씩 생성할지 계획을 세운다.
    반환: [(category, sentiment, count), ...]
    """
    plan = []
    for category, cat_ratio in CATEGORY_RATIO.items():
        for sentiment, sent_ratio in SENTIMENT_RATIO.items():
            count = round(total * cat_ratio * sent_ratio)
            plan.append((category, sentiment, count))
    return plan


def generate_batch(tokenizer, model, category: str, sentiment: str, n: int, device,
                    max_new_tokens: int = 100, batch_size: int = 8) -> list:
    """
    LM으로 기본 리뷰 텍스트를 생성한 뒤, entity_injection으로
    품종/나이/체중/알러지반응/관심건강정보/aspect 표현을 확률적으로 삽입한다.
    species(DOG/CAT)는 리뷰마다 무작위로 배정 (카테고리와 무관하게 둘 다 구매 가능하므로).
    """
    prompt = f"[{category}][{sentiment}]"
    results = []
    seed_counter = 0

    while len(results) < n:
        current_batch_size = min(batch_size, n - len(results))
        inputs = tokenizer([prompt] * current_batch_size, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.92,
                repetition_penalty=1.3,
                pad_token_id=tokenizer.pad_token_id,
            )

        for output in outputs:
            text = tokenizer.decode(output, skip_special_tokens=True)
            # 프롬프트 부분 제거하고 실제 생성된 리뷰 텍스트만 추출
            base_text = text.replace(prompt, "").strip()
            if not base_text:
                continue

            species = random.choice(["DOG", "CAT"])
            injected = inject_attributes(base_text, species, sentiment, seed=seed_counter)
            seed_counter += 1

            results.append({
                "category_bucket": category,
                "sentiment_label": sentiment,
                "species": species,
                "review_text": injected["final_text"],
                "ground_truth": injected["ground_truth"],
            })

    return results[:n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models/review_generator")
    parser.add_argument("--output_dir", default="data/amazon/generated")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model(args.model_dir)

    all_generated = []

    for target_name, target_count in [("train", TRAIN_TARGET), ("val", VAL_TARGET)]:
        print(f"\n=== {target_name} 셋 생성 (목표 {target_count}건) ===")
        plan = build_generation_plan(target_count)

        generated = []
        for category, sentiment, count in plan:
            print(f"  [{category}][{sentiment}] {count}건 생성 중...")
            batch_results = generate_batch(tokenizer, model, category, sentiment, count, device)
            generated.extend(batch_results)

        random.shuffle(generated)
        output_path = os.path.join(args.output_dir, f"{target_name}_reviews.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for item in generated:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"{target_name} 셋 저장: {output_path} ({len(generated)}건)")
        all_generated.append((target_name, generated))

    print("\n=== 최종 요약 ===")
    for name, items in all_generated:
        from collections import Counter
        cat_counts = Counter(i["category_bucket"] for i in items)
        sent_counts = Counter(i["sentiment_label"] for i in items)
        species_counts = Counter(i["species"] for i in items)
        breed_mentioned = sum(1 for i in items if i["ground_truth"]["breed"])
        allergen_mentioned = sum(1 for i in items if i["ground_truth"]["allergen_mentioned"])
        concern_mentioned = sum(1 for i in items if i["ground_truth"]["concern_mentioned"])
        aspect_mentioned = sum(1 for i in items if i["ground_truth"]["aspect_mentioned"])

        print(f"\n{name}: 총 {len(items)}건")
        print(f"  카테고리: {dict(cat_counts)}")
        print(f"  감성: {dict(sent_counts)}")
        print(f"  종: {dict(species_counts)}")
        print(f"  품종 언급: {breed_mentioned}건 ({breed_mentioned/len(items)*100:.1f}%)")
        print(f"  알러지 언급: {allergen_mentioned}건 ({allergen_mentioned/len(items)*100:.1f}%)")
        print(f"  관심건강정보 언급: {concern_mentioned}건 ({concern_mentioned/len(items)*100:.1f}%)")
        print(f"  aspect 표현 언급: {aspect_mentioned}건 ({aspect_mentioned/len(items)*100:.1f}%)")


if __name__ == "__main__":
    main()