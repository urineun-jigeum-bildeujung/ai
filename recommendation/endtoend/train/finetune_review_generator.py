# -*- coding: utf-8 -*-
"""
Polyglot-Ko-5.8B LoRA 파인튜닝 (리뷰 생성용 언어모델).

각 리뷰를 "[카테고리][감성] 리뷰텍스트" 형태로 포맷팅해서 학습시킨다.
이렇게 학습하면 생성 시점에 "[SUPPLEMENT][POSITIVE]" 같은 조건을 프롬프트로 주고
이어쓰게 해서, 카테고리/감성 비율을 소스 데이터 분포와 무관하게 직접 통제할 수 있다.

실행 (GPU 필요, Dockerfile.gpu로 빌드한 이미지에서):
    docker run --rm --gpus all -v $(pwd):/app endtoend-gpu \
        python3 train/finetune_review_generator.py \
            --input_path data/amazon/filtered_reviews.jsonl \
            --output_dir models/review_generator

주의: 이 스크립트는 GPU(권장 VRAM 16GB 이상)에서 실행하는 것을 전제로 한다.
      로컬(Apple Silicon, CPU 등)에서는 실행하지 않는다.
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
import mlflow

MODEL_NAME = "EleutherAI/polyglot-ko-1.3b"
MAX_LENGTH = 256


def format_training_text(item: dict) -> str:
    """
    리뷰 하나를 "[카테고리][감성] 텍스트" 형태로 포맷팅.
    감성은 별점 기준(4,5=POSITIVE / 1,2,3=NEGATIVE)으로 결정 -- 라벨링 가이드라인과 동일 규칙.
    """
    sentiment = "POSITIVE" if item["rating"] >= 4 else "NEGATIVE"
    category = item["category_bucket"]
    text = item["review_text_ko"]
    return f"[{category}][{sentiment}] {text}"


def load_dataset(input_path: str, tokenizer):
    with open(input_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    texts = [format_training_text(item) for item in items]
    print(f"학습 텍스트 예시: {texts[0]}")

    dataset = Dataset.from_dict({"text": texts})

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    return dataset, len(items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="data/amazon/filtered_reviews.jsonl")
    parser.add_argument("--output_dir", default="models/review_generator")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"GPU 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    print(f"\n토크나이저/모델 로딩: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,  # RTX 5090은 bfloat16 지원, 메모리 절약 + 속도 향상
        device_map="auto",
    )

    # --- LoRA 설정 ---
    # 데이터가 4,835건으로 크지 않은 규모라, 전체 파라미터를 다 학습시키기보다
    # LoRA(저랭크 어댑터)만 학습시켜 과적합 위험을 줄이고 학습 시간도 단축한다.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["query_key_value"],  # GPT-NeoX 계열(Polyglot-Ko) 어텐션 모듈명
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\n데이터 준비")
    dataset, n_samples = load_dataset(args.input_path, tokenizer)
    print(f"학습 샘플: {n_samples}건")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )

    mlflow.set_experiment("review_generator_finetune")
    with mlflow.start_run():
        mlflow.log_param("base_model", MODEL_NAME)
        mlflow.log_param("method", "LoRA")
        mlflow.log_param("lora_r", lora_config.r)
        mlflow.log_param("lora_alpha", lora_config.lora_alpha)
        mlflow.log_param("num_train_epochs", args.num_train_epochs)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("train_size", n_samples)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )

        print("\n학습 시작")
        train_result = trainer.train()
        mlflow.log_metric("final_train_loss", train_result.training_loss)

        print(f"\nLoRA 어댑터 저장: {args.output_dir}")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        mlflow.log_param("output_dir", args.output_dir)

    print("\n완료. 다음 단계: generate_reviews.py로 실제 리뷰 생성")


if __name__ == "__main__":
    main()