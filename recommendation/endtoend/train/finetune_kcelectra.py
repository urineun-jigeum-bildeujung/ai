# -*- coding: utf-8 -*-
"""
KcELECTRA 파인튜닝 스크립트.

실행 (더미 데이터, 코드 검증용):
    python3 train/finetune_kcelectra.py

실행 (실제 생성 데이터로 학습):
    python3 train/finetune_kcelectra.py \
        --train_path data/amazon/generated_v2/train_reviews.jsonl \
        --val_path data/amazon/generated_v2/val_reviews.jsonl

모델: beomi/KcELECTRA-base (HuggingFace 공개 사전학습 모델)
"""

import argparse
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "sentiment"))
from dataset import load_labeled_reviews, load_generated_reviews, train_val_split, LABEL2ID, ID2LABEL  # noqa: E402

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import mlflow

MODEL_NAME = "beomi/KcELECTRA-base"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "kcelectra")
MAX_LENGTH = 128


def build_hf_dataset(labeled_reviews: list, tokenizer):
    texts = [r["text"] for r in labeled_reviews]
    label_ids = [r["label_id"] for r in labeled_reviews]

    ds = Dataset.from_dict({"text": texts, "label": label_ids})

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    ds = ds.map(tokenize_fn, batched=True)
    return ds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default=None, help="생성된 train_reviews.jsonl 경로 (없으면 더미 데이터 사용)")
    parser.add_argument("--val_path", default=None, help="생성된 val_reviews.jsonl 경로 (없으면 더미 데이터 사용)")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    mlflow.set_experiment("kcelectra_sentiment")

    with mlflow.start_run():
        print(f"모델 로딩: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        print("데이터 준비")
        if args.train_path and args.val_path:
            train_set, val_set = load_generated_reviews(args.train_path, args.val_path)
            data_source = f"generated ({args.train_path})"
        else:
            labeled = load_labeled_reviews()
            train_set, val_set = train_val_split(labeled)
            data_source = "dummy_reviews (파이프라인 검증용, 실데이터 아님)"
        print(f"학습 {len(train_set)}건 / 검증 {len(val_set)}건 (source: {data_source})")

        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("train_size", len(train_set))
        mlflow.log_param("val_size", len(val_set))
        mlflow.log_param("data_source", data_source)

        train_ds = build_hf_dataset(train_set, tokenizer)
        val_ds = build_hf_dataset(val_set, tokenizer)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        num_train_epochs = args.num_train_epochs
        learning_rate = 2e-5
        mlflow.log_param("num_train_epochs", num_train_epochs)
        mlflow.log_param("learning_rate", learning_rate)

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=num_train_epochs,
            weight_decay=0.01,
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to=[],  # wandb 등 외부 로깅 비활성화 (MLflow는 별도 연동)
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        print("학습 시작")
        trainer.train()

        print("최종 검증 결과")
        metrics = trainer.evaluate()
        print(metrics)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key.replace("eval_", ""), value)

        print(f"모델 저장: {args.output_dir}")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        mlflow.log_param("model_output_dir", args.output_dir)

        if "dummy" in data_source:
            print("\n주의: 이번 학습은 더미 데이터(47건) 기준이라 성능 지표 자체는 의미가 크지 않음.")


if __name__ == "__main__":
    main()