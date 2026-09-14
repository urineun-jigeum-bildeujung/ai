# -*- coding: utf-8 -*-
"""
DeepFM 학습 스크립트 (정형 aspect 평점 기반 버전).

실행: python3 train/train_deepfm.py

[변경 이력]
KcELECTRA/tagging.py를 더 이상 사용하지 않기로 확정됨에 따라(정형 aspect 평점
1~3점을 추천의 핵심 데이터로 사용, 리뷰 텍스트 감성분석은 MVP 이후 부가 기능으로
보류), 학습 데이터 생성 시 더 이상 감성분석/키워드 태깅을 거치지 않고
rating_converter로 변환된 -1~1 aspect score를 바로 사용한다.

데이터: 지금은 더미 pet_profile/product_master/reviews + 더미 order_items -- 파이프라인 검증 목적.
        실제 합성 order_items(10만 건)가 오면 order_items 로딩 부분만 교체하면 된다.
"""

import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # src.data_access import용
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data", "dummy"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "aspect"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "features"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "recommend"))

import torch
import torch.nn as nn
import mlflow
from sklearn.metrics import roc_auc_score, accuracy_score

from dummy_data import PET_PROFILES, PRODUCTS
from src.data_access.reviews_repository import load_reviews_with_reviewer_pet
from rating_converter import ASPECT_FIELD_TO_CODE, convert_rating_to_score
from deepfm_features import build_interaction_features
from deepfm_labeling import build_training_pairs
from deepfm_model import FeatureEncoder, DeepFM
from reviewer_profile_similarity import compute_weighted_aspect_scores

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "deepfm")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 더미 order_items (실제 합성 데이터로 교체 예정)
# -----------------------------
DUMMY_ORDERS = {
    "order_001": {"order_status": "PAID", "paid_at": "2026-06-01", "ordered_at": "2026-06-01"},
    "order_002": {"order_status": "PAID", "paid_at": "2026-06-10", "ordered_at": "2026-06-10"},
    "order_003": {"order_status": "PARTIAL_REFUND", "paid_at": "2026-07-01", "ordered_at": "2026-07-01"},
    "order_004": {"order_status": "PAID", "paid_at": "2026-06-15", "ordered_at": "2026-06-15"},
    "order_005": {"order_status": "PAID", "paid_at": "2026-07-05", "ordered_at": "2026-07-05"},
}
DUMMY_ORDER_ITEMS = [
    {"order_id": "order_001", "product_id": "prod_001", "pet_id": "pet_001", "quantity": 1, "cancelled_quantity": 0, "returned_quantity": 0, "item_status": "PAID"},
    {"order_id": "order_002", "product_id": "prod_003", "pet_id": "pet_002", "quantity": 2, "cancelled_quantity": 0, "returned_quantity": 0, "item_status": "PAID"},
    {"order_id": "order_003", "product_id": "prod_004", "pet_id": "pet_003", "quantity": 3, "cancelled_quantity": 0, "returned_quantity": 1, "item_status": "PARTIAL_RETURN"},
    {"order_id": "order_004", "product_id": "prod_005", "pet_id": "pet_002", "quantity": 1, "cancelled_quantity": 0, "returned_quantity": 0, "item_status": "PAID"},
    {"order_id": "order_005", "product_id": "prod_003", "pet_id": "pet_001", "quantity": 1, "cancelled_quantity": 0, "returned_quantity": 0, "item_status": "PAID"},
]


def build_reviews_by_product():
    """
    reviews -> {reviewer_pet, ratings} 형태로 변환, 상품별로 묶어서 반환.
    pipeline.py의 build_reviews_with_ratings()와 동일한 로직.
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for review in load_reviews_with_reviewer_pet():
        ratings = {}
        for field_name, aspect_code in ASPECT_FIELD_TO_CODE.items():
            raw_rating = review.get(field_name)
            ratings[aspect_code] = convert_rating_to_score(raw_rating) if raw_rating is not None else None
        grouped[review["product_id"]].append({
            "reviewer_pet": review["reviewer_pet"],
            "ratings": ratings,
        })
    return grouped


def build_dataset():
    pets_by_id = {p["pet_id"]: p for p in PET_PROFILES}
    products_by_id = {p["product_id"]: p for p in PRODUCTS}

    pairs = build_training_pairs(DUMMY_ORDER_ITEMS, DUMMY_ORDERS, pets_by_id, PRODUCTS)
    reviews_by_product = build_reviews_by_product()

    samples = []
    for pair in pairs:
        pet = pets_by_id[pair["pet_id"]]
        product = products_by_id[pair["product_id"]]

        # (pet, product) 쌍마다 유사도 가중 aspect score를 새로 계산 -- pet마다 다른 값이 나옴
        product_reviews = reviews_by_product.get(product["product_id"], [])
        weighted_result = compute_weighted_aspect_scores(pet, product_reviews)
        summary = {"weighted_aspect_scores_by_code": weighted_result["weighted_aspect_scores"]}

        feat = build_interaction_features(pet, product, summary)
        feat["label"] = pair["label"]
        samples.append(feat)
    return samples


def train_val_split(samples: list, val_ratio: float = 0.2, seed: int = 42):
    import random
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio))
    return shuffled[val_size:], shuffled[:val_size]


def evaluate(model, batch):
    model.eval()
    with torch.no_grad():
        preds = model(batch).numpy()
    labels = batch["label"].numpy()
    pred_labels = (preds >= 0.5).astype(int)
    acc = accuracy_score(labels, pred_labels)
    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = float("nan")
    return acc, auc


def main():
    mlflow.set_experiment("deepfm_recommendation")

    with mlflow.start_run():
        print("데이터 준비 (정형 aspect 평점 + 리뷰 작성자 프로필 유사도 반영)")
        samples = build_dataset()
        print(f"전체 샘플: {len(samples)}건 (positive: {sum(1 for s in samples if s['label']==1)}, "
              f"negative: {sum(1 for s in samples if s['label']==0)})")

        train_samples, val_samples = train_val_split(samples)
        print(f"학습 {len(train_samples)}건 / 검증 {len(val_samples)}건")

        encoder = FeatureEncoder()

        train_encoded = [encoder.encode(s) for s in train_samples]
        val_encoded = [encoder.encode(s) for s in val_samples]
        train_batch = encoder.collate(train_encoded)
        val_batch = encoder.collate(val_encoded)

        embed_dim = 8
        learning_rate = 1e-3
        num_epochs = 30

        mlflow.log_param("embed_dim", embed_dim)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("train_size", len(train_samples))
        mlflow.log_param("val_size", len(val_samples))
        mlflow.log_param("data_source", "dummy order_items (파이프라인 검증용, 실데이터 아님)")
        mlflow.log_param("aspect_score_method", "structured_rating_reviewer_similarity_weighted")

        model = DeepFM(encoder, embed_dim=embed_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.BCELoss()

        print("학습 시작")
        for epoch in range(1, num_epochs + 1):
            model.train()
            optimizer.zero_grad()
            preds = model(train_batch)
            loss = loss_fn(preds, train_batch["label"])
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 or epoch == 1:
                val_acc, val_auc = evaluate(model, val_batch)
                print(f"epoch {epoch:3d} | train_loss: {loss.item():.4f} | val_acc: {val_acc:.4f} | val_auc: {val_auc:.4f}")
                mlflow.log_metric("train_loss", loss.item(), step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                if val_auc == val_auc:
                    mlflow.log_metric("val_auc", val_auc, step=epoch)

        final_val_acc, final_val_auc = evaluate(model, val_batch)
        print(f"\n최종 검증 accuracy: {final_val_acc:.4f}, AUC: {final_val_auc:.4f}")
        mlflow.log_metric("final_val_accuracy", final_val_acc)

        model_path = os.path.join(OUTPUT_DIR, "deepfm_model.pt")
        encoder_path = os.path.join(OUTPUT_DIR, "feature_encoder.json")
        config_path = os.path.join(OUTPUT_DIR, "model_config.json")

        torch.save(model.state_dict(), model_path)
        with open(encoder_path, "w", encoding="utf-8") as f:
            json.dump(encoder.sparse_vocabs, f, ensure_ascii=False, indent=2)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"embed_dim": embed_dim, "dnn_hidden": [64, 32]}, f, ensure_ascii=False, indent=2)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(encoder_path)
        mlflow.log_artifact(config_path)

        print(f"모델 저장: {model_path}")
        print(f"vocab 저장: {encoder_path}")
        print(f"config 저장: {config_path}")
        print("\n주의: 이번 학습은 더미 데이터 기준이라 성능 지표 자체는 의미가 크지 않음.")
        print("학습 코드/파이프라인이 정상 동작하는지 확인하는 것이 이번 실행의 목적.")


if __name__ == "__main__":
    main()