# -*- coding: utf-8 -*-
"""
DB 연동 end-to-end 테스트: 실제 DB에서 pet/products/reviews를 가져와서
recommend_for_pet()까지 정상적으로 돌아가는지 확인.

main.py의 sys.path 설정 방식을 그대로 따라간다 (동일한 flat import 구조 재사용).

실행: (endtoend 프로젝트 루트에서, port-forward 켜진 상태로)
    USE_DUMMY_DATA=false python3 scripts/test_pipeline_e2e.py
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

sys.path.append(str(REPO_ROOT))
sys.path.append(str(SRC_DIR))
sys.path.append(str(SRC_DIR / "aspect"))
sys.path.append(str(SRC_DIR / "features"))
sys.path.append(str(SRC_DIR / "recommend"))

from pipeline import build_reviews_with_ratings, recommend_for_pet
from deepfm_model import load_deepfm

from src.data_access.pet_repository import get_pet_by_id
from src.data_access.product_repository import list_products
from src.data_access.reviews_repository import load_reviews_with_reviewer_pet

DEEPFM_MODEL_DIR = str(REPO_ROOT / "models" / "deepfm")

PET_ID = 1  # DBeaver에서 확인한 실제 pet id로 바꿔도 됨

print(f"1) pet 조회 (id={PET_ID})...")
pet = get_pet_by_id(PET_ID)
if pet is None:
    raise SystemExit(f"pet_id={PET_ID} 를 찾을 수 없음")
print("   ok:", pet["species"], pet["breed"])

print("2) products 조회...")
products = list_products()
print(f"   ok: {len(products)}개")

print("3) reviews 조회 + 집계...")
raw_reviews = load_reviews_with_reviewer_pet()
print(f"   raw reviews: {len(raw_reviews)}건")
reviews_by_product = build_reviews_with_ratings(raw_reviews)
print(f"   상품 {len(reviews_by_product)}개에 리뷰 매핑됨")

print("4) DeepFM 모델 로드...")
encoder, model = load_deepfm(DEEPFM_MODEL_DIR)
print("   ok")

print("5) recommend_for_pet 실행...")
results = recommend_for_pet(pet, products, reviews_by_product, encoder, model)
print(f"   ok: {len(results)}건 추천")
print("\n상위 3개:")
for r in results[:3]:
    print(" -", r)