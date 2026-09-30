# -*- coding: utf-8 -*-
"""
AI 안전성 검증 - §3 편향(Bias) 검증 스크립트.

실제 DB(member_db/product_db/review_db) + 재학습된 DeepFM 모델로
안전성 검증 보고서 §3.1/3.2에 적힌 방법론을 그대로 실행한다:
1) 체구(소형/중형/대형견)별 평균 score, PENALIZED/PENDING 비율 비교
2) 알레르기 있는 pet vs 없는 pet 간 평균 score, PENALIZED/PENDING 비율 비교
3) 상위 추천 노출 브랜드 분포(HHI) vs 전체 카탈로그 브랜드 분포(HHI) 비교

실행: (endtoend 프로젝트 루트에서, port-forward 켜진 상태로)
    USE_DUMMY_DATA=false python3 scripts/bias_verification.py
"""
import sys
import statistics
from pathlib import Path
from collections import Counter, defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.append(str(REPO_ROOT))
sys.path.append(str(SRC_DIR))
sys.path.append(str(SRC_DIR / "aspect"))
sys.path.append(str(SRC_DIR / "features"))
sys.path.append(str(SRC_DIR / "recommend"))

from pipeline import build_reviews_with_ratings, recommend_for_pet
from deepfm_model import load_deepfm
from reviewer_profile_similarity import _calc_breed_size

from src.data_access.db import get_connection
from src.data_access.pet_repository import get_pets_by_ids, MEMBER_DB_ENV
from src.data_access.product_repository import list_products
from src.data_access.reviews_repository import load_reviews_with_reviewer_pet

DEEPFM_MODEL_DIR = str(REPO_ROOT / "models" / "deepfm")
SAMPLE_SIZE_PER_SPECIES = 40
TOP_K_FOR_BRAND_CHECK = 5


def sample_pet_ids(species, limit):
    conn = get_connection(MEMBER_DB_ENV)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM pet WHERE species = %s ORDER BY random() LIMIT %s",
                (species, limit),
            )
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def hhi(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return sum((c / total) ** 2 for c in counter.values()) * 10000


print("1) 샘플 pet 추출...")
dog_ids = sample_pet_ids("DOG", SAMPLE_SIZE_PER_SPECIES)
cat_ids = sample_pet_ids("CAT", SAMPLE_SIZE_PER_SPECIES)
pets_by_id = {}
pets_by_id.update(get_pets_by_ids(dog_ids))
pets_by_id.update(get_pets_by_ids(cat_ids))
print(f"   dog={len(dog_ids)} cat={len(cat_ids)} (조회된 실제 pet: {len(pets_by_id)})")

print("2) products/reviews/모델 로드 (1회만, 재사용)...")
products = [p for p in list_products() if p.get("status") == "ON_SALE"]
raw_reviews = load_reviews_with_reviewer_pet()
reviews_by_product = build_reviews_with_ratings(raw_reviews)
encoder, model = load_deepfm(DEEPFM_MODEL_DIR)
product_map = {p["product_id"]: p for p in products}
print(f"   상품 {len(products)}개, 리뷰 {len(raw_reviews)}건")

print("3) 전체 catalog 브랜드 분포 (기준값)...")
catalog_brand_counts = Counter(p.get("brand_name") for p in products if p.get("brand_name"))
catalog_hhi = hhi(catalog_brand_counts)
print(f"   브랜드 {len(catalog_brand_counts)}개, catalog HHI={catalog_hhi:.0f}")

print("4) pet별 추천 실행 + 집계...")
breed_size_scores = defaultdict(list)
breed_size_flags = defaultdict(lambda: defaultdict(int))
allergy_group_scores = {"has_allergy": [], "no_allergy": []}
allergy_group_flags = {"has_allergy": defaultdict(int), "no_allergy": defaultdict(int)}
top_recommended_brands = Counter()

n_done = 0
for pet in pets_by_id.values():
    try:
        results = recommend_for_pet(pet, products, reviews_by_product, encoder, model)
    except Exception as e:
        print(f"   [스킵] pet_id={pet['pet_id']} 에러: {e}")
        continue
    if not results:
        continue
    n_done += 1

    scores = [r["score_100"] for r in results]
    avg_score = statistics.mean(scores)

    has_allergy = len(pet.get("allergy_codes") or []) > 0
    group = "has_allergy" if has_allergy else "no_allergy"
    allergy_group_scores[group].append(avg_score)
    for r in results:
        allergy_group_flags[group]["total"] += 1
        allergy_group_flags[group][r["allergy_status"]] += 1

    if pet["species"] == "DOG":
        size = _calc_breed_size(pet["weight"])
        breed_size_scores[size].append(avg_score)
        for r in results:
            breed_size_flags[size]["total"] += 1
            breed_size_flags[size][r["allergy_status"]] += 1

    for r in results[:TOP_K_FOR_BRAND_CHECK]:
        product = product_map.get(r["product_id"])
        if product and product.get("brand_name"):
            top_recommended_brands[product["brand_name"]] += 1

print(f"   완료: pet {n_done}마리 추천 실행됨")

print("\n=== [1] 체구별 (DOG만) ===")
for size in ["SMALL", "MEDIUM", "LARGE"]:
    scores = breed_size_scores.get(size, [])
    flags = breed_size_flags.get(size, {})
    if not scores:
        print(f"  {size}: 샘플 없음")
        continue
    total = flags.get("total", 0)
    pen = flags.get("PENALIZED", 0)
    pend = flags.get("PENDING", 0)
    print(f"  {size}: pet {len(scores)}마리, 평균 score={statistics.mean(scores):.1f}, "
          f"PENALIZED율={pen/total*100:.1f}%, PENDING율={pend/total*100:.1f}% (전체 추천 {total}건)")

print("\n=== [2] 알레르기 유무 ===")
for group in ["has_allergy", "no_allergy"]:
    scores = allergy_group_scores[group]
    flags = allergy_group_flags[group]
    if not scores:
        print(f"  {group}: 샘플 없음")
        continue
    total = flags.get("total", 0)
    pen = flags.get("PENALIZED", 0)
    pend = flags.get("PENDING", 0)
    print(f"  {group}: pet {len(scores)}마리, 평균 score={statistics.mean(scores):.1f}, "
          f"PENALIZED율={pen/total*100:.1f}%, PENDING율={pend/total*100:.1f}% (전체 추천 {total}건)")

print("\n=== [3] 브랜드 분포 (상위 5개 추천 기준) ===")
top_hhi = hhi(top_recommended_brands)
print(f"  전체 catalog HHI={catalog_hhi:.0f} vs 추천 노출 HHI={top_hhi:.0f}")
print(f"  (참고: HHI 1500 미만=분산, 2500 이상=집중 시장으로 보는 게 통상 기준)")
print(f"  추천 상위 브랜드 Top 5: {top_recommended_brands.most_common(5)}")