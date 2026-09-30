# -*- coding: utf-8 -*-
"""
AI 기능 평가 보고서 - §2 정량 평가 스크립트.

실제 DB(member_db/product_db/review_db/order_db) + 재학습된 DeepFM 모델로
평가 보고서 §2.1에 정의된 지표를 실측한다.

[방법론]
1) 홈 추천 (NDCG@9 / Precision@9 / Recall@9 / Hit Rate@9)
   - order_db의 실제 구매 이력이 있는 사용자를 대상으로 leave-one-out 평가.
   - 사용자의 가장 최근 구매 상품 1개를 "정답(relevant)"으로 숨기고,
     나머지 구매 이력만 구매 이력 임베딩으로 사용해 recommend_for_pet 실행.
   - 상위 9개 추천 안에 숨겨둔 실제 구매 상품이 포함되는지로 NDCG/Precision/
     Recall/Hit Rate 계산.
   - Baseline A(인기순: sales_count 내림차순), Baseline B(룰 기반: 알레르기
     필터만 적용 후 최신순)와 동일 방식으로 비교.
2) 대체 상품 추천 (카테고리 일치율 / 평균 코사인 유사도)
   - 판매중 상품 중 N개를 base_product로 샘플링해 find_substitute_products 실행.
   - 반환된 후보들이 base_product와 subcategory_code가 실제로 같은지, 그리고
     embedding 코사인 유사도 평균이 얼마인지 계산.
3) 알레르기 판정 정확도
   - pet 샘플 각각에 대해 recommend_for_pet 결과의 allergy_status를,
     실제 pet.allergy_codes / product.allergen_flags 교집합으로 직접 재계산한
     "정답"과 대조해 정확도 산출.

실행 (endtoend 프로젝트 루트에서, port-forward 켜진 상태로):
    USE_DUMMY_DATA=false python3 scripts/functional_evaluation.py
"""
import sys
import math
import random
import statistics
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.append(str(REPO_ROOT))
sys.path.append(str(SRC_DIR))
sys.path.append(str(SRC_DIR / "aspect"))
sys.path.append(str(SRC_DIR / "features"))
sys.path.append(str(SRC_DIR / "recommend"))

from pipeline import build_reviews_with_ratings, recommend_for_pet
from deepfm_model import load_deepfm
from substitute_recommendation import find_substitute_products, build_review_summary_by_product

from src.data_access.db import get_connection
from src.data_access.pet_repository import get_pets_by_ids, MEMBER_DB_ENV, get_pet_by_id
from src.data_access.product_repository import list_products, get_product_by_id
from src.data_access.reviews_repository import load_reviews_with_reviewer_pet
from src.data_access.order_embedding_repository import get_product_embeddings

DEEPFM_MODEL_DIR = str(REPO_ROOT / "models" / "deepfm")
TOP_K = 9
SUB_SAMPLE_SIZE = 30
SUB_TOP_K = 4
LOO_MIN_ORDERS = 2  # leave-one-out 평가에 포함하려면 최소 2건 이상 구매해야 함 (1건은 숨길 정답, 나머지는 이력)
LOO_MAX_USERS = 60  # 평가 대상 사용자 수 상한 (실행 시간 제한)


def get_users_with_orders(min_orders):
    """order_db에서 구매 건수가 min_orders 이상인 member_id 목록과, 각자의 (product_id, ordered_at) 리스트."""
    conn = get_connection("ORDER_DATABASE_URL")
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT o.member_id, oi.product_id, o.created_at
                FROM orders o
                JOIN order_items oi ON oi.order_id = o.id
                ORDER BY o.member_id, o.created_at
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    by_user = defaultdict(list)
    for member_id, product_id, created_at in rows:
        by_user[member_id].append((product_id, created_at))

    return {u: items for u, items in by_user.items() if len(items) >= min_orders}


def get_pet_ids_for_user(user_id):
    conn = get_connection(MEMBER_DB_ENV)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM pet WHERE member_id = %s", (user_id,))
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def dcg_at_k(relevant_flags):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevant_flags))


def ndcg_at_k(rank_of_hit, k):
    """rank_of_hit: 정답 상품이 추천 리스트에서 몇 번째(0-base)에 있었는지. None이면 못 맞춘 것."""
    if rank_of_hit is None:
        return 0.0
    relevant_flags = [1 if i == rank_of_hit else 0 for i in range(k)]
    dcg = dcg_at_k(relevant_flags)
    idcg = dcg_at_k([1] + [0] * (k - 1))  # 정답이 1개뿐이므로 이상적 DCG는 1위에 있을 때
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(get_ranked_product_ids_fn, users_data, products_by_id, k=TOP_K):
    """
    get_ranked_product_ids_fn(user_id, pet_id, visible_history_ids) -> [product_id, ...] (순위대로)
    각 사용자에 대해 마지막 구매를 정답으로 숨기고 나머지로 추천 실행, 정답이 top-k 안에 있는지 평가.
    """
    ndcgs, precisions, recalls, hits = [], [], [], []
    n_evaluated = 0

    user_ids = list(users_data.keys())
    random.shuffle(user_ids)
    user_ids = user_ids[:LOO_MAX_USERS]

    for user_id in user_ids:
        items = users_data[user_id]  # [(product_id, created_at), ...] 시간순 정렬됨
        held_out_product_id = items[-1][0]
        visible_ids = [pid for pid, _ in items[:-1]]

        pet_ids = get_pet_ids_for_user(user_id)
        if not pet_ids:
            continue
        pet_id = pet_ids[0]

        try:
            ranked = get_ranked_product_ids_fn(user_id, pet_id, visible_ids)
        except Exception as e:
            print(f"   [스킵] user_id={user_id} 에러: {e}")
            continue
        if not ranked:
            continue

        n_evaluated += 1
        top_k_ids = ranked[:k]
        rank_of_hit = top_k_ids.index(held_out_product_id) if held_out_product_id in top_k_ids else None

        hit = 1 if rank_of_hit is not None else 0
        hits.append(hit)
        precisions.append(hit / k)  # 정답 1개 기준이라 precision@k = hit/k
        recalls.append(hit)  # 정답이 1개뿐이므로 recall@k = hit/1
        ndcgs.append(ndcg_at_k(rank_of_hit, k))

    if n_evaluated == 0:
        return None

    return {
        "n": n_evaluated,
        "ndcg": statistics.mean(ndcgs),
        "precision": statistics.mean(precisions),
        "recall": statistics.mean(recalls),
        "hit_rate": statistics.mean(hits),
    }


def main():
    print("0) 데이터/모델 로드...")
    products = [p for p in list_products() if p.get("status") == "ON_SALE"]
    product_map = {p["product_id"]: p for p in products}
    raw_reviews = load_reviews_with_reviewer_pet()
    reviews_by_product = build_reviews_with_ratings(raw_reviews)
    encoder, model = load_deepfm(DEEPFM_MODEL_DIR)
    print(f"   상품 {len(products)}개, 리뷰 {len(raw_reviews)}건")

    print("1) 구매 이력 2건 이상인 사용자 조회...")
    users_data = get_users_with_orders(LOO_MIN_ORDERS)
    print(f"   대상 사용자 {len(users_data)}명 (평가에는 최대 {LOO_MAX_USERS}명 랜덤 샘플)")

    if not users_data:
        print("   [중단] 구매 이력 2건 이상인 사용자가 없어 NDCG/Precision/Recall/HitRate 평가 불가.")
        print("   → 정량 평가 §2.4는 '실제 구매 로그 부족으로 측정 불가' 로 기록하고 대안(정성 평가로 보완) 필요.")
    else:
        def deepfm_ranker(user_id, pet_id, visible_ids):
            pet = get_pet_by_id(pet_id)
            purchase_history_embeddings = (
                list(get_product_embeddings(visible_ids).values()) if visible_ids else []
            )
            results = recommend_for_pet(
                pet, products, reviews_by_product, encoder, model,
                purchase_history_embeddings=purchase_history_embeddings,
            )
            return [r["product_id"] for r in results]

        def popularity_ranker(user_id, pet_id, visible_ids):
            return [p["product_id"] for p in sorted(products, key=lambda p: -(p.get("sales_count") or 0))]

        def rule_based_ranker(user_id, pet_id, visible_ids):
            pet = get_pet_by_id(pet_id)
            allergy_codes = set(pet.get("allergy_codes") or [])
            safe = [
                p for p in products
                if not (set(p.get("allergen_flags") or []) & allergy_codes)
            ]
            safe.sort(key=lambda p: p.get("created_at") or "", reverse=True)
            return [p["product_id"] for p in safe]

        print("2) DeepFM(제안 모델) 평가 실행...")
        deepfm_result = evaluate_ranking(deepfm_ranker, users_data, product_map)
        print("3) Baseline A(인기순) 평가 실행...")
        pop_result = evaluate_ranking(popularity_ranker, users_data, product_map)
        print("4) Baseline B(룰 기반) 평가 실행...")
        rule_result = evaluate_ranking(rule_based_ranker, users_data, product_map)

        print("\n=== [1] 홈 추천 비교표 (NDCG@9 / Precision@9 / Recall@9 / HitRate@9) ===")
        for name, r in [("Baseline A(인기순)", pop_result), ("Baseline B(룰기반)", rule_result), ("DeepFM(제안)", deepfm_result)]:
            if r is None:
                print(f"  {name}: 평가 불가 (데이터 부족)")
            else:
                print(f"  {name}: n={r['n']}, NDCG@9={r['ndcg']:.3f}, Precision@9={r['precision']:.3f}, "
                      f"Recall@9={r['recall']:.3f}, HitRate@9={r['hit_rate']:.3f}")

    print("\n5) 대체 상품 추천 평가 (카테고리 일치율 / 평균 코사인 유사도)...")
    on_sale = [p for p in products]
    random.shuffle(on_sale)
    base_products = on_sale[:SUB_SAMPLE_SIZE]

    match_count = 0
    total_subs = 0
    all_similarities = []

    for base_product in base_products:
        candidates = [
            p for p in products
            if p["product_id"] != base_product["product_id"] and p.get("status") == "ON_SALE"
            and p.get("category_code") == base_product.get("category_code")
        ]
        candidate_ids = [p["product_id"] for p in candidates]
        if not candidate_ids:
            continue
        embeddings = get_product_embeddings([base_product["product_id"]] + candidate_ids)
        if base_product["product_id"] not in embeddings:
            continue

        pet_sample_ids = get_pet_ids_for_user(1)  # 임시 pet 컨텍스트 (유사도 계산 자체는 pet 무관 임베딩 비교)
        pet = get_pet_by_id(pet_sample_ids[0]) if pet_sample_ids else None
        if pet is None:
            continue

        review_summary = build_review_summary_by_product(pet, candidate_ids, reviews=raw_reviews)
        try:
            subs = find_substitute_products(
                base_product=base_product, candidate_products=candidates,
                product_embeddings=embeddings, pet=pet, pet_age_group=None,
                review_summary_by_product=review_summary, top_k=SUB_TOP_K,
            )
        except Exception as e:
            print(f"   [스킵] base_product_id={base_product['product_id']} 에러: {e}")
            continue

        for sub in subs:
            total_subs += 1
            sub_product = product_map.get(sub["product_id"])
            if sub_product and sub_product.get("subcategory_code") == base_product.get("subcategory_code"):
                match_count += 1
            if "similarity" in sub:
                all_similarities.append(sub["similarity"])
            elif "score" in sub:
                all_similarities.append(sub["score"])

    if total_subs:
        print(f"   대체 추천 {total_subs}건 (base_product {len(base_products)}개 샘플 기준)")
        print(f"   서브카테고리 일치율: {match_count / total_subs * 100:.1f}%")
        if all_similarities:
            print(f"   평균 유사도(score): {statistics.mean(all_similarities):.3f}")
    else:
        print("   [경고] 대체 추천 결과가 없어 평가 불가 (product_embeddings 권한 이슈일 수 있음 — 모델 카드 §4 참조)")

    print("\n6) 알레르기 판정 정확도 평가...")
    conn = get_connection(MEMBER_DB_ENV)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM pet ORDER BY random() LIMIT 40")
            sample_pet_ids = [row[0] for row in cur.fetchall()]
    finally:
        conn.close()

    pets_by_id = get_pets_by_ids(sample_pet_ids)
    total_checked = 0
    mismatches = 0
    for pet in pets_by_id.values():
        try:
            results = recommend_for_pet(pet, products, reviews_by_product, encoder, model)
        except Exception:
            continue
        allergy_codes = set(pet.get("allergy_codes") or [])
        for r in results:
            product = product_map.get(r["product_id"])
            if not product:
                continue
            allergen_flags = set(product.get("allergen_flags") or [])
            has_conflict = bool(allergy_codes & allergen_flags)
            expected_status = "PENALIZED" if has_conflict else "SAFE"
            total_checked += 1
            if r["allergy_status"] not in (expected_status, "PENDING"):
                mismatches += 1
            elif has_conflict and r["allergy_status"] == "SAFE":
                mismatches += 1

    if total_checked:
        accuracy = (total_checked - mismatches) / total_checked * 100
        print(f"   검사 {total_checked}건, 불일치 {mismatches}건, 정확도 {accuracy:.2f}%")
    else:
        print("   [경고] 검사 대상 없음")

    print("\n완료.")


if __name__ == "__main__":
    main()