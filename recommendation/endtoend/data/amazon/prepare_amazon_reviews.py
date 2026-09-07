# -*- coding: utf-8 -*-
"""
아마존 Pet Supplies 리뷰 데이터 필터링 + 샘플링.

원본 파일이 매우 커서(Pet_Supplies.json 3.31GB, 654만 건) 메모리에 통째로 올리지 않고
한 줄씩 스트리밍으로 읽으며 처리한다.

실행 (로컬, 도커 컨테이너 안):
    python3 data/amazon/prepare_amazon_reviews.py \
        --meta_path /data/meta_Pet_Supplies.json \
        --reviews_path /data/Pet_Supplies.json \
        --output_path data/amazon/sampled_reviews.jsonl

주의: 원본 파일(3.31GB, 2.68GB)은 용량이 커서 이 리포지토리에는 포함하지 않는다.
      도커 실행 시 -v 옵션으로 원본 파일이 있는 로컬 경로를 컨테이너에 마운트해서 사용한다.
"""

import argparse
import json
import random
import re

# -----------------------------
# 카테고리 매핑 규칙
# -----------------------------
# 아마존 category 배열(계층 구조) 안에 아래 키워드가 포함되어 있으면 해당 버킷으로 분류.
# 우리 프로젝트의 category_code(FOOD/SUPPLEMENT/TREAT)에 대응.
CATEGORY_RULES = {
    "FOOD": ["Food"],  # "Food Storage" 같은 오탐 방지는 아래에서 별도 처리
    "TREAT": ["Treats"],
    "SUPPLEMENT": ["Health Supplies", "Vitamins", "Supplements", "Joint Care", "Hip & Joint Care"],
}
FOOD_EXCLUDE_KEYWORDS = ["Storage", "Bowl", "Feeder", "Container"]  # Food와 헷갈리는 오탐 제외

SPECIES_KEYWORDS = {"DOG": ["Dogs", "Dog"], "CAT": ["Cats", "Cat"]}


def classify_category(category_list: list):
    """category(계층 리스트)를 보고 (category_bucket, species_guess) 반환. 매칭 안 되면 (None, None)."""
    if not category_list:
        return None, None

    joined = " > ".join(category_list)

    category_bucket = None
    for bucket, keywords in CATEGORY_RULES.items():
        for kw in keywords:
            if kw in joined:
                if bucket == "FOOD" and any(ex in joined for ex in FOOD_EXCLUDE_KEYWORDS):
                    continue
                category_bucket = bucket
                break
        if category_bucket:
            break

    species_guess = None
    for species, keywords in SPECIES_KEYWORDS.items():
        if any(kw in joined for kw in keywords):
            species_guess = species
            break

    return category_bucket, species_guess


def build_asin_map(meta_path: str) -> dict:
    """meta_Pet_Supplies.json을 스트리밍으로 읽어 asin -> {category_bucket, species_guess} 매핑 구축."""
    asin_map = {}
    total = 0
    matched = 0

    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            if total % 50000 == 0:
                print(f"  meta 처리 중... {total}건 읽음, {matched}건 매칭")
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            asin = item.get("asin")
            category_list = item.get("category", [])
            if not asin:
                continue

            bucket, species = classify_category(category_list)
            if bucket:
                asin_map[asin] = {"category_bucket": bucket, "species_guess": species}
                matched += 1

    print(f"meta 처리 완료: 전체 {total}건 중 {matched}건이 FOOD/TREAT/SUPPLEMENT로 매칭됨")
    return asin_map


def is_valid_text(text: str, min_words: int = 3, max_words: int = 100) -> bool:
    if not text:
        return False
    word_count = len(text.split())
    return min_words <= word_count <= max_words


def sample_reviews(reviews_path: str, asin_map: dict, quotas: dict, seed: int = 42) -> list:
    """
    Pet_Supplies.json을 스트리밍으로 읽으며,
    - asin_map에 있는(카테고리 매칭된) 상품의 리뷰만
    - 텍스트 길이 조건을 만족하고
    - 별점 그룹(POSITIVE: 4,5 / NEGATIVE: 1,2,3)별 목표 건수(quotas)를 채울 때까지 수집.
    각 그룹 저장소 크기가 목표치를 넘으면 이후 해당 그룹은 확률적으로 교체(reservoir 유사 방식)
    하지 않고, 목표 건수 도달 즉시 해당 그룹 수집을 멈춘다 (단순하고 충분히 랜덤한 순서 가정).
    """
    random.seed(seed)
    collected = {"POSITIVE": [], "NEGATIVE": []}
    total_read = 0
    matched_count = 0

    with open(reviews_path, "r", encoding="utf-8") as f:
        for line in f:
            total_read += 1
            if total_read % 200000 == 0:
                print(f"  reviews 처리 중... {total_read}건 읽음, "
                      f"POSITIVE {len(collected['POSITIVE'])}/{quotas['POSITIVE']}, "
                      f"NEGATIVE {len(collected['NEGATIVE'])}/{quotas['NEGATIVE']}")

            # 두 그룹 다 목표 달성하면 조기 종료
            if len(collected["POSITIVE"]) >= quotas["POSITIVE"] and len(collected["NEGATIVE"]) >= quotas["NEGATIVE"]:
                print(f"목표 건수 도달 -> 조기 종료 (총 {total_read}건 스캔)")
                break

            line = line.strip()
            if not line:
                continue
            try:
                review = json.loads(line)
            except json.JSONDecodeError:
                continue

            asin = review.get("asin")
            if asin not in asin_map:
                continue

            rating = review.get("overall")
            text = review.get("reviewText", "")
            if rating is None or not is_valid_text(text):
                continue

            group = "POSITIVE" if rating >= 4 else "NEGATIVE"
            if len(collected[group]) >= quotas[group]:
                continue  # 이 그룹은 이미 목표 달성, 더 이상 안 모음

            meta = asin_map[asin]
            collected[group].append({
                "asin": asin,
                "category_bucket": meta["category_bucket"],
                "species_guess": meta["species_guess"],
                "rating": rating,
                "review_text": text,
                "summary": review.get("summary", ""),
            })
            matched_count += 1

    print(f"\n최종 수집: POSITIVE {len(collected['POSITIVE'])}건, NEGATIVE {len(collected['NEGATIVE'])}건 "
          f"(전체 {total_read}건 스캔, {matched_count}건 매칭)")

    all_reviews = collected["POSITIVE"] + collected["NEGATIVE"]
    random.shuffle(all_reviews)
    return all_reviews


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_path", required=True, help="meta_Pet_Supplies.json 경로")
    parser.add_argument("--reviews_path", required=True, help="Pet_Supplies.json 경로")
    parser.add_argument("--output_path", default="data/amazon/sampled_reviews.jsonl")
    parser.add_argument("--target_total", type=int, default=6000, help="샘플링할 전체 건수 목표")
    parser.add_argument("--positive_ratio", type=float, default=0.6, help="긍정(별점 4,5) 비율")
    args = parser.parse_args()

    quotas = {
        "POSITIVE": int(args.target_total * args.positive_ratio),
        "NEGATIVE": int(args.target_total * (1 - args.positive_ratio)),
    }
    print(f"목표 건수: POSITIVE {quotas['POSITIVE']}건, NEGATIVE {quotas['NEGATIVE']}건")

    print("\n1단계: meta 파일에서 asin -> 카테고리 매핑 구축")
    asin_map = build_asin_map(args.meta_path)

    print("\n2단계: reviews 파일 스트리밍 필터링 + 샘플링")
    sampled = sample_reviews(args.reviews_path, asin_map, quotas)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for item in sampled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n저장 완료: {args.output_path} ({len(sampled)}건)")

    # 카테고리 분포 요약
    bucket_counts = {}
    for item in sampled:
        b = item["category_bucket"]
        bucket_counts[b] = bucket_counts.get(b, 0) + 1
    print(f"카테고리 분포: {bucket_counts}")


if __name__ == "__main__":
    main()