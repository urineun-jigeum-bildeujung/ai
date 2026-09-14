# -*- coding: utf-8 -*-
"""
번역 품질 필터링.

1단계 (규칙 기반, 빠름): 길이 비율 / 반복 어구 / 미번역 영어 잔존 비율
2단계 (의미 유사도): 다국어 문장 임베딩(paraphrase-multilingual-MiniLM-L12-v2)으로
                     영어 원문과 한국어 번역문의 코사인 유사도 계산 -> 의미가 너무 다르면 제외
                     (back-translation 없이, 같은 임베딩 공간에서 직접 비교하는 방식이라 훨씬 가벼움)

실행:
    python3 data/amazon/filter_translations.py \
        --input_path data/amazon/translated_reviews.jsonl \
        --passed_path data/amazon/filtered_reviews.jsonl \
        --rejected_path data/amazon/rejected_reviews.jsonl \
        --similarity_threshold 0.5
"""

import argparse
import json
import re
from collections import Counter


# ---------------------------------------------------------------------------
# 1단계: 규칙 기반 필터
# ---------------------------------------------------------------------------

def check_length_ratio(en_text: str, ko_text: str, min_ratio: float = 0.2, max_ratio: float = 3.0) -> bool:
    """한/영 문자 길이 비율이 정상 범위인지 (너무 짧거나 길면 번역 실패 신호)."""
    if len(en_text) == 0:
        return False
    ratio = len(ko_text) / len(en_text)
    return min_ratio <= ratio <= max_ratio


def check_min_length(ko_text: str, min_words: int = 2) -> bool:
    """번역 결과가 너무 짧지 않은지."""
    return len(ko_text.split()) >= min_words


def check_repetition(ko_text: str, max_repeat: int = 2) -> bool:
    """동일한 3-gram(어절 기준)이 반복되는지 (번역 붕괴 신호).
    자연스러운 한국어 문장에서 동일한 3어절 조합이 2번 이상 나오는 경우는 드물어서,
    엄격하게(2회 이상 반복 시 거부) 잡는다."""
    words = ko_text.split()
    if len(words) < 6:
        return True  # 너무 짧아서 반복 여부 판단 불가 -> 통과 (다른 필터에서 걸러짐)

    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    counts = Counter(trigrams)
    most_common_count = counts.most_common(1)[0][1] if counts else 0
    return most_common_count < max_repeat


def check_latin_ratio(ko_text: str, max_latin_ratio: float = 0.3) -> bool:
    """번역 결과에 영어(라틴 문자)가 과도하게 남아있는지 (미번역 신호)."""
    letters = [c for c in ko_text if c.isalpha()]
    if not letters:
        return True
    latin_count = sum(1 for c in letters if re.match(r"[a-zA-Z]", c))
    return (latin_count / len(letters)) <= max_latin_ratio


def apply_heuristic_filters(en_text: str, ko_text: str) -> tuple:
    """모든 규칙 기반 필터를 적용, (통과 여부, 실패 사유) 반환."""
    if not check_min_length(ko_text):
        return False, "too_short"
    if not check_length_ratio(en_text, ko_text):
        return False, "length_ratio_abnormal"
    if not check_repetition(ko_text):
        return False, "repetition_detected"
    if not check_latin_ratio(ko_text):
        return False, "untranslated_latin_text"
    return True, None


# ---------------------------------------------------------------------------
# 2단계: 의미 유사도 필터 (다국어 임베딩)
# ---------------------------------------------------------------------------

def compute_similarities(pairs: list, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> list:
    """
    pairs: [(en_text, ko_text), ...]
    반환: 각 쌍의 코사인 유사도 리스트 (0~1)
    """
    from sentence_transformers import SentenceTransformer, util

    print(f"다국어 임베딩 모델 로딩: {model_name}")
    model = SentenceTransformer(model_name)

    en_texts = [p[0] for p in pairs]
    ko_texts = [p[1] for p in pairs]

    print("임베딩 계산 중 (영어)...")
    en_embeddings = model.encode(en_texts, batch_size=32, show_progress_bar=True, convert_to_tensor=True)
    print("임베딩 계산 중 (한국어)...")
    ko_embeddings = model.encode(ko_texts, batch_size=32, show_progress_bar=True, convert_to_tensor=True)

    similarities = util.pairwise_cos_sim(en_embeddings, ko_embeddings).tolist()
    return similarities


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="data/amazon/translated_reviews.jsonl")
    parser.add_argument("--passed_path", default="data/amazon/filtered_reviews.jsonl")
    parser.add_argument("--rejected_path", default="data/amazon/rejected_reviews.jsonl")
    parser.add_argument("--similarity_threshold", type=float, default=0.5)
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    print(f"전체 {len(items)}건 필터링 시작")
    print("\n1단계: 규칙 기반 필터")

    heuristic_passed = []
    heuristic_rejected = []
    reject_reason_counts = Counter()

    for item in items:
        ok, reason = apply_heuristic_filters(item["review_text"], item["review_text_ko"])
        if ok:
            heuristic_passed.append(item)
        else:
            item["reject_reason"] = reason
            heuristic_rejected.append(item)
            reject_reason_counts[reason] += 1

    print(f"1단계 통과: {len(heuristic_passed)}건, 제외: {len(heuristic_rejected)}건")
    print(f"제외 사유: {dict(reject_reason_counts)}")

    print("\n2단계: 의미 유사도 필터 (다국어 임베딩)")
    pairs = [(item["review_text"], item["review_text_ko"]) for item in heuristic_passed]
    similarities = compute_similarities(pairs)

    final_passed = []
    final_rejected = []
    for item, sim in zip(heuristic_passed, similarities):
        item["semantic_similarity"] = round(sim, 4)
        if sim >= args.similarity_threshold:
            final_passed.append(item)
        else:
            item["reject_reason"] = "low_semantic_similarity"
            final_rejected.append(item)

    print(f"2단계 통과: {len(final_passed)}건, 제외: {len(final_rejected)}건 "
          f"(threshold={args.similarity_threshold})")

    all_rejected = heuristic_rejected + final_rejected

    with open(args.passed_path, "w", encoding="utf-8") as f:
        for item in final_passed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(args.rejected_path, "w", encoding="utf-8") as f:
        for item in all_rejected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n=== 최종 결과 ===")
    print(f"입력: {len(items)}건")
    print(f"통과: {len(final_passed)}건 ({len(final_passed)/len(items)*100:.1f}%) -> {args.passed_path}")
    print(f"제외: {len(all_rejected)}건 ({len(all_rejected)/len(items)*100:.1f}%) -> {args.rejected_path}")


if __name__ == "__main__":
    main()