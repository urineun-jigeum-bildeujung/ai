"""영양성분 정규화 모듈 (8/20)

핵심:
- 시드 11 (allergen_ingredient_map) normalized exact-match 매핑
- NFKC + lowercase + strip 정규화
- 결측치 = "not_stated" 통일
- 단위 정규화 (PERCENT / KCAL_PER_KG / MG / G)

사용법:
    from ai.scripts.nutrition.normalizer import normalize_ingredient, normalize_ingredient_list, normalize_nutrient, not_stated_for_missing
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any
from allergen_service import product_allergen_refs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_11_PATH = PROJECT_ROOT / "data" / "raw" / "seed_11_allergen_ingredient_map.json"


def _load_seed_11() -> dict:
    return json.loads(SEED_11_PATH.read_text(encoding="utf-8"))


SEED_11 = _load_seed_11()


def _normalize_text(s: str) -> str:
    """NFKC + lowercase + strip — 시드 11 match_rule.normalization 동일"""
    return unicodedata.normalize("NFKC", s).lower().strip()


def normalize_ingredient(raw: str) -> dict:
    """원시 성분 → 시드 11 매핑 결과.

    Returns:
        {
            "raw": str,
            "normalized": str | None,         # 시드 11 ingredient_code[0] 또는 raw.upper().replace(" ", "_")
            "matched_allergen": str | None,    # 시드 11 allergen_code
            "matched_keyword": str | None,
            "allergen_name_ko": str | None,
            "severity": str | None,            # critical/severe/moderate
        }
    """
    if not raw or not isinstance(raw, str):
        return {"raw": raw or "", "normalized": None, "matched_allergen": None, "matched_keyword": None, "allergen_name_ko": None, "severity": None}
    raw_norm = _normalize_text(raw)
    for ref in product_allergen_refs("normalizer", [raw]):
        if ref.get("allergen_code") and ref["mapping_method"] in {"CANONICAL_EXACT", "CANONICAL_ALIAS"}:
            return {"raw": raw, "normalized": ref["allergen_code"].upper(), "matched_allergen": ref["allergen_code"],
                    "matched_keyword": ref["matched_text"], "allergen_name_ko": None, "severity": None}
    return {
        "raw": raw,
        "normalized": raw.upper().replace(" ", "_").replace("-", "_"),
        "matched_allergen": None,
        "matched_keyword": None,
        "allergen_name_ko": None,
        "severity": None,
    }


def normalize_ingredient_list(raw_list: list[str] | None) -> list[dict]:
    """여러 원시 성분 일괄 정규화"""
    if not raw_list:
        return []
    return [normalize_ingredient(r) for r in raw_list]


def not_stated_for_missing(value: Any) -> str:
    """결측치 통일 — "not_stated" / "ok"

    매칭: None, "", "-", "?", "N/A", "n/a", "unknown", "null", "None" → not_stated
    """
    if value is None:
        return "not_stated"
    if isinstance(value, str) and value.strip() in ("", "-", "?", "N/A", "n/a", "unknown", "Unknown", "null", "NULL", "None", "not_stated"):
        return "not_stated"
    return "ok"


def normalize_nutrient(value: Any, unit: str = "PERCENT") -> dict:
    """단위 정규화

    Returns:
        {"value": float | None, "unit": str, "state": "ok" | "not_stated"}
    """
    state = not_stated_for_missing(value)
    if state == "not_stated":
        return {"value": None, "unit": unit, "state": "not_stated"}
    try:
        v = float(value)
    except (ValueError, TypeError):
        return {"value": None, "unit": unit, "state": "not_stated"}
    return {"value": v, "unit": unit, "state": "ok"}


def extract_allergens_from_text(text: str) -> list[dict]:
    """원시 텍스트 → exact token 알레르기 매칭 결과.

    예: "닭고기, 쌀, 비트펄프" → [{"allergen_code": "chicken", "matched_keyword": "닭고기", ...}]
    """
    if not text or not isinstance(text, str):
        return []
    tokens = {_normalize_text(token) for token in text.split(",") if token.strip()}
    matched = []
    for item in SEED_11["items"]:
        for kw in item["matched_keywords"]:
            kw_norm = _normalize_text(kw)
            if kw_norm in tokens:
                matched.append({
                    "allergen_code": item["allergen_code"],
                    "allergen_name_ko": item["allergen_name_ko"],
                    "severity": item["severity"],
                    "matched_keyword": kw,
                })
                break
    return matched


if __name__ == "__main__":
    # 1건 self-test
    samples = ["닭고기", "chicken", "닭가슴살", "연어", "쌀", "옥수수", "비트펄프", "오메가-3"]
    print("=== normalize_ingredient self-test ===")
    for s in samples:
        r = normalize_ingredient(s)
        flag = "[알레르기]" if r["matched_allergen"] else "[일반]"
        print(f"  {s:15s} -> {flag} {r['normalized']} (allergen={r['matched_allergen']}, severity={r['severity']})")

    print()
    print("=== not_stated_for_missing self-test ===")
    for v in [None, "", "-", "?", "23.5", 0, "N/A"]:
        print(f"  {repr(v):8s} -> {not_stated_for_missing(v)}")

    print()
    print("=== extract_allergens_from_text self-test ===")
    txt = "닭고기, 쌀, 비트펄프, 오메가3"
    print(f"  '{txt}' -> {[m['allergen_code'] for m in extract_allergens_from_text(txt)]}")
