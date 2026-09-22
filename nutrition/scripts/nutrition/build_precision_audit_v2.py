"""Deterministic human-review sample builder; it never assigns review labels."""
from __future__ import annotations
import csv
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/eval/allergen_mapping_precision_audit_p2_v2.csv"
MANIFEST = ROOT / "data/eval/allergen_precision_audit_sampling_manifest_p2.json"
SEED = 20260915
QUOTAS = {"OPFF": 20, "OEM": 20, "GLOBAL": 20}
MAX_PER_PRODUCT = 2
FIELDS = ["product_id", "source_dataset", "raw_ingredient_text", "normalized_text", "segmented_text", "matched_text", "matched_alias", "allergen_code", "mapping_method", "confidence", "dictionary_version", "review_label", "reviewer_note"]

def text_class(row):
    text = f"{row.get('raw_ingredient_text','')} {row.get('segmented_text','')}"
    if not text.strip(): return "UNKNOWN"
    if "�" in text or re.search(r"(?:\b[a-z]\s){4,}|[\*_]{2,}", text, re.I): return "OCR_NOISY"
    if re.search(r"[,;/]|\b(and|et|및)\b", text, re.I): return "COMPOUND"
    hangul, latin, non_ascii = bool(re.search(r"[가-힣]", text)), bool(re.search(r"[A-Za-z]", text)), bool(re.search(r"[^\x00-\x7f]", text))
    if hangul and latin: return "MULTILINGUAL"
    if non_ascii: return "NON_ENGLISH"
    if latin: return "EN"
    return "UNKNOWN"

def priority(row):
    text = row["raw_ingredient_text"].lower()
    edge = int(bool(re.search(r"gluten|poultry|by-product|flavor|sweet potato|farine|dinde", text)))
    diverse = int(text_class(row) in {"COMPOUND", "MULTILINGUAL", "NON_ENGLISH", "OCR_NOISY"})
    exact = int(row["mapping_method"] == "CANONICAL_EXACT")
    return edge * 8 + diverse * 4 + exact * 2

def choose(rows, quota, rng):
    """Maximise product diversity first, then allow a second row per product."""
    grouped = defaultdict(list)
    for row in rows: grouped[row["product_id"]].append(row)
    for items in grouped.values():
        rng.shuffle(items); items.sort(key=priority, reverse=True)
    products = list(grouped); rng.shuffle(products)
    products.sort(key=lambda p: priority(grouped[p][0]), reverse=True)
    selected, seen = [], set()
    for product in products:
        if len(selected) == quota: break
        selected.append(grouped[product][0]); seen.add((product, grouped[product][0]["raw_ingredient_text"], grouped[product][0].get("allergen_code")))
    if len(selected) < quota:
        candidates = [r for r in rows if (r["product_id"], r["raw_ingredient_text"], r.get("allergen_code")) not in seen]
        rng.shuffle(candidates); candidates.sort(key=priority, reverse=True)
        per_product = Counter(r["product_id"] for r in selected)
        for row in candidates:
            if len(selected) == quota: break
            if per_product[row["product_id"]] >= MAX_PER_PRODUCT: continue
            selected.append(row); per_product[row["product_id"]] += 1
    return selected

def main():
    refs = json.loads((ROOT / "data/processed/product_allergen_refs_p1.json").read_text())["refs"]
    rng = random.Random(SEED); selected = []
    for source, quota in QUOTAS.items():
        rows = [row for row in refs if row["source_dataset"] == source and row["usable_for_safety"]]
        selected.extend(choose(rows, quota, rng))
    if len(selected) != sum(QUOTAS.values()): raise RuntimeError("source quota could not be met")
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader()
        for row in selected:
            writer.writerow({key: row.get(key, "") for key in FIELDS[:-2]} | {"review_label": "", "reviewer_note": ""})
    product_counts = Counter(row["product_id"] for row in selected)
    manifest = {"audit_version": "p2_v2", "source_file": "data/processed/product_allergen_refs_p1.json", "created_at": datetime.now(timezone.utc).isoformat(), "sampling_seed": SEED, "target_rows": 60, "source_quotas": QUOTAS, "max_rows_per_product": MAX_PER_PRODUCT, "unique_products": len(product_counts), "mapping_method_counts": dict(Counter(row["mapping_method"] for row in selected)), "text_class_counts": dict(Counter(text_class(row) for row in selected)), "selection_rule": "source quota 20 each; maximise unique products before selecting a second row; maximum two rows/product; prioritise deterministic edge, text-diversity, and CANONICAL_EXACT candidates; labels remain blank."}
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False))

if __name__ == "__main__": main()
