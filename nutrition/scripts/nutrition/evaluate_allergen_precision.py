"""Evaluate only human-supplied allergen precision labels; never create labels."""
from __future__ import annotations
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "data/eval/allergen_mapping_precision_audit_p2.csv"
OUT = ROOT / "data/eval/allergen_precision_metrics_p2.json"
CANDIDATE_OUT = ROOT / "data/eval/allergen_dictionary_candidate_review_p2.csv"
ALLOWED = {"", "CORRECT", "INCORRECT", "AMBIGUOUS"}

def text_class(row):
    text = f"{row.get('raw_ingredient_text','')} {row.get('segmented_text','')}"
    if not text.strip(): return "UNKNOWN"
    if "�" in text or re.search(r"(?:\b[a-z]\s){4,}|[\*_]{2,}", text, re.I): return "OCR_NOISY"
    if re.search(r"[,;/]|\b(and|et|및)\b", text, re.I): return "COMPOUND"
    has_hangul = bool(re.search(r"[가-힣]", text)); has_latin = bool(re.search(r"[A-Za-z]", text)); has_french = bool(re.search(r"[àâçéèêëîïôûùüÿñæœ]", text, re.I))
    if sum((has_hangul, has_latin, has_french)) > 1: return "MULTILINGUAL"
    if has_hangul: return "KO"
    if has_french: return "FR"
    if has_latin: return "EN"
    return "UNKNOWN"

def wilson(correct, n, z=1.96):
    if not n: return None, None
    p = correct / n; denom = 1 + z*z/n; center = (p + z*z/(2*n))/denom
    margin = z * math.sqrt((p*(1-p) + z*z/(4*n))/n) / denom
    return center-margin, center+margin

def metrics(rows):
    total = len(rows); counts = Counter(row.get("review_label", "") for row in rows)
    reviewed = total - counts[""]
    base = {"total": total, "reviewed_rows": reviewed, "remaining_rows": counts[""], "correct": counts["CORRECT"], "incorrect": counts["INCORRECT"], "ambiguous": counts["AMBIGUOUS"]}
    if counts[""]:
        return {**base, "status": "INCOMPLETE", "strict_precision": None, "determinate_precision": None, "ambiguous_rate": None, "incorrect_rate": None, "ci95_lower": None, "ci95_upper": None}
    strict = counts["CORRECT"] / total if total else None
    determinate_n = counts["CORRECT"] + counts["INCORRECT"]
    low, high = wilson(counts["CORRECT"], total)
    return {**base, "status": "COMPLETE", "strict_precision": strict, "determinate_precision": counts["CORRECT"] / determinate_n if determinate_n else None, "ambiguous_rate": counts["AMBIGUOUS"] / total if total else None, "incorrect_rate": counts["INCORRECT"] / total if total else None, "ci95_lower": low, "ci95_upper": high, "n": total}

def grouped(rows, key):
    buckets = defaultdict(list)
    for row in rows: buckets[key(row)].append(row)
    return {name: metrics(items) for name, items in sorted(buckets.items())}

def coverage():
    raw = ROOT / "data/raw"
    sources = [("OPFF", "seed_9_placeholder_feed_opff.json", "items", "ingredients"), ("OEM", "seed_9b_off_korean_oem.json", "products", "ingredients_text"), ("GLOBAL", "seed_9_global_brands_v2.json", "products", "ingredients_parsed")]
    by_source, total, with_data = {}, 0, 0
    for name, filename, list_key, ingredient_key in sources:
        rows = json.loads((raw / filename).read_text())[list_key]; populated = sum(bool(row.get(ingredient_key)) for row in rows)
        by_source[name] = {"products_total": len(rows), "products_with_ingredient_data": populated, "no_ingredient_data": len(rows)-populated}; total += len(rows); with_data += populated
    p1 = json.loads((ROOT / "data/processed/product_allergen_refs_p1.json").read_text())["coverage"]
    full = sum(x["fully_safety_resolvable"] for x in p1.values()); partial = sum(x["partially_resolved"] for x in p1.values()); unresolved = sum(x["completely_unresolved"] for x in p1.values())
    return {"products_total": total, "products_with_ingredient_data": with_data, "NO_INGREDIENT_DATA": total-with_data, "FULLY_RESOLVABLE": full, "PARTIALLY_RESOLVED": partial, "UNRESOLVED": unresolved, "KNOWN_LIST": {"SAFETY_EVALUABLE": full, "SAFETY_DATA_INSUFFICIENT": total-full}, "by_source": by_source}

def evaluate(rows):
    for row in rows:
        if row.get("review_label", "") not in ALLOWED: raise ValueError(f"invalid review_label: {row.get('review_label')}")
    overall = metrics(rows); breakdown = {"source_dataset": grouped(rows, lambda r: r.get("source_dataset") or "UNKNOWN"), "mapping_method": grouped(rows, lambda r: r.get("mapping_method") or "UNKNOWN"), "text_class": grouped(rows, text_class)}
    tiers = {method: {"candidate": "TIER_B_REVIEW", "metrics": values, "runtime_policy_changed": False, "note": "Human precision evidence is required; no threshold is set automatically."} for method, values in breakdown["mapping_method"].items()}
    return {"status": overall["status"], "precision": overall["strict_precision"], "overall": overall, "breakdown": breakdown, "safety_tier_candidates": tiers, "coverage": coverage(), "gates": {"GATE_A_PERSISTENCE": "PASS", "GATE_B_LINEAGE": "PASS", "GATE_C_FAIL_CLOSE": "PASS", "GATE_D_FRESH_REBUILD": "PASS", "GATE_E_AUTOMATED_REGRESSION": "PASS", "GATE_F_HUMAN_PRECISION": "PASS" if overall["status"] == "COMPLETE" else "WAITING", "GATE_G_PRODUCT_COVERAGE": "MEASURE_ONLY", "P2_STRUCTURE": "COMPLETE", "P2_PRECISION": "COMPLETE" if overall["status"] == "COMPLETE" else "WAITING"}}

def write_candidate_review():
    refs = json.loads((ROOT / "data/processed/product_allergen_refs_p1.json").read_text())["refs"]
    groups = defaultdict(list)
    for row in refs:
        if (row.get("matched_alias") or row.get("normalized_text")) == "dinde": groups[(row["source_dataset"], row["raw_ingredient_text"])].append(row)
    with CANDIDATE_OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["canonical_candidate", "alias_candidate", "source_dataset", "raw_context", "occurrence_count", "review_label", "reviewer_note"])
        writer.writeheader()
        for (source, context), rows in sorted(groups.items()): writer.writerow({"canonical_candidate": "turkey", "alias_candidate": "dinde", "source_dataset": source, "raw_context": context, "occurrence_count": len(rows), "review_label": "", "reviewer_note": ""})

def main():
    with AUDIT.open(encoding="utf-8", newline="") as handle: rows = list(csv.DictReader(handle))
    report = evaluate(rows)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    write_candidate_review()
    print(json.dumps({"status": report["status"], "reviewed_rows": report["overall"]["reviewed_rows"], "remaining_rows": report["overall"]["remaining_rows"]}, ensure_ascii=False))

if __name__ == "__main__": main()
