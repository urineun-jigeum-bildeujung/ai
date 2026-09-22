"""Compare the frozen v2 human-audit rows with current operational evidence."""
from __future__ import annotations
import csv
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "data/eval/allergen_mapping_precision_audit_p2_v2.csv"
MANIFEST = ROOT / "data/eval/allergen_precision_audit_sampling_manifest_p2.json"
OUT = ROOT / "data/eval/allergen_precision_audit_v2_parity_p2.json"
DB = ROOT / "data/processed/allergen_evidence_catalog_p2.db"
FIELDS = ("product_id", "source_dataset", "raw_ingredient_text", "normalized_text", "matched_text", "matched_alias", "allergen_code", "mapping_method", "dictionary_version")

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

def signature(row): return tuple((row.get(field) or "") for field in FIELDS)

def main():
    with AUDIT.open(encoding="utf-8", newline="") as handle: audit = list(csv.DictReader(handle))
    connection = sqlite3.connect(DB); connection.row_factory = sqlite3.Row
    catalog = [dict(row) for row in connection.execute("select evidence_id,component_occurrence_id," + ",".join(FIELDS) + " from product_allergen_refs")]
    version = dict(connection.execute("select dictionary_version,pipeline_version from product_allergen_refs limit 1").fetchone())
    connection.close()
    index = {}
    for row in catalog: index.setdefault(signature(row), []).append(row)
    differences = []
    counts = Counter()
    for number, row in enumerate(audit, 1):
        matches = index.get(signature(row), [])
        if len(matches) == 1:
            counts["exact_semantic_match"] += 1
            differences.append({"row_number": number, "status": "exact_semantic_match", "evidence_id": matches[0]["evidence_id"], "component_occurrence_id": matches[0]["component_occurrence_id"]})
        elif not matches:
            counts["missing_from_current"] += 1
            differences.append({"row_number": number, "status": "missing_from_current", "audit_signature": dict(zip(FIELDS, signature(row)))})
        else:
            counts["ambiguous_multiple_match"] += 1
            differences.append({"row_number": number, "status": "ambiguous_multiple_match", "evidence_ids": [x["evidence_id"] for x in matches]})
    result = {"audit_rows": len(audit), "exact_matches": counts["exact_semantic_match"], "changed_mapping": 0, "missing": counts["missing_from_current"], "ambiguous": counts["ambiguous_multiple_match"], "current_catalog_version": version["pipeline_version"], "dictionary_version": version["dictionary_version"], "parity_status": "PASS" if counts == Counter({"exact_semantic_match": len(audit)}) else "FAIL", "row_differences": differences}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    manifest = json.loads(MANIFEST.read_text())
    manifest["text_class_counts"] = dict(Counter(text_class(row) for row in audit))
    manifest["text_class_rule"] = "COMPOUND와 OCR_NOISY를 우선 분류한다. ASCII 영문은 EN, 비ASCII 단일 언어 판별 불가 텍스트는 NON_ENGLISH, 한글과 라틴 문자가 함께 있을 때만 MULTILINGUAL, 그 외는 UNKNOWN이다."
    manifest["canonical_parity_status"] = result["parity_status"]
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__": main()
