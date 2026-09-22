"""Deterministic P1 batch: PRODUCT_LABEL -> auditable product_allergen_refs.

No external requests, fuzzy promotion, or mutation of source records occurs here.
"""
from __future__ import annotations
import json, re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from allergen_service import DICTIONARY, SAFE_METHODS, product_allergen_refs
from allergen_repository import replace_all_refs
from allergen_catalog_versions import DICTIONARY_VERSION, PIPELINE_VERSION

ROOT = Path(__file__).resolve().parents[2]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
SOURCES = [
    ("OPFF", "seed_9_placeholder_feed_opff.json", "items", "ingredients"),
    ("OEM", "seed_9b_off_korean_oem.json", "products", "ingredients_text"),
    ("GLOBAL", "seed_9_global_brands_v2.json", "products", "ingredients_parsed"),
]

def components(value):
    if isinstance(value, list): return [str(x).strip() for x in value if str(x).strip()]
    return [x.strip() for x in re.split(r"[,，;；/]", str(value)) if x.strip()]

def build():
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    evidence, coverage, unresolved, candidates = [], {}, Counter(), defaultdict(lambda: {"aliases": set(), "sources": set(), "products": set()})
    for source, filename, list_key, ingredient_key in SOURCES:
        rows = json.loads((RAW / filename).read_text(encoding="utf-8"))[list_key]
        stats = Counter(products_with_data=0, total_components=0, resolved_safe=0, resolved_non_safe=0, unresolved=0,
                        fully_safety_resolvable=0, partially_resolved=0, completely_unresolved=0)
        for index, row in enumerate(rows):
            value = row.get(ingredient_key)
            if not value: continue
            parts = components(value); stats["products_with_data"] += 1; stats["total_components"] += len(parts)
            pid = str(row.get("id") or row.get("product_id") or f"{source}:{index}")
            refs = product_allergen_refs(pid, parts, source="PRODUCT_LABEL", source_version=filename)
            safe_raw = {r["raw_text"] for r in refs if r["mapping_method"] in SAFE_METHODS}
            for r in refs:
                r.update({"raw_ingredient_text": r.pop("raw_text"), "evidence_scope": r.pop("source"), "source_dataset": source, "ingredient_source": "PRODUCT_LABEL", "segmented_text": components(r["normalized_text"]),
                          "matched_alias": r.get("matched_text"), "evidence_status": "RESOLVED" if r["mapping_method"] in SAFE_METHODS else "UNRESOLVED",
                          "usable_for_safety": r["mapping_method"] in SAFE_METHODS, "pipeline_version": PIPELINE_VERSION,
                          "created_at": created_at, "source_row_id": index})
                evidence.append(r)
                if not r["usable_for_safety"]:
                    phrase = r.get("matched_text") or r["normalized_text"]
                    unresolved[(source, phrase)] += 1
                    if r.get("allergen_code") == "turkey":
                        c = candidates["turkey"]; c["aliases"].add("dinde"); c["sources"].add(source); c["products"].add(pid)
            safe_count = len(safe_raw); stats["resolved_safe"] += safe_count; stats["unresolved"] += len(parts) - safe_count
            if safe_count == len(parts): stats["fully_safety_resolvable"] += 1
            elif safe_count: stats["partially_resolved"] += 1
            else: stats["completely_unresolved"] += 1
        stats["resolved_non_safe"] = 0
        coverage[source] = dict(stats)
    evidence.sort(key=lambda r: (r["ingredient_source"], r["product_id"], r["raw_ingredient_text"], str(r["allergen_code"])))
    pareto = []
    total = sum(unresolved.values())
    for rank, ((source, phrase), count) in enumerate(unresolved.most_common(), 1):
        pareto.append({"rank": rank, "normalized_unresolved_phrase": phrase, "count": count, "source": source,
                       "candidate": "turkey" if phrase == "dinde" else None, "action": "REVIEW_REQUIRED"})
    candidate_rows = [{"candidate_code": code, "aliases": sorted(v["aliases"]), "languages": ["fr"],
                       "occurrence_count": sum(x["count"] for x in pareto if x["candidate"] == code),
                       "example_products": sorted(v["products"])[:10], "evidence_source": sorted(v["sources"]),
                       "proposed_status": "PROPOSED", "review_required": True} for code, v in sorted(candidates.items())]
    payload = {"schema_version": "product_allergen_refs_v1", "pipeline_version": PIPELINE_VERSION,
               "dictionary_version": DICTIONARY_VERSION, "created_at": created_at, "refs": evidence,
               "coverage": coverage, "pareto": pareto,
               "pareto_share": {str(n): sum(x["count"] for x in pareto[:n]) / total if total else 0 for n in (20,50,100)}}
    (OUT / "product_allergen_refs_p1.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["catalog_row_count"] = replace_all_refs(evidence)
    (OUT / "allergen_dictionary_candidates.json").write_text(json.dumps({"dictionary_version": DICTIONARY["version"], "candidates": candidate_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload

if __name__ == "__main__":
    result = build(); print(len(result["refs"]), result["coverage"])
