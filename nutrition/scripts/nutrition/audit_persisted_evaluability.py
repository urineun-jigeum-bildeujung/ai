"""Recompute local-artifact input evaluability with the shared readiness SoT."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


# This script is also executed directly from the repository root.  Make the
# sibling ``scripts`` directory importable so nutrition_readiness can use the
# canonical pipeline constants instead of duplicating them here.
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from nutrition_readiness import evaluate_nutrition_readiness, resolve_reference_stage
from product_input_adapter import ROOT, _read, load_product_input


OUT = ROOT / "data" / "eval" / "nutrition_input_evaluability_after_contract_v1.json"
SOURCES = (
    ("seed_9_placeholder_feed_opff.json", "items"),
    ("seed_9b_off_korean_oem.json", "products"),
    ("seed_9_global_brands_v2.json", "products"),
)
BEFORE = {"READY": 0, "PARTIAL": 24, "INSUFFICIENT_DATA": 370, "SAFETY_BLOCKED": 0, "UNSUPPORTED": 0}


def main() -> None:
    ids = sorted({row["product_id"] for filename, key in SOURCES for row in _read(filename).get(key, []) if row.get("product_id")})
    counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    rows = []
    stage = resolve_reference_stage({"life_stage": "adult", "age_years": 3})
    for product_id in ids:
        loaded = load_product_input(product_id)
        product = loaded["product"]
        target = product.get("target_species")
        compatible_species = target if target in {"dog", "cat"} else "dog"
        readiness = evaluate_nutrition_readiness(product, compatible_species, stage)
        status = readiness["input_readiness"]
        counts[status] += 1
        reason_counts.update(readiness["reason_codes"])
        rows.append({
            "product_id": product_id,
            "input_readiness": status,
            "reason_codes": readiness["reason_codes"],
            "source_dataset": loaded["provenance"]["product_source_dataset"],
        })
    after = {key: counts.get(key, 0) for key in BEFORE}
    OUT.write_text(json.dumps({
        "scope": "local artifact-backed input completeness; no user allergy profile applied",
        "before": BEFORE,
        "after": after,
        "delta": {key: after[key] - BEFORE[key] for key in BEFORE},
        "reason_code_counts": dict(sorted(reason_counts.items())),
        "product_count": len(ids),
        "rows": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"before": BEFORE, "after": after, "delta": {key: after[key] - BEFORE[key] for key in BEFORE}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
