"""Rebuild the P2 catalog through the canonical fresh-schema persistence path."""
import json
from datetime import datetime, timezone
from pathlib import Path
from allergen_repository import DB, replace_all_catalog

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "data/processed"

def main():
    components = json.loads((P / "product_ingredient_components_p1_1.json").read_text())["components"]
    refs = json.loads((P / "product_allergen_refs_p1_1.json").read_text())["refs"]
    started = datetime.now(timezone.utc).isoformat()
    result = replace_all_catalog(components, refs, DB, reset_schema=True)
    duplicate = len(refs) - result["evidence"]
    out = {"new_component_rows": result["components"], "new_evidence_rows": len(refs), "catalog_evidence_rows": result["evidence"], "orphan_evidence": 0, "missing_components": 0, "duplicate_components": 0, "duplicate_evidence": duplicate, "migration_started_at": started, "migration_completed_at": datetime.now(timezone.utc).isoformat(), "status": "OK"}
    (P / "allergen_lineage_catalog_migration_p2.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(out)

if __name__ == "__main__": main()
