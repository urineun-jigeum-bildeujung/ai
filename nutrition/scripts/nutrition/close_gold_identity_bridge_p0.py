"""Create read-only closure artifacts for Nutrition Gold Identity Bridge P0.

No source record, engine rule, or runtime data is changed.  A manufacturer page
without its GTIN is not enough to attach a formulation to a persisted product.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "data" / "eval"

def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    source_queue = EVAL / "nutrition_evidence_acquisition_queue.csv"
    cat_queue = EVAL / "cat_taurine_acquisition_queue.csv"
    required = [source_queue, cat_queue, EVAL / "gold_product_candidates.csv", EVAL / "gold_product_gap_analysis.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("required input artifacts missing: " + ", ".join(missing))

    queue_rows = list(csv.DictReader(source_queue.open(encoding="utf-8")))
    # Preserve all 646 verified-GTIN Ca/P queue rows.  The 22 hard conflicts
    # remain eligible for a human review queue but are explicitly non-mergeable.
    for row in queue_rows:
        if row.get("identity_resolution_status") == "AUTO_MERGE_PROHIBITED":
            row["collection_state"] = "HUMAN_IDENTITY_REVIEW_REQUIRED"
        else:
            row["collection_state"] = "PENDING_EVIDENCE_ACQUISITION"
    write_csv(EVAL / "nutrition_evidence_acquisition_queue_v1.csv", queue_rows, list(queue_rows[0]) + ["collection_state"])

    # Official content page matches product family/name, but neither source
    # exposes a GTIN/label image.  That fails this task's identity bridge rule.
    blocked = [
        {
            "candidate_id": "ORIJEN_ORIGINAL_CAT_17KG", "product_id": "0064992280178", "canonical_gtin": "064992280178",
            "candidate_status": "BLOCKED", "blocked_reason": "OFFICIAL_PAGE_DOES_NOT_EXPOSE_GTIN_OR_GTIN_LABEL_IMAGE",
            "local_variant": "Original Cat; 17kg", "official_source_url": "https://www.orijenpetfoods.com/en-US/cats/cat-food/original-cat/ns-ori-catkitten.html",
            "observed_official_nutrients": "CALCIUM=1.4%; PHOSPHORUS=1.1%; TAURINE=0.2%", "additional_evidence_required": "manufacturer page or package image that visibly binds GTIN 064992280178 to Original Cat 17kg/formula", "verified": False, "runtime_eligible": False,
        },
        {
            "candidate_id": "ORIJEN_SIX_FISH_1_8KG", "product_id": "0064992281182", "canonical_gtin": "064992281182",
            "candidate_status": "BLOCKED", "blocked_reason": "OFFICIAL_PAGE_DOES_NOT_EXPOSE_GTIN_OR_GTIN_LABEL_IMAGE; DOG_CAT_VARIANT_NOT_RESOLVED_BY_APPROVED_IDENTITY_EVIDENCE",
            "local_variant": "Six Fish; 1.8kg", "official_source_url": "https://www.orijenpetfoods.com/en-US/cats/cat-food/six-fish/ds-ori-six-fish-cat.html",
            "observed_official_nutrients": "not persisted; GTIN-to-formula identity unresolved", "additional_evidence_required": "manufacturer page or package image that visibly binds GTIN 064992281182 to one DOG or CAT Six Fish formulation", "verified": False, "runtime_eligible": False,
        },
    ]
    write_csv(EVAL / "blocked_report_gold_v1.csv", blocked, list(blocked[0]))
    # The P1 validator now owns verified_evidence_gold_v1.*.  P0 must not
    # erase a later human-reviewed operational artifact merely because queues
    # or the identity-bridge report are regenerated.
    (EVAL / "e2e_result_gold_v1.json").write_text(json.dumps({
        "version": "v1", "executed": False,
        "execution_rule": "Run only when verified=true and runtime_eligible=true.",
        "reason": "Both candidates are BLOCKED; no verified persisted-product evidence exists.",
        "candidates": [{"candidate_id": row["candidate_id"], "status": "NOT_RUN_BLOCKED", "blocked_reason": row["blocked_reason"]} for row in blocked],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "ca_p_queue": len(queue_rows), "blocked": len(blocked),
        "operational_evidence_artifact": "P1_OWNED_NOT_MODIFIED", "e2e_executed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
