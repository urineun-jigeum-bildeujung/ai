import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "scripts"
sys.path[:0] = [str(S), str(S / "nutrition")]
from allergen_repository import get_refs, upsert


class P2Catalog(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((S.parent / "data/processed/product_allergen_refs_p1.json").read_text())
        self.lineage = json.loads((S.parent / "data/processed/product_allergen_refs_p1_1.json").read_text())
        self.temp = tempfile.NamedTemporaryFile(suffix=".db", delete=True)

    def tearDown(self):
        self.temp.close()

    def test_idempotent_catalog_and_provenance(self):
        self.assertEqual(upsert(self.data["refs"], self.temp.name), 2852)
        self.assertEqual(upsert(list(reversed(self.data["refs"])), self.temp.name), 2852)
        self.assertEqual({r["source_dataset"] for r in self.data["refs"]}, {"OPFF", "OEM", "GLOBAL"})

    def test_read_and_stale(self):
        row = self.lineage["refs"][0]
        self.assertEqual(get_refs(row["product_id"], "allergen_sot_v1", "product_allergen_refs_p1.0")[1], "PRECOMPUTED")
        self.assertEqual(get_refs(row["product_id"], "other", "product_allergen_refs_p1.0")[1], "STALE_EVIDENCE")
        self.assertEqual(get_refs("missing", "allergen_sot_v1", "product_allergen_refs_p1.0")[1], "NO_EVIDENCE")

    def test_root_cause_denominators(self):
        report = json.loads((S.parent / "data/processed/allergen_unresolved_root_causes_p2.json").read_text())
        self.assertEqual(sum(sum(v["root_causes"].values()) for v in report.values()), 2416)
