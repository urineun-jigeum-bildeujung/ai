import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "scripts"
sys.path[:0] = [str(S), str(S / "nutrition")]
import api_nutrition
from allergen_repository import DB, TERMINAL_COMPONENT_STATUSES, get_refs


class Lineage(unittest.TestCase):
    def setUp(self):
        self.components = json.loads((ROOT / "data/processed/product_ingredient_components_p1_1.json").read_text())["components"]
        self.refs = json.loads((ROOT / "data/processed/product_allergen_refs_p1_1.json").read_text())["refs"]
        self.counts = Counter(row["component_occurrence_id"] for row in self.refs)
        self.component_ids = {row["component_occurrence_id"] for row in self.components}
        self.catalog = sqlite3.connect(DB)

    def tearDown(self):
        self.catalog.close()

    def _copied_catalog(self):
        handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        shutil.copy2(DB, handle.name)
        return handle.name

    def test_component_table_exists(self):
        self.assertIn("product_ingredient_components", {r[0] for r in self.catalog.execute("select name from sqlite_master where type='table'")})

    def test_persisted_component_count(self):
        self.assertEqual(len(self.components), 2814)
        self.assertEqual(self.catalog.execute("select count(*) from product_ingredient_components").fetchone()[0], 2814)

    def test_component_ids_are_unique(self):
        self.assertEqual(len(self.component_ids), 2814)

    def test_terminal_status_is_present_and_valid(self):
        self.assertTrue(all(row.get("processing_status") in TERMINAL_COMPONENT_STATUSES for row in self.components))

    def test_source_counts_sum_to_component_total(self):
        self.assertEqual(Counter(r["source_dataset"] for r in self.components), {"OPFF": 443, "OEM": 1470, "GLOBAL": 901})

    def test_status_counts_sum_to_component_total(self):
        self.assertEqual(Counter(r["processing_status"] for r in self.components), {"RESOLVED": 398, "PARTIALLY_RESOLVED": 5, "UNRESOLVED": 2411})

    def test_every_evidence_has_component_id(self):
        self.assertTrue(all(row.get("component_occurrence_id") for row in self.refs))

    def test_every_evidence_parent_exists(self):
        self.assertTrue(all(row["component_occurrence_id"] in self.component_ids for row in self.refs))

    def test_orphan_evidence_is_zero(self):
        self.assertEqual(self.catalog.execute("select count(*) from product_allergen_refs r left join product_ingredient_components c on r.component_occurrence_id=c.component_occurrence_id where c.component_occurrence_id is null").fetchone()[0], 0)

    def test_broken_parent_is_not_precomputed(self):
        path = self._copied_catalog()
        pid = self.refs[0]["product_id"]
        c = sqlite3.connect(path)
        parent = c.execute("select component_occurrence_id from product_allergen_refs where product_id=? limit 1", (pid,)).fetchone()[0]
        c.execute("delete from product_ingredient_components where component_occurrence_id=?", (parent,)); c.commit(); c.close()
        self.assertEqual(get_refs(pid, "allergen_sot_v1", "product_allergen_refs_p1.0", path)[1], "LINEAGE_INTEGRITY_ERROR")

    def test_invalid_terminal_status_is_not_precomputed(self):
        path = self._copied_catalog()
        pid = self.refs[0]["product_id"]
        c = sqlite3.connect(path)
        parent = c.execute("select component_occurrence_id from product_allergen_refs where product_id=? limit 1", (pid,)).fetchone()[0]
        c.execute("update product_ingredient_components set processing_status='' where component_occurrence_id=?", (parent,)); c.commit(); c.close()
        self.assertEqual(get_refs(pid, "allergen_sot_v1", "product_allergen_refs_p1.0", path)[1], "LINEAGE_INTEGRITY_ERROR")

    def test_dictionary_stale_fails_closed(self):
        self.assertEqual(get_refs(self.refs[0]["product_id"], "stale", "product_allergen_refs_p1.0")[1], "STALE_EVIDENCE")

    def test_pipeline_stale_fails_closed(self):
        self.assertEqual(get_refs(self.refs[0]["product_id"], "allergen_sot_v1", "stale")[1], "STALE_EVIDENCE")

    def test_resolved_components_have_evidence(self):
        self.assertTrue(all(self.counts[r["component_occurrence_id"]] >= 1 for r in self.components if r["processing_status"] == "RESOLVED"))

    def test_partially_resolved_has_evidence_and_reason(self):
        self.assertTrue(all(self.counts[r["component_occurrence_id"]] >= 1 and r.get("processing_reason") for r in self.components if r["processing_status"] == "PARTIALLY_RESOLVED"))

    def test_unresolved_has_reason(self):
        self.assertTrue(all(r.get("processing_reason") for r in self.components if r["processing_status"] == "UNRESOLVED"))

    def test_multiple_evidence_components_supported(self):
        self.assertEqual(sum(v > 1 for v in self.counts.values()), 37)
        self.assertEqual(max(self.counts.values()), 3)

    def test_raw_child_evidence_cardinality(self):
        self.assertEqual(sum(self.counts.values()), 2856)
        self.assertEqual((sum(v == 1 for v in self.counts.values()), sum(v == 2 for v in self.counts.values()), sum(v >= 3 for v in self.counts.values())), (2777, 32, 5))

    def test_catalog_deduplicates_only_four_evidence_rows(self):
        self.assertEqual(self.catalog.execute("select count(*) from product_allergen_refs").fetchone()[0], 2852)
        self.assertEqual(len(self.refs) - 2852, 4)

    def test_migration_report_matches_rebuild_counts(self):
        report = json.loads((ROOT / "data/processed/allergen_lineage_catalog_migration_p2.json").read_text())
        self.assertEqual((report["status"], report["new_component_rows"], report["new_evidence_rows"], report["duplicate_evidence"]), ("OK", 2814, 2856, 4))

    def test_normal_precomputed_path_is_preserved(self):
        self.assertEqual(get_refs(self.refs[0]["product_id"], "allergen_sot_v1", "product_allergen_refs_p1.0")[1], "PRECOMPUTED")

    def test_integrity_failure_reaches_shared_fail_close_contract(self):
        original = api_nutrition.get_refs
        api_nutrition.get_refs = lambda *_args: ([], "LINEAGE_INTEGRITY_ERROR")
        self.addCleanup(setattr, api_nutrition, "get_refs", original)
        pet = api_nutrition.PetIn(id="p", species="dog", age_years=2, weight_kg=4, allergies=["chicken"])
        product = api_nutrition.ProductIn(id="x", name="x", target_species="dog", aafco_life_stage="ADULT_MAINTENANCE")
        result = api_nutrition._allergy_gate(pet, product)
        self.assertEqual((result["safety_status"], result["evidence_trace"]), ("SAFETY_DATA_INSUFFICIENT", "LINEAGE_INTEGRITY_ERROR"))
