import json
import sqlite3
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "scripts"
sys.path[:0] = [str(S), str(S / "nutrition")]
import api_nutrition
from allergen_catalog_versions import DICTIONARY_VERSION, PIPELINE_VERSION
from allergen_repository import EVIDENCE_IDENTITY_FIELDS, connect, get_refs, identity, replace_all_catalog, upsert_refs


class FreshPersistence(unittest.TestCase):
    def setUp(self):
        self.components = json.loads((ROOT / "data/processed/product_ingredient_components_p1_1.json").read_text())["components"]
        self.refs = json.loads((ROOT / "data/processed/product_allergen_refs_p1_1.json").read_text())["refs"]
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=True)

    def _fresh(self):
        return replace_all_catalog(self.components, self.refs, self.tmp.name)

    def tearDown(self):
        self.tmp.close()

    def test_empty_db_bootstrap_has_component_id_column(self):
        c = connect(self.tmp.name)
        self.assertIn("component_occurrence_id", {r[1] for r in c.execute("pragma table_info(product_allergen_refs)")})
        self.assertIn("product_ingredient_components", {r[0] for r in c.execute("select name from sqlite_master where type='table'")})
        c.close()

    def test_fresh_build_persists_lineage_and_returns_precomputed(self):
        self.assertEqual(self._fresh(), {"components": 2814, "evidence": 2852})
        c = sqlite3.connect(self.tmp.name)
        self.assertEqual(c.execute("select count(*) from product_allergen_refs where component_occurrence_id='' or component_occurrence_id is null").fetchone()[0], 0)
        self.assertEqual(c.execute("select count(*) from product_allergen_refs r left join product_ingredient_components p on r.component_occurrence_id=p.component_occurrence_id where p.component_occurrence_id is null").fetchone()[0], 0)
        c.close()
        self.assertEqual(get_refs(self.refs[0]["product_id"], DICTIONARY_VERSION, PIPELINE_VERSION, self.tmp.name)[1], "PRECOMPUTED")

    def test_fresh_api_gate_uses_precomputed_evidence(self):
        self._fresh()
        original = api_nutrition.get_refs
        api_nutrition.get_refs = lambda product_id, dictionary, pipeline: get_refs(product_id, dictionary, pipeline, self.tmp.name)
        self.addCleanup(setattr, api_nutrition, "get_refs", original)
        pet = api_nutrition.PetIn(id="p", species="dog", age_years=2, weight_kg=4, allergies=[])
        product = api_nutrition.ProductIn(id=self.refs[0]["product_id"], name="x", target_species="dog", aafco_life_stage="ADULT_MAINTENANCE")
        self.assertEqual(api_nutrition._allergy_gate(pet, product)["evidence_trace"], "PRECOMPUTED")

    def test_identity_excludes_created_at(self):
        changed = deepcopy(self.refs[0]); changed["created_at"] = "2099-01-01T00:00:00+00:00"
        self.assertEqual(identity(self.refs[0]), identity(changed))
        self.assertNotIn("created_at", EVIDENCE_IDENTITY_FIELDS)

    def test_identity_distinguishes_semantic_provenance(self):
        for field in ("source_dataset", "allergen_code", "component_occurrence_id"):
            changed = deepcopy(self.refs[0]); changed[field] = f"other-{field}"
            self.assertNotEqual(identity(self.refs[0]), identity(changed), field)

    def test_source_dataset_has_its_own_identity(self):
        changed = deepcopy(self.refs[0]); changed["source_dataset"] = "other-source"
        self.assertNotEqual(identity(self.refs[0]), identity(changed))

    def test_allergen_code_has_its_own_identity(self):
        changed = deepcopy(self.refs[0]); changed["allergen_code"] = "other-allergen"
        self.assertNotEqual(identity(self.refs[0]), identity(changed))

    def test_component_occurrence_has_its_own_identity(self):
        changed = deepcopy(self.refs[0]); changed["component_occurrence_id"] = "other-component"
        self.assertNotEqual(identity(self.refs[0]), identity(changed))

    def test_unique_identity_parity_and_four_real_duplicates(self):
        self._fresh()
        c = sqlite3.connect(self.tmp.name)
        self.assertEqual(c.execute("select count(*) from product_allergen_refs").fetchone()[0], len({identity(r) for r in self.refs}))
        self.assertEqual(len(self.refs) - len({identity(r) for r in self.refs}), 4)
        c.close()

    def test_rerun_keeps_ids_and_counts(self):
        first = self._fresh()
        c = sqlite3.connect(self.tmp.name); ids1 = {r[0] for r in c.execute("select evidence_id from product_allergen_refs")}; c.close()
        changed = deepcopy(self.refs)
        for row in changed: row["created_at"] = "2099-01-01T00:00:00+00:00"
        self.assertEqual(replace_all_catalog(self.components, changed, self.tmp.name), first)
        c = sqlite3.connect(self.tmp.name); ids2 = {r[0] for r in c.execute("select evidence_id from product_allergen_refs")}; c.close()
        self.assertEqual(ids1, ids2)

    def test_full_rebuild_failure_preserves_prior_catalog(self):
        self._fresh()
        broken = deepcopy(self.refs); broken[0]["component_occurrence_id"] = "missing"
        with self.assertRaises(ValueError): replace_all_catalog(self.components, broken, self.tmp.name)
        c = sqlite3.connect(self.tmp.name); self.assertEqual(c.execute("select count(*) from product_allergen_refs").fetchone()[0], 2852); c.close()

    def test_incremental_upsert_keeps_unrelated_rows(self):
        self._fresh()
        extra = deepcopy(self.refs[0]); extra["product_id"] = "unrelated-product"; extra["created_at"] = "2099-01-01T00:00:00+00:00"
        self.assertEqual(upsert_refs([extra], self.tmp.name), 2853)
        c = sqlite3.connect(self.tmp.name); self.assertEqual(c.execute("select count(*) from product_allergen_refs where product_id=?", (self.refs[1]["product_id"],)).fetchone()[0] > 0, True); c.close()

    def test_malformed_serialization_fails_closed(self):
        self._fresh(); pid = self.refs[0]["product_id"]
        c = sqlite3.connect(self.tmp.name); c.execute("update product_allergen_refs set segmented_text='{' where product_id=?", (pid,)); c.commit(); c.close()
        self.assertEqual(get_refs(pid, DICTIONARY_VERSION, PIPELINE_VERSION, self.tmp.name)[1], "LINEAGE_INTEGRITY_ERROR")

    def test_missing_mandatory_evidence_field_fails_closed(self):
        self._fresh(); pid = self.refs[0]["product_id"]
        c = sqlite3.connect(self.tmp.name); c.execute("update product_allergen_refs set source_dataset='' where product_id=?", (pid,)); c.commit(); c.close()
        self.assertEqual(get_refs(pid, DICTIONARY_VERSION, PIPELINE_VERSION, self.tmp.name)[1], "LINEAGE_INTEGRITY_ERROR")

    def test_api_status_and_card_prioritize_safety_insufficiency(self):
        original_gate, original_run = api_nutrition._allergy_gate, api_nutrition._run_p0d
        api_nutrition._allergy_gate = lambda *_: {"safety_status": "SAFETY_DATA_INSUFFICIENT", "evidence_trace": "LINEAGE_INTEGRITY_ERROR", "excluded": True}
        api_nutrition._run_p0d = lambda *_: {"nutrition_comparison_status": "KNOWN", "aafco_pass": None, "nutrition_comparison": {}, "nutrition_items": [], "warnings": []}
        self.addCleanup(setattr, api_nutrition, "_allergy_gate", original_gate); self.addCleanup(setattr, api_nutrition, "_run_p0d", original_run)
        pet = api_nutrition.PetIn(id="p", species="dog", age_years=2, weight_kg=4)
        product = api_nutrition.ProductIn(id="x", name="x", target_species="dog", aafco_life_stage="ADULT_MAINTENANCE")
        result = api_nutrition._analyze_product(api_nutrition.AnalyzeRequest(pet=pet, product=product))
        self.assertEqual(result["analysis_status"], "INSUFFICIENT_DATA")
        self.assertEqual(result["input_readiness"]["input_readiness"], "INSUFFICIENT_DATA")
        self.assertIn("P0D_RESULT_CONTRACT_INVALID", result["input_readiness"]["reason_codes"])
        self.assertIn("안전 판정을 보류", result["consumer_card"])

    def test_version_sot_is_shared_by_api_and_builder(self):
        import build_product_allergen_refs
        self.assertEqual((api_nutrition.DICTIONARY_VERSION, api_nutrition.PIPELINE_VERSION), (DICTIONARY_VERSION, PIPELINE_VERSION))
        self.assertEqual((build_product_allergen_refs.DICTIONARY_VERSION, build_product_allergen_refs.PIPELINE_VERSION), (DICTIONARY_VERSION, PIPELINE_VERSION))

    def test_parent_query_is_scoped_to_product_result(self):
        self._fresh(); pid = self.refs[0]["product_id"]
        rows, state = get_refs(pid, DICTIONARY_VERSION, PIPELINE_VERSION, self.tmp.name)
        self.assertEqual(state, "PRECOMPUTED")
        self.assertEqual(len(rows), len({identity(r) for r in self.refs if r["product_id"] == pid}))
