import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "scripts"
sys.path[:0] = [str(S), str(S / "nutrition")]
from allergen_service import evaluate_safety
from build_product_allergen_refs import components


class P1BatchContracts(unittest.TestCase):
    def test_segmentation_and_case_are_deterministic(self):
        self.assertEqual(components("a, b; c"), components("a, b; c"))
        self.assertEqual(components("A, B"), ["A", "B"])

    def test_precomputed_unresolved_is_not_safety_evidence(self):
        pet = {"id":"p", "species":"dog", "age_years":3, "weight_kg":1, "allergies":["chicken"], "allergy_profile_status":"KNOWN_LIST"}
        product = {"id":"x", "category":"food", "target_species":"dog", "aafco_life_stage":"MAINTENANCE", "ingredient_list":["ignored"],
                   "product_allergen_refs":[{"raw_text":"dinde", "mapping_method":"UNRESOLVED", "allergen_code":"turkey"}]}
        r = evaluate_safety(pet, product)
        self.assertEqual((r["safety_status"], r["evidence_trace"]), ("SAFETY_DATA_INSUFFICIENT", "PRECOMPUTED"))

    def test_runtime_fallback_is_explicit(self):
        r = evaluate_safety({"id":"p", "allergies":[], "allergy_profile_status":"KNOWN_NONE"}, {"id":"x", "category":"treat", "ingredient_list":["chicken"]})
        self.assertEqual(r["evidence_trace"], "RUNTIME_FALLBACK")
