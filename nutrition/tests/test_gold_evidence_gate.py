from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "nutrition"))
from gold_evidence_gate import evaluate_runtime_eligibility
from pipeline_p1c_v1 import compute_nutrition_comparison_status


class GoldEvidenceGateTests(unittest.TestCase):
    def base(self):
        return {
            "candidate_id": "GOLD_CAT_036000291452",
            "product_id": "local-cat-1",
            "canonical_gtin": "036000291452",
            "gtin_valid": True,
            "identity_state": "VERIFIED",
            "identity_conflict": "NO_CONFLICT",
            "manufacturer": "Example Manufacturer",
            "manufacturer_product_name": "Example Cat Food",
            "product_variant": "Adult Dry 1kg",
            "package_size": "1kg",
            "market_region": "US",
            "formula_version": "LABEL_2026_01",
            "identity_source_type": "MANUFACTURER_PACKAGE_IMAGE_WITH_GTIN",
            "identity_source_url_or_document": "https://manufacturer.example/package-image",
            "identity_source_authority": "Example Manufacturer",
            "species": "CAT",
            "life_stage": "ADULT_MAINTENANCE",
            "life_stage_evidence_state": "VERIFIED_AUTHORITATIVE",
            "product_form": "DRY",
            "product_form_evidence_state": "VERIFIED_AUTHORITATIVE",
        }

    def evidence(self):
        identity = self.base()
        return [{
            **{field: identity[field] for field in (
                "candidate_id", "product_id", "canonical_gtin", "manufacturer", "manufacturer_product_name",
                "product_variant", "package_size", "market_region", "formula_version",
            )},
            "verified": True,
            "nutrient_source_type": "MANUFACTURER_GUARANTEED_ANALYSIS",
            "nutrient_code": code,
            "value": 1.0,
            "unit": "PERCENT",
            "basis": "AS_FED",
            "guarantee_type": "GUARANTEED",
            "nutrient_source_url_or_document": "https://manufacturer.example/guaranteed-analysis",
            "nutrient_source_authority": "Example Manufacturer",
            "observed_at": "2026-09-20T00:00:00Z",
            "retrieved_at": "2026-09-20T00:00:00Z",
            "verification_method": "HUMAN_REVIEWED_DOCUMENT",
        } for code in ("CRUDE_PROTEIN", "CRUDE_FAT", "MOISTURE", "CALCIUM", "PHOSPHORUS", "TAURINE")]

    def test_invalid_or_unverified_identity_fails_closed(self):
        item = self.base(); item["identity_state"] = "UNVERIFIED"
        self.assertFalse(evaluate_runtime_eligibility(item, self.evidence())["runtime_eligible"])

    def test_invalid_gtin_fails_closed(self):
        item = self.base(); item["gtin_valid"] = False
        self.assertIn("INVALID_GTIN", evaluate_runtime_eligibility(item, self.evidence())["reason_codes"])
        item = self.base(); item.pop("gtin_valid")
        self.assertIn("INVALID_GTIN", evaluate_runtime_eligibility(item, self.evidence())["reason_codes"])

    def test_hard_conflict_fails_closed(self):
        item = self.base(); item["identity_conflict"] = "HARD_IDENTITY_CONFLICT"
        self.assertFalse(evaluate_runtime_eligibility(item, self.evidence())["runtime_eligible"])

    def test_official_page_without_verified_link_fails_closed(self):
        item = self.base(); item["identity_source_type"] = "OFFICIAL_MANUFACTURER"
        self.assertFalse(evaluate_runtime_eligibility(item, self.evidence())["runtime_eligible"])

    def test_missing_required_nutrient_fails_closed(self):
        self.assertIn("TAURINE", evaluate_runtime_eligibility(self.base(), self.evidence()[:-1])["missing_nutrients"])

    def test_complete_verified_evidence_is_eligible(self):
        self.assertTrue(evaluate_runtime_eligibility(self.base(), self.evidence())["runtime_eligible"])

    def test_dog_growth_requires_detail(self):
        item = self.base(); item.update({"candidate_id": "GOLD_DOG_036000291452", "product_id": "local-dog-1", "species": "DOG", "life_stage": "GROWTH_REPRODUCTION"})
        dog = self.evidence()[:-1]
        for row in dog:
            row.update({"candidate_id": item["candidate_id"], "product_id": item["product_id"]})
        self.assertIn("DOG_GROWTH_LIFE_STAGE_DETAIL_UNRESOLVED", evaluate_runtime_eligibility(item, dog)["reason_codes"])
        item["life_stage_detail"] = "EARLY_GROWTH_AND_REPRODUCTION"
        item["life_stage_detail_evidence_state"] = "KEYWORD_INFERRED"
        self.assertFalse(evaluate_runtime_eligibility(item, dog)["runtime_eligible"])
        item["life_stage_detail_evidence_state"] = "VERIFIED_AUTHORITATIVE"
        self.assertTrue(evaluate_runtime_eligibility(item, dog)["runtime_eligible"])

    def test_evidence_from_another_identity_fails_closed(self):
        evidence = self.evidence()
        for row in evidence:
            row["canonical_gtin"] = "9501101530003"
        result = evaluate_runtime_eligibility(self.base(), evidence)
        self.assertFalse(result["runtime_eligible"])
        self.assertIn("EVIDENCE_PROVENANCE_INSUFFICIENT", result["reason_codes"])

    def test_unverified_attribute_provenance_fails_closed(self):
        item = self.base(); item["life_stage_evidence_state"] = "KEYWORD_INFERRED"
        self.assertFalse(evaluate_runtime_eligibility(item, self.evidence())["runtime_eligible"])
        item = self.base(); item["life_stage"] = "NOT_A_STAGE"
        self.assertFalse(evaluate_runtime_eligibility(item, self.evidence())["runtime_eligible"])
        item = self.base(); item["product_form"] = "NOT_A_FORM"
        self.assertFalse(evaluate_runtime_eligibility(item, self.evidence())["runtime_eligible"])

    def test_unusable_nutrient_evidence_fails_closed(self):
        evidence = self.evidence()
        evidence[0]["value"] = "1.0"
        evidence[1]["nutrient_source_type"] = "RETAILER_TRANSCRIPTION"
        result = evaluate_runtime_eligibility(self.base(), evidence)
        self.assertFalse(result["runtime_eligible"])
        self.assertIn("EVIDENCE_PROVENANCE_INSUFFICIENT", result["reason_codes"])

    def test_false_rule_engine_result_is_not_promoted(self):
        rows = [{"product_id": "fixture", "nutrient_code": code, "nias_compare_status": status}
                for code, status in (("CRUDE_PROTEIN", "IN_RANGE"), ("CRUDE_FAT", "IN_RANGE"),
                                     ("MOISTURE", "OUT_OF_RANGE"), ("CALCIUM", "IN_RANGE"), ("PHOSPHORUS", "IN_RANGE"))]
        result = compute_nutrition_comparison_status(rows, {"fixture": "DOG"}, {"fixture": "ADULT_MAINTENANCE"})["fixture"]
        self.assertEqual(result["nutrition_comparison_status"], "FALSE")


if __name__ == "__main__":
    unittest.main()
