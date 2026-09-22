import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "nutrition"))

from gold_evidence_intake import (  # noqa: E402
    INTAKE_FIELDS,
    build_operational_evidence_artifact,
    read_human_gold_evidence_intake,
    validate_human_gold_evidence_intake,
    write_operational_evidence_artifact,
)


class GoldEvidenceIntakeTests(unittest.TestCase):
    validated_at = "2026-09-20T00:00:00+00:00"

    def candidate(self):
        # This is a test-only local candidate context.  It deliberately does
        # not provide manufacturer/variant/market/formula authority; those
        # values must arrive consistently in each human-reviewed intake row.
        return {
            "candidate_id": "GOLD_CAT_036000291452",
            "canonical_gtin": "036000291452",
            "allowed_product_ids": ["local-cat-1"],
            "species": "CAT",
            "life_stage": "ADULT_MAINTENANCE",
            "product_form": "DRY",
            "identity_conflict": "NO_HARD_IDENTITY_CONFLICT",
        }

    def row(self, nutrient_code="CRUDE_PROTEIN", value="1.0"):
        return {
            "candidate_id": "GOLD_CAT_036000291452",
            "product_id": "local-cat-1",
            "canonical_gtin": "036000291452",
            "manufacturer": "Example Manufacturer",
            "manufacturer_product_name": "Example Cat Food",
            "species": "CAT",
            "life_stage": "ADULT_MAINTENANCE",
            "life_stage_detail": "",
            "product_form": "DRY",
            "product_variant": "Adult Dry 1kg",
            "package_size": "1kg",
            "market_region": "US",
            "formula_version": "LABEL_2026_01",
            "effective_date": "2026-01-01",
            "identity_source_type": "MANUFACTURER_PACKAGE_IMAGE_WITH_GTIN",
            "identity_source_url_or_document": "https://manufacturer.example/package-image",
            "identity_source_authority": "Example Manufacturer",
            "nutrient_code": nutrient_code,
            "value": value,
            "unit": "PERCENT",
            "basis": "AS_FED",
            "value_qualifier": "MINIMUM",
            "guarantee_type": "GUARANTEED",
            "nutrient_source_type": "MANUFACTURER_GUARANTEED_ANALYSIS",
            "nutrient_source_url_or_document": "https://manufacturer.example/guaranteed-analysis",
            "nutrient_source_authority": "Example Manufacturer",
            "observed_at": "2026-09-20T00:00:00Z",
            "retrieved_at": "2026-09-20T00:00:00Z",
            "verification_method": "HUMAN_REVIEWED_DOCUMENT",
            "identity_state": "VERIFIED",
            "life_stage_evidence_state": "VERIFIED_AUTHORITATIVE",
            "life_stage_detail_evidence_state": "",
            "product_form_evidence_state": "VERIFIED_AUTHORITATIVE",
            "evidence_state": "VERIFIED",
            "verified": "true",
            # This submitted value is intentionally not trusted by validation.
            "runtime_eligible": "false",
            "review_note": "test fixture only",
        }

    def full_rows(self):
        return [self.row(code) for code in (
            "CRUDE_PROTEIN", "CRUDE_FAT", "MOISTURE", "CALCIUM", "PHOSPHORUS", "TAURINE",
        )]

    def parsed(self, directory, rows):
        path = Path(directory) / "human_gold_evidence_intake_v1.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=INTAKE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return read_human_gold_evidence_intake(path)

    def validate(self, directory, rows):
        return validate_human_gold_evidence_intake(
            self.parsed(directory, rows), [self.candidate()], validated_at=self.validated_at,
        )

    def test_empty_or_missing_intake_builds_zero_row_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            parsed = read_human_gold_evidence_intake(Path(directory) / "does-not-exist.csv")
            result = validate_human_gold_evidence_intake(parsed, [self.candidate()], validated_at=self.validated_at)
            artifact = build_operational_evidence_artifact(result)
            self.assertTrue(result["input_missing"])
            self.assertEqual(result["human_intake_rows"], 0)
            self.assertEqual(result["accepted"], 0)
            self.assertEqual(result["runtime_eligible_count"], 0)
            self.assertEqual(artifact["operational_evidence"], [])
            csv_path, json_path = Path(directory) / "verified.csv", Path(directory) / "verified.json"
            write_operational_evidence_artifact(artifact, csv_path=csv_path, json_path=json_path)
            with csv_path.open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 0)
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["operational_evidence"], [])

    def test_complete_fixture_pipeline_preserves_provenance_and_calculates_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.validate(directory, self.full_rows())
        self.assertEqual(result["accepted"], 6)
        self.assertEqual(result["rejected"], 0)
        self.assertEqual(result["runtime_eligible_count"], 1)
        evidence = result["accepted_evidence"][0]
        self.assertEqual(evidence["validation_status"], "ACCEPTED")
        self.assertEqual(evidence["validation_version"], "gold_evidence_intake_v1")
        self.assertEqual(evidence["validated_at"], self.validated_at)
        self.assertEqual(evidence["identity_source_url_or_document"], "https://manufacturer.example/package-image")
        self.assertEqual(evidence["nutrient_source_url_or_document"], "https://manufacturer.example/guaranteed-analysis")
        self.assertTrue(evidence["runtime_eligible"])

    def test_malformed_gtin_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self.full_rows()
            for row in rows:
                row["canonical_gtin"] = "036000291459"
            result = self.validate(directory, rows)
        self.assertEqual(result["accepted"], 0)
        self.assertIn("INVALID_GTIN", result["rejection_reason_counts"])
        self.assertIn("CANONICAL_GTIN_MISMATCH", result["rejection_reason_counts"])

    def test_matching_gtin_with_product_id_outside_cluster_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self.full_rows()
            for row in rows:
                row["product_id"] = "another-local-product"
            result = self.validate(directory, rows)
        self.assertEqual(result["accepted"], 0)
        self.assertIn("PRODUCT_ID_NOT_IN_CANDIDATE_CLUSTER", result["rejection_reason_counts"])

    def test_official_nutrition_page_cannot_replace_gtin_identity_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self.full_rows()
            for row in rows:
                row["identity_source_type"] = "OFFICIAL_MANUFACTURER"
            result = self.validate(directory, rows)
        self.assertEqual(result["accepted"], 0)
        self.assertIn("IDENTITY_SOURCE_NOT_GTIN_BINDING", result["rejection_reason_counts"])

    def test_identity_without_nutrition_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self.full_rows()
            rows[0]["nutrient_source_authority"] = ""
            result = self.validate(directory, rows)
        self.assertEqual(result["accepted"], 0)
        self.assertIn("NUTRIENT_PROVENANCE_FIELD_MISSING:nutrient_source_authority", result["rejection_reason_counts"])

    def test_product_name_keyword_does_not_verify_life_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self.full_rows()
            for row in rows:
                row["manufacturer_product_name"] = "Example Adult Cat Food"
                row["life_stage"] = ""
                # A state flag by itself cannot infer this missing value from
                # the word "Adult" in the product name.
                row["life_stage_evidence_state"] = "VERIFIED_AUTHORITATIVE"
            result = self.validate(directory, rows)
        self.assertEqual(result["accepted"], 0)
        self.assertIn("LIFE_STAGE_MISMATCH_OR_MISSING", result["rejection_reason_counts"])

    def test_raw_metadata_form_cannot_be_used_as_authoritative_form_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self.full_rows()
            for row in rows:
                row["product_form"] = ""
                row["product_form_evidence_state"] = "UNVERIFIED_LOCAL_METADATA"
            result = self.validate(directory, rows)
        self.assertEqual(result["accepted"], 0)
        self.assertIn("PRODUCT_FORM_PROVENANCE_UNVERIFIED", result["rejection_reason_counts"])
        self.assertIn("PRODUCT_FORM_MISMATCH_OR_MISSING", result["rejection_reason_counts"])

    def test_submitted_runtime_eligible_does_not_bypass_missing_nutrient(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self.full_rows()[:-1]
            for row in rows:
                row["runtime_eligible"] = "true"
            result = self.validate(directory, rows)
        self.assertEqual(result["accepted"], 5)
        self.assertEqual(result["runtime_eligible_count"], 0)
        self.assertEqual(result["operational_candidates"][0]["missing_nutrients"], ["TAURINE"])
        self.assertFalse(result["accepted_evidence"][0]["runtime_eligible"])

    def test_same_gtin_variant_or_formula_conflict_rejects_entire_candidate_group(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self.full_rows()
            rows[-1]["product_variant"] = "Different Formula 1kg"
            rows[-1]["formula_version"] = "LABEL_2026_02"
            result = self.validate(directory, rows)
        self.assertEqual(result["accepted"], 0)
        self.assertIn("INTERNAL_BINDING_CONFLICT:product_variant", result["rejection_reason_counts"])
        self.assertIn("INTERNAL_BINDING_CONFLICT:formula_version", result["rejection_reason_counts"])

    def test_authoritative_dog_growth_detail_can_fill_absent_raw_detail(self):
        candidate = {
            **self.candidate(),
            "candidate_id": "GOLD_DOG_036000291452",
            "species": "DOG",
            "life_stage": "GROWTH_REPRODUCTION",
            "life_stage_detail": "",
        }
        rows = self.full_rows()[:-1]  # DOG has no TAURINE runtime requirement.
        for row in rows:
            row.update({
                "candidate_id": candidate["candidate_id"],
                "species": "DOG",
                "life_stage": "GROWTH_REPRODUCTION",
                "life_stage_detail": "EARLY_GROWTH_AND_REPRODUCTION",
                "life_stage_detail_evidence_state": "VERIFIED_AUTHORITATIVE",
            })
        with tempfile.TemporaryDirectory() as directory:
            result = validate_human_gold_evidence_intake(
                self.parsed(directory, rows), [candidate], validated_at=self.validated_at,
            )
        self.assertEqual(result["accepted"], 5)
        self.assertEqual(result["rejected"], 0)
        self.assertEqual(result["runtime_eligible_count"], 1)


if __name__ == "__main__":
    unittest.main()
