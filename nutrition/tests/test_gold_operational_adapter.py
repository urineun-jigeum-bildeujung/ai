"""Fixture-only tests for the Gold operational evidence adapter layer.

These synthetic rows exercise plumbing, not real persisted-product Gold E2E.
They are written only inside ``TemporaryDirectory`` and never to ``data/``.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path[:0] = [str(SCRIPTS), str(SCRIPTS / "nutrition")]

import nutrition.product_input_adapter as adapter
from gold_evidence_intake import (  # noqa: E402
    INTAKE_FIELDS,
    build_operational_evidence_artifact,
    read_human_gold_evidence_intake,
    validate_human_gold_evidence_intake,
    write_operational_evidence_artifact,
)


class GoldOperationalAdapterTests(unittest.TestCase):
    product_id = "036000291452"
    canonical_gtin = "036000291452"

    def raw_product(self) -> dict:
        return {
            "product_id": self.product_id,
            "name": "Raw label product",
            "target_species": "DOG",
            "target_life_stage": None,
            "product_type": "DRY_FOOD",
            "ingredients": ["raw ingredient"],
        }

    def raw_nutrients(self) -> list[dict]:
        return [{
            "nutrient_code": "CRUDE_PROTEIN",
            "value": 20.0,
            "unit": "PERCENT",
            "basis": "AS_FED",
            "source": "RAW_TEST_FIXTURE",
        }]

    def operational_rows(self) -> list[dict]:
        common = {
            "candidate_id": "GOLD_CAT_036000291452",
            "product_id": self.product_id,
            "canonical_gtin": self.canonical_gtin,
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
            "life_stage_detail": None,
            "life_stage_detail_evidence_state": "VERIFIED_AUTHORITATIVE",
            "product_form": "DRY",
            "product_form_evidence_state": "VERIFIED_AUTHORITATIVE",
            "verified": True,
            "runtime_eligible": True,
            "validation_status": "ACCEPTED",
            "validation_version": "gold_intake_v1",
            "validated_at": "2026-09-20T00:00:00Z",
            "nutrient_source_type": "MANUFACTURER_GUARANTEED_ANALYSIS",
            "unit": "PERCENT",
            "basis": "AS_FED",
            "guarantee_type": "GUARANTEED",
            "nutrient_source_url_or_document": "https://manufacturer.example/guaranteed-analysis",
            "nutrient_source_authority": "Example Manufacturer",
            "observed_at": "2026-09-20T00:00:00Z",
            "retrieved_at": "2026-09-20T00:00:00Z",
            "verification_method": "HUMAN_REVIEWED_DOCUMENT",
        }
        return [{**common, "nutrient_code": code, "value": 1.0} for code in (
            "CRUDE_PROTEIN", "CRUDE_FAT", "MOISTURE", "CALCIUM", "PHOSPHORUS", "TAURINE",
        )]

    def load_with_artifact(self, payload: dict | None) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "verified_evidence_gold_v1.json"
            if payload is not None:
                path.write_text(json.dumps(payload), encoding="utf-8")
            with patch.object(adapter, "_find_product", return_value=("GLOBAL", self.raw_product())), patch.object(
                adapter, "_nutrition_items", return_value=self.raw_nutrients()
            ):
                return adapter.load_product_input(self.product_id, operational_evidence_path=path)

    def test_accepted_layer_replaces_raw_nutrition_and_preserves_provenance(self):
        loaded = self.load_with_artifact({"artifact_version": "verified_evidence_gold_v1", "operational_evidence": self.operational_rows()})
        product, provenance = loaded["product"], loaded["provenance"]

        self.assertEqual(product["target_species"], "cat")
        self.assertEqual(product["aafco_life_stage"], "ADULT_MAINTENANCE")
        self.assertEqual(product["product_form"], "DRY")
        self.assertEqual({row["nutrient_code"] for row in product["nutrition_items"]}, {
            "CRUDE_PROTEIN", "CRUDE_FAT", "MOISTURE", "CALCIUM", "PHOSPHORUS", "TAURINE",
        })
        self.assertTrue(all(row["source"] == "MANUFACTURER_GUARANTEED_ANALYSIS" for row in product["nutrition_items"]))
        self.assertEqual(provenance["raw_nutrition_item_count"], 1)
        self.assertEqual(provenance["nutrition_input_source"], "VERIFIED_GOLD_OPERATIONAL_EVIDENCE")
        gold = provenance["gold_operational_evidence"]
        self.assertEqual(gold["status"], "APPLIED")
        self.assertEqual(gold["canonical_gtin"], self.canonical_gtin)
        self.assertEqual(gold["artifact_version"], "verified_evidence_gold_v1")
        self.assertEqual(gold["validation_versions"], ["gold_intake_v1"])
        self.assertEqual(gold["evidence_rows"][0]["nutrient_source_url_or_document"], "https://manufacturer.example/guaranteed-analysis")

    def test_empty_or_missing_operational_layer_preserves_raw_fail_close_input(self):
        loaded = self.load_with_artifact({"version": "v1", "operational_evidence": []})
        self.assertEqual(loaded["product"]["nutrition_items"], self.raw_nutrients())
        self.assertEqual(loaded["product"]["target_species"], "dog")
        self.assertIsNone(loaded["product"]["aafco_life_stage"])
        self.assertEqual(loaded["provenance"]["gold_operational_evidence"]["status"], "NO_OPERATIONAL_EVIDENCE")
        self.assertEqual(loaded["provenance"]["gold_operational_evidence"]["artifact_version"], "v1")

        loaded = self.load_with_artifact(None)
        self.assertEqual(loaded["product"]["nutrition_items"], self.raw_nutrients())
        self.assertEqual(loaded["provenance"]["gold_operational_evidence"]["status"], "ARTIFACT_NOT_FOUND")

    def test_same_gtin_with_a_different_product_id_is_not_applied(self):
        rows = self.operational_rows()
        for row in rows:
            row["product_id"] = "other-local-product"
        loaded = self.load_with_artifact({"version": "v1", "operational_evidence": rows})
        self.assertEqual(loaded["product"]["nutrition_items"], self.raw_nutrients())
        self.assertEqual(loaded["provenance"]["gold_operational_evidence"]["status"], "NO_MATCHING_OPERATIONAL_EVIDENCE")

    def test_unaccepted_or_conflicting_artifact_cannot_bypass_the_gate(self):
        rows = self.operational_rows()
        rows[-1]["product_variant"] = "Different variant"
        loaded = self.load_with_artifact({"version": "v1", "operational_evidence": rows})
        self.assertEqual(loaded["product"]["nutrition_items"], self.raw_nutrients())
        self.assertEqual(loaded["provenance"]["gold_operational_evidence"]["status"], "GOLD_GATE_REJECTED")

        rows = self.operational_rows()
        for row in rows:
            row["validation_status"] = "REJECTED"
        loaded = self.load_with_artifact({"version": "v1", "operational_evidence": rows})
        self.assertEqual(loaded["product"]["nutrition_items"], self.raw_nutrients())
        self.assertEqual(loaded["provenance"]["gold_operational_evidence"]["status"], "GOLD_GATE_REJECTED")

    def test_fixture_based_gold_pipeline_integration(self):
        """Synthetic fixture only: intake -> validator -> artifact -> adapter -> gate.

        This verifies plumbing, not a real persisted-product Gold E2E and never
        writes an operational record below ``data/``.
        """
        candidate = {
            "candidate_id": "GOLD_CAT_036000291452",
            "canonical_gtin": self.canonical_gtin,
            "allowed_product_ids": [self.product_id],
            "species": "CAT",
            "life_stage": "ADULT_MAINTENANCE",
            "product_form": "DRY",
            "identity_conflict": "NO_HARD_IDENTITY_CONFLICT",
        }
        intake_rows = []
        for evidence in self.operational_rows():
            intake_rows.append({
                **{field: "" for field in INTAKE_FIELDS},
                **{key: evidence.get(key, "") for key in INTAKE_FIELDS},
                "life_stage": "ADULT_MAINTENANCE",
                "life_stage_detail": "",
                "product_form": "DRY",
                "effective_date": "2026-01-01",
                "value_qualifier": "MINIMUM",
                "evidence_state": "VERIFIED",
                "review_note": "synthetic fixture only",
                "verified": "true",
                "runtime_eligible": "false",
            })
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            intake_path = directory_path / "human_gold_evidence_intake_v1.csv"
            with intake_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=INTAKE_FIELDS)
                writer.writeheader()
                writer.writerows(intake_rows)
            validation = validate_human_gold_evidence_intake(
                read_human_gold_evidence_intake(intake_path), [candidate],
                validated_at="2026-09-20T00:00:00Z",
            )
            self.assertEqual(validation["accepted"], 6)
            self.assertEqual(validation["runtime_eligible_count"], 1)
            artifact = build_operational_evidence_artifact(validation)
            artifact_csv = directory_path / "verified_evidence_gold_v1.csv"
            artifact_json = directory_path / "verified_evidence_gold_v1.json"
            write_operational_evidence_artifact(
                artifact, csv_path=artifact_csv, json_path=artifact_json,
            )
            with patch.object(adapter, "_find_product", return_value=("GLOBAL", self.raw_product())), patch.object(
                adapter, "_nutrition_items", return_value=self.raw_nutrients()
            ):
                loaded = adapter.load_product_input(
                    self.product_id, operational_evidence_path=artifact_json,
                )
        self.assertEqual(loaded["provenance"]["gold_operational_evidence"]["status"], "APPLIED")
        self.assertEqual(loaded["provenance"]["nutrition_input_source"], "VERIFIED_GOLD_OPERATIONAL_EVIDENCE")
        self.assertEqual(len(loaded["product"]["nutrition_items"]), 6)


if __name__ == "__main__":
    unittest.main()
