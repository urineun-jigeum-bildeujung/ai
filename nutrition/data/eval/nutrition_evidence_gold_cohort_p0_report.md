# Nutrition Product Identity / Nutrition Evidence Gold Cohort P0

## Baseline

- local source records: 394; valid-GTIN identity clusters: 323
- This is a read-only projection. The Nutrition Rule Engine, reference artifacts, API, raw seeds, and database were not changed.

## Portfolio Metrics

- source records: 394; valid GTIN clusters: 323
- structured nutrition clusters: 25; Ca/P-present clusters: 0; CAT taurine-present clusters: 0
- Gold-ready: 0; Gold-near-ready: 12

## GTIN Normalization

`gtin_normalization_audit.csv` verifies allowed GTIN length and check digit. Invalid or unsupported source IDs are not join keys. UPC-A/EAN-13 leading-zero equivalence is canonicalized only after a valid check digit.

## Identity vs Attribute Conflicts

`identity_conflict_classification.csv` separates same-GTIN identity conflicts from metadata variation. No conflict result authorizes a merge; every multi-source cluster remains a projection.

## Field-level Complementarity

`field_level_provenance_projection.csv` records observed values and source records without selecting a source priority.

## Ca/P Recovery Candidates / CAT Taurine Gap

The two queues only identify missing structured values and the required identity verification. No external page was requested, scraped, or inserted.

## Open Pet Food Facts Evidence Availability

{"opff_records": 108, "front_image_url_present": 99, "nutrition_image_metadata_present": 0, "ingredients_image_metadata_present": 0, "packaging_image_metadata_present": 0, "interpretation": "URL presence only; no image retrieval or OCR performed."}

## Danawa Mapping Pilot

The 30 rows are candidate-only. `pcode` is not a GTIN, and no fuzzy match is a verified identity.

## Gold Product Candidates

`gold_product_candidates.csv` includes only the requested DOG adult dry, DOG growth dry, CAT adult dry, and CAT adult canned profiles. `GOLD_READY` means the local minimum-runtime fields and ingredients are present; it is not a clinical suitability or comprehensive-coverage claim. `gold_product_gap_analysis.csv` ranks gaps without assigning a default species or missing nutrient value.

## Minimum Acquisition Gap

`nutrition_evidence_acquisition_queue.csv` and `cat_taurine_acquisition_queue.csv` are verification work queues. A queue entry has no acquired external nutrient value.

## Official manufacturer evidence candidates

`official_manufacturer_evidence_candidates_v1.csv` preserves the observed manufacturer-page values separately from local product data. Its `verified` value is `false`: official-page content was checked, but its GTIN-to-formula identity is not exposed by the page. The request-scoped E2E artifact is not a persisted-product validation. The CAT candidate returns `TRUE`; the DOG candidate returns `FALSE` because the current NIAS comparison marks its 12% moisture `OUT_OF_RANGE`, despite Ca/P being `IN_RANGE`.

## Provenance Contract (proposal only)

`product_id`, `canonical_gtin`, `nutrient_code`, `value`, `unit`, `basis`, `guarantee_type`, `source_type`, `source_record_id`, `source_authority`, `observed_at`, `verified`, `verification_method`.

## 확인 불가

Actual package-label nutrient values, manufacturer/importer evidence, and a verified Danawa pcode-to-GTIN mapping are not present in the local inputs. Gold readiness is analysis-only and does not assert nutritional adequacy.

## Next Step

Human-review the GTIN/identity rows required by the Ca/P and taurine queues, then acquire and verify only package-label or authoritative evidence against that approved identity. Do not use a Danawa candidate or fuzzy name similarity as a merge key.
