# Local product identity exact-join projection

## 범위

barcode/source-id exact match만 projected join으로 사용했다. normalized brand+name match는 candidate-only이며 merge하지 않았다.

## Exact join matrix

`[{"left_source": "OPFF", "right_source": "OEM", "left_records": 108, "right_records": 189, "exact_identity_keys": 60, "exact_pairs": 60, "ambiguous_exact_pairs": 0, "brand_conflicts": 41, "metadata_conflict_pairs": 60, "candidate_name_pairs_not_approved": 2, "unmatched_left": 48, "unmatched_right": 129}, {"left_source": "OPFF", "right_source": "GLOBAL", "left_records": 108, "right_records": 97, "exact_identity_keys": 5, "exact_pairs": 5, "ambiguous_exact_pairs": 0, "brand_conflicts": 2, "metadata_conflict_pairs": 5, "candidate_name_pairs_not_approved": 0, "unmatched_left": 103, "unmatched_right": 92}, {"left_source": "OEM", "right_source": "GLOBAL", "left_records": 189, "right_records": 97, "exact_identity_keys": 3, "exact_pairs": 3, "ambiguous_exact_pairs": 0, "brand_conflicts": 0, "metadata_conflict_pairs": 1, "candidate_name_pairs_not_approved": 0, "unmatched_left": 186, "unmatched_right": 94}]`

## Projected coverage

`{"scope": "exact barcode/source-id join projection only; no raw merge", "identity_clusters": 329, "current_field_coverage": {"products": 329, "species": 329, "nutrition": 25, "ingredients": 103, "stage": 87, "form": 92}, "projected_field_coverage": {"products": 329, "species": 329, "nutrition": 25, "ingredients": 103, "stage": 92, "form": 97}, "delta": {"form": 5, "nutrition": 0, "ingredients": 0, "species": 0, "stage": 5, "products": 0}, "projected_ready_candidate_count": 0}`

## Existing mapping

`{"verified_mapping_artifact": null, "legacy_evidence": ["scripts/p2_sql_v1.py:to_ean13()", "docs/canonical_data_contract_v1.md: OFF_KR prefix normalization discussion"], "legacy_product_master": {"path": "data/processed/verify_P2_v1.db", "product_count": 286, "fields": ["product_id", "name", "category", "target_species"]}}`

## Danawa read-only inventory

`{"products": 508, "stable_source_id": "pcode", "barcode_or_gtin_present": 0, "species_from_source_split": {"DOG": 298, "CAT": 210}, "life_stage_known": 443, "nutrition_nonempty": 464, "calcium_present": 346, "phosphorus_present": 348, "moisture_present": 331, "ingredients_present": 257, "form_explicit": "not a separate adapter field; inferred from description and therefore not identity evidence"}`

## 제한

Danawa `pcode`는 current OPFF/GLOBAL/OEM barcode와 직접 join key가 아니다. 실제 BE schema는 이 repository에 없어 docs의 schema 주장만 확인 가능하다.
