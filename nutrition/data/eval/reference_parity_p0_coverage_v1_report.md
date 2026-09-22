# Nutrition Reference Parity P0 — persisted-product applicability 분석

## 실측 범위

- local product records: 394
- source-confirmed FOOD: 205
- 이 분석은 raw 데이터를 수정하거나 누락값을 보정하지 않았다.

## Reference reconciliation

`{"legacy_rows": 325, "new_threshold_rows": 293, "legacy_mapping_counts": {"SPLIT_OR_APPLICABILITY_RESTORED": 66, "REMOVED_OR_UNMATCHED": 106, "MAPPED": 153}, "new_rows_without_legacy_four_dimension_match": 8, "unmatched_legacy_by_nutrient": {"ARACHIDONIC_ACID": 2, "BIOTIN": 4, "CHOLINE": 8, "FOLIC_ACID": 8, "IODINE": 8, "ISOLEUCINE": 8, "LEUCINE": 8, "NIACIN": 8, "PANTOTHENIC_ACID": 8, "PHENYLALANINE_TYROSINE": 8, "SELENIUM": 8, "VALINE": 8, "VITAMIN_B12": 8, "VITAMIN_B6": 8, "VITAMIN_K": 4}, "explanation": "new rows are threshold-level rules; form/detail splits can increase rules while unsupported legacy nutrient mappings can reduce total rows"}`

## Source inventory

`{"OPFF": {"product_count": 108, "category:food": 108, "species:UNKNOWN": 108, "life_stage:UNKNOWN": 108, "nutrition_items_present": 25, "ingredients_present": 28, "moisture_present": 6, "explicit_product_form_present": 0}, "OEM": {"product_count": 189, "category:UNKNOWN": 189, "species:DOG": 176, "life_stage:UNKNOWN": 189, "nutrition_items_present": 0, "ingredients_present": 44, "moisture_present": 0, "explicit_product_form_present": 0, "species:CAT": 13}, "GLOBAL": {"product_count": 97, "category:food": 97, "species:CAT": 35, "life_stage:ADULT_MAINTENANCE": 11, "nutrition_items_present": 0, "ingredients_present": 47, "moisture_present": 0, "explicit_product_form_present": 97, "species:DOG": 33, "life_stage:ALL_LIFE_STAGES": 74, "life_stage:GROWTH_REPRODUCTION": 7, "species:UNKNOWN": 29, "life_stage:UNKNOWN": 5}}`

## Form / life-stage coverage

`{"ALL|MOISTURE_DERIVED|DRY": 4, "FOOD|MOISTURE_DERIVED|DRY": 4, "ALL|UNKNOWN|UNKNOWN": 291, "FOOD|UNKNOWN|UNKNOWN": 102, "ALL|MOISTURE_DERIVED|CANNED": 2, "FOOD|MOISTURE_DERIVED|CANNED": 2, "ALL|EXPLICIT|DRY": 88, "FOOD|EXPLICIT|DRY": 88, "ALL|EXPLICIT|CANNED": 9, "FOOD|EXPLICIT|CANNED": 9}`

`{"UNKNOWN": 302, "ADULT_MAINTENANCE": 11, "ALL_LIFE_STAGES": 74, "GROWTH_REPRODUCTION": 7}`

## Nutrient coverage

`{"ALL_PRODUCTS:CALCIUM": {"present": 0, "valid": 0, "missing": 394, "unit_missing": 0, "basis_missing": 0, "invalid": 0}, "ALL_PRODUCTS:CRUDE_FAT": {"present": 22, "valid": 19, "missing": 372, "unit_missing": 0, "basis_missing": 0, "invalid": 0}, "ALL_PRODUCTS:CRUDE_PROTEIN": {"present": 22, "valid": 20, "missing": 372, "unit_missing": 0, "basis_missing": 0, "invalid": 0}, "ALL_PRODUCTS:MOISTURE": {"present": 6, "valid": 6, "missing": 388, "unit_missing": 0, "basis_missing": 0, "invalid": 0}, "ALL_PRODUCTS:PHOSPHORUS": {"present": 0, "valid": 0, "missing": 394, "unit_missing": 0, "basis_missing": 0, "invalid": 0}, "ALL_PRODUCTS:TAURINE": {"present": 0, "valid": 0, "missing": 394, "unit_missing": 0, "basis_missing": 0, "invalid": 0}, "DOG:CRUDE_PROTEIN": {"present": 0, "valid": 0, "missing": 209, "unit_missing": 0, "basis_missing": 0, "invalid": 0}, "DOG:CRUDE_FAT": {"present": 0, "valid": 0, "missing": 209, "unit_missing": 0, "basis_missing": 0, "invalid": 0}, "DOG:MOISTURE": {"present": 0, "valid": 0, "missing": 209, "unit_missing": 0, "basis_missing": 0, "invalid": 0}, "DOG:CALCIUM": {"present": 0, "valid": 0, "missing": 209, "unit_missing": 0, "basis_missing": 0, "invalid": 0}, "DOG:PHOSPHORUS": {"present": 0, "valid": 0, "missing": 209, "unit_missing": 0, "basis_missing": 0, "invalid": 0}, "CAT:CRUDE_PROTEIN": {"present": 0, "valid": 0, "missing": 48, "unit_missing": 0, "basis_missing": 0, "invalid": 0}, "CAT:CRUDE_FAT": {"present": 0, "valid": 0, "missing": 48, "unit_missing": 0, "basis_missing": 0, "invalid": 0}, "CAT:MOISTURE": {"present": 0, "valid": 0, "missing": 48, "unit_missing": 0, "basis_missing": 0, "invalid": 0}, "CAT:CALCIUM": {"present": 0, "valid": 0, "missing": 48, "unit_missing": 0, "basis_missing": 0, "invalid": 0}, "CAT:PHOSPHORUS": {"present": 0, "valid": 0, "missing": 48, "unit_missing": 0, "basis_missing": 0, "invalid": 0}, "CAT:TAURINE": {"present": 0, "valid": 0, "missing": 48, "unit_missing": 0, "basis_missing": 0, "invalid": 0}}`

## Evaluability

`{"INSUFFICIENT_DATA": 205, "UNSUPPORTED": 189}`

## Reference applicability and transition

`{"NO_VALUE": 7, "NO_REF_STAGE_OR_SPECIES_UNKNOWN": 53}`

`{"NOT_COMPARABLE": 60}`

## PR-CS 사전 조건

`{"total_products": 394, "calcium_present": 0, "phosphorus_present": 0, "both_present": 0, "both_valid": 0, "same_basis": 0, "ratio_computable": 0, "reference_resolvable": 0}`

## 해석 제한

- product form 또는 life-stage detail이 없는 경우 임의 추정하지 않아 `NO_REF` 계열로 남긴다.
- 결과는 local artifact의 metadata completeness이며, 임상적 적합성 또는 전체 상품 population을 의미하지 않는다.
