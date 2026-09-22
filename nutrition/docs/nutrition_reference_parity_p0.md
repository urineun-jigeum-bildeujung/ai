# Nutrition Reference Parity P0

본 문서는 영양성분 분석 AI의 NIAS reference applicability 보존 작업을 설명한다. AAFCO 적합성, PR-CS Ca:P product-level 판정, BE API 계약의 완료를 의미하지 않는다.

## 활성 canonical reference

- 활성 artifact: `data/processed/nutrition_reference_nias_2024_parity_p0_v1.json`
- 생성기: `scripts/nutrition/reference_parity.py`
- 원문 입력: `data/raw/seed_46_nias_2024_nutrient_tables.json`
- legacy 보존: `data/raw/seed_14_nutrition_reference_v5.json`은 수정하거나 삭제하지 않는다.

각 threshold row의 필수 선택 차원은 `species`, `life_stage`, `nutrient_code`, `basis`, `threshold_type`이다. `reference_form_canonical`과 `life_stage_detail`은 raw table이 실제로 구분할 때에만 추가 적용한다. `source_name_original`, `reference_form_raw`, `source_table`, `source_page`, `authority`, `edition`, `source_version`은 provenance다.

## 선택 정책

- 명시적 `product_form` (`DRY_FOOD`/`WET_FOOD`)이 있으면 moisture-derived form보다 우선한다.
- 명시적 form이 없으면 기존 moisture 분류의 `DRY`/`WET`만 `DRY`/`CANNED` reference form으로 변환한다.
- `MID`와 `UNKNOWN`은 임의 form으로 변환하지 않는다.
- form-specific 또는 life-stage-detail-specific rule을 해소할 수 없으면 selector는 `NO_REF`를 반환한다. 이는 영양소 부족 판정이 아니다.
- generic rule은 product form이 `UNKNOWN`이어도 form 때문에 차단하지 않는다.

## 이번 범위의 정보 보존

- CAT `TAURINE`과 `COPPER`는 raw `건사료`/`통조림 사료` 구분 및 `DRY_MATTER`/`PER_1000KCAL` basis를 별도 rule로 보존한다.
- DOG 성장·번식 raw header의 `초기 성장 및 번식`과 `후기 성장`은 각각 `EARLY_GROWTH_AND_REPRODUCTION`, `LATE_GROWTH`의 `MINIMUM` rule이다. 값의 대소로 MIN/MAX를 추론하지 않는다.
- CAT 성장·번식의 raw cell이 실제로 `성장`/`번식`을 별도 표기한 경우에만 해당 detail rule을 생성한다.

## 재생성 및 검증

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/nutrition/reference_parity.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/tests -v
```

생성기는 duplicate applicability key, form-specific raw label 손실, DOG growth detail 손실을 invariant로 검사한다. 비교 결과는 `data/eval/nutrition_reference_parity_p0_before_after_v1.json`에 저장한다. 이 artifact의 row 수 증가는 성능 지표가 아니라 applicability 정보 보존의 trace다.
