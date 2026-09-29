# Nutrition Canonical Input Contract

## 목적

Nutrition Rule Engine은 AWS Service DB row나 FE request schema에 직접 결합하지 않는다. source data는 Adapter를 거쳐 Canonical Pet Input과 Canonical Product Input으로 변환한다.

```text
Service DB source row
→ Repository
→ Service DB Adapter
→ Canonical Pet / Product Input
→ Nutrition Rule Engine
```

현재 `scripts/nutrition/product_input_adapter.py`는 local persisted artifact를 동일한 Product input shape로 변환하는 demonstrator다. 2026-09-28 추가된 `service_db_adapter.py`는 조회 완료된 Service source의 순수 변환 경계이며, 실제 AWS Repository는 아직 없다. 최신 입력·누락 정책은 [P0~P3 보고서](service_integration_preparation_p0_p3.md)를 따른다.

## 현재 Canonical input에 필요한 logical field

### Pet input

- `id`
- `species`
- `age_years`
- `weight_kg`
- `allergies`
- `allergy_profile_status`
- `life_stage`
- `life_stage_detail`

### Product input

- `id`, `name`, `category`
- `target_species`, `aafco_life_stage`, `product_form`
- `ingredient_list`와 ingredient source/provenance
- `nutrition_items[]`: `nutrient_code`, `value`, `unit`, `basis`, `source`

이 목록은 현재 `api_nutrition.py`의 Pydantic input과 `product_input_adapter.py`가 전달하는 runtime shape를 문서화한 것이다. 이는 아직 AWS table/column contract가 아니다.

## 변환 규칙

- Adapter는 source에 명시된 값만 전달한다.
- category, species, life-stage, moisture, nutrient value, unit, basis, reference를 추론하거나 기본값으로 만들지 않는다.
- product target life-stage와 pet life-stage는 서로 대체하지 않는다.
- Service Product `target_age_group`은 마케팅 타깃 메타데이터다. `service_target_age_group`과 provenance에만 보존하며 `aafco_life_stage`를 생성하지 않는다. 현재 Service 경로는 별도 AAFCO label evidence 계약이 없으므로 `None` / `UNKNOWN`을 유지한다. exact GTIN 연결도 그 증명을 대신하지 않는다.
- verified Gold operational evidence는 product ID와 GTIN binding 및 gate를 통과한 경우에만 raw nutrition layer를 대체할 수 있다.
- 변환할 수 없는 source field는 `None` 또는 빈 evidence로 남기고, downstream Rule Engine이 `UNKNOWN`/`INSUFFICIENT_DATA`를 반환하도록 한다.

## 후속 구현 경계

Service source 조회 및 Gateway/ownership 연결이 확인되면 준비된 Adapter에 연결한다. Rule Engine의 rule/reference/safety policy를 DB schema에 맞추어 변경하지 않는다.

외부 입력 필드와 누락 정책은 [서비스 입력 계약](service_source_input_contract.md), 실제 데이터 수령 후 검증 절차는 [P2 체크리스트](p2_production_validation_checklist.md)를 따른다.
