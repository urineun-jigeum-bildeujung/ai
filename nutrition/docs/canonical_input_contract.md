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

현재 `scripts/nutrition/product_input_adapter.py`는 local persisted artifact를 동일한 Product input shape로 변환하는 demonstrator다. AWS Service DB Adapter는 아직 구현하지 않았다.

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
- verified Gold operational evidence는 product ID와 GTIN binding 및 gate를 통과한 경우에만 raw nutrition layer를 대체할 수 있다.
- 변환할 수 없는 source field는 `None` 또는 빈 evidence로 남기고, downstream Rule Engine이 `UNKNOWN`/`INSUFFICIENT_DATA`를 반환하도록 한다.

## 후속 구현 경계

AWS schema가 확인되면 Service DB Adapter가 source record를 이 contract로 변환한다. 그때도 Rule Engine의 rule/reference/safety policy를 DB schema에 맞추어 변경하지 않는다.
