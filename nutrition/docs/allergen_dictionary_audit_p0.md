# P0 알레르기 사전 감사 및 단일 SoT

본 문서는 영양성분 분석 AI 전체 중 알레르기 안전성 판단 서브시스템의 검증 문서이다. NIAS 영양 기준 비교, AAFCO 판정, 보증성분 정규화 등 영양성분 전체 기능의 완료를 의미하지 않는다.

## 실제 소비자

| Dictionary | 행/code 수 | alias 수 | 실제 사용 코드 | 서비스 영향 | 유지 여부 |
|---|---:|---:|---|---|---|
| `seed_feed_codes.json` | 13 entries | 142 normalized aliases | 기존 `match_v1.py` signature | 과거 matcher 입력 | Deprecated compatibility input |
| `seed_11_allergen_ingredient_map.json` | 11 | 142 declared keyword instances | 기존 `normalizer.py` | 과거 화면 정규화 | Deprecated compatibility source |
| `seed_11_allergen_ingredient_map_v3.json` | 29 actual rows | 298 normalized aliases | `nutrition/allergen_service.py` | canonical safety input | ACTIVE SoT source |
| `seed_15_ingredient_synonym_v2.json` | 40 actual rows | ingredient synonym table | 직접 safety consumer 없음 | 원료 일반화 보조 | safety input 아님 |

## P0 계약

`nutrition/allergen_service.py`는 runtime 판단을 담당하는 단일 서비스다. v3에서 `allergen_code`, `canonical_name`, `aliases`, `language`, `source`, `source_version`, `status`를 정규화해 사용한다. alias 충돌은 자동 해소하지 않고 `DICTIONARY["conflicts"]`로 기록한다. 모든 safety caller는 `evaluate_safety`를 사용하며 `match_v1.match_allergens`는 compatibility adapter로만 유지한다.

## 이관과 한계

- 상품 원료는 read-time에 `product_allergen_refs`로 변환한다. 원천 파일은 덮어쓰지 않으며 `PRODUCT_LABEL`과 `INGREDIENT_REFERENCE`를 혼합하지 않는다.
- `CANONICAL_EXACT`, `CANONICAL_ALIAS`, `STRUCTURED_SOURCE`만 safety 판단 근거가 될 수 있다. `OCR_INFERRED`, `FUZZY`, `UNRESOLVED`는 safety 근거가 될 수 없다.
- `dinde`는 active v3 allergen entry가 아니므로 turkey의 `UNRESOLVED` evidence로 기록되어 KNOWN_LIST profile에서 fail-close된다. `turkey` / `dinde`는 `PROPOSED` 상태이고 Human Precision Audit 및 후보 검토 전 자동 승인하지 않는다.
- 생애주기 매핑은 `allergen_service._pet_stage`에 문서화되어 있다. 명시적 puppy/kitten/growth는 `GROWTH_REPRODUCTION`, adult/senior/maintenance는 `ADULT_MAINTENANCE`로 매핑하며, 명시값이 없을 때만 age `< 1`을 사용한다.
