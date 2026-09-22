# Nutrition AI 로컬 아티팩트 기반 persisted-product E2E 감사 v1

본 문서는 영양성분 분석 AI 전체 중 **로컬 아티팩트 기반 persisted-product E2E**의 현재 상태를 기록한다. 운영 DB나 배포 서비스에 연결된 E2E가 아니며, request-scoped E2E와 같은 완료 상태로 해석하지 않는다.

## 범위와 검증 기준

- **Request-scoped E2E — VERIFIED**: `POST /api/nutrition/analyze`에 `pet`, `product`, `nutrition_items`를 직접 전달하면 `pipeline_p1c_v1`까지 실행된다.
- **로컬 아티팩트 기반 persisted-product E2E — PARTIAL**: `POST /api/nutrition/analyze/by-product-id`가 `product_id`로 `data/raw/`의 상품·보증성분을 읽어 기존 엔진에 전달한다. 운영 DB 조회나 정식 백엔드 계약은 아직 연결되지 않았다.
- HTTP 200은 전송 성공일 뿐 `READY`, 영양 판정 성공, 또는 안전성 검증 완료를 의미하지 않는다.
- `input_readiness`는 **현재 런타임 규칙 엔진을 실행할 입력 완전성**이며, `analysis_status`·AAFCO 결과와 별도다. 입력이 완전해도 비교·안전 판정이 성공했다는 뜻은 아니다.

## 결정적 fixture

| 항목 | 실제 값 |
|---|---|
| product_id | `0064992280178` |
| source_dataset | `OPFF` |
| product source | `data/raw/seed_9_placeholder_feed_opff.json` |
| nutrition source | `data/raw/seed_13_guaranteed_analysis.json` |
| nutrition item count | 4 (`CRUDE_PROTEIN`, `CRUDE_FAT`, `CRUDE_FIBER`, `MOISTURE`) |
| species | `BOTH` (명시됨) |
| product target life-stage | 없음 |
| moisture | 10% 존재 |
| ingredient | 20개 존재 |

fixture에는 값을 보충하거나, 상품명·카테고리에서 종/생애주기를 추론하지 않았다.

## 실제 런타임 READY 기준과 이전 5개 기준의 차이

현재 공유 SoT는 `scripts/pipeline_p1c_v1.py`의 `ESSENTIAL_NUTRIENTS`이며, `scripts/nutrition/nutrition_readiness.py`가 이를 직접 읽는다. 현재 실제 required set은 다음과 같다.

| 요청 종 / reference stage | 런타임 필수 영양소 |
|---|---|
| DOG / `ADULT_MAINTENANCE`, `GROWTH_REPRODUCTION` | `CRUDE_PROTEIN`, `CRUDE_FAT`, `MOISTURE`, `CALCIUM`, `PHOSPHORUS` |
| CAT / `ADULT_MAINTENANCE`, `GROWTH_REPRODUCTION` | 위 5개 + `TAURINE` |

이전 5개 기준은 DOG의 현재 런타임 set에는 맞지만 CAT의 `TAURINE`을 빠뜨린 예비 기준이다. 따라서 “5개 보증성분 보유 = runtime READY”라고 일반화할 수 없다. fixture는 CAT 요청 기준에서 `CALCIUM`, `PHOSPHORUS`, `TAURINE`이 없어 `PARTIAL`이고, product target life-stage도 없다.

`required_nutrients`의 **실행 SoT**는 현재 `pipeline_p1c_v1.ESSENTIAL_NUTRIENTS` 한 곳이다. `data/processed/required_nutrient_matrix_v2.csv`와 `docs/required_nutrient_matrix_v2.md`는 17종(DOG)/18종(CAT) 및 Ca:P 기준으로 기존 집합을 supersede한다고 문서화하지만, 실제 `compute_nutrition_comparison_status()`는 아직 그 matrix를 읽지 않는다. 즉 문서/매트릭스 기준과 runtime 기준은 현재 다르며, runtime에 적용되기 전에는 matrix v2를 readiness 기준으로 사용하지 않는다.

## Minimum Runtime Nutrient Set과 Comprehensive Nutrient Matrix

두 기준은 삭제·강제 통합 대상이 아니라 역할이 다른 별도 SoT다.

| 개념 | SoT | 용도 | 현재 runtime gate 여부 |
|---|---|---|---|
| Minimum Runtime Nutrient Set | `pipeline_p1c_v1.ESSENTIAL_NUTRIENTS` | 현재 Rule Engine의 최소 reference 비교 입력 | 예 |
| Comprehensive Nutrient Matrix | `data/processed/required_nutrient_matrix_v2.csv` | tier별로 어느 nutrient까지 분석 coverage가 있는지 표시 | 아니오 |

`READY`는 “현재 지원되는 Rule Engine의 필수 입력과 safety gate가 충족되어 해당 분석 요청의 구조화된 결과를 반환할 수 있는 상태”다. 영양학적 완전성, matrix 전 항목의 검증 완료, 임상적 적합성을 뜻하지 않는다.

### 상태 축 계약

| 축 | 의미 | 다른 축과의 관계 |
|---|---|---|
| `input_readiness` | Minimum Runtime Nutrient Set 기준 입력 완결성 | `READY`여도 reference 비교 성공을 보장하지 않음 |
| `nutrition_coverage` | Comprehensive Nutrient Matrix 기준 구조화·비영값 nutrient 범위 | runtime `READY`를 올리거나 내리지 않음 |
| `nutrition_comparison_status` | 실제 NIAS/reference 비교 결과 | coverage 또는 HTTP 상태의 alias가 아님 |
| `safety_status` | 종·생애주기·알레르기 fail-close 결과 | nutrition 비교와 독립 |
| `analysis_status` | API가 위 사실을 종합한 최종 처리 상태 | nutritional adequacy의 선언이 아님 |

`nutrition_coverage` 상태는 `UNKNOWN`, `NONE`, `LIMITED`, `STANDARD`, `COMPREHENSIVE`다. `STANDARD`는 Minimum Runtime Nutrient Set의 구조화된 값이 존재하지만 matrix tier1/tier2 전체 coverage는 아닌 상태다. `COMPREHENSIVE`는 matrix의 현재 적용 가능 tier1/tier2 input이 모두 구조화되어 있다는 뜻일 뿐, reference comparison 통과나 의료적 적합성을 뜻하지 않는다. `CA_P_RATIO`는 derived 비교이며 label input 수로 세지 않고, `tier3_unavailable`은 trace로 반환하지만 gate에 사용하지 않는다.

따라서 runtime readiness ≠ nutritional adequacy, minimum nutrients ≠ comprehensive profile, README matrix ≠ executable runtime, input completeness ≠ analysis success, READY ≠ clinically suitable, HTTP 200 ≠ READY다.

## 단계별 실행 결과

fixture를 adult CAT 요청으로 in-process HTTP 실행해 확인한 결과다.

| 단계 | 상태 | 근거 |
|---|---|---|
| product lookup | PASS | OPFF 로컬 아티팩트에서 fixture ID 조회 |
| nutrition lookup | PASS | `seed_13_guaranteed_analysis.json`에서 4개 row 조회 |
| 입력 스키마 검증 | PASS | 현재 row의 unit/basis가 모두 명시돼 `ProductIn` 검증 통과 |
| normalization / unit alignment / AS_FED-DM | PASS | 기존 `pipeline_p1c_v1` 함수 실행 |
| NIAS lookup | UNKNOWN | 필수 `CALCIUM`·`PHOSPHORUS`·`TAURINE` 결측으로 비교 결과 확정 불가 |
| nutrition comparison | UNKNOWN | `nutrition_comparison_status=UNKNOWN`, `aafco_pass=null` |
| species | PASS | `BOTH`를 요청 종(CAT)에 맞춰 명시적으로 해석 |
| pet reference stage | PASS | 요청의 adult를 `ADULT_MAINTENANCE`로 명시 변환 |
| product target stage | UNKNOWN | 저장 상품의 `target_life_stage` 없음; 요청 반려동물 stage로 대체하지 않음 |
| allergen gate | INSUFFICIENT_DATA | 알레르기 evidence/제품 정보의 fail-close 결과 |
| final status | INSUFFICIENT_DATA | `input_readiness=PARTIAL` 및 분석/안전 보류를 그대로 반환 |
| response serialization | PASS | 200 응답과 `input_provenance` 반환 확인 |

## 단위·basis 계약 감사

`seed_13_guaranteed_analysis.json`의 현재 60개 보증성분 row는 unit과 basis가 모두 명시되어 있다(모두 unit 존재, basis 결측 0). basis는 `DRY_MATTER` 54개와 `AS_FED` 6개다.

- adapter는 `unit`을 `PERCENT`, `basis`를 `AS_FED`로 기본 대입하지 않는다.
- 새 row가 unit/basis를 누락하면 Pydantic 검증 오류를 HTTP 500으로 숨기지 않고 `source_validation_status=INVALID_SOURCE_DATA`, `analysis_status=INSUFFICIENT_DATA`로 반환한다.
- 현재 아티팩트에서 값이 모두 있다는 사실은 **행별 명시값**의 근거일 뿐, 다른 소스의 기본값 계약 또는 추론 근거가 아니다.

## 카테고리 계약 감사

| source_dataset | food 인정 조건 | 실제 처리 |
|---|---|---|
| OPFF | 원본 `category == "food"` | food |
| GLOBAL | 원본 `product_type`이 `DRY_FOOD` 또는 `WET_FOOD` | food |
| OEM | food를 보장하는 닫힌 원본 enum 없음; category 결측도 140/189 | `None`으로 보존, `CATEGORY_NOT_SOURCE_CONFIRMED` |

OEM 189개를 임의로 food로 강제하지 않는다. 이들은 재계산에서 `UNSUPPORTED`로 분류되며, 이는 영양값이 없다는 주장과 다른 **소스 분류 미확정** 상태다.

## 생애주기 계약 및 silent fallback 감사

반려동물 reference stage와 상품 target stage는 다른 사실이다.

- **pet reference stage**: `puppy`/`kitten`/`growth` → `GROWTH_REPRODUCTION`, `adult`/`maintenance` → `ADULT_MAINTENANCE`. life-stage가 없을 때만 유효한 `age_years`의 명시 규칙을 사용한다.
- **unsupported pet stage**: `senior` 등은 adult로 fallback하지 않고 `UNSUPPORTED_PET_LIFE_STAGE`와 `UNKNOWN`/`INSUFFICIENT_DATA`로 남긴다.
- **product target stage**: 저장 라벨의 `ADULT`, `MAINTENANCE`, `GROWTH`, `ALL_LIFE_STAGES`만 명시 변환한다. 누락은 `PRODUCT_TARGET_STAGE_MISSING`, 지원하지 않는 값은 `UNSUPPORTED_PRODUCT_TARGET_STAGE`다.
- product target stage 누락을 pet reference stage로 대체하지 않는다.

따라서 source default는 추론 근거가 아니며, pet life-stage와 product target life-stage도 서로 대체할 수 없다.

`compute_nutrition_comparison_status()`도 명시적으로 전달된 `(species, stage)` 조합이 현재 `ESSENTIAL_NUTRIENTS`에 없으면 DOG adult set으로 대체하지 않고 `UNKNOWN` 및 `UNSUPPORTED_REFERENCE_COMBINATION`을 반환한다. map 자체가 전혀 없는 오래된 직접 호출의 legacy default는 남아 있으므로, 새 caller는 반드시 species/stage map을 전달해야 한다.

### Legacy fallback caller 감사

`compute_nutrition_comparison_status()` 및 compatibility alias `compute_aafco_pass()`의 실제 caller를 코드 검색으로 분류했다.

| 분류 | 위치 | 근거 |
|---|---|---|
| `SAFE_NEW_CALLER` | `scripts/api_nutrition.py`의 `_run_p0d` 두 호출 | species/stage map을 모두 명시 전달 |
| `LEGACY_EXPLICIT` | `tests/test_nutrition_readiness.py`의 unsupported-reference test | 지원하지 않는 map을 명시 전달해 fail-close 검증 |
| `LEGACY_IMPLICIT_DEFAULT` | `scripts/pipeline_p1c_v1.py:1107-1117` | `compare_nias`와 `compute_aafco_pass`에 species map만 전달, stage map 없음 |
| `LEGACY_IMPLICIT_DEFAULT` | `scripts/e2e1_p2_final_v1.py:123-129`, `scripts/p2_regen_v1.py:101-111` | 동일한 batch regeneration 경로 |
| `LEGACY_IMPLICIT_DEFAULT` | `tests/test_p0_contracts.py:55-61`, `tests/test_p0_nutrition_safety.py:119-124` | old no-map 동작을 사용하는 단위 테스트 |

`verify_7_queries*`의 동명 함수는 `pipeline_p1c_v1.compute_nutrition_comparison_status()`가 아니라 해당 파일 내부의 다른 legacy 구현이므로 이 caller 집계에서 제외했다.

안전한 migration 순서는 (1) batch 원천의 per-product life-stage map 존재 여부를 실측, (2) `pipeline_p1c_v1.main`과 재생성 스크립트가 map을 명시 전달하도록 변경, (3) no-map 호출을 `UNKNOWN`/`UNSUPPORTED_REFERENCE_COMBINATION`으로 fail-close, (4) legacy 단위 테스트를 explicit-map fixture로 교체하는 것이다. 현재는 이전 batch 산출물의 비교 가능성에 영향을 줄 수 있으므로 이 fallback을 무조건 삭제하지 않았다.

## Product-level input evaluability 재계산

`scripts/nutrition/audit_persisted_evaluability.py`가 394개 고유 로컬 상품을 공유 readiness evaluator로 재계산했고, 결과는 `data/eval/nutrition_input_evaluability_after_contract_v1.json`에 보관한다. 사용자 알레르기 프로필을 적용하지 않았으므로 `SAFETY_BLOCKED`는 이 분류에서 0이다.

| 상태 | 이전 예비 집계 | 계약 정리 후 | 변화 |
|---|---:|---:|---:|
| READY | 0 | 0 | 0 |
| PARTIAL | 24 | 24 | 0 |
| INSUFFICIENT_DATA | 370 | 181 | -189 |
| SAFETY_BLOCKED | 0 | 0 | 0 |
| UNSUPPORTED | 0 | 189 | +189 |

주요 reason code는 `CATEGORY_NOT_FOOD` 189건, `NUTRITION_ITEMS_MISSING` 181건, `MISSING_REQUIRED_NUTRIENTS` 205건, `PRODUCT_TARGET_STAGE_MISSING` 108건, `UNSUPPORTED_PRODUCT_TARGET_STAGE` 5건이다. 하나의 상품에 여러 reason code가 함께 있을 수 있으므로 reason code 합계는 상품 수와 일치하지 않는다.

row-level 값 존재는 product-level evaluability가 아니다. 특히 현재 로컬 아티팩트에는 runtime `READY` 상품이 없다.

## READY fixture 후보 재탐색

`scripts/nutrition/audit_ready_fixture_candidates.py`가 OEM 및 source-confirmed food가 아닌 상품을 제외하고 DOG/CAT adult audit request scenario별 후보를 재계산했다. 결과는 `data/eval/ready_fixture_candidates_minimum_runtime_v1.json`에 보관한다. adult는 후보를 비교하기 위한 명시적 audit request scenario일 뿐 상품 label stage를 추론한 값이 아니다.

- DOG 상위 후보: `0064992280178`, `0064992281182`. 둘 다 `CRUDE_PROTEIN`, `CRUDE_FAT`, `MOISTURE`가 있고 `CALCIUM`, `PHOSPHORUS`, `PRODUCT_TARGET_STAGE`가 부족하다.
- CAT 상위 후보: 동일 product ID들이지만 CAT minimum에는 `TAURINE`이 추가로 부족하다.
- 첫 후보: `0064992280178`의 DOG scenario. CAT보다 minimum missing field가 하나 적고, category·species·moisture·ingredients·unit/basis provenance가 명시돼 있다. 단 상품 target stage와 calcium/phosphorus는 여전히 없으므로 `READY` fixture가 아니다.

후보 ranking은 source collection을 시작하지 않았으며, missing source field 목록으로만 이후 수집 범위를 결정하기 위한 artifact다.

## API 계약 불일치 및 compare 방향 감사

| 항목 | 실제 local runtime | 의도된 문서 계약 |
|---|---|---|
| 분석 경로 | `POST /api/nutrition/analyze`, `POST /api/nutrition/analyze/by-product-id` | `POST /nutrition/analyze` |
| 요청 | embedded product 또는 `pet` + `product_id` | product/pet ID 기반 BE/FE 계약 초안 |
| 응답 | input readiness, 비교/안전 상태, provenance | 문서의 canonical schema 확정 필요 |
| 비교 | `/api/nutrition/compare`는 501 | `/nutrition/diff` 미구현 |
| exclusions | 없음 | `/nutrition/exclusions` 미구현 |

현재 compare route의 주석은 LLM 비교 방향을 언급하지만, 이 감사에서 구현하지 않았다. 이후 비교 기능의 데이터 코어는 **canonical nutrition A vs B를 결정론적으로 비교**하고, LLM은 선택적 설명 레이어로만 둬야 한다. endpoint rename·계약 확정·compare 구현은 이번 범위에서 수행하지 않았다.

## 현재 결론

로컬 아티팩트 기반 adapter와 API route의 연결 및 raw schema fail-close는 확인됐다. 그러나 `READY=0`이므로, 완전한 저장 상품 기준 영양 E2E나 운영 DB 연동이 완료됐다고 말할 수 없다. 다음 데이터 수집은 자동으로 시작하지 않았으며, 수집 전에 runtime 기준과 별도 matrix/문서 기준의 최종 SoT 결정을 먼저 해야 한다.
