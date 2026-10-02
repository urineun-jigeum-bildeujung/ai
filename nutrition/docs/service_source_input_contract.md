# Nutrition 외부 서비스 최소 입력 계약

> **2026-09-30 v1.0 최신 상태**
>
> PR #141에서 Service DB SELECT Repository와 Gateway trust boundary를 구현했다. `member_db` / `product_db`는 SELECT only이며 Pet ownership은 SQL에서 `pet.id + member_id + deleted_at IS NULL`로 강제한다.
> `POST /api/nutrition/analyze/by-service-id`는 더 이상 항상-503 stub이 아니다. source/auth 미구성 시 fail-close하고, 구성이 있으면 실제 Service DB source를 조회한다.
> `/ready`는 local mode에서는 artifact readiness, service mode에서는 artifact + member_db + product_db + service auth/source readiness를 본다.
> AI Result Table / upsert는 v1.0 필수 범위가 아니다. 실제 dev Secret/NetworkPolicy/Gateway route/Service DB row/FE E2E는 아직 미검증이다.
> 아래의 상충하는 “Repository 미구현”, “항상 503”, “Result Store 필수” 설명은 historical record로 읽는다.


## 범위와 상태

2026-09-29 기준 Nutrition이 요구하는 **조회 완료된 내부 DTO**의 계약이다. 현재 BE 응답 또는 운영 DB column 전체와 동일하다는 뜻이 아니다. 구현 근거는 `scripts/nutrition/service_db_adapter.py`의 `adapt_pet`, `adapt_product`, `evaluate_service_safety`와 `scripts/api_nutrition.py`의 `analyze_service_records`다.

실제 source/auth/ownership/result persistence는 미연결이다. 다른 팀 코드와 실제 DB는 수정하지 않았다. 이 문서가 배포 또는 DB 쓰기 승인을 대신하지 않는다.

## Pet 최소 입력

| 외부에서 확보할 항목 | Adapter 내부 표현 및 검증 | 누락/모순 처리 |
|---|---|---|
| `pet_id` | `id`, 양의 정수 또는 ASCII 숫자 문자열 → canonical 문자열 ID | 식별자 오류로 거절 |
| `member_id` / ownership verified context | 인증된 principal의 소유 Pet임을 호출자가 확인한 뒤 내부 함수 호출 | 신뢰 경계 미연결이면 서비스 endpoint 활성화 금지 |
| `species` | DOG/CAT → dog/cat | 거절, 임의 species 없음 |
| `age` | 현재 합의된 년 단위 → `age_years`, 유한한 0 이상 값 | 거절, SQL numeric에서 단위 추론 금지 |
| `weight` | 현재 합의된 kg → `weight_kg`, 유한한 양수 | 거절 |
| `allergies` | 서비스 알레르겐 code 배열. 객체 배열이면 호출자가 명시적으로 code 추출 | 목록 없음은 빈 배열일 수 있지만 KNOWN_NONE의 근거가 아님 |
| `allergy_profile_status` | UNKNOWN / KNOWN_NONE / KNOWN_LIST | 상태 생략, 미지원, 상태-목록 모순 → UNKNOWN |
| `life_stage`, 필요 시 `life_stage_detail` | 명시적 canonical 단계 전달. 나이/상품명 기반 fallback 없음 | 단계 없음 → UNKNOWN, Safety fail-close |

`KNOWN_NONE`은 명시 상태 + 빈 목록, `KNOWN_LIST`는 명시 상태 + 비어 있지 않은 목록일 때만 일관적이다. `pet_allergy` 0행 또는 상태 미등록을 알레르기 없음으로 해석하지 않는다. Pet profile이 SoT이며 Product가 이를 덮어쓸 수 없다.

Service `AllergenCode`는 `sever/dev` revision `230e598833f684c6c9f2ce605776e230a5b6f236`와 AI v3 사전의 명시 alias를 기준으로 변환한다. SALMON/TUNA → fish, CHEESE/WHEY → dairy, CRUSTACEAN → shellfish, WHEAT_GLUTEN → wheat, OAT_BARLEY → oat + barley, SWEET_POTATO/TAPIOCA → potato를 사용한다. 지원된 정확한 enum 값만 변환하며 원문 `service_allergy_codes`를 보존한다. v3에 근거 없는 enum은 `SERVICE_CODE:` unresolved로 유지하고 parent group, fuzzy text, 독성 namespace를 알레르겐으로 임의 확장하지 않는다.

최신 Pet domain/entity/DTO/migration에도 명시 allergy_profile_status 또는 life-stage 저장 필드가 없다. 따라서 알레르기 0행은 UNKNOWN이며 KNOWN_NONE persistence gap은 남는다. 기존 Mock Nutrition의 age-rule 호환 경로는 Feeding 정책과 분리한다. Feeding은 실제 Service stage와 승인 계수 없음을 명시하고 `MER_COEFFICIENT_UNRESOLVED`, `daily_serving_g=null`로 반환한다. 기존 bcs/is_neutered/birth_date 컬럼을 read-only 추가 조회하되 생애주기를 새로 추정하지 않는다.

## Product 최소 입력

| 외부에서 확보할 항목 | Adapter 처리 | 누락/한계 |
|---|---|---|
| `product_id` | `id`로 전달, 서비스 ID를 응답에 보존 | 로컬 evidence ID로 치환 금지 |
| `sku` | ASCII GTIN-8/12/13/14 + check digit 검증, 기존 canonical 규칙만 사용 | INVALID_SKU / UNMATCHED / AMBIGUOUS이면 영양값 조회 안 함 |
| `category_code` | FOOD/TREAT/SUPPLEMENT만 변환 | 누락/미지원 거절 |
| `target_species` | 관계 source에서 집계한 DOG/CAT 배열 | 미지원/누락 None, Safety fail-close |
| `target_age_group` | `service_target_age_group` 및 provenance 메타데이터 | AAFCO 단계로 승격 금지 |
| `allergen_flags` | 서비스 알레르겐 code 배열 | 교집합 없음은 안전 근거 아님 |
| `ingredient_codes` | dictionary ingredient code exact match 우선 STRUCTURED_SOURCE; 미일치 시 충돌 없는 정확한 v3 alias를 CANONICAL_ALIAS | 미등록/복수 매핑은 UNRESOLVED |

같은 service namespace에서 Pet allergies와 Product flags가 정확히 교차하면 명시적 충돌 근거로 차단할 수 있다. 교집합 부재는 원료 evidence/profile/species/life-stage gate를 우회하지 않는다. `product_allergens`의 생성 주체 및 provenance 보장 정책은 외부 담당자가 확정해야 한다.

## 별도 Nutrition evidence

보증성분은 `nutrient_code`, `value`, `unit`, `basis`, `source`와 source provenance가 별도로 필요하다. Product 마케팅 metadata가 이를 대신하지 않는다. 값·수분·단위·basis를 생성하거나 평균으로 보정하지 않는다.

현재 exact SKU 연결 시 기존 local `load_product_input` 및 Gold gate를 재사용하지만 **운영 guaranteed-analysis 저장/조회 계약이 확정된 것은 아니다**. `operating_identity_verified=false`를 유지한다. 동일 GTIN에 명시적 종/분류 충돌이면 AMBIGUOUS로 보류한다.

AAFCO claim은 별도 authoritative label evidence가 필요하다. 현재 Service adapter에는 이를 승인해서 전달하는 계약이 없으므로 `aafco_life_stage=None`, `aafco_life_stage_evidence_status=UNKNOWN`이다. 요청에 임의 필드를 추가하거나 local artifact에 stage가 있다는 것만으로 승격하지 않는다. 이 때문에 FOOD의 service-source 분석은 현재 label 부족으로 Safety 차단될 수 있으며 이를 숨기지 않는다.

## 상태 및 endpoint 계약

| 상황 | Safety/제외 계약 |
|---|---|
| 알레르기 충돌 | SAFETY_BLOCKED, `excluded=true` |
| Safety 근거 부족 | SAFETY_DATA_INSUFFICIENT, `excluded=true` |
| Safety blocker 없는 Nutrition-only 부족 | `analysis_status=INSUFFICIENT_DATA`, `excluded=false` |

`analysis_status`만 보고 추천 제외를 결정하지 않는다. 기존 요청 기반/local persisted API의 호환 계약은 유지하며 이 문서는 신규 Service 입력 경계를 구분한다.

`POST /api/nutrition/analyze/by-service-id`는 양의 정수 `pet_id`, `product_id`만 받는다. 현재 유효 요청은 **503 / SERVICE_SOURCE_NOT_CONFIGURED**다. 신뢰할 수 없는 `X-Member-Id` 또는 local seed/persisted artifact로 운영 source/auth를 대체하지 않는다. 내부 `analyze_service_records` 테스트는 실제 인증·ownership·AWS 검증이 아니다.

## 운영 연결 전 필요한 외부 확정

1. source 조회 방식 및 접속/권한, 관계 데이터 집계 책임과 provenance.
2. authenticated principal → member → Pet ownership의 호출 책임 및 신뢰 경계. 현재 함수에 boolean을 넣는 방식으로 인증을 대신하지 않는다.
3. 명시 allergy profile status 제공, 서비스 allergen code 생성·검증 책임.
4. 실제 Service Product 표본의 exact identity 검증 및 authoritative AAFCO/보증성분 evidence 전달 계약.
5. AI 결과 저장 schema, 저장 주체, upsert key, transaction 및 저장 실패 응답 계약.

AI 결과 저장 및 새 schema는 미확정이다. 필요 시 DDL은 `nutrition/docs/` 안에 `PROPOSED` / `NOT_APPLIED`로만 제안한다. 현재 CREATE/ALTER/GRANT/REVOKE/INSERT/UPDATE/DELETE 실행은 모두 금지다.
