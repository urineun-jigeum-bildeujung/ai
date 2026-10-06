# Nutrition 외부 서비스 최소 입력 계약

> **2026-10-05 로컬 수정 계약, 서버 반영 미확인**
>
> 현재 통합 계약과 실제 Gateway 관찰 결과는 [통합 수정 기록](service_integration_resolution_20261005.md), 38개 Service 코드 목록은 [매핑 CSV](service_allergen_mapping_v2.csv)를 따른다. 아래 2026-09-30 기록의 배포 상태는 현재 상태를 의미하지 않는다.
>
> **2026-09-30 v1.0 기록**
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
| `birth_date` / `age` | birth_date 우선, KST 현재 날짜의 완료 개월 수 / 12 → age_years. birth_date 없으면 기존 년 단위 age 검증 | 잘못된 날짜/미래 날짜 거절, birth_date 없는 production stage는 UNKNOWN 유지 |
| `weight` | 현재 합의된 kg → `weight_kg`, 유한한 양수 | 거절 |
| `allergies` | 서비스 알레르겐 code 배열. 객체 배열이면 호출자가 명시적으로 code 추출 | 목록 없음은 빈 배열일 수 있지만 KNOWN_NONE의 근거가 아님 |
| `allergy_profile_status` | UNKNOWN / KNOWN_NONE / KNOWN_LIST | 요청 생략 시 Repository 상태 유지; 명시 null, 미지원, 상태-목록 모순 → UNKNOWN |
| `life_stage`, 필요 시 `life_stage_detail` | birth_date <12개월 GROWTH_REPRODUCTION, >=12개월 ADULT_MAINTENANCE. 생년월일 사용 시 임신/수유 detail을 만들거나 전달하지 않음 | birth_date 없으면 기존 명시 stage/UNKNOWN 정책 유지 |

by-service-id와 compare의 FE 상태는 조회된 `pet_allergy` 목록과 교차 검증한다. 요청에서 생략하면 Repository 상태를 유지하고, 명시 null 또는 저장된 known 상태와 충돌하는 선언은 UNKNOWN으로 처리한다. Backend 로컬 수정의 V3 migration은 `pet.allergy_profile_status`를 저장하며 기존 비어 있지 않은 목록만 KNOWN_LIST로 이관한다. 빈 목록은 UNKNOWN으로 남긴다. 아직 컬럼이 없는 DB에서도 AI 조회는 가능하며 비어 있지 않은 목록만 KNOWN_LIST로 해석한다.

`KNOWN_NONE`은 명시 상태 + 빈 목록, `KNOWN_LIST`는 명시 상태 + 비어 있지 않은 목록일 때만 일관적이다. `pet_allergy` 0행 또는 상태 미등록을 알레르기 없음으로 해석하지 않는다. Pet profile이 SoT이며 Product가 이를 덮어쓸 수 없다.

Service `AllergenCode`는 `sever/dev` revision `c23e7a1ab1b1a31ff32a84e19a50eab7c28bf5c9`의 정확한 enum/표시명과 기존 v3 alias를 기준으로 변환한다. Service 전용 identity 사전은 DUCK/TURKEY 및 SALMON/TUNA/BONITO/ANCHOVY 등을 개별 코드로 보존하며 Pet의 특정 어종을 fish로 확대하지 않는다. 기존 임상 사전이나 라벨 근거를 변경하지 않는다. 33개 코드는 지원하며, 4개 독성 코드는 독성 namespace에 보존하고 OTHER는 미해결로 남긴다. 전체 목록과 근거 범위는 매핑 CSV와 통합 수정 기록을 따른다.

Backend 로컬 수정에는 명시 allergy profile 저장과 등록/수정/상세 DTO가 포함된다. 실제 migration 및 배포는 별도 검증 대상이다. Pet 생애주기는 AI가 유효한 birth_date로 계산하며, 나이만으로 production stage를 대신 정하지 않는다. 기존 Mock Nutrition의 age-rule 호환 경로는 Feeding 정책과 분리한다. 실제 Service Feeding에서 승인 계수 없음을 명시하고 `MER_COEFFICIENT_UNRESOLVED`, `daily_serving_g=null`로 반환하는 정책은 유지한다.

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

한글 원료 exact alias는 기존 v3 매핑을 유지한다. Toxic/Caution code는 알레르기 매핑 및 allergen flag 교집합에서 제외하며, 독성 평가는 `product_cautions`만 사용한다.

같은 service allergy namespace에서 Pet allergies와 Product flags가 정확히 교차하면 명시적 충돌 근거로 차단할 수 있다. 교집합 부재는 원료 evidence/profile/species/life-stage gate를 우회하지 않는다. `product_allergens`의 생성 주체 및 provenance 보장 정책은 외부 담당자가 확정해야 한다.

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

`POST /api/nutrition/analyze/by-service-id`는 양의 정수 `pet_id`, `product_id`와 optional `allergy_profile_status`(UNKNOWN / KNOWN_NONE / KNOWN_LIST)를 받는다. 현재 유효 요청은 **503 / SERVICE_SOURCE_NOT_CONFIGURED**다. 신뢰할 수 없는 `X-Member-Id` 또는 local seed/persisted artifact로 운영 source/auth를 대체하지 않는다. 내부 `analyze_service_records` 테스트는 실제 인증·ownership·AWS 검증이 아니다.

## 운영 연결 전 필요한 외부 확정

1. source 조회 방식 및 접속/권한, 관계 데이터 집계 책임과 provenance.
2. authenticated principal → member → Pet ownership의 호출 책임 및 신뢰 경계. 현재 함수에 boolean을 넣는 방식으로 인증을 대신하지 않는다.
3. 명시 allergy profile status 제공, 서비스 allergen code 생성·검증 책임.
4. 실제 Service Product 표본의 exact identity 검증 및 authoritative AAFCO/보증성분 evidence 전달 계약.
5. AI 결과 저장 schema, 저장 주체, upsert key, transaction 및 저장 실패 응답 계약.

AI 결과 저장 및 새 schema는 미확정이다. 필요 시 DDL은 `nutrition/docs/` 안에 `PROPOSED` / `NOT_APPLIED`로만 제안한다. 현재 CREATE/ALTER/GRANT/REVOKE/INSERT/UPDATE/DELETE 실행은 모두 금지다.

## 2026-10-02 생년월일 경계

현재 날짜는 Asia/Seoul 기준으로 계산한다. 12개월은 달력 개월 경계이며 2월 29일의 다음 해 경계는 2월 말일이다. PostgreSQL DATE와 정확한 YYYY-MM-DD 문자열만 허용한다. 생년월일이 없으면 production의 stage 누락 fail-close를 유지한다. 별도 Mock fixture의 기존 Nutrition AGE_RULE 호환 경로는 legacy regression을 위해 유지하며 Feeding으로 승격하지 않는다.

## 2026-10-06 스키마 Mock 범위 보완

전체 서비스 상품은 스키마로 만든 Mock이라는 사용자 확인을 반영한다. 기존 `MOCK-*` 경로와 함께 `mock_service_identity_v1.json`의 정확한 Service ID·SKU 조합을 합성 입력 대상으로 사용한다. 조회하지 못한 상품을 만들어 반환하지 않는다. 302/ONF-004도 이 목록을 통해 기존 영양·급여 프로필을 사용한다. 당근·비트·호박·연어오일의 합성 identity는 `mock_ingredient_identity_v1.json`에서 별도로 관리하며 실제 상품 사전의 안전 근거로 승격하지 않는다. 상세 결과는 [Mock 근거 보완](service_evidence_gap_resolution_20261006.md)을 참고한다.
