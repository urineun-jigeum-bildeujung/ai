# Nutrition 서비스 통합 준비 — P0~P3

> **2026-09-30 v1.0 최신 상태**
>
> PR #141에서 Service DB SELECT Repository와 Gateway trust boundary를 구현했다. `member_db` / `product_db`는 SELECT only이며 Pet ownership은 SQL에서 `pet.id + member_id + deleted_at IS NULL`로 강제한다.
> `POST /api/nutrition/analyze/by-service-id`는 더 이상 항상-503 stub이 아니다. source/auth 미구성 시 fail-close하고, 구성이 있으면 실제 Service DB source를 조회한다.
> `/ready`는 local mode에서는 artifact readiness, service mode에서는 artifact + member_db + product_db + service auth/source readiness를 본다.
> AI Result Table / upsert는 v1.0 필수 범위가 아니다. 실제 dev Secret/NetworkPolicy/Gateway route/Service DB row/FE E2E는 아직 미검증이다.
> 아래의 상충하는 “Repository 미구현”, “항상 503”, “Result Store 필수” 설명은 historical record로 읽는다.


## 범위와 기준

2026-09-28, 로컬 `ai-nutrition-main`의 `main` HEAD `8aeb918`에서 시작했다.
작업 시작 시 Git 작업 트리는 비어 있었다. 수정 범위는 이 worktree의 `nutrition/**`뿐이다.
구형 `ai/scripts/**`는 읽기 전용 비교 자료로만 사용했다. 파일 전체 덮어쓰기, Git write, AWS 접근은 하지 않았다.

본 문서는 **영양성분 분석 AI의 기존 기능을 서비스에 연결하기 위한 준비 상태**다.
운영 데이터 identity 검증 완료, 전체 영양 적합성 검증 완료, AWS E2E 완료를 뜻하지 않는다.

## 단계별 결과

| 단계 | 상태 | 구현 및 한계 |
|---|---|---|
| P0 | DONE | Pet allergy SoT와 기존 non-food 상태축 유지. additive Safety 설명, `/ready`, `/metrics`, 요청 ID 추가. catalog 조회는 읽기 전용 |
| P1 | DONE | 조회 완료된 source record → Canonical Input 순수 변환. 실제 Repository나 BE DTO 클라이언트는 아직 없음 |
| P2 | PARTIAL | strict GTIN bridge, 정확한 서비스 코드 충돌, dictionary ingredient code 연결 구현. 실제 서비스 상품 표본은 없음 |
| P3 | PARTIAL | 내부 source→Adapter→기존 Engine 연결 및 식별자 endpoint 실패 계약 준비. 실제 source/auth/result persistence 연결은 BLOCKED |

## P0 — 원래 판정 보존

- `scripts/nutrition/allergen_service.py:127`의 기존 판정은 `_evaluate_safety`로 유지하고, `:195`의 공개 함수에서 설명만 추가한다.
- `safety_reason_codes`, `conflicting_allergens`, `safety_message`는 기존 분기와 evidence의 투영이다. 새로운 영양값·원료명·검증 완료 주장을 만들지 않는다.
- Product metadata가 Pet allergy를 덮어쓰는 구형 구현은 가져오지 않았다.
- 명확한 충돌은 `SAFETY_BLOCKED`, 근거 부족은 `SAFETY_DATA_INSUFFICIENT`; 두 경우 모두 `excluded=true`다.
- Nutrition-only 부족은 `analysis_status=INSUFFICIENT_DATA`여도 `excluded=false`일 수 있다. 추천 제외 기준은 `excluded`다.
- 기존 직접 요청 API의 생략/default 호환 정책은 이번 작업에서 일괄 변경하지 않았다. 신규 Service adapter는 별도 엄격한 입력 경계를 적용한다.
- `get_refs`는 이제 `mode=ro`로 연결한다. 누락된 SQLite DB를 생성하거나 조회 중 schema DDL을 실행하지 않는다. 누락/조회 실패는 `LINEAGE_INTEGRITY_ERROR`다.

### 관측 계약

| 경로 | 의미 |
|---|---|
| `GET /health` | 기존 liveness 응답 유지. 서비스 통합 성공을 의미하지 않음 |
| `GET /ready` | 로컬 JSON 10종, coverage CSV, SQLite catalog 점검. 필수 의존성 DOWN이면 503 |
| `GET /metrics` | HTTP 상태와 `analysis_status`/`safety_status`를 별도 counter로 노출 |

`/ready`의 `runtime_mode=local_artifact`, `service_source=NOT_CONFIGURED`를 함께 읽어야 한다.
readiness는 파일 구조/가용성 점검이지 모든 evidence의 의미적 정확성이나 운영 연결 검증이 아니다.
Gold operational evidence는 선택적 계층이며 현재 0건도 정상적인 fail-close 상태다.

유효한 불투명 `X-Request-ID`는 전달하고, 잘못되거나 중복된 ID는 새 UUID로 대체한다.
ID는 response header 및 request state에만 둔다. metric label에는 사용자/상품 ID, 원문, query, credential, 예외 내용을 넣지 않는다.
등록되지 않은 path는 `__unmatched__`로 모으고 `/metrics` 자신의 scrape는 집계에서 제외한다.
counter는 프로세스별이며 재시작 시 초기화된다. 다중 worker 집계/접근 제어는 실제 배포 단계 과제다.

## P1 — Canonical Input

`scripts/nutrition/service_db_adapter.py`는 DB 연결을 만들지 않는다.
입력은 source를 조회한 호출자가 아래 관계를 **명시적으로 집계한 내부 DTO**다.
이 DTO를 실제 DB column 또는 현재 BE JSON과 동일하다고 주장하지 않는다.

| 입력 | 처리 |
|---|---|
| Pet `id`, `species`, `age`, `weight` | 양의 identifier, DOG/CAT, 년→`age_years`, kg→`weight_kg`; 결측/비정상 측정값은 고정 오류 코드로 거절 |
| Pet `allergies` | 서비스 코드 문자열 배열. BE `AllergyOption` 객체라면 호출자가 code를 명시적으로 추출해야 함 |
| `allergy_profile_status` | 명시 KNOWN_NONE+빈 목록 / KNOWN_LIST+목록만 수용. 누락·모순은 UNKNOWN |
| Pet `life_stage` | 명시 값만 전달, 누락은 UNKNOWN. 나이/제품명으로 보완하지 않음 |
| Product `id`, `sku`, `category_code` | 서비스 products.id 보존. FOOD/TREAT/SUPPLEMENT만 변환, category 누락 거절 |
| Product `target_species` | 관계 테이블에서 집계한 DOG/CAT 배열. 누락/미지원은 None |
| Product `target_age_group` | `service_target_age_group` 메타데이터 및 provenance에 보존. AAFCO label로 승격하지 않음. Service 경로의 `aafco_life_stage=None`, evidence status는 UNKNOWN |
| Product `ingredient_codes`, `allergen_flags` | 각각 관계 테이블에서 가져온 배열. 별도 service namespace metadata 보존 |
| 구조화된 보증성분 없음 | `nutrition_items=[]`; 단위·basis·수분·영양값 생성 금지 |

age/weight 단위는 이번 사용자 확정 계약을 적용했다. 로컬 SQL numeric 타입만으로 단위를 추론한 것이 아니다.
`pet_allergy` 0행의 의미는 여전히 명시 상태 없이는 구분 불가이므로 UNKNOWN이다.

## P2 — Identity와 근거

`bridge_sku`는 공백·하이픈·Unicode 숫자·체크디지트 오류를 거절한다.
기존 `gtin_validation.is_valid_gtin` 및 `product_input_adapter`의 GTIN canonical 규칙을 재사용한다.
허용 길이는 8/12/13/14다. 기존 규칙의 leading-zero EAN-13/GTIN-12 동등성 외 새로운 padding/fuzzy 규칙은 추가하지 않았다.

- 후보 1개: MATCHED. 이 경우에만 기존 `load_product_input`/Gold gate 경로를 이용한다.
- 후보 0개: UNMATCHED. 형식 오류: INVALID_SKU. 복수 후보: AMBIGUOUS.
- MATCHED여도 명시된 종/분류가 충돌하면 AMBIGUOUS로 보류한다. 서비스 타깃 연령과 로컬 AAFCO 단계는 서로 다른 개념이므로 충돌 비교에 사용하지 않는다.
- exact GTIN 연결만으로 AAFCO label evidence가 검증되지는 않는다. 현재 Service adapter는 source의 임의 `aafco_life_stage`나 로컬 stage를 채택하지 않는다. 별도 authoritative label evidence의 승인된 전달 계약은 후속 과제다.
- MATCHED는 exact identifier 연결 결과일 뿐 운영 제조사 identity 검증 승인이 아니다. `operating_identity_verified=false` 유지.
- 원래 source의 영양값·unit·basis·provenance를 보존한다. 결측 unit/basis는 downstream validation을 우회하지 않는다.
- 서비스 상품 ID로 로컬 catalog를 다시 찾지 않는다. 로컬 evidence ID와 서비스 응답 product_id를 분리한다.
- `SALMON→fish`, `SWEET_POTATO→potato` 등 namespace alias 추론을 하지 않는다. 미지원 서비스 코드는 unresolved profile로 남긴다.
- Service Pet/Product의 **같은 코드** 교집합은 명시적 충돌로 추가 차단한다. 교집합이 없다는 사실로 안전하다고 판정하지 않는다.
- 기존 dictionary의 `ingredient_codes`와 정확히 일치하는 경우만 STRUCTURED_SOURCE다. 미등록/충돌 코드는 UNRESOLVED다. dictionary 원본은 수정하지 않았다.

### 실제 로컬 dump 계수

DB 접속 없이 `pg_restore --data-only --table=products --file=-`로 2026-09-24 dump를 읽었다.

| 항목 | 실측 |
|---|---:|
| 로컬 Nutrition 유효 GTIN cluster | 323 |
| 복수 source record가 연결된 cluster | 62 |
| dump의 서비스 products | 0 |
| SKU 존재 / valid GTIN / MATCHED / UNMATCHED / AMBIGUOUS / INVALID_SKU | 모두 0 |

0행이므로 운영 identity 연결률은 **산출 불가**다. 0% 성능이나 연결 성공으로 표현하지 않는다.

```text
Nutrition internal GTIN clusters: 323
Ambiguous internal GTIN clusters: 62
Service products sampled: 0
Service SKU → Nutrition GTIN coverage: NOT MEASURABLE
```

323/62는 Nutrition 내부 evidence identity 집계이며 서비스 Product 매칭 결과가 아니다.
실제 데이터 수령 시 [P2 운영 검증 체크리스트](p2_production_validation_checklist.md)를 사용한다.
외부 담당자에게 전달할 최소 요구사항은 [서비스 입력 계약](service_source_input_contract.md)에 정리했다.
dump SHA256: `c1d6c6a140866d762f8615add9352c2395a479a1c08686d7d33762a449c65f16`.

재현 명령(`nutrition/`에서, 로컬 dump와 pg_restore 경로를 직접 지정):

```bash
python -B scripts/nutrition/audit_service_identity.py \
  --dump ../../petflow-db-dump/data/product_db.dump \
  --pg-restore /opt/homebrew/opt/libpq/bin/pg_restore
```

이 명령은 파일/DB를 쓰지 않고 집계 JSON만 stdout으로 출력한다. 원본 상품 행은 출력하지 않는다.

## P3 — 준비된 연결 경계와 차단 사항

`analyze_service_records`는 ownership 확인이 끝난 source만 받는 내부 함수다.
순수 Adapter → Pydantic validation → Service exact conflict 및 기존 Safety → 기존 Rule Engine → 기존 response 축으로 연결된다.
함수 자체는 인증이나 ownership을 수행하지 않으므로 이를 서비스에 노출해서는 안 된다.

`POST /api/nutrition/analyze/by-service-id`의 본문은 양의 정수 `pet_id`, `product_id`만 허용한다.
`member_id`와 다른 추가 필드는 422다. 현재는 헤더 유무와 무관하게 503 `SERVICE_SOURCE_NOT_CONFIGURED`다.
임의 `X-Member-Id`를 보내도 성공하지 않는다. 실제 source 및 Gateway 신뢰 경계가 확인되기 전 활성화하지 않는다.

실제 Pet/Product 조회 방식, Gateway→AI 내부 인증, ownership 호출, 결과 저장소 계약이 남아 있다.
SQL, 새 table/column, credential, 배포 route는 만들지 않았다. `/compare`는 기존 501을 유지한다.
`/safety`는 기존 consumer-card 검증 응답, `/report`는 기존 요약 응답을 유지한다. 이 두 route를 전체 Safety response와 동일하다고 해석하지 않는다.

## 외부 담당자별 요청 — 다른 파트 수정 없음

다음 BE/FE 근거는 읽어 본 source snapshot이다. 운영 배포 여부의 증거는 아니다.
BE snapshot: `4982a238be86566cd59926183fb1dd6ac8f5c2f1`, FE snapshot: `ba6e22dac6a8024e39e11d8fc38975511ddda9d3`.

### Recommendation

- 현재 코드: `recommendation/endtoend/src/pipeline.py:365`의 `recommend_for_pet`는 `:375`에서 EXCLUDE를 results에 추가하고 `:439`에서 추천과 함께 반환한다.
- 필요한 변경: 최종 사용자 추천 후보에서 `response.excluded == true`를 제거한다. 제외 설명 목록을 별도로 보여줄지 여부는 담당자 결정이다.
- Nutrition 제공 계약: SAFETY_BLOCKED/SAFETY_DATA_INSUFFICIENT는 제외, Nutrition-only INSUFFICIENT_DATA는 자동 제외 근거 아님.
- `recommendation/endtoend/src/recommend/allergy_filter.py:18`의 빈 집합 교차 검사만으로 UNKNOWN을 판정할 수 없다. 공통 Safety 경계 적용이 필요하다.

### BE

- `platform/api-gateway/src/main/java/com/golajugaenyang/gateway/filter/JwtClaimForwardingFilter.java:61`에서 JWT memberId를 `X-Member-Id`로 전달한다.
- `services/member-service/src/main/java/com/golajugaenyang/member/adapter/in/web/InternalMemberController.java:43`의 Pet 조회는 `PetService.getPetDetail`로 연결된다.
- `services/member-service/src/main/java/com/golajugaenyang/member/application/PetService.java:224`의 상세 조회는 `:228`에서 memberId ownership을 검사한다. 재구현 대신 이 경계 재사용 가능 여부를 확정해야 한다.
- 같은 파일 `:56`에서는 allergies 미입력을 빈 목록으로 바꾼다. 명시 없음과 미등록을 구별하는 상태 전달이 필요하다.
- 요청: AI가 사용할 실제 Pet/Product source 방식과 접속 계약, service 코드 배열 변환, explicit profile status, 운영 보증성분 저장/조회 및 결과 저장소 계약.
- 나이/무게 단위나 products.id 의미를 다시 묻지 않는다. 이미 확정된 계약을 적용했다.

### FE

- `src/views/product-detail/ui/product-detail-view.tsx:8`은 적합도·영양 분석이 mock임을 명시한다. `:161`의 PET_MATCHES, `:340`의 MatchPanel 연결이 실제 API 결과 연결 대상이다.
- 요청: source/auth 연결 완료 후 `safety_status`, `warnings`, `excluded`, `exclude_reasons`를 사용한다. 신규 3개 설명 필드와 `analysis_status`도 사용 가능하다.
- `analysis_status=INSUFFICIENT_DATA`를 곧바로 allergy 차단으로 표시하지 않는다. 서비스 endpoint의 현재 503은 데이터 결과가 아니라 미연결 오류다.

### Infra/GitOps

- 요청: AI→source network path, 내부 인증/secret 전달, Gateway→Nutrition route/ingress, 결과 저장소 접근 정책.
- `/health`와 로컬 `/ready`의 목적을 구분한다. 현재 `/ready` 200으로 AWS source 가용성을 주장하지 않는다.
- `/metrics` 접근 범위와 다중 worker 집계는 운영 배포 시 확정한다. Nutrition Dockerfile/Jenkins/공용 CI/GitOps 파일은 이번에 수정하지 않았다.

## 이전 작업의 검증 근거 (2026-09-28 기록)

- 수정 전 전체 Nutrition pytest: 120 passed, 1 warning.
- 신규 회귀 83개 추가 후 전체 Nutrition pytest: **203 passed, 1 warning**. 기존 테스트 삭제·완화 없음.
- Python 33개 파일을 메모리에서 compile하여 통과했고 `git diff --check`도 통과했다. pyc 파일 생성 없음.
- 변경 전 HEAD의 Safety 및 category 함수를 메모리에서 별도로 로드했다. 3 profile × 2 category × 3 target species × 4 product stage × 3 ingredient 입력의 **216개 domain 결과** 및 같은 **216개 HTTP 요청**을 비교했다.
- 신규 3개 필드만 제거하고 dictionary/list 중첩까지 비교했다. JSON key 순서만 무시하고 list 순서는 보존했다. 기존 결과는 모두 동일했다.
- 이 비교는 모든 가능한 입력의 수학적 증명이 아니다. 원본 100상품 seed/v2 manifest를 이 worktree로 복사하거나 재승인한 것도 아니다.
- 신규 회귀는 `tests/test_observability_safety_contract.py`, `tests/test_service_integration_boundary.py`에 있다. 테스트 더블은 경계 검증용이며 실제 운영 evidence나 AWS E2E 근거가 아니다.
- 전체 테스트 명령: `PYTHONDONTWRITEBYTECODE=1 python -B -m pytest tests -q -p no:cacheprovider`.
- 기존 warning은 Starlette/AnyIO의 BlockingPortal deprecated alias다.
- 작업 전후 SHA256 대조: 구형 `ai/` 자료 726개, curated Nutrition 70개, 대상 Nutrition 데이터 20개, 대상 Recommendation 43개 및 Repurchase 175개 모두 byte 변경 0건. cache/venv/Git 내부 파일은 비교 대상에서 제외했다.
- 수정 7개·신규 6개는 모두 `nutrition/**` 안이다. Rule Engine/NIAS/reference/dictionary/evidence 원본은 변경하지 않았다.

## 배포 전 재검토 (2026-09-29)

- 기존 13개 변경 파일 전체와 HEAD diff를 검토했다. `target_age_group → aafco_life_stage` 자동 변환을 발견하여 제거했다. Rule/Reference/Safety 판정 함수 및 원본 evidence는 변경하지 않았다.
- 변경 전 전체 테스트 203개 통과를 재현했다. 신규 회귀 5건이 수정 전 실패하는 것을 확인한 후 자동 승격을 제거했다.
- 새로운 회귀는 ADULT/GROWTH/SENIOR/결측 타깃 연령, exact identity만 있는 로컬 stage의 비승격을 고정한다.
- 기존 원료 Safety 및 Nutrition-only 부족 테스트는 명시적 canonical 테스트 label로 해당 분기를 계속 검증한다. 이 label은 운영 evidence가 아니다. 실제 Service adapter의 label 결측 차단도 별도로 검증한다.
- 서비스 식별자 endpoint는 인증처럼 보이는 헤더의 유무와 관계없이 503을 반환한다. 로컬 상품 조회나 adapter를 호출하면 테스트가 실패하도록 고정했다.
- 실측 최종 테스트·compile·범위 검증 결과는 아래 체크리스트 문서의 실행 기록을 따른다.

## 최종 준비 수준

- Nutrition 단독 QA: YES — 로컬 API/Adapter/회귀 범위.
- Nutrition service integration code readiness: PARTIAL — 입력/판정 경계 및 실패 route 준비, 실제 source/auth/result store는 미연결.
- 실제 전체 서비스 E2E QA: NO — AWS 연결·쓰기·FE 통합 실행 없음.

## 수정 파일 전체 목록

아래 경로는 `ai-nutrition-main` 기준이다. 기존 파일 7개 수정, 신규 파일 8개 추가다.

```text
nutrition/README.md
nutrition/docs/canonical_input_contract.md
nutrition/docs/service_db_contract.md
nutrition/docs/service_integration_architecture.md
nutrition/docs/service_integration_preparation_p0_p3.md
nutrition/docs/service_source_input_contract.md
nutrition/docs/p2_production_validation_checklist.md
nutrition/scripts/api_nutrition.py
nutrition/scripts/nutrition/allergen_repository.py
nutrition/scripts/nutrition/allergen_service.py
nutrition/scripts/nutrition/audit_service_identity.py
nutrition/scripts/nutrition/observability.py
nutrition/scripts/nutrition/service_db_adapter.py
nutrition/tests/test_observability_safety_contract.py
nutrition/tests/test_service_integration_boundary.py
```
