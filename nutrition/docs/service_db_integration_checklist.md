# Nutrition AI Service DB 연동 사전 확인 목록

본 문서는 다음 `feat/nutrition-service-db-integration` 작업 전에 확정해야 할 운영 연동 정보를 정리한다. 현재 repository에는 AWS Service DB schema·접속 정보·Repository가 없으므로 table, column, SQL을 추정하지 않는다.

## 현재 검증 범위와 분리

- Local Runtime HTTP E2E: `tests/test_runtime_e2e.py`로 검증
- Persisted Local Product E2E: repository의 로컬 artifact를 사용하는 demonstrator로 검증
- AWS Service DB E2E: **NOT IMPLEMENTED**
- FE → AI → AWS DB → AI Result Store → FE: **NOT VERIFIED**

## 연동 전 확보할 정보

| 확인 항목 | 필요한 결정 또는 근거 |
|---|---|
| DB engine | 실제 사용 DB 종류와 지원 driver |
| connection method | runtime 환경에서의 접속 방식, secret 주입 경로, 네트워크 접근 범위 |
| pet source | pet 식별자 형식, 필요한 species/life-stage/allergy profile의 authoritative source |
| product source | product 식별자 형식, category/species/life-stage/product form의 authoritative source |
| nutrition evidence source | 보증성분 value/unit/basis/provenance의 source와 갱신 정책 |
| ingredient source | PRODUCT_LABEL ingredient와 allergen evidence의 source·version·lineage |
| result ownership | AI Result Store의 소유 주체, 읽기/쓰기 권한, 결과 조회 책임 |
| result policy | upsert 또는 history 보존, rule/reference/dictionary version 기록, 재실행 정책 |
| transaction | source 조회와 결과 저장의 transaction/실패 처리 요구사항 |
| authentication and authorization | authenticated principal과 `pet_id` ownership을 검사하는 책임 경계 |
| deployment boundary | DB credential은 FE에 노출하지 않으며, CORS를 authorization 대체 수단으로 사용하지 않음 |

## 구현 시작 조건

1. 실제 schema와 identifier contract를 담당자에게서 받는다.
2. Service DB Adapter가 Canonical Pet/Product Input에 전달할 필드를 명시한다.
3. 누락값은 추정하지 않고 현재 Runtime의 `UNKNOWN`/`INSUFFICIENT_DATA`/`SAFETY_DATA_INSUFFICIENT` fail-close 계약을 유지한다.
4. 인증 주체가 요청한 `pet_id`에 접근할 수 있는지 확인하는 책임 경계를 정한다.
5. 실제 Service DB integration test 환경과 테스트용 credential 관리 방식을 합의한다.
