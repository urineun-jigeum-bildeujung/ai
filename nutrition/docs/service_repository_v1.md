# Nutrition Service Repository v1 구현 및 검증

## 기준선과 구현 범위

기준: `origin/develop`의 PR #138 merge `1f1d3d4fc31615fabd518abe685c07099b735917`.
기존 main 체크아웃의 미커밋 15개 파일을 보존하고 별도 깨끗한 체크아웃에서 작업했다.

이 문서는 이전 `service_source_input_contract.md`의 **source 미연결** 상태를 아래 구현 범위에 한해 갱신한다. 운영 배포 완료를 뜻하지 않는다.

- `service_repository.py`: psycopg2-binary 2.9.9, MEMBER_DATABASE_URL / PRODUCT_DATABASE_URL, SELECT 전용.
- 연결 제한 3초, statement 제한 3초, lock 제한 1초. read-only / REPEATABLE READ transaction, 성공·실패 모두 연결 종료. DB write/DDL/commit 없음.
- `get_pet`: id + member_id + deleted_at IS NULL을 SQL에서 동시 확인. 미존재·타인 소유·삭제는 같은 PET_NOT_FOUND.
- 알레르기 관계 0행 UNKNOWN, 비어 있지 않으면 KNOWN_LIST. life_stage는 None. age 년 / weight kg를 기존 adapter에 전달하며 추론하지 않는다.
- `get_product`: is_active 상품과 species/allergens/ingredients 관계를 동일 snapshot에서 조회한다. target_age_group은 마케팅 metadata다.
- 기존 exact GTIN bridge/Gold gate/Service positive conflict/Rule Engine을 재사용한다. Rule·Reference·dictionary·evidence 원본 변경 없음.

## SQL 근거와 schema 차이

확인한 BE dev snapshot: `eb48f6c`.

- `services/member-service/src/main/java/com/golajugaenyang/member/adapter/out/persistence/entity/PetJpaEntity.java`
- 같은 디렉터리 `PetAllergyJpaEntity.java`
- `services/product-service/src/main/java/com/golajugaenyang/product/domain/product/Product.java`
- 로컬 `petflow-db-dump/member_db_schema.sql`, `product_db_schema.sql`

최신 JPA에는 ingredient `sort_order`가 있으나 2026-09-24 dump에는 없다. Safety는 순서가 아니라 exact code 집합을 사용하므로 공통으로 확인된 ingredient_code만 조회하며 정렬한다. 원재료 함량/원래 표시 순서 계약으로 사용하지 않는다. 라이브 DB schema 대조는 별도로 필요하다.

## 인증 및 HTTP 오류

본문은 기존 양의 정수 pet_id/product_id만 받는다. member_id 본문 추가는 422다. Gateway가 JWT 검증 후 주입한 단일 X-Internal-Secret / X-Member-Id를 사용한다. secret은 constant-time bytes 비교, member는 양의 PostgreSQL bigint 범위를 검사한다. 중복 헤더도 거절한다.

| 조건 | HTTP / 고정 코드 |
|---|---|
| DB URL 미설정 | 503 SERVICE_SOURCE_NOT_CONFIGURED |
| expected secret 미설정 | 503 SERVICE_AUTH_NOT_CONFIGURED |
| secret/member 헤더 누락·오류 | 401 SERVICE_UNAUTHORIZED |
| Pet 미존재·ownership 불일치 | 404 PET_NOT_FOUND |
| 상품 미존재·비활성 | 404 PRODUCT_NOT_FOUND |
| DB 연결/쿼리/종료 오류 | 503 SERVICE_DB_UNAVAILABLE |
| source schema/측정값 오류 | 422 SERVICE_SOURCE_INVALID |

DB 오류 내용, DSN, secret, SQL 원문을 응답에 포함하지 않는다. 인증 실패 후 DB를 조회하지 않는다. local seed fallback은 없다. 내부 함수 `analyze_service_records`는 HTTP 인증을 대체하지 않는다.

## 관측과 배포 의존성

기본 local 모드는 기존 artifact readiness를 유지한다. `NUTRITION_RUNTIME_MODE=service` 또는 하나 이상의 DB URL이 있으면 service readiness를 사용한다. 일부 설정만으로 local ready가 되지 않는다. 두 DB `SELECT 1`, 내부 secret 설정, 기존 artifact가 모두 필요하며 하나라도 DOWN이면 503이다. health는 liveness 200을 유지한다. metrics label 정책은 변경하지 않았다.

DB UP은 SQL source schema/실제 rows/운영 E2E 성공을 보증하지 않는다. 순차 DB 연결 및 쿼리 제한과 artifact 점검을 고려해 배포 readiness timeout은 15초 이상으로 검토한다.

GitOps binding/NetworkPolicy, Gateway RewritePath, 새 이미지 배포는 별도 PR 의존성이다. 현재 로컬 과정의 성공을 AWS 또는 FE E2E로 표현하지 않는다. AI result store, compare, Cart/Checkout, 모델 재학습은 구현하지 않았다.

## 회귀 재현

```bash
cd nutrition
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest tests -q -p no:cacheprovider
git diff --check
```

Python 3.11 / requirements.lock 환경에서 실행한다. 새 Repository 테스트의 connection mock은 SQL 조건·정리·오류 경계용이며 운영 DB 증거가 아니다. 기존 208개를 먼저 재현했고 추가 테스트는 별도로 측정한다. compile은 파일 생성 없이 메모리 compile로 수행한다.

실행 결과: BEFORE 208 passed / ADDED 34 / AFTER **242 passed, 1 warning**. scripts/tests 35개 compile PASS, `git diff --check` PASS. 신규 테스트를 위해 지정 psycopg2-binary 2.9.9를 별도 임시 dependency 경로에 설치했으며 기존 가상환경은 변경하지 않았다. tracked Nutrition 데이터 20개를 HEAD bytes와 대조하여 변경 0개를 확인했다. 로컬 `/health`, `/ready`, `/metrics`는 200이며 `/ready`의 모드는 local_artifact다.

## 실제 환경 검증 차단 사항

- 현재 셸에 MEMBER_DATABASE_URL / PRODUCT_DATABASE_URL / INTERNAL_GATEWAY_SECRET 미설정. credential을 문서나 Git에 복사하지 않는다.
- 실제 서비스 배포·Gateway 연결과 승인된 사용자 JWT/owned pet/product 표본 검증 미실시.
- operational Gold evidence 0건을 변경하지 않았다. FOOD AAFCO/생애주기 근거 결측은 계속 fail-close. 실제 authoritative life-stage mismatch 검증은 BLOCKED_BY_NO_AUTHORITATIVE_PRODUCT.
- GitOps·Gateway 등 다른 저장소 변경은 이번 요청의 확장 범위와 이전 Nutrition-only 제한 간 충돌로 도구 검토에서 차단되어 사용자 재확인 대기다.
