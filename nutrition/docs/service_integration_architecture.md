# Nutrition 서비스 통합 아키텍처

> **2026-09-30 v1.0 최신 상태**
>
> PR #141에서 Service DB SELECT Repository와 Gateway trust boundary를 구현했다. `member_db` / `product_db`는 SELECT only이며 Pet ownership은 SQL에서 `pet.id + member_id + deleted_at IS NULL`로 강제한다.
> `POST /api/nutrition/analyze/by-service-id`는 더 이상 항상-503 stub이 아니다. source/auth 미구성 시 fail-close하고, 구성이 있으면 실제 Service DB source를 조회한다.
> `/ready`는 local mode에서는 artifact readiness, service mode에서는 artifact + member_db + product_db + service auth/source readiness를 본다.
> AI Result Table / upsert는 v1.0 필수 범위가 아니다. 실제 dev Secret/NetworkPolicy/Gateway route/Service DB row/FE E2E는 아직 미검증이다.
> 아래의 상충하는 “Repository 미구현”, “항상 503”, “Result Store 필수” 설명은 historical record로 읽는다.


2026-09-28 구현 상태: Service source의 순수 Canonical Adapter와 미연결 시 503으로 실패하는 identifier route를 준비했다. 실제 Repository/auth/result persistence/AWS E2E는 아직 없다. 아래는 목표 아키텍처와 기존 결정 기록이며 최신 구현·차단 상태는 [P0~P3 보고서](service_integration_preparation_p0_p3.md)를 따른다.

## 2026-09-21 Architecture Decision

현재 Nutrition 서비스 통합의 공식 목표 구조는 다음과 같다.

```text
FE
↓
Nutrition AI FastAPI
↓
AWS Service DB
↓
Pet / Product / Nutrition / Ingredient source data 조회
↓
Canonical Input Adapter
↓
Deterministic Nutrition Rule Engine
↓
Nutrition / Safety Result
↓
AI-owned Nutrition Result Table 저장
↓
FE Response
```

이 결정은 이전 `FE → BE → AI → BE → FE` 구조를 대체한다. 이전 `/internal/v1/nutrition/*` 문서는 historical target contract로 보존하되, 현재 서비스 통합 경로로 해석하지 않는다.

## 책임과 데이터 접근

아래는 목표 책임 구분이다. 현재 실제 인프라 DB 작업은 승인되지 않았으며 로컬 dump 분석만 수행한다. 결과 저장은 미구현·미승인이고 DDL 및 데이터 쓰기를 실행하지 않는다.

| 경계 | 책임 | 허용 동작 |
|---|---|---|
| FE → Nutrition AI | 인증된 요청과 최소 식별자 전달 | `pet_id`, `product_id` 전달 |
| Nutrition AI → Service DB | 분석에 필요한 source data 조회 | SELECT only |
| Service DB Adapter | source data를 Canonical Input으로 변환 | 명시된 값만 전달, 누락값 추론 금지 |
| Nutrition Rule Engine | 기준 비교와 safety 판정 | deterministic rule 실행 |
| AI-owned Result Store | 재현 가능한 결과와 provenance 저장 목표 | 현재 금지, schema/ownership 확정 및 별도 승인 필요 |
| Nutrition AI → FE | domain status와 구조화된 결과 반환 | HTTP status와 domain status 분리 |

Service source data와 AI 결과 저장소의 실제 table/column/schema는 아직 확인되지 않았다. 이 문서는 logical boundary만 정의하며 AWS SQL, table명, column명을 정의하지 않는다.

## 결정론과 안전 정책

- 동일 Canonical Input + Rule Version + Reference Version은 동일 결과를 반환해야 한다.
- Nutrition/Safety 판정은 deterministic Rule Engine이 담당한다. LLM은 판정값을 변경하지 않는다.
- 원본 서비스 데이터는 AI가 수정하지 않는다.
- missing nutrient, species, life-stage, ingredient/allergy evidence는 기본값·추정값·PASS로 보정하지 않는다.
- 명확한 allergy/species/life-stage 충돌은 `SAFETY_BLOCKED` 계열로 처리한다.
- safety evidence가 부족하면 `UNKNOWN` 또는 `SAFETY_DATA_INSUFFICIENT`로 fail-close한다.

## 현재 구현과 후속 범위

현재 구현된 FastAPI는 request-scoped 분석과 로컬 persisted-product artifact demonstrator다. AWS Service DB 연결, 실제 Repository, AI-owned result persistence, FE identifier 기반 AWS E2E는 아직 구현하지 않았다.

후속 작업에서 실제 source 접근·인증 및 결과 저장 계약을 확정한 뒤 준비된 Service DB Adapter에 Repository를 연결한다. 이번 로컬 준비 작업에서는 branch/PR을 생성하지 않았다.

2026-09-29 입력 계약과 차단 조건은 [서비스 입력 계약](service_source_input_contract.md)을 따른다. Product 마케팅 타깃 연령은 AAFCO label evidence가 아니며 현재 Service 경로에서 자동 변환하지 않는다.
