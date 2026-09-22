# Nutrition 서비스 통합 아키텍처

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

| 경계 | 책임 | 허용 동작 |
|---|---|---|
| FE → Nutrition AI | 인증된 요청과 최소 식별자 전달 | `pet_id`, `product_id` 전달 |
| Nutrition AI → Service DB | 분석에 필요한 source data 조회 | SELECT only |
| Service DB Adapter | source data를 Canonical Input으로 변환 | 명시된 값만 전달, 누락값 추론 금지 |
| Nutrition Rule Engine | 기준 비교와 safety 판정 | deterministic rule 실행 |
| AI-owned Result Store | 재현 가능한 결과와 provenance 저장 | INSERT / UPDATE |
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

후속 `feat/nutrition-service-db-integration` PR에서 AWS schema가 확정된 뒤 실제 Service DB Adapter와 Repository를 구현한다.
