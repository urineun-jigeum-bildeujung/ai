# 골라주개냥 영양성분 분석 AI

결정론적 Evidence-grounded Nutrition/Safety Engine이다. 보증성분과 원재료 근거를 정규화해 NIAS reference 비교와 알레르기·종·생애주기 safety 판정을 수행한다. LLM은 Nutrition 또는 Safety 판정값을 변경하지 않는다.

## 현재 서비스 통합 구조

2026-09-30 v1.0 기준의 서비스 통합 경로는 다음과 같다.

```text
FE → API Gateway → Nutrition AI FastAPI
   → Service DB (member_db / product_db, SELECT only)
   → Canonical Input Adapter → Nutrition Rule Engine / Safety Gate
   → synchronous HTTP response → FE
```

- 서비스 원본 데이터는 AI가 읽기만 하며 INSERT / UPDATE / DELETE / DDL을 수행하지 않는다.
- v1.0 완료 조건은 동기식 분석 응답이며 AI-owned Result Table / upsert는 필수 범위가 아니다.
- 동일 Canonical Input + Rule Version + Reference Version은 동일 결과를 반환해야 한다.
- 이전 `/internal/v1/nutrition/*` 문서는 historical target contract이며 현재 Runtime SoT가 아니다.
- PR #141은 Service DB SELECT Repository와 Gateway trust boundary 코드를 구현한다. GitOps secret/network, Gateway route, 실제 dev DB row 및 FE E2E는 별도 검증이 필요하다.

상세 결정과 DB/입력 경계는 [service_integration_architecture.md](docs/service_integration_architecture.md), [service_db_contract.md](docs/service_db_contract.md), [canonical_input_contract.md](docs/canonical_input_contract.md)를 확인한다.

## 현재 구현 Runtime

2026-09-30 Service Repository 구현은 [v1 구현 및 검증](docs/service_repository_v1.md)을 따른다. DB URL과 Gateway 인증 구성이 갖춰진 경우 `by-service-id`가 실제 조회→기존 Engine으로 연결되며, 미설정 환경은 기존 503을 유지한다. 실제 AWS/dev 배포 및 E2E 완료는 아직 아니다. 아래 이전 준비 단계의 미구현 설명은 이 변경 범위에 한해 갱신된다.

현재 FastAPI 구현은 `scripts/api_nutrition.py`에 있다. Service DB SELECT Repository와 인증 경계는 구현했으며 실제 dev 연결·배포·E2E는 미완료다. AI result persistence는 v1.0 필수 범위가 아니다.

- `GET /health`
- `GET /ready` — local mode에서는 artifact readiness, service mode에서는 artifact + member_db + product_db + service auth/source readiness를 점검. DB UP만으로 schema/row/E2E 성공을 의미하지 않음
- `GET /metrics` — HTTP와 도메인 상태를 별도로 집계하는 프로세스별 counter
- `POST /api/nutrition/analyze` — request에 `pet`, `product`, `nutrition_items`를 직접 전달
- `POST /api/nutrition/analyze/by-product-id` — 로컬 선택 artifact에서 `product_id`를 조회하는 demonstrator
- `POST /api/nutrition/safety`
- `POST /api/nutrition/report`
- `POST /api/nutrition/compare` — **Future / Not Implemented**, 현재 HTTP 501. 비교 기능은 현재 Runtime scope 밖이며 구현 완료 API가 아니다.

`POST /api/nutrition/analyze/by-service-id`는 양의 정수 `pet_id`, `product_id`만 받는다. DB URL 미설정은 503 `SERVICE_SOURCE_NOT_CONFIGURED`, 내부 secret 미설정은 503 `SERVICE_AUTH_NOT_CONFIGURED`다. 구성이 있으면 Gateway 인증·SQL ownership 검증 후 실제 source를 조회하며 로컬 seed로 대체하지 않는다.

2026-09-28 로컬 통합 준비 변경과 담당자별 요청은 [P0~P3 보고서](docs/service_integration_preparation_p0_p3.md)를 따른다. Safety 응답에 `safety_reason_codes`, `conflicting_allergens`, `safety_message`를 추가했으며 기존 판정과 상태축은 유지한다.

2026-09-30 PR #141 기준 로컬 검증: 전체 242 tests PASS, warning 1개, scripts/tests compile 35 PASS, `git diff --check` PASS. Service `target_age_group`은 AAFCO label evidence로 사용하지 않는다. 실제 Service DB row 및 Gateway E2E는 아직 검증하지 않았다. [외부 서비스 입력 계약](docs/service_source_input_contract.md)과 [P2 실제 데이터 검증 체크리스트](docs/p2_production_validation_checklist.md)를 참고한다. 서비스 상품 dump는 0행이므로 운영 SKU coverage는 NOT MEASURABLE이다.

## 코드와 데이터 범위

- NIAS reference 비교 및 Rule Engine: `scripts/pipeline_p1c_v1.py`
- FastAPI API: `scripts/api_nutrition.py`
- Canonical product input adapter와 supporting module: `scripts/nutrition/`
- request/response historical target contract: `API/`
- 선택된 실행·검증용 reference 및 artifact: `data/`
- runtime regression tests: `tests/`

`data/raw/`와 `data/processed/`에는 이 baseline 실행에 필요한 선택 파일만 포함한다. 전체 수집물, crawler 결과, archive, backup은 포함하지 않는다.

## 데이터 흐름과 안전 정책

현재 로컬 demonstrator의 흐름은 `persisted product artifact → product_input_adapter → Canonical Product Input → Rule Engine`이다. 운영 전환 시 DB row를 엔진에 직접 전달하지 않고 Service DB Adapter를 통해 같은 Canonical Pet/Product Input 형태로 변환한다.

- 분석 가능한 nutrient만 비교하고, 비교 불가능한 nutrient는 `UNKNOWN`으로 남긴다.
- 입력 evidence가 분석 자체에 부족하면 `INSUFFICIENT_DATA` 계열 상태를 반환한다.
- 알레르기 충돌, 종 불일치, 생애주기 불일치는 Nutrition Coverage와 별도 safety 영역이며 명확한 충돌은 `SAFETY_BLOCKED`로 처리한다.
- ingredient/allergy evidence가 부족하면 `UNKNOWN` 또는 `SAFETY_DATA_INSUFFICIENT`로 fail-close한다.

`UNKNOWN != PASS`, `UNKNOWN != FAIL`, `INSUFFICIENT_DATA != 영양 부적합`, `READY != nutritional completeness`, `HTTP 200 != nutritional suitability`이다.

### E2E 검증 범위

- **Local Runtime HTTP E2E — VERIFIED**: `tests/test_runtime_e2e.py`가 FastAPI `TestClient`로 health, request-scoped 분석, safety fail-close, persisted local product 조회, unknown ID 404, non-food, determinism, runtime artifact 존재를 검증한다.
- **Persisted Local Product E2E — VERIFIED**: 위 테스트의 `by-product-id` 경로는 repository에 포함된 로컬 artifact를 `product_input_adapter`로 읽어 Rule Engine까지 전달한다. 해당 fixture의 domain 결과가 `READY`라는 뜻은 아니다.
- **AWS Service DB E2E — NOT VERIFIED**: source 조회 구현은 connection mock으로 검증했다. 실제 Gateway/DB/FE E2E는 미검증이며 AI Result Store는 이번 v1.0 범위 밖이다.

## 실행과 검증

- Python 3.11을 사용한다.
- 의존성 설치: `python -m pip install -r requirements.lock`
- 문법 확인: `python -m py_compile scripts/*.py scripts/nutrition/*.py`
- 테스트: `python -m pytest tests -q`
- FastAPI: `python -m uvicorn scripts.api_nutrition:app --reload --port 8002`

`nutrition-ci` workflow는 `nutrition/**` 또는 workflow 파일 변경 시 Python 3.11 의존성 설치, compile check, pytest, diff whitespace check를 수행한다. 실제 AWS DB credential을 사용하지 않는다.

## 계약 상태와 한계

- `API/nutrition_request.json`, `API/nutrition_response.json`, `API/error_response.json`, `docs/nutrition_ai_api_spec_v3_draft.md`의 `/internal/v1/nutrition/*`는 **SUPERSEDED historical target contract**다.
- `READY`는 현재 runtime의 최소 입력과 safety gate 충족 상태이며, 영양학적 완전성이나 임상적 적합성을 뜻하지 않는다.
- `data/processed/baseline_manifest_v3.json`은 2026-09-04 historical reproduction manifest이며, 현재 Git branch/commit metadata가 아니다.
- Pet/Product SELECT Repository와 HTTP trust boundary는 PR #141에서 구현했다. 실제 dev schema/row·Secret/NetworkPolicy·Gateway route·FE E2E는 남아 있다.
- AI result persistence는 v1.0 필수 범위가 아니며 신규 Result Table을 이번 마감에 추가하지 않는다.
- Human Gold expansion, Feeding Engine, 대규모 데이터 확대는 Future Scope다. Recommendation Safety 처리 방식은 PR #129의 penalty 정책과 기존 hard-filter 기획 충돌을 해소한 뒤 확정한다.
