# 골라주개냥 영양성분 분석 AI

결정론적 Evidence-grounded Nutrition/Safety Engine이다. 보증성분과 원재료 근거를 정규화해 NIAS reference 비교와 알레르기·종·생애주기 safety 판정을 수행한다. LLM은 Nutrition 또는 Safety 판정값을 변경하지 않는다.

## 현재 서비스 통합 구조

2026-09-21 결정 기준의 목표 경로는 다음과 같다.

```text
FE → Nutrition AI FastAPI → AWS Service DB (SELECT only)
   → Canonical Input Adapter → Nutrition Rule Engine
   → AI-owned Nutrition Result Table (INSERT / UPDATE) → FE
```

- 서비스 원본 데이터는 AI가 읽기만 한다.
- AI는 분석 결과 저장소만 기록한다.
- 동일 Input + Rule Version + Reference Version은 동일 결과를 반환해야 한다.
- 이전 `FE → BE → AI` 전제의 `/internal/v1/nutrition/*` 문서는 historical target contract이며 현재 서비스 통합 경로가 아니다.

상세 결정과 DB/입력 경계는 [service_integration_architecture.md](docs/service_integration_architecture.md), [service_db_contract.md](docs/service_db_contract.md), [canonical_input_contract.md](docs/canonical_input_contract.md)를 확인한다.

## 현재 구현 Runtime

현재 FastAPI 구현은 `scripts/api_nutrition.py`에 있으며, AWS Service DB 연결이나 AI result persistence는 아직 구현하지 않았다.

- `GET /health`
- `POST /api/nutrition/analyze` — request에 `pet`, `product`, `nutrition_items`를 직접 전달
- `POST /api/nutrition/analyze/by-product-id` — 로컬 선택 artifact에서 `product_id`를 조회하는 demonstrator
- `POST /api/nutrition/safety`
- `POST /api/nutrition/report`
- `POST /api/nutrition/compare` — 현재 HTTP 501

FE의 최종 최소 identifier 요청은 논리적으로 `{ "pet_id": "...", "product_id": "..." }`이지만, 실제 AWS schema·Repository·E2E가 없는 현재에는 runtime endpoint로 구현하지 않았다.

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
- AWS Service DB schema, Pet/Product Repository, AI result persistence, 실제 AWS E2E는 후속 `feat/nutrition-service-db-integration` 범위다.
- Human Gold expansion, Functional Matrix, Scale Regression, Feeding Engine, Recommendation SafetyDecision integration은 별도 범위다.
