# 골라주개냥 영양성분 분석 AI

이 디렉터리는 전체 AI 저장소에서 영양성분 분석에 필요한 코드, 최소 데이터, 평가 근거, API 계약 문서만 분리한 제출용 묶음이다.

## 포함 범위

- NIAS reference 비교 및 Rule Engine: `scripts/pipeline_p1c_v1.py`
- FastAPI 영양 분석 API: `scripts/api_nutrition.py`
- 알레르기 P0/P1/P2 persistence·lineage·precision evaluator, GTIN/Gold evidence, readiness, product input adapter, reference parity, Danawa adapter: `scripts/nutrition/`
- API request/response 예시: `API/`
- 선택된 raw/processed reference, P1.1/P2 allergen catalog, Gold evidence 및 evaluability 평가 근거: `data/`
- P0 계약, P1 Gold evidence, P2 persistence·lineage·precision, persisted-product adapter, reference parity, Danawa adapter 테스트: `tests/`
- 상태 계약, reference edition, API 초안, persisted-product E2E, P0/P1/P2 safety·lineage 문서: `docs/`

## 실행 환경

- Python 3.11 사용을 전제로 한다.
- 패키지 루트에서 의존성 설치: `python -m pip install -r requirements.lock`
- 패키지 루트에서 테스트 실행: `python -m pytest tests -q`
- FastAPI 실행: `python -m uvicorn scripts.api_nutrition:app --reload --port 8002`

## API 계약 구분

### 현재 Runtime API

`scripts/api_nutrition.py`에 실제 구현된 경로는 다음과 같다.

- `GET /health`
- `POST /api/nutrition/analyze`
- `POST /api/nutrition/analyze/by-product-id`
- `POST /api/nutrition/safety`
- `POST /api/nutrition/report`
- `POST /api/nutrition/compare` (현재 HTTP 501)

### Target BE integration contract draft

`API/nutrition_request.json`, `API/nutrition_response.json`, `API/error_response.json`은 목표 BE 연동 계약 초안이다. 이 문서의 `/internal/v1/nutrition/*` 경로는 현재 runtime에 구현되어 있지 않으며, 실제 FastAPI route 또는 응답 스키마로 해석하면 안 된다.

## 현재 구현 범위

완료된 범위는 deterministic runtime, persisted-product adapter, P0-D, 알레르기·종·생애주기 safety, P1.1/P2 allergen persistence·lineage, human precision evaluation, 5축 상태 계약이다.

아직 구현 중이거나 미구현인 범위는 최종 BE internal API, Shared SafetyDecision, Feeding, Human Gold 확장, 10K functional matrix, 100K regression이다.

이 패키지의 현재 회귀 테스트는 Python 3.11에서 137 passed, 1 warning이다. 이는 제출 패키지에 포함된 테스트 결과이며, 원본 저장소의 historical SQL parity test는 현 runtime과 다른 legacy policy를 검증하므로 이 패키지에 포함하지 않았다.

## 해석 주의

- `READY`는 현재 런타임의 최소 입력과 safety gate가 충족된 상태이며, 영양학적 완전성이나 임상적 적합성을 의미하지 않는다.
- HTTP 200은 분석 성공이나 `READY`를 의미하지 않는다.
- `data/raw/`와 `data/processed/`에는 이 패키지 실행·감사에 필요한 선택 파일만 포함한다. 전체 로컬 원천 데이터와 archive는 포함하지 않는다.
- 현재 API runtime에 적용되지 않은 BE 변경안은 이 패키지에 포함하지 않았다.
- `data/processed/baseline_manifest_v3.json`은 2026-09-04 당시 생성된 historical reproduction manifest이며, 현재 GitHub branch/commit metadata를 의미하지 않는다.

자세한 현재 상태와 한계는 `docs/nutrition_persisted_product_e2e_audit_v1.md`를 기준으로 확인한다.
