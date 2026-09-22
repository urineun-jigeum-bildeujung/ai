# 골라주개냥 영양성분 분석 AI

이 디렉터리는 전체 AI 저장소에서 영양성분 분석에 필요한 코드, 최소 데이터, 평가 근거, API 계약 문서만 분리한 제출용 묶음이다.

## 포함 범위

- NIAS reference 비교 및 Rule Engine: `scripts/pipeline_p1c_v1.py`
- FastAPI 영양 분석 API: `scripts/api_nutrition.py`
- 알레르기, GTIN, readiness, product input adapter, reference parity, Danawa adapter: `scripts/nutrition/`
- API request/response 예시: `API/`
- 선택된 raw/processed reference 및 Gold evidence 평가 근거: `data/`
- P0 계약, persisted-product adapter, Gold evidence, reference parity, Danawa adapter 테스트: `tests/`
- 상태 계약, reference edition, API 초안, persisted-product E2E 감사 문서: `docs/`

## 실행 환경

- Python 3.11 사용을 전제로 한다.
- 의존성 설치: `python -m pip install -r requirements.lock`
- 테스트 실행: `python -m pytest tests -q`

이 브랜치를 만들 때 Python 3.11 문법 검사는 통과했다. 다만 해당 로컬 환경에는 `pytest`가 설치되어 있지 않아 전체 테스트를 새로 실행하지 않았다.

## 해석 주의

- `READY`는 현재 런타임의 최소 입력과 safety gate가 충족된 상태이며, 영양학적 완전성이나 임상적 적합성을 의미하지 않는다.
- HTTP 200은 분석 성공이나 `READY`를 의미하지 않는다.
- `data/raw/`와 `data/processed/`에는 이 패키지 실행·감사에 필요한 선택 파일만 포함한다. 전체 로컬 원천 데이터와 archive는 포함하지 않는다.
- 현재 API runtime에 적용되지 않은 BE 변경안은 이 패키지에 포함하지 않았다.

자세한 현재 상태와 한계는 `docs/nutrition_persisted_product_e2e_audit_v1.md`를 기준으로 확인한다.
