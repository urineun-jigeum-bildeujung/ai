# 골라주개 AI

반려동물 커머스 **골라주개**의 재구매 예측, 개인화 상품 추천, 리뷰 AI 및 모델 서빙을 담당하는 저장소입니다.

## 담당 모듈

| 모듈 | 역할 |
|---|---|
| `repurchase` | 구매 데이터 파이프라인, 재구매 시점 예측, 예측 근거 생성 |
| `recommendation` | 대체·교차상품 추천, 랭킹, 안전 필터 |
| `review_ai` | 유사 리뷰 검색, 근거 기반 요약, 리뷰 데이터 처리 |
| `api` | FastAPI 기반 모델 서빙 및 요청·응답 스키마 |
| `common` | 공통 설정, 로깅, 데이터 모델과 유틸리티 |

## 개발 환경

- Python 3.11
- 패키지·도구 설정: `pyproject.toml`
- 테스트: Pytest
- 코드 품질: Ruff

## 시작하기

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp .env.example .env
```

## 로컬 검증

```bash
ruff check .
ruff format --check .
pytest
```

## 디렉터리 구조

```text
src/gollajugae_ai/
├── api/
├── common/
├── recommendation/
├── repurchase/
└── review_ai/
tests/
configs/
data/
artifacts/
notebooks/
scripts/
```

`data/`의 원본 데이터와 `artifacts/`의 모델 파일은 Git에 올리지 않습니다. 저장소에는 데이터 출처·버전·생성 방법과 재현 가능한 코드만 기록합니다.

## 협업 흐름

1. `develop`에서 작업 브랜치를 생성합니다.
2. 구현과 로컬 검증을 완료합니다.
3. 일반 작업 PR은 `develop`을 대상으로 생성합니다.
4. 배포 가능한 변경만 `develop → main` PR로 반영합니다.

자세한 규칙은 [CONTRIBUTING.md](CONTRIBUTING.md)를 확인하세요.
