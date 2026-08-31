# 실제 데이터 분석

공개 이커머스 구매 데이터의 구조와 재구매 패턴을 분석해 백엔드 목데이터 생성 기준을 정의합니다. 공개 데이터는 최종 학습 원본이 아니라 분포·조건부 관계·예외 패턴을 관찰하는 참고 원천으로 사용합니다.

## 분석 대상

1. UCI Online Retail II
2. Complete Journey

## 확인 항목

- 사용자·주문·상품·구매 시각 식별 가능 여부
- 취소·반품·비회원 거래의 비율과 처리 기준
- 사용자별 주문 수와 반복 구매 사용자 비율
- 동일 상품 및 상품군 재구매 간격 분포
- 주문당 상품 수량·금액 분포와 극단값
- 관측 기간과 마지막 구매의 중도절단 비율
- 시간 순서 기반 학습·검증·테스트 분할 가능 여부
- 콜드스타트·이력 부족·비정상 구매 시나리오 비율
- 사용자 활동 세그먼트별 비율과 세그먼트 내부 주문 수 분포
- 주문당 상품 행 수와 사용자 전체 주문 간격 분포
- 사건 가중 구매 간격과 사용자·상품 쌍 가중 대표 주기의 차이
- 이전 구매 수량과 다음 구매 간격의 관계
- 같은 카테고리 내 상품 전환·이른 전환·다음 주문 유지·복귀 패턴

## 산출물

- 데이터 품질 및 분포 분석 코드
- 데이터셋별 분석 결과
- 두 데이터셋의 공통·차이점 비교
- 백엔드 전달용 목데이터 생성 가이드
- 목데이터 생성·검증용 데이터셋별 JSON 프로파일

## 개발 환경

- 재구매 예측 파트의 공식 Python 버전은 3.11.15입니다.
- 이 디렉터리의 `.python-version`은 재구매 파트에만 적용합니다.
- 추천·영양성분 분석 등 다른 AI 파트의 Python 환경은 각 파트에서 별도로 관리합니다.

## 실행

```bash
cd repurchase/data_analysis
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m scripts.download_datasets
.venv/bin/python -m scripts.profile_repurchase
.venv/bin/python -m scripts.profile_mock_generation --dataset all
.venv/bin/python -m scripts.visualize_profiles
.venv/bin/python -m scripts.validate_uci_preprocessing
.venv/bin/python -m scripts.validate_uci_events
.venv/bin/python -m scripts.validate_uci_preprocessing_e2e
.venv/bin/python -m scripts.run_uci_baseline_e2e
.venv/bin/python -m scripts.visualize_uci_events
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

## 자동 검증과 전체 데이터 검증의 구분

`repurchase-quality`는 `develop`과 `main` Ruleset의 필수 상태 검사이므로 모든 대상 Pull Request와 push에서 상태를 보고합니다. workflow 수준에서 경로 필터로 실행을 건너뛰면 필수 검사가 Pending으로 남아 병합을 막기 때문에, 먼저 Git diff로 변경 경로를 확인한 뒤 실행할 단계를 결정합니다.

- `repurchase/**` 또는 `.github/workflows/repurchase-ci.yml` 변경: Python 3.11 환경을 만들고 아래 검사를 실행
- 그 외 변경: Python과 의존성을 설치하지 않고 생략 사유만 출력한 뒤 성공 상태 보고
- 수동 실행: 변경 경로와 관계없이 전체 재구매 검사 실행

재구매 관련 변경에서는 다음 빠른 검사를 자동 실행합니다.

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

`pytest`에는 작은 고정 표본으로 전처리 전체 연결을 검사하는 E2E 테스트가 포함됩니다. 이 검사는 외부 네트워크와 로컬 원본 파일에 의존하지 않으므로 모든 PR에서 재현할 수 있습니다.

실제 UCI 전체 ZIP을 사용하는 아래 검증은 대용량 외부 데이터에 의존하므로 PR CI에서는 실행하지 않습니다. 데이터 다운로드·품질 분류·사건 집계·라벨 로직을 변경했을 때 수동으로 실행하고 결과 보고서를 함께 검토합니다.

```bash
.venv/bin/python -m scripts.validate_uci_preprocessing_e2e
```

## 시각화 결과

- `reports/figures/dataset_quality_comparison.png`: 유효 구매와 품질 이슈 비교
- `reports/figures/repurchase_behavior_comparison.png`: 반복 구매와 중도절단 비교
- `reports/figures/repurchase_interval_quantiles.png`: 상품·카테고리 재구매 간격 비교
- `reports/figures/pet_category_repurchase_profile.png`: 반려동물 카테고리별 특성 비교
- `reports/figures/monthly_order_trends.png`: 월별 주문 및 구매 사용자 추이
- `reports/figures/uci_purchase_event_duplicate_sensitivity.png`: 구매 사건 중복 후보 영향률과 수량 차이 분위수

## 목데이터 생성 기준 결과

- `reports/uci_online_retail_ii_mock_generation_profile.json`: UCI 사용자·주문·상품 주기 분포
- `reports/complete_journey_mock_generation_profile.json`: Complete Journey 전체 행동 분포
- `reports/complete_journey_pet_mock_generation_profile.json`: 반려동물 카테고리 및 상품 전환 조건부 분포
- `reports/synthetic_data_generation_guidelines.md`: 관찰값·초기 생성값·도메인 가정값을 구분한 백엔드 전달 가이드
- `reports/uci_preprocessing_validation.json`: 전처리 행 보존·사유별 건수·불변조건 검증 결과
- `reports/uci_preprocessing_validation.md`: 사람이 검토하기 위한 UCI 전처리 검증 요약
- `reports/uci_purchase_event_validation.json`: 구매 사건 집계·중복 민감도·시각 변동 검증 결과
- `reports/uci_purchase_event_validation.md`: 사람이 검토하기 위한 구매 사건 검증 요약
- `reports/uci_preprocessing_e2e_validation.json`: 원본 로드부터 재구매·우측검열 라벨까지 단계별 검증 결과
- `reports/uci_preprocessing_e2e_validation.md`: E2E 건수·라벨 분포·불변조건 검토 요약
- `reports/uci_baseline_e2e_evaluation.json`: 시간 분할·베이스라인 학습·평가·현재 예측 전체 결과
- `reports/uci_baseline_e2e_evaluation.md`: 전역·계층형 중앙값 성능과 fallback 사용 비율 검토 요약

## 파일 관리

- 원본 배포 파일은 `data/raw/`에 보관하며 Git에 커밋하지 않습니다.
- UCI 워크북은 분석할 때만 임시로 추출하고 종료 후 삭제해 중복 저장하지 않습니다.
- 전처리 결과를 저장할 경우 `data/processed/`에 두고 원본에서 재생성합니다.
- 출처·버전·라이선스·SHA-256 검증 결과는 `reports/data_source_manifest.json`에 기록합니다.
- 표·그래프·요약 결과는 `reports/`에 저장합니다.
