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

## 실행

```bash
cd repurchase/data_analysis
python -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m scripts.download_datasets
.venv/bin/python -m scripts.profile_repurchase
.venv/bin/python -m scripts.profile_mock_generation --dataset all
.venv/bin/python -m scripts.visualize_profiles
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

## 시각화 결과

- `reports/figures/dataset_quality_comparison.png`: 유효 구매와 품질 이슈 비교
- `reports/figures/repurchase_behavior_comparison.png`: 반복 구매와 중도절단 비교
- `reports/figures/repurchase_interval_quantiles.png`: 상품·카테고리 재구매 간격 비교
- `reports/figures/pet_category_repurchase_profile.png`: 반려동물 카테고리별 특성 비교
- `reports/figures/monthly_order_trends.png`: 월별 주문 및 구매 사용자 추이

## 목데이터 생성 기준 결과

- `reports/uci_online_retail_ii_mock_generation_profile.json`: UCI 사용자·주문·상품 주기 분포
- `reports/complete_journey_mock_generation_profile.json`: Complete Journey 전체 행동 분포
- `reports/complete_journey_pet_mock_generation_profile.json`: 반려동물 카테고리 및 상품 전환 조건부 분포
- `reports/synthetic_data_generation_guidelines.md`: 관찰값·초기 생성값·도메인 가정값을 구분한 백엔드 전달 가이드

## 파일 관리

- 원본 배포 파일은 `data/raw/`에 보관하며 Git에 커밋하지 않습니다.
- UCI 워크북은 분석할 때만 임시로 추출하고 종료 후 삭제해 중복 저장하지 않습니다.
- 전처리 결과를 저장할 경우 `data/processed/`에 두고 원본에서 재생성합니다.
- 출처·버전·라이선스·SHA-256 검증 결과는 `reports/data_source_manifest.json`에 기록합니다.
- 표·그래프·요약 결과는 `reports/`에 저장합니다.
