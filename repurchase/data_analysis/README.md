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
.venv/bin/python -m scripts.run_uci_lightgbm_feature_comparison
.venv/bin/python -m scripts.run_uci_conditional_validation
.venv/bin/python -m scripts.run_uci_conditional_bootstrap
.venv/bin/python -m scripts.run_uci_lightgbm_bc_bootstrap
.venv/bin/python -m scripts.visualize_uci_lightgbm_calibration
.venv/bin/python -m scripts.visualize_uci_events
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

`run_uci_lightgbm_feature_comparison`은 동일 Train·Validation 표본에서 횟수만 사용하는
A, 구매 간격 중앙값을 추가한 B, 불규칙성까지 추가한 C를 비교합니다. 결측 행은
보존하며 Test 예측·평가는 실행하지 않습니다. 피처 목록·모델 설정·실행 환경과
Brier·Calibration 결과는 `reports/uci_lightgbm_feature_comparison.json` 및
동명의 Markdown에 저장합니다. 단계별 차이는 앞선 피처가 주어진 조건에서의
효과이며, 개별 피처의 독립적인 인과 효과를 뜻하지 않습니다.

`run_uci_lightgbm_bc_bootstrap`은 B와 C를 각각 한 번 학습·예측한 뒤 같은
Validation 사용자를 1,000회 복원추출합니다. `Brier(B) - Brier(C)`의 점추정,
95% 구간과 양수 비율을 저장하며 Test 표본은 사용하지 않습니다. 이 구간은 고정된
모델 예측과 IPCW 가중치 아래의 평가 표본 불확실성만 나타냅니다.

## 현재 시점 재구매 확률 (AFT 후보)

`build_current_features_from_valid_purchases()`는 취소·반품을 반영해 확정한
`user_id`, `order_id`, `product_id`, `paid_at` 구매 사건 전체에서 현재 피처를 만듭니다.
`predict_current_repurchase_probability()`는 저장된 XGBoost AFT 후보를 읽은 뒤
마지막 구매 후 경과 시간 `t`에 아직 같은 상품을 재구매하지 않았다는 조건에서
앞으로 `h`일 이내 재구매 확률 `1 - S(t+h)/S(t)`를 계산합니다. 모델 아티팩트 ID와
기준 시각을 결과에 남기며, 피처는 학습 코드와 같은 함수를 재사용합니다.

LightGBM 후보의 구매 후 고정 30일 확률을 현재 시점 조건부 확률로 재해석하지
않습니다. 이 함수는 LightGBM 아티팩트를 명시적으로 거절합니다. AFT 후보 역시
현재 시점 조건부 확률에 대한 별도의 시간별 Calibration·운영 검증을 마치기 전에는
실제 구매 알림이나 API 응답으로 사용하지 않습니다. 원본 주문 상태 정규화,
상품군·반려동물 단위 정의와 운영 모델 승인은 후속 작업입니다.

`run_uci_conditional_bootstrap`은 구매 후 0·7·14·30일 Validation 위험집단에서
고정 AFT 후보와 각 시점의 Train 전체확률 기준선 간 IPCW Brier 차이를 사용자
단위로 1,000회 복원추출합니다. 같은 사용자의 구매 행을 함께 뽑되 모델과 IPCW
가중치는 고정하므로, 구간은 평가 표본의 변동만 나타냅니다. 요약·반복별 원자료는
각각 `reports/uci_aft_conditional_bootstrap.json`과
`reports/uci_aft_conditional_bootstrap_trials.json.gz`에 저장합니다.

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

테스트가 실행되면 성공·실패 내역과 소요시간을 JUnit XML로 생성하고,
`repurchase-tests-<실행 ID>-<재실행 번호>` 아티팩트로 14일간 보관합니다.
GitHub의 **Actions → Repurchase CI → 해당 실행 → Artifacts**에서 다운로드할 수 있습니다.
테스트가 실패해도 결과 파일을 업로드하며 CI 실패 상태는 그대로 유지합니다.
이전 단계 실패나 비관련 변경으로 테스트를 실행하지 않았거나 실행이 취소된 경우에는
업로드하지 않습니다. 테스트가 실행됐는데 결과 파일이 없으면 업로드 단계도 실패로 표시합니다.

업로드 대상은 러너 임시 디렉터리의 `repurchase-test-results/junit.xml` 하나입니다.
원본 데이터·모델·보고서 디렉터리는 포함하지 않고 표준 출력 로그도 XML에 첨부하지 않습니다.
다만 실패 메시지에는 테스트 값이 포함될 수 있으므로 테스트 입력에는 실제 개인정보나
비밀정보를 사용하지 않습니다. 이 파일은 코드 검증 기록이며 실제 데이터의 모델 성능 보고서가 아닙니다.

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
- `reports/uci_baseline_e2e_product_concentration_trials.json`: Validation 최악 5%와 비교한 동일 크기 무작위 표본 1,000회의 상품 집중도 원자료

## 파일 관리

- 원본 배포 파일은 `data/raw/`에 보관하며 Git에 커밋하지 않습니다.
- UCI 워크북은 분석할 때만 임시로 추출하고 종료 후 삭제해 중복 저장하지 않습니다.
- 전처리 결과를 저장할 경우 `data/processed/`에 두고 원본에서 재생성합니다.
- 출처·버전·라이선스·SHA-256 검증 결과는 `reports/data_source_manifest.json`에 기록합니다.
- 표·그래프·요약 결과는 `reports/`에 저장합니다.

## 재구매 배치 Runtime

재구매 배치는 API 서버처럼 계속 실행되지 않습니다. 입력과 모델을 읽어 한 번의
작업을 수행한 뒤 종료되는 Job입니다. Docker 이미지는 Python 코드와 실행
라이브러리만 포함하며 모델 아티팩트·입력 데이터·출력 결과·Secret은 실행 시
외부에서 주입합니다.

`scripts.modeling.cloud_source_reader`에는 호출자가 전달한 PostgreSQL 연결로
`order_db`의 주문·상태·클레임과 `member_db`의 반려동물 생일을 읽는 함수를
추가했습니다. 각 DB는 독립된 `REPEATABLE READ READ ONLY` 트랜잭션으로 읽고
추출 시각을 반환합니다. 두 DB가 동일한 시점의 스냅샷이라는 보장은 없으며,
`audit_cloud_source_reader` 명령은 명시한 관측 컷의 유효 구매·반려동물 이력
건수를 검사하고 원천 행이나 접속 정보를 출력하지 않습니다. 실제 dev DB에서
읽기 전용 감사까지 실행했으며, 운영 배치 추론·결과 적재와는 아직 연결하지 않았습니다.
공용 DB 원천 점검은 유효 구매에 지정된 반려동물이 주문 회원 소유인지도 확인하며,
불일치가 있으면 반려동물별 예측 입력을 만들지 않고 오류로 종료합니다.

로컬 실행 시 먼저 `order_db`와 `member_db`에 접속 가능한 포트포워딩을 준비하고,
비밀번호를 제외한 연결 문자열을 각각 `REPURCHASE_ORDER_DATABASE_DSN`,
`REPURCHASE_MEMBER_DATABASE_DSN` 환경변수로 전달합니다. 비밀번호를 터미널
명령이나 저장소에 적지 않으려면 `--prompt-password`를 사용합니다. `--as-of`는
시간대가 포함된 확정된 관측 종료 시각이어야 하며, 추출 시각보다 늦으면 거절합니다.

```bash
.venv/bin/python -m scripts.audit_cloud_source_reader \
  --as-of '<확정된-ISO-관측-종료-시각>' --prompt-password
```

이 명령은 DB를 변경하지 않습니다. 성공 시 각 원천 테이블 수, 시점별 유효 구매
수, 반려동물 생일 불일치로 제외된 행 수와 두 DB의 추출 시각을 JSON으로 출력합니다.
상태 이력이 없거나 현재 주문 상태와 마지막 이력이 불일치하는 주문은 관련
주문상품·클레임·상태 이력과 함께 입력에서 격리하고, 사유별 건수를 별도 출력합니다.
동일한 격리 규칙을 CSV 기반 서비스 모델 비교 입력에도 적용하며, 원본 파일의
해시와 제외 건수를 비교 결과에 함께 남깁니다. 격리 사실만으로 원천 정합성이
해결된 것은 아니며 운영 알림·실제 배치 실행은 별도 연결이 필요합니다.
2026-10-03 dev DB 감사에서는 이력 누락 주문 26건(결제 15건, 상품 42건),
최종 상태 불일치 주문 4건(상품 8건)을 격리한 뒤 유효 주문상품 98,008건과
전체 구매 사건 98,004건을 확인했습니다. 해당 감사의 현재 시각 컷은 학습용
관측 종료 시각으로 확정한 값이 아닙니다.
두 DB의 추출 시각이 다르므로 갱신이 진행 중인 환경에서 완전한 교차 DB 정합성
증명으로 해석하지 않습니다.

`run_cloud_service_model_comparison`은 동일한 읽기 전용 원천을 메모리에서
서비스 Validation 비교기로 전달합니다. 원천 행을 CSV로 다시 저장하지 않고,
먼저 관측 컷·반려동물 소유 관계를 점검합니다. 아래 명령은 모델 평가 실험이며
운영 예측값 적재나 모델 아티팩트 발행을 하지 않습니다. 실행 전 BE가 확정한
시간대 포함 관측 종료 시각을 지정하고, 결과 파일은 개인정보가 없는 집계
지표·설정·두 DB의 추출 시각·모델 코드 SHA-256만 담지만 저장 위치의 접근 권한을 확인합니다.

```bash
.venv/bin/python -m scripts.run_cloud_service_model_comparison \
  --observation-end-at '<확정된-ISO-관측-종료-시각>' \
  --bootstrap-replicates 1000 --prompt-password \
  --output '<접근-제한된-결과-JSON-경로>'
```

CSV 비교기와 동일하게 `--train-fraction`·`--validation-fraction`으로 시간
구간을 옮기고, `--aft-round-candidates 5 20 50 100` 또는
`--aft-scale-candidates 0.5 1.0 2.0`으로 Train 내부 선택 실험을 실행할 수
있습니다. 두 후보군은 한 번에 지정하지 않으며, 내부 컷 비율은
`--inner-train-ratio`로 지정합니다. 잘못된 비율·후보는 DB 접속 전에
거절됩니다. 이 옵션은 평가용이며 운영 모델 선택이나 배치 적재를 의미하지
않습니다.

같은 Validation 행에 상품군 확률 기준선을 추가하려면
`--product-group-smoothing-candidates 1 2 4 8`을 지정합니다. 기존 계층형
확률 기준선의 상품 키에 서비스 `target_id`(상품군)를 명시적으로 연결하고,
전체 Train의 IPCW 사건 확률을 상품군 prior로 사용합니다. 각 후보의 수축 강도는
Train 내부 시간 분할의 IPCW Brier로만 선택하며, 선택 후 전체 Train으로 다시
학습해 AFT·LightGBM과 같은 바깥 Validation 행에서 Brier·C-index·ECE를
계산합니다. 후보 숫자는 사전 지정된 탐색 범위이지 검증된 운영 설정이 아닙니다.
기본 실행은 기존 두 모델 비교 그대로이며, 상품군 기준선은 옵션을 지정할 때만
결과에 추가됩니다. 이때 같은 Validation 행에서 사용자를 복원추출해
`Brier(상품군 기준선) - Brier(AFT)`를 1,000회 쌍 비교하고,
`product_group_aft_bootstrap`에 점추정·95% 구간·양수 반복 비율과 반복별
결과를 기록합니다. 양수는 AFT의 오차가 더 작다는 뜻입니다. 이 구간은 고정된
모델의 평가 표본 불확실성이지, 재학습·모델 선택의 변동성을 포함하지 않습니다.
최종 모델 선정은 별도 검증이 필요합니다.

현재 시점 조건부 확률의 Validation 검증을 추가하려면 같은 비교 명령에
`--conditional-landmarks 0 7 14 30`을 지정합니다. 각 시점까지 재구매하거나
관측이 끝난 행을 제외한 뒤, 이미 Train에서 학습한 AFT 모델의 향후 30일
조건부 확률을 IPCW Brier와 Calibration/ECE로 평가합니다. 시점별 위험집단
크기와 사건·검열 제외 건수는 `conditional_aft_landmarks`에 따로 기록합니다.
0일 Brier는 기존 AFT Validation Brier와 같아야 합니다. 시점별 모집단이
서로 달라 Brier·ECE의 단순 증감을 모델 개선 효과로 해석하지 않습니다.
이 옵션은 Validation에서 모델을 다시 선택하거나 Test를 열지 않으며,
날짜·범위·신뢰도 표시를 자동 승인하지 않습니다.

2026-10-03 09:39 UTC dev DB 읽기 전용 스냅샷에서 같은 관측 컷으로 실행한
시점별 Validation 결과는 아래와 같습니다. 각 행은 서로 다른 위험집단이므로
행 간 Brier를 개선량으로 비교하지 않습니다.

| 구매 후 경과 | 위험집단 | 정답 확인 | IPCW Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| 0일 | 19,189 | 13,625 | 0.121258 | 0.028450 |
| 7일 | 17,088 | 11,848 | 0.120861 | 0.036713 |
| 14일 | 14,999 | 10,151 | 0.112399 | 0.033736 |
| 30일 | 11,047 | 6,896 | 0.094063 | 0.024656 |

0일 Brier는 기존 AFT Validation과 일치합니다. 7·14일의 ECE는 0일보다
높아, 이 결과만으로 현재 시점 확률을 프론트에 그대로 표시하지 않습니다.
고정 모델의 Validation 평가이며 Test·운영 확률 보정은 아직 수행하지 않았습니다.

기존 시점별 결과 JSON의 보정 오차 방향은 아래 명령으로 재현합니다. 새 모델을
학습하거나 Test를 평가하지 않고, IPCW calibration 구간의 가중 평균과 구간별
표본·가중치를 요약합니다. 입력 JSON의 SHA-256을 결과에 남기며 입력은 수정하지
않습니다.

```bash
.venv/bin/python -m scripts.summarize_landmark_calibration \
  --input /private/tmp/repurchase-service-landmarks-live-20261003.json \
  --output /private/tmp/repurchase-landmark-calibration-diagnostics-20261003.json
```

| 구매 후 경과 | 가중 평균 예측 | 가중 관측 재구매율 | 예측−관측 | 과소예측 구간의 IPCW 가중치 비율 |
| --- | ---: | ---: | ---: | ---: |
| 0일 | 13.05% | 15.88% | −2.83%p | 99.98% |
| 7일 | 11.78% | 15.46% | −3.67%p | 100% |
| 14일 | 10.58% | 13.95% | −3.37%p | 100% |
| 30일 | 8.69% | 11.15% | −2.47%p | 100% |

각 시점의 위험집단·검열 구성은 다릅니다. 이 표는 Validation에서의 오차
방향을 보여줄 뿐, 단일 확률 보정 계수를 추정하거나 확률 노출을 승인하지
않습니다. 특히 0일의 예측 60~70% 구간은 3건, 14일의 40~50% 구간은 1건으로
꼬리 구간의 개별 오차를 일반화하지 않습니다. 보정 후보는 별도의 Train 내부
구간에서 학습하고 외부 Validation에서 평가해야 합니다.

`scripts.run_service_probability_calibration`은 AFT 부스팅 횟수와 scale을
`--aft-boost-rounds`, `--aft-loss-distribution-scale`로 명시할 수 있습니다.
기본값 20회/1.0은 기존 실험을 재현하기 위한 값입니다. 최종 평가 후보로
기록된 20회/2.0 재검증은
`reports/service_probability_calibration_fixed_aft_20261003.md`에 정리했습니다.
0일 Brier는 보정 후 소폭 악화되고 사용자 Bootstrap 구간이 0을 포함하므로,
네 시점 전체에 Isotonic 보정을 적용하거나 Test·운영 확률 노출을 승인하지 않습니다.

별도 합성 개발·평가 생성본의 동결 평가와 새 시드 반복은
`reports/independent_synthetic_final_evaluation_20261005.md` 및
`reports/independent_synthetic_replication_20261005.md`에 기록했습니다.
두 최종 평가에서 경과 0일·저이력 구간의 점추정 악화가 반복됐습니다.
`scripts.summarize_independent_synthetic_final`은 결과 JSON의 경과일별·저이력
쌍 비교를 검사해 요약하지만, 학습·후보 선택이나 운영 공개 승인은 하지 않습니다.
같은 생성 규칙의 합성 반복이므로 실제 사용자 분포의 검증으로 해석하지 않습니다.
후속 [저이력 진단](reports/independent_synthetic_history_diagnostic_20261005.md)에서는
앞선 개발본에 간격 0개·1~2개 사용자가 없었음을 확인했습니다. 사용자 유입을
분산한 새 개발본에서도 경과 0일·간격 0개 Brier가 보정 후 악화돼 새 최종 평가본은
열지 않았습니다. `scripts.diagnose_independent_calibration_history`는 개발본만으로
이 구간을 검사하며, 통과 여부가 운영 확률 공개 승인을 의미하지는 않습니다.

최종 평가 사전검증은 Validation 결과에 기록된 코드 SHA-256도 현재 코드·승인
manifest와 대조합니다. 이전 형식의 결과 JSON에는 이 지문이 없어 재실행이
필요합니다. 모듈 로드 시점과 비교 실행 전후의 디스크 지문을 대조하지만, Python이
이미 메모리에 읽은 코드가 그 파일 바이트에서 왔다는 증거는 아닙니다. 최종 Test
실행에는 별도로 고정된 이미지 또는 불변 체크아웃의 식별자·실행 기록을 확인해야
하며, 이 사전검증만으로 모델 선정·Test 실행을 승인하지 않습니다.
`scripts.freeze_service_models`는 이 사전검증을 통과한 Validation 결과와 원천
CSV 6개를 받아 Validation 종료 컷까지 재학습한 AFT·LightGBM을 새 디렉터리에
저장합니다. `freeze-record.json`에 표본 수, 원천·코드·결과 해시와 모델 ID를
남기고 기존 결과는 덮어쓰지 않습니다. Test 라벨·지표는 생성하지 않으며,
로컬 아티팩트의 보존·배포는 별도 절차가 필요합니다.
주문 격리는 Validation 종료 이전 이력으로 판단하며, 그 이후 상태 전이가 있는
주문의 현재 상태를 과거 상태와 직접 대조하지 않습니다. 컷 이후 결제 주문은
재학습에서 제외하지만, 컷 이전 주문의 이후 클레임은 수량 검증에 남깁니다.
`scripts.validate_frozen_service_test_readiness`는 고정 기록·Validation 결과·원천
6개·두 모델 파일의 해시와 추론 계약을 다시 대조합니다. 통과해도 Test 행은
생성하지 않으며, 최종 평가 실행 또는 모델 운영 채택을 승인한다는 뜻은 아닙니다.
`scripts.run_frozen_service_test`는 리뷰·병합 이후 명시적인
`--confirm-final-test`로만 실행합니다. Validation 종료 이후부터 고정 관측 종료
시각까지의 구매 앵커를 같은 30일 IPCW 모집단에서 두 고정 모델로 평가하고,
Brier·C-index·calibration·사용자 단위 쌍 Bootstrap을 기록합니다. 실행 직전
고정 디렉터리 옆에 `<고정 디렉터리명>-final-test`를 독점 생성합니다. 실패해도
실행 기록을 남겨 자동 재실행을 막으며, 수동 재시도 여부는 원인 검토 후 별도
결정해야 합니다. 이 장치는 해당 로컬 경로의 중복 실행만 막고 다른 호스트로
복사한 모델의 전역 중복 실행까지 보장하지는 않습니다. Test 결과로 모델을
재선택하지 않습니다. 2026-10-04에 고정된 로컬 스냅샷으로 1회 실행한 결과와
사용자 노출·dev 연동의 판단은
[`reports/service_final_test_20261004.md`](reports/service_final_test_20261004.md)에 기록했습니다.
2026-10-03 dev DB에서 상품군 기준선을 포함해 실행한 결과, Validation
19,189행 중 정답 확인 13,625행에서 IPCW Brier는 상품군 기준선 0.134545,
AFT 0.121258, LightGBM 0.123063이었습니다. 상품군 수축 강도는 Train 내부
후보 1·2·4·8 중 8이 선택됐습니다. 같은 Validation에서 사용자 2,786명을
1,000회 재표본추출한 `Brier(상품군 기준선) - Brier(AFT)`는 점추정
+0.013286, 95% percentile 구간 [+0.011110, +0.015447], 양수 반복 비율
100%였습니다. 이는 2026-10-03 09:01 UTC에 읽은 dev DB 스냅샷의
고정 모델·평가 표본에 대한 결과입니다. 두 DB의 원자적 스냅샷이 아니고
모델 재학습 변동성도 포함하지 않습니다. 이 문단의 Validation 비교 시점에는
Test를 평가하지 않았으며, 이후 1회 평가 결과는 위 최종 보고서를 참조합니다.
DB별 추출 시각이 다르므로 갱신 중에는 원자적 교차 DB 스냅샷으로 해석하지
않습니다. 비교 결과는 검증 후 별도 모델 선정 결정에 사용합니다.
현재 시점 복원은 주문 상태·클레임 이력을 사용합니다. 주문상품의 구매 당시 키와
수량이 사후에 추가·수정될 수 있는지는 아직 확인되지 않아, 과거 컷의 완전한
재현성을 주장하지 않습니다. 해당 필드의 불변성 또는 변경 이력은 BE와 확인해야 합니다.
요청 중인 `EXCHANGE` 클레임은 구매 수량에 반영하지 않습니다. 실제 데이터에서
확인된 교환 4건은 모두 `REQUESTED`였으며, 향후 `COMPLETED` 교환이 생기면
원상품·대체상품 연결 규칙이 확정되기 전까지 점검을 거절합니다.

서비스 상품군 운영 모델은 아직 선정되지 않았으며, 이 비교 명령은 예측값을
클라우드 DB에 적재하지 않습니다. `contract-check` 명령은 고정된 원천 데이터와
결과 발행 계약을 검사할 뿐 운영 추론이 아닙니다. 별도 `shadow-check`와
`shadow-run`은 사전 고정 AFT의 내부 검증 전용이며 사용자 노출용 발행은 하지 않습니다.

### 로컬 계약 점검

`repurchase/data_analysis`에서 다음 명령을 실행합니다.

```bash
.venv/bin/python -m scripts.run_repurchase_batch contract-check \
  --source-contract tests/fixtures/cloud_contract/source_orders.json \
  --publication-contract tests/fixtures/cloud_contract/prediction_publications.json
```

성공하면 입력 파일의 SHA-256과 단계별 처리 건수가 JSON 한 줄로 출력되고 종료
코드 `0`을 반환합니다. 계약 위반은 `2`, 예상하지 못한 시스템 장애는 `1`입니다.

### AFT 내부 검증 배치

`shadow-check`는 완료된 로컬 스냅샷의 해시·행 수와 모델 아티팩트 ID를 확인하고,
주문 상태·클레임 시점 복원 → 사용자·반려동물·상품군 피처 → AFT 조건부 확률까지
계산합니다. DB 연결이나 결과 파일 생성은 없습니다. 2026-10-03 스냅샷을
`2026-10-03T11:03:00Z` 컷으로 확인한 결과, 원천 주문 44,777건 중 19건을
격리하고 내부 검증 대상 50,517건을 만들었습니다. 이 수치는 dev DB의 현재값이나
모델 성능 지표가 아닙니다.

```bash
.venv/bin/python -m scripts.run_repurchase_batch shadow-check \
  --snapshot-directory data/raw/local_service_snapshots/service-20261003T110358Z \
  --model-directory data/processed/pretest-frozen-review-20261004/xgboost_aft \
  --artifact-id ec25eb1bcd8f24ce36a2397c1544cb087cb970af532365de7f8ec900aeb4b7fb \
  --as-of 2026-10-03T11:03:00Z --window-days 30
```

`shadow-run`은 DB가 가동 중이고 002 마이그레이션이 적용된 환경에서만 실행합니다.
수동 EKS 실행 이후의 변경 입력·실패 시 미발행 검증과 인프라 담당 경계는
[SHADOW 검증 인계](SHADOW_VALIDATION_HANDOFF.md)에 분리해 기록합니다.
`REPURCHASE_ORDER_DATABASE_DSN`, `REPURCHASE_MEMBER_DATABASE_DSN`,
`REPURCHASE_RESULT_DATABASE_DSN`을 Secret으로 주입하며, 셋째 연결은 반드시
`repurchase_db`여야 합니다. 명시적 `--allow-shadow-write`, 시간대가 포함된
`--as-of`·`--created-at`, 재시도에도 불변인 `--publication-id`, 고정
`--artifact-id`가 없으면 적재하지 않습니다. 원천은 DB별 읽기 전용 스냅샷으로
조회하고, 결과는 트랜잭션 안에서 `SHADOW`로 완료합니다. `latest_predictions`는
`PUBLISHED`만 조회하므로 내부 결과를 사용자에게 제공하지 않습니다. 소스 DB 간
동일 시각 스냅샷은 보장되지 않습니다. 필요한 Secret은 수동 Job에 주입됐지만
정기 CronJob은 아직 활성화하지 않았습니다. 실제 dev DB의 수동 SHADOW 적재는
아래처럼 검증했습니다.

2026-10-04 일회용 로컬 PostgreSQL 16의 세 DB에 위 스냅샷을 적재해
`shadow-run` 전체 경로를 검증했습니다. 주문 44,777건 중 19건을 격리하고
예측 50,517건을 `SHADOW`로 적재했으며, 동일 실행 ID 재시도는 추가 적재
없이 성공했습니다. DB의 예상·실제 결과는 모두 50,517건이고
`latest_predictions` 조회 결과는 0건입니다. 이는 공용 dev DB 실행 결과가 아닙니다.

2026-10-04 공용 dev DB에서 같은 고정 AFT·관측 컷으로 `shadow-run`을 실행해
`publication_id=shadow-dev-20261003T110300Z-ec25eb1b-v1`의 SHADOW 배치
1건·예측 50,517건을 저장했습니다. 동일 ID 재실행 결과 `inserted=false`, 전체
배치·예측 건수는 1건·50,517건으로 유지됐고 `latest_predictions`는 0건입니다.
이는 내부 적재·멱등성 검증이며 사용자 노출용 확률의 성능 승인이나 정기 실행
검증이 아닙니다. 실행에 사용한 비밀번호와 원천 행은 문서·Git에 기록하지 않습니다.

2026-10-04 EKS 수동 Job에서는 별도 실행 ID
`shadow-eks-20261003T110300Z-ec25eb1b-v1`로 같은 50,517건을 `SHADOW`에
저장했습니다. 별도 재시도 Job에서 같은 ID의 `inserted=false`를 확인했고,
DB의 배치 1건·예측 50,517건과 사용자 최신 조회 뷰 0건은 그대로였습니다.
이 결과는 위 dev 수동 실행 ID와 구분해서 관리합니다.

```bash
python -m scripts.run_repurchase_batch shadow-run \
  --model-directory /models/xgboost_aft \
  --artifact-id '<고정-AFT-artifact-id>' \
  --as-of '<시간대-포함-관측-컷>' \
  --created-at '<재시도에도-고정할-배치-생성-시각>' \
  --publication-id '<재시도에도-고정할-실행-ID>' \
  --window-days 30 --allow-shadow-write
```

### Docker 이미지와 smoke test

저장소 루트에서 재구매 디렉터리를 빌드 컨텍스트로 사용합니다.

```bash
docker build \
  --tag gollajugaenyang-repurchase-batch:local \
  repurchase/data_analysis
```

고정 예제는 이미지에 복사하지 않고 읽기 전용으로 마운트합니다.

```bash
docker run --rm \
  --volume "$PWD/repurchase/data_analysis/tests/fixtures/cloud_contract:/input:ro" \
  gollajugaenyang-repurchase-batch:local \
  contract-check \
  --source-contract /input/source_orders.json \
  --publication-contract /input/prediction_publications.json
```

### 이미지에 포함하지 않는 항목

- 원본 데이터와 분석 리포트
- 테스트 코드와 로컬 가상환경
- 모델 아티팩트와 예측 결과
- DB 주소·계정·비밀번호·API 키

실제 모델 추론 단계에서는 검증된 아티팩트를 읽기 전용 볼륨 또는 객체 저장소로
주입합니다. 클라우드 연결 정보는 Secret 관리 기능으로 전달하고 로그에 값을
출력하지 않습니다. 내부 검증용 DB 어댑터와 SHADOW 적재 경로를 구현했고,
dev DB의 002 마이그레이션·수동 SHADOW 적재도 검증했습니다. 스케줄러 연결과
사용자 노출 결정은 별도 단계입니다.

운영·CI의 Linux AMD64 환경은 GPU 의존성을 포함하지 않는 `xgboost-cpu`를
사용합니다. XGBoost 3.2에서 CPU 전용 wheel을 제공하지 않는 ARM64·macOS는 같은
버전의 일반 `xgboost`를 사용합니다. 따라서 Apple Silicon의 로컬 이미지는 운영용
AMD64 이미지보다 클 수 있으며, 실제 배포 이미지 크기는 CI의 AMD64 빌드 결과를
기준으로 판단합니다. Python 기반 이미지는 태그가 가리키는 내용이 바뀌지 않도록
멀티 아키텍처 manifest digest까지 고정합니다.

### GHCR 이미지 등록

`main`에 Dockerfile·런타임 의존성·실행 스크립트 변경이 병합되면
GitHub Actions가 Linux AMD64 이미지를 다시 빌드하고 계약 smoke test를 수행합니다.
검증을 통과한 이미지만 아래 GitHub Container Registry 경로에 등록합니다.

```text
ghcr.io/urineun-jigeum-bildeujung/ai-repurchase-batch
```

한 번 발행한 커밋 버전을 다시 찾을 수 있도록 `sha-<Git commit SHA>` 태그를
불변 버전으로 사용합니다. `stable`은 최신 검증본을 확인하기 위한 이동 태그이며,
PR 검증과 수동 워크플로 실행에서는 Registry에 이미지를 등록하지 않습니다.
Runtime 소스 변경 없이 의도적으로 이미지를 다시 발행해야 할 때는
`image-release.txt`의 정수만 올립니다. CI 파일을 수정한 것만으로는 새 Runtime
이미지를 만들지 않습니다.

클라우드 배포 시에는 `stable`을 직접 참조하기보다 Actions 실행 요약에 기록된
digest를 사용합니다. digest를 고정하면 이후 `stable`이 새 이미지로 이동해도
실행 중인 배포가 의도치 않게 바뀌지 않고 같은 이미지를 재현하거나 롤백할 수
있습니다. 이미지 등록과 실제 클라우드 배포는 분리하며, 배포 자동화는 실행 환경과
승인 정책이 확정된 뒤 별도 워크플로로 구성합니다.

같은 커밋의 워크플로를 다시 실행해도 기존 `sha-<commit>` 태그는 덮어쓰지 않고
revision 라벨을 확인한 뒤 재사용합니다. 현재 `stable` 이미지의 커밋이 실행 후보보다
최신이면 태그 갱신을 건너뛰어 과거 워크플로 재실행이 최신 이미지를 되돌리지 못하게
합니다. main push는 커밋별 독립 실행해 중간 커밋의 Runtime 변경 감지가 대기열에서
사라지지 않도록 합니다. `stable`은 로컬 이미지를 다시 push하지 않고 Registry의
검증된 SHA manifest를 digest로 복사하며, 등록 후 두 태그의 digest가 같은지 다시
검사합니다. 품질 검사는 커밋별로 실행하되 이미지 게시 Job은 공통 concurrency
그룹에서 직렬 실행합니다. 따라서 stable 조회와 게시 사이에 다른 게시 Job이
끼어들 수 없습니다. `queue: max`로 최대 100개의 대기 Job을 보존하며, 큐 한도를
넘어 취소된 실행은 재실행이 필요합니다. 대기 진입 순서는 커밋 순서와 다를 수
있으므로 기존 revision 선후 관계 검사도 유지합니다. 이 잠금은 같은 그룹을
사용하는 Job에만 적용되므로 다른 워크플로에서 stable을 쓰면 안 됩니다.
