# 재구매 독립 목데이터 생성 및 검증

`scripts.generate_independent_synthetic_sources`는 기존 Test 행을 복제하지 않고 보정 개발용과 최종 평가용 원천 6종을 각각 생성한다. 원천 주문·회원·반려동물·상품 키, 난수 시드, 관측 기간을 분리하고 생성 설정과 파일 해시를 기록한다. 출력은 무시된 로컬 `data/raw/` 아래에 두며 클라우드 DB에 적재하지 않는다.

2026-10-05 로컬 생성본 `synthetic-20261005-v1`은 보정 개발 890개 주문·120명, 최종 평가 971개 주문·120명이다. 기존 Test 종료 후 별도 시계열 기간에서 30일 이상 관측 가능한 주문은 각각 756개, 841개다. 두 생성본 모두 파일 해시·행 수·필수 컬럼·시간/키 분리 사전검사를 통과했다. 원천 구간 파이프라인에 대한 소규모 자동 테스트도 통과했다.

현재 생성 분포는 사용자·상품군별 잠재 구매 간격 18~62일, 로그 잡음 0.18의 단순 시뮬레이션이다. 클레임은 생성하지 않는다. 따라서 이 결과는 입력 계약 및 파이프라인 검증용이지, 실제 사용자 분포의 확률 보정, 모델 우열, 사용자 공개 승인을 증명하지 않는다. 특히 최종 평가용 생성본을 튜닝에 사용해서는 안 된다.

로컬 실행 예:

```sh
cd repurchase/data_analysis
.venv/bin/python -m scripts.generate_independent_synthetic_sources \
  --output-root data/raw/independent_evaluation/synthetic-run-v1 \
  --start-at 2026-10-04T00:00:00+00:00
```

기존 Test·보정 개발·최종 평가 분리 여부는 `scripts.validate_independent_dataset_preflight`로 확인한다. 별도 원천 감사와 실사용 분포의 독립 데이터 검증 전에는 확률 공개 판단을 바꾸지 않는다.
