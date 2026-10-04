# 재구매 독립 목데이터 수용 사전검사 계약

## 목적과 경계

기존 봉인 Test 이후에 별도로 생성할 **보정 개발용**과 **최종 평가용** 목데이터의 파일 무결성·시간 구간·평가 주문 키 분리를 먼저 확인한다. 이 검사는 생성 전 코드와 가상 fixture로 준비했으며 **새 데이터는 아직 없고 실제 수용 판정도 하지 않았다**. 목데이터 통과를 실제 사용자 성능이나 확률 공개 승인으로 해석하지 않는다.

## 전달 형식

두 데이터셋은 각각 별도 디렉터리로 보관한다. 각 디렉터리에는 `orders.csv`, `order_items.csv`, `histories.csv`, `claims.csv`, `claim_items.csv`, `pets.csv`와 기존 로컬 추출본 형식의 `manifest.json`(파일명·행 수·SHA-256), 아래 `evaluation-plan.json`이 필요하다. 여섯 CSV의 필수 열은 현재 서비스 원천 6종 계약과 같다. 과거 구매를 피처 문맥으로 포함해도 되지만, **평가 대상 주문**은 `evaluation_start_at` 이상·`observation_end_at` 이하에 결제된 주문으로 구분한다.

```json
{
  "schema_version": 1,
  "dataset_role": "calibration_development",
  "dataset_run_id": "고유 생성 실행 ID",
  "generator_version": "생성 코드 버전",
  "source_dataset_version": "생성 기준 데이터 버전",
  "config_hash": "설정의 소문자 SHA-256 64자리",
  "random_seed": 42,
  "evaluation_start_at": "2026-11-01T00:00:00+09:00",
  "observation_end_at": "2026-12-31T00:00:00+09:00"
}
```

최종 평가용은 `dataset_role`을 `final_evaluation`로 쓴다. 두 실행은 서로 다른 `dataset_run_id`로 식별하고, 생성 버전·설정 해시·시드 조합도 같지 않아야 한다. 날짜는 예시이며 실제 컷으로 확정해야 한다.

## 통과 조건과 후속 게이트

- 기준 스냅샷의 원천 지문이 기존 최종 Test 결과와 일치하고, 새 데이터 각각의 6개 CSV 행 수·SHA-256·필수 열이 manifest와 일치한다.
- 기존 Test 관측 종료 < 보정 개발 평가 시작 < 보정 개발 관측 종료 < 최종 평가 시작 < 최종 평가 관측 종료 순서다.
- 각 데이터셋에 평가 주문과 30일 결과를 관측할 수 있는 주문이 적어도 1건 있다. `paid_at`과 평가 계획의 시각에는 시간대가 명시되어야 하며, 관측 종료 이후 결제 주문은 포함하지 않는다.
- 각 데이터셋의 평가 주문 ID는 기존 스냅샷 및 서로의 평가 주문 ID와 겹치지 않는다. 과거 문맥 주문과 회원·pet ID의 재사용은 여기서 일괄 금지하지 않는다.

위 조건을 통과해도 **원천 키·주문 상태 이력·클레임 수량·반려동물 연결 감사, 피처 시점 복원, 보정 개발/최종 평가 실행은 아직 남는다**. 기존 Test를 보정에 재사용하지 않고, 최종 평가용은 보정 방법을 고정한 뒤 한 번만 연다. 사용자 확률 노출과 정기 CronJob은 별도 운영 판단 전까지 보류한다.

검사 명령은 `python -m scripts.validate_independent_dataset_preflight --baseline-directory <기존 스냅샷> --final-test-result <기존 Test result.json> --development-directory <보정 개발용> --evaluation-directory <최종 평가용>`이다. 결과에는 집계 건수와 데이터셋 실행 ID만 출력하며 원천 행이나 비밀번호는 출력하지 않는다.
