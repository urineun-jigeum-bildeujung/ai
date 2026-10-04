# 재구매 결과 저장 구조

- 원천 목데이터는 BE, 결과 테이블과 예측값 적재는 AI가 관리한다.
- `001_prediction_storage.sql`은 PostgreSQL 15 이상용 첫 스키마 버전이다.
- 2026-10-02 개발 환경의 `repurchase_db`에 `ai_dev`로 `001_prediction_storage.sql`을 적용했다. `repurchase` 스키마, 테이블 2개, `latest_predictions` 뷰를 확인했고 두 테이블의 초기 행 수는 모두 0건이었다.
- 접속·스키마/테이블 생성·1건 입력·조회 권한은 롤백 트랜잭션으로 검증했다. 이는 운영 배치 실행이나 FE 조회 권한 검증을 의미하지 않는다.
- 기존 스키마에 덮어쓰지 않는다. 이미 같은 이름이 있으면 중단하고 충돌을 확인한다.
- 사용자 키는 현재 AI 계약에 맞춰 text이며 서비스 member.id를 문자열로 변환한다. pet_id는 서비스 bigint, 미지정은 NULL이다.
- 저장 구조, 발행 안전성 제약, 최신 결과 뷰와 독립된 DB 적재 함수까지 구현했다. 운영 배치의 추론 흐름에는 아직 연결하지 않았다.
- `002_shadow_publication.sql`은 내부 검증용 `SHADOW` 완료 상태를 추가한다. `latest_predictions`는 여전히 `PUBLISHED`만 조회하므로 shadow 결과는 서비스 조회 대상이 아니다. 2026-10-04에 개발 `repurchase_db`에 002를 적용하고 SHADOW 배치 1건·예측 50,517건을 확인했다. 같은 실행 ID 재시도에서 추가 적재는 0건, 조회 뷰의 노출 결과는 0건이었다.
- `003_latest_full_snapshot.sql`은 아직 개발 DB에 적용하지 않은 후속 마이그레이션이다. 현재 배치는 전체 대상 스냅샷이므로 최신 `PUBLISHED` 배치 한 건의 결과만 조회하게 한다. 새 배치에서 사라진 키에 과거 예측이 남는 것을 막는다. 부분 재계산 발행을 도입하려면 별도 조회 정책이 필요하다.

## 후속 적재 로직

AI 쓰기 로직은 한 트랜잭션에서 STAGING 생성 → 결과 저장 → PUBLISHED 전환을 수행한다. DB 트리거는 전환 시 실제/예상 건수를 검사하고, 발행 후 결과 및 배치 변경을 거절한다. 결과 변경과 발행 전환은 부모 배치 행 잠금으로 직렬화한다. 적재 함수는 멱등키를 직렬화해 기존 내용과 비교하고, 동일 내용 재실행은 추가 쓰기 없이 성공 처리한다. 로컬 PostgreSQL에서 실패 롤백도 검증했다.

내부 검증에서는 같은 원자적 절차의 마지막 상태만 `SHADOW`로 전환한다. `SHADOW`도 건수 일치와 불변성 제약을 받지만 `latest_predictions`에는 나타나지 않는다. 기존 001 적용 DB에는 002를 적용한 뒤에만 SHADOW 적재가 가능하다. 이 상태 추가는 실제 AFT 추론 배치 연결이나 dev DB 적재 완료를 뜻하지 않는다.

스키마 소유/마이그레이션 권한, 배치 쓰기 권한, 사용자별 인증된 조회 권한은 별도로 분리한다. latest_predictions 뷰 자체는 사용자 접근 통제가 아니며 브라우저에 DB 자격 증명을 제공하지 않는다. 여기서는 계정 생성이나 GRANT를 수행하지 않는다.

## 적용·검증

최초 적용은 승인된 DB에 `psql -X -v ON_ERROR_STOP=1 -f 001_prediction_storage.sql`로 실행한다. 접속 자격 증명은 명령이나 저장소에 넣지 않는다. 재실행은 마이그레이션 이력에서 차단하며, 후속 변경은 번호가 다른 SQL로 추가한다.

기존 001 적용 DB의 SHADOW 확장은 `psql -X -v ON_ERROR_STOP=1 -f 002_shadow_publication.sql`로 한 번만 적용한다. 개발 DB에는 이미 적용했으므로 **다시 실행하지 않는다**. 다른 환경에 적용할 때는 전후 `prediction_batches` 상태 제약과 `latest_predictions` 뷰가 `PUBLISHED`만 노출하는지 확인한다.

003은 로컬 검증 후 별도 승인된 적용 창에서 한 번만 적용한다. 003 적용 전 개발 DB의 기존 뷰는 여전히 키별 과거 발행분을 섞을 수 있다. 현재 `SHADOW` 결과 50,517건은 003 적용 여부와 관계없이 노출되지 않는다.

`test_prediction_storage.sql`은 로컬 일회용 DB에서만 실행한다. 최신 결과, 건수 불일치 발행 거절, 발행 후 수정 거절, 중복 멱등키, 확률 제약을 확인하고 트랜잭션 종료 시 데이터를 롤백한다. 운영 DB를 테스트 대상으로 사용하지 않는다.

`scripts.modeling.prediction_storage.publish_prediction_publication()`은 호출자가 전달한 psycopg autocommit 연결에서 한 배치를 원자적으로 발행한다. 동일 멱등키와 동일 내용은 추가 저장 없이 성공, 다른 내용은 오류다. 연결 자격 증명은 코드·로그에 기록하지 않는다. 로컬 통합 테스트는 별도 `repurchase_writer_test` DB의 `REPURCHASE_TEST_DATABASE_DSN`이 설정될 때만 실행하고 스키마를 만들었다가 제거한다. 001·002는 개발 DB에 적용했고 AFT 추론에서 SHADOW 적재까지 검증했다. 사용자 노출용 `PUBLISHED` 발행은 하지 않았다.
