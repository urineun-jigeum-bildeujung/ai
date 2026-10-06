# 추천 Gateway 인증 및 Pet 소유권

2026-10-05 로컬 수정. 이전 [알레르기 상태 작업](integration_followup_20261005.md)에 이어 적용했다. 커밋·푸시·배포는 수행하지 않았다.

## 2026-10-06 최종 확인

추천 테스트를 다시 실행해 **90 passed**를 확인했다. 변경 Python 구문·공백 검사와 `git diff --check`도 통과했다. GitHub develop은 계속 `8f0c086ec732ced1eb653be1bb84a645821e42f9`이며, 현재 추천 values에서도 `INTERNAL_GATEWAY_SECRET` 주입은 확인되지 않았다. 로컬 구현·회귀 검증은 완료했지만 배포 설정 연결과 실제 Gateway/Pod 확인은 남아 있다. 아래 PostgreSQL 결과는 10월 5일 수행한 일회용 합성 DB 검증 기록이다.

## 원인과 수정

기존 `/recommend/home`, `/recommend/substitute`는 요청 body의 `pet_id`로 회원 확인 없이 조회했다. 홈 추천은 조회한 Pet의 `user_id`로 구매 이력까지 읽었다. Gateway의 로그인 확인만으로 해당 Pet 소유권이 검증되는 구조가 아니었다.

[Backend Gateway 필터](https://github.com/urineun-jigeum-bildeujung/sever/blob/6625cbae6bac22dc144c49f81b5f80d9b702e9f2/platform/api-gateway/src/main/java/com/golajugaenyang/gateway/filter/JwtClaimForwardingFilter.java)는 클라이언트의 `X-Member-Id`, `X-Internal-Secret`, `X-Auth-Id`를 제거하고 검증한 JWT에서 회원 ID와 내부 Secret을 설정한다. 이 현재 원격 코드와 로컬 사본의 SHA256 일치를 확인했다. Nutrition에서 이미 사용하는 신뢰 계약을 추천에도 적용했다.

- 두 추천 경로는 `INTERNAL_GATEWAY_SECRET`과 단일 `X-Internal-Secret` 헤더를 상수 시간 비교로 검증한다.
- 단일 `X-Member-Id`는 양의 PostgreSQL bigint 범위 ASCII 정수만 허용한다. 중복·위조·누락 헤더는 401이며 DB·모델을 호출하지 않는다. Secret 미설정은 503이다.
- 요청 body의 회원 ID를 신뢰하지 않는다. `p.id = 요청 Pet AND p.member_id = 인증 회원 AND p.deleted_at IS NULL`을 SQL에서 강제한다.
- 다른 회원 Pet, 삭제 Pet, 없는 Pet은 모두 기존 404 응답을 사용하며 상품·리뷰·구매·모델 접근 전에 종료한다.
- 내부 리뷰 작성자 일괄 조회와 배치 조회는 추천 대상 소유권 경로와 구분해 유지한다. 더미 모드에서도 소유권/삭제 검사를 생략하지 않는다.
- `/health`는 기존대로 공개한다. 프론트 요청 body와 인증 Bearer 토큰 전달 방식은 변경하지 않는다.

## 검증

- 수정 전 두 경로의 무인증 요청 테스트가 실패하는 것을 재현했다. 기대 401 대신 Pet 조회 후 404였다.
- 추천 전체 **90 passed**. 이전 44개 프로필 테스트와 신규 46개 인증·소유권 회귀를 포함한다. 소유 Pet 성공, 누락/잘못된/중복 헤더, Secret 미설정, 본문 회원 ID 무시, 후속 접근 차단, SQL 조건, 더미 경로 등을 검증했다.
- PostgreSQL 16 일회용 합성 DB에서 실제 `get_owned_pet_by_id` 실행 **6/6 passed**. 소유/타인/삭제/없는 Pet, 이전 스키마 빈 목록/등록 목록을 확인하고 컨테이너를 제거했다.
- `git diff --check` 통과. 기존 Starlette/AnyIO deprecation warning 1개.
- 실제 Gateway/운영 DB/모델/Pod E2E는 이번 작업에서 실행하지 않았다. 모델 점수 회귀는 합성 대역을 사용한다.

## 배포 전제

[현재 dev 추천 values](https://github.com/urineun-jigeum-bildeujung/gitops-value/blob/a28d4884a4cfef57fde3516fada9d04d462cb2e4/values/dev/services/recommendation/values.yaml)는 Gateway에서 오는 8000 포트만 허용하는 NetworkPolicy를 선언하지만 `INTERNAL_GATEWAY_SECRET` 환경변수 주입은 없다. 실제 클러스터 적용 상태는 확인하지 않았다.

따라서 이 코드는 **현재 values 그대로 배포할 수 없다**. 배포 담당자가 Gateway와 동일한 비밀값을 추천 namespace의 승인된 Secret 경로로 제공하고 env에 연결해야 한다. Secret 이름/값을 추정하거나 새 값을 코드에 만들지 않았다. 수동 GitOps 수정이나 배포도 하지 않았다. Secret이 없으면 추천 요청이 503으로 닫히는 것이 의도된 동작이다.

배포 시 소유 Pet 성공, 다른 회원 Pet 404, 삭제 Pet 404, 일반 클라이언트의 내부 헤더 위조 차단을 공개 Gateway에서 검증하고 실제 추천 Pod 이미지·Ready를 확인해야 한다. 설정 주입만으로 인증 E2E 통과를 주장하지 않는다.
