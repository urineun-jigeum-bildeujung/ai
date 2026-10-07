# 파트 간 연동 점검 및 추천 알레르기 상태 수정

2026-10-05 GitHub 조회와 로컬 코드 검증 결과. 이 문서는 배포 완료 보고가 아니다.

## 확인한 작업

| 파트 | 확인한 현재 기준 | 결과 |
| --- | --- | --- |
| AI / Nutrition | [develop 8f0c086](https://github.com/urineun-jigeum-bildeujung/ai/tree/8f0c086ec732ced1eb653be1bb84a645821e42f9), [PR #306](https://github.com/urineun-jigeum-bildeujung/ai/pull/306) | 저장된 알레르기 상태 보존과 정확한 Service 코드 매핑 반영 |
| Backend | [dev 6625cba](https://github.com/urineun-jigeum-bildeujung/sever/tree/6625cbae6bac22dc144c49f81b5f80d9b702e9f2), [PR #222](https://github.com/urineun-jigeum-bildeujung/sever/pull/222) | 프로필 상태 저장·응답 및 V3~V6 migration 코드 반영 |
| Frontend | [dev 52d38f7](https://github.com/urineun-jigeum-bildeujung/web/tree/52d38f735bc10ef072ab5031f7acbbf86367fbb4), [PR #664](https://github.com/urineun-jigeum-bildeujung/web/pull/664), [PR #665](https://github.com/urineun-jigeum-bildeujung/web/pull/665) | 없음/미응답 저장·복원, 상품 상세 영양 막대 실제 API 연결 반영 |
| 추천 | 같은 AI develop의 recommendation/endtoend tree `0bf68e67e5086dd8d566d888b5de1f5fb994d96a` | 저장 프로필 상태를 조회하지 않고, 충돌 없는 빈 목록을 SAFE로 표시하는 누락 확인 |
| 재구매 | [PR #303](https://github.com/urineun-jigeum-bildeujung/ai/pull/303), [시연 계약](../../../repurchase/DEMO_MODEL_DECISION.md) | SHADOW 시연 배치와 목계정 조회 함수 구현. 문서에 HTTP 인증·BE/FE 연결은 별도라고 명시 |
| GitOps | [main a28d488](https://github.com/urineun-jigeum-bildeujung/gitops-value/tree/a28d4884a4cfef57fde3516fada9d04d462cb2e4) | 최근 web/member-service/nutrition 이미지 갱신 커밋 확인. 실제 Pod 상태는 이번 작업에서 확인하지 않음 |

최상위 `ai/`는 과거 브랜치와 기존 미커밋 작업이 있어 수정하지 않았다. 기존 작업본 `/private/tmp/nutrition-integration-20261005-ai`의 HEAD `9b807e5`는 최신 develop `8f0c086`과 전체 파일 diff가 없음을 확인했다. 수정은 이 작업본의 `recommendation/endtoend/**` 안에만 있다. Git commit/push/merge, DB 운영 쓰기, 배포는 수행하지 않았다.

## 이번 수정

- 단일·일괄 Pet 조회에서 `to_jsonb(p)->>'allergy_profile_status'`를 읽는다. 이전 스키마에는 null이 반환되어 누락 컬럼 오류를 피한다.
- 명시 KNOWN_NONE + 빈 목록만 없음으로 인정한다. KNOWN_LIST + 비어 있지 않은 목록은 등록 목록으로 인정하고, UNKNOWN·잘못된 상태·상태/목록 모순은 판단 보류한다. 이전 스키마의 비어 있지 않은 등록 목록은 Nutrition과 같은 KNOWN_LIST 호환 규칙을 따른다.
- 홈 추천에서 미확인 프로필은 기존 PENDING 0.7배 감점을 적용한다. 실제 충돌은 상태 모순이 있더라도 기존 PENALIZED 0.3배 감점을 유지한다. 종 필터와 모델 입력 구조는 바꾸지 않는다.
- 프로필 미확인 사유는 “반려동물의 알레르기 정보를 확인해 주세요.”로 구분한다. 상품 성분 미확인 문구는 기존대로 유지한다.
- 대체상품에도 실제 판정 상태를 전달하고 API의 상태 누락 기본값을 PENDING으로 바꾼다. 대체상품의 기존 충돌 제외·연령 필터·유사도 점수 정책은 유지한다.
- SAFE는 기존 추천의 코드 교집합 판정이다. Nutrition의 evidence 기반 안전 판정이나 임상적 안전 보장을 대신하지 않는다. 추천 후보의 완전 제외 정책으로 변경하지 않았다.

## 검증

- 수정 전 코드를 `git show HEAD:...`로 읽어 동일 합성 Pet(UNKNOWN/빈 목록)·상품·0.8 모델 출력을 비교: **SAFE/80 → PENDING/56** 재현.
- 추천 회귀 **44 passed**: 상태 일관성, 단일/일괄 Repository, 홈·대체상품 실제 FastAPI 경로, 점수·사유, 충돌 제외, 종 필터와 정렬.
- 네트워크를 끈 일회용 PostgreSQL 16에서 **4/4 passed**: 이전/새 스키마 × 단일/일괄 실제 SELECT. 합성 테이블만 사용했고 컨테이너는 제거했다.
- Nutrition 전체 **616 passed**. 첫 실행은 기존 `/private/tmp/ai-nutrition-test-deps`에 psycopg2 패키지 본문이 빠져 1건 실패했고, 완전한 기존 `/private/tmp/nutrition-test-deps`로 재실행해 통과했다. 제품 코드를 이 환경 문제에 맞춰 변경하지 않았다.
- 두 pytest 실행 모두 기존 Starlette/AnyIO deprecation warning 1개. `git diff --check` 통과.
- 추천 모델 출력은 합성 대역이다. 실제 학습 모델 아티팩트가 로컬에 없어 모델 품질·실제 Service DB/Gateway E2E는 실행하지 않았다. 새 코드의 CI/배포/Pod는 미검증이다.

재현 명령(기존 curated Python 사용):

```sh
cd /private/tmp/nutrition-integration-20261005-ai/recommendation/endtoend
PYTHONPATH=/private/tmp/recommendation-profile-test-deps \
  '/Users/aku/Documents/통합 프로젝트(우리는지금빌드중)/ai-nutrition-curated/.venv/bin/python' \
  -B -m pytest tests -q -p no:cacheprovider
```

테스트 의존성 torch/python-dotenv는 `/private/tmp/recommendation-profile-test-deps`에만 설치했다. 프로젝트 의존성 파일과 모델은 변경하지 않았다.

## 남은 작업

1. 이 로컬 수정의 리뷰·배포 후 소유 Pet의 KNOWN_NONE/UNKNOWN/KNOWN_LIST를 Gateway와 추천 화면에서 검증해야 한다. [FE 추천 계약](https://github.com/urineun-jigeum-bildeujung/web/blob/52d38f735bc10ef072ab5031f7acbbf86367fbb4/src/entities/recommendation/README.md)은 PENDING 전용 배지를 표시하지 않으며, 이번 변경은 기존 reason_text로 미확인 이유를 전달한다. 화면까지 검증한 것은 아니다.
2. 재구매는 10/6 일회성 SHADOW 시연 Job의 실제 실행과 목계정 전용 BE 인증·조회/FE 표시 연결 확인이 남아 있다. 후속 조회한 [이슈 #302](https://github.com/urineun-jigeum-bildeujung/ai/issues/302)에 따라 10/7 시연은 취소됐고 Job 삭제 확인이 남아 있다. 운영 PUBLISHED 공개는 현재 계약에서 NO-GO이며 이를 임의로 해제하지 않았다.
3. Nutrition 상품 141 당근·비트의 알레르기 근거와 상품 302 성분·생애주기 근거는 [기존 통합 기록](../../../nutrition/docs/service_integration_resolution_20261005.md)의 미해결 항목이다. 이번에 제조사 원문을 확보한 것은 아니며 근거를 생성하지 않았다.
4. 추천의 충돌 상품 감점 노출과 Nutrition의 excluded는 서로 다른 명시 정책이다. 자동으로 통일하지 않았으며, 서비스 전체 노출 정책을 바꾸려면 담당자 합의가 필요하다.
