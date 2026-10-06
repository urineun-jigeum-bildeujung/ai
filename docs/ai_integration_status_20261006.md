# AI 연동 작업 현황 — 2026-10-06

## 작업 목적
2026-10-06까지의 AI·Frontend 연동 작업을 정리하고, 최신 로컬 변경·병합 이력·실제 dev 실행 증거와 미완료 항목을 구분한다.

## 현재 확인된 결과
### Nutrition
- 현재 dev 이미지: `8f0c086ec732ced1eb653be1bb84a645821e42f9`. 이번 조회에서 Pod `generic-service-d5d76c5d9-s4h42` Ready=true, restart=0, Argo CD Synced/Healthy를 확인했다. 별도 재배포 기록은 10/6 09:40 KST /health·/ready HTTP 200 및 Member/Product DB UP를 확인했다. 이번 게시 작업에서 화면 E2E를 실행한 것은 아니다.
- 최신 로컬 변경: 기존 286개 MOCK-*와 명시 등록한 36개 Service ID/SKU, 총 322개를 스키마 기반 Mock으로 사용. 상품 141 당근·비트, 302 성견 단계·영양값 5개·호박·연어오일을 연결했다. `SCHEMA_DRIVEN_SYNTHETIC`, `production_evidence=false`를 유지한다.
- 동일 DOG/ADULT/KNOWN_NONE의 보관 스냅샷 재현: NOT_APPLICABLE 155, 종 불일치 84, 생애주기 불일치 31, 상품 생애주기 근거 부족 52. 이 수치는 현재 DB 조회나 적합도 숫자 점수 분포가 아니다.
- 이번 재검증: Nutrition **632 passed**, 기존 deprecation warning 1개. 이전 로컬 HTTP 기록 **13요청·58 assertion**, 실제 DB/Gateway E2E 아님.
- 322개 전체 `suitability.match_score` 분포 검증은 완료 산출물이 없어 **미검증**. fixture READY·HTTP 200을 점수 생성 성공으로 취급하지 않고, 차단/근거 부족의 null을 0점으로 바꾸지 않는다.
- 배포된 8f0c086 이미지에는 이번 322개 보완이 포함되지 않는다. 게시/리뷰와 후속 배포가 별도다.

### Frontend·Backend
- web PR #664: 없음/미응답 저장·복원 구분, PR #665: 상품 상세 영양 막대 API 값 연결. 두 PR의 `merged=true`, `merged_at`을 이번 조회에서 확인했다. #665 merge SHA `52d38f735bc10ef072ab5031f7acbbf86367fbb4`.
- Frontend 병합은 현재 브라우저 노출·실제 화면 E2E 통과를 뜻하지 않는다.

### 추천
- UNKNOWN 또는 상태/목록 모순을 SAFE로 표시하던 누락 수정. 홈 추천은 기존 PENDING 0.7배, 실제 충돌은 PENALIZED 0.3배를 유지하며 Nutrition excluded 정책과 임의 통일하지 않는다.
- 홈/대체 추천에서 Gateway 내부 인증 및 Pet 소유권/삭제 검사를 적용. 무효 헤더 401, 타인·삭제·없는 Pet 404, Secret 미설정 503.
- 이번 재검증: 추천 **90 passed**, 기존 warning 1개. 과거 일회용 합성 PostgreSQL 검사 4/4 및 소유권 6/6 통과 기록.
- 실제 모델 품질·Service DB/Gateway/화면 E2E는 미검증. 배포 전 승인된 `INTERNAL_GATEWAY_SECRET` 주입이 필요하며 이번 코드 PR은 Draft로 유지한다.

### 재구매
- PR #259, #303(develop), #304(main)의 실제 병합 여부와 시각을 이번 조회에서 확인했다.
- 이슈 #302의 10/5 테스트 기록: 일회성 SHADOW Job 성공, 50,517건 저장, 공개 뷰 노출 0건. 이번 게시 작업에서 DB를 다시 검증하지 않았다.
- 10/6 시연 Job 실행/DB 결과, 취소된 10/7 Job 삭제, 목계정 BE 인증·FE 표시 연결은 확인이 남았다.
- 사전 고정 AFT로 30일 확률만 시연. 운영 PUBLISHED 발행·정기 CronJob은 승인하지 않는다. 실제 사용자 품질이나 AFT 우위를 주장하지 않는다.

## 남은 확인
- [ ] 최신 322개 Mock 보완 코드 리뷰·병합·배포
- [ ] 322개 전체 match_score 숫자/null 및 원인 분포 검증
- [ ] 추천 Gateway Secret 설정 및 인증·소유권 실제 E2E
- [ ] Frontend 실제 화면에서 Pet 변경·영양값·null·오류/재시도 확인
- [ ] 재구매 10/6 시연 결과·10/7 Job 삭제와 목계정 조회 경계 확인

## 영향 범위 및 리뷰 요청사항
- 변경 대상: Nutrition Mock 경로, 추천 프로필/인증·소유권, 재구매 일정 문서, 누적 작업 보고서.
- 앱 추가 배포·DB write·GitOps/IAM/DNS 수정은 이번 게시 작업에서 수행하지 않는다.
- 민감정보·키·원본 회원/주문 데이터는 게시하지 않는다. 합성 provenance와 fail-close 및 배포 전 설정을 중점 검토한다.


근거: [작업 정리 이슈 #307](https://github.com/urineun-jigeum-bildeujung/ai/issues/307), [Nutrition 로컬 변경](../nutrition/docs/service_evidence_gap_resolution_20261006.md), [추천 인증 계약](../recommendation/endtoend/docs/gateway_ownership_contract.md), [재구매 #302](https://github.com/urineun-jigeum-bildeujung/ai/issues/302).
