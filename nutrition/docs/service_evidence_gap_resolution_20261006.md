# 스키마 기반 Mock 근거 보완 — 2026-10-06

사용자가 전체 상품을 실제 상품이 아닌 스키마 기반 Mock으로 확정했다. 제조사 자료를 요구하던 이전 조사 결론을 이 기준으로 대체한다. DB 조회·인증 경로는 유지하며, 조회된 상품에 합성 입력을 연결한다.

## 변경

- 기존 `MOCK-*` 외에 보관된 Service 상품 스냅샷의 36개 ID·SKU 조합을 `data/integration/mock_service_identity_v1.json`에 명시했다. ONF-004는 ID 302와 정확히 일치해야 한다. 임의 ONF-* 또는 불일치 ID에는 적용하지 않는다.
- 302번은 기존 DOG_ADULT_DRY 합성 프로필을 사용한다. 영양값 5개, 합성 ADULT 단계 및 기존 급여 에너지를 연결한다. 제조사 AAFCO 적합 선언을 뜻하지 않는다.
- `mock_ingredient_identity_v1.json`은 당근→carrot, 비트→beet, 호박→pumpkin, 연어오일→salmon·fish를 합성 비교용 identity로 정의한다. NON_ALLERGEN 판정이 아니다. Mock 경로에서만 적용하고 원재료별 synthetic 출처를 기록한다.
- 모든 합성 입력은 `SCHEMA_DRIVEN_SYNTHETIC`, `production_evidence=false`로 구분한다. API 최종 판정은 기존 엔진이 계산한다.
- 알레르기 UNKNOWN/OTHER, 새 미등록 원재료, 독성, 종·생애주기 불일치, partial/unavailable 시나리오를 유지한다. SENIOR를 영양 기준의 ADULT로 임의 변경하지 않는다.

## 검증 결과

보관된 상품 스냅샷과 합성 성견 프로필을 사용했다. 현재 DB·Gateway·Pod 검증이 아니다. [재현 JSON](service_evidence_gap_replay_20261006.json)에 수정 전 6개와 수정 후 10개 결과 및 스냅샷 SHA256을 보존했다.

| 상품 | 등록 알레르기 | 수정 후 결과 | 영양값 |
| --- | --- | --- | --- |
| 141 | 없음 | NOT_APPLICABLE | 5개 |
| 141 | 닭고기 | NO_CONFLICT_DETECTED | 5개 |
| 141 | 양고기 | SAFETY_BLOCKED / ALLERGY_CONFLICT | 5개 |
| 302 | 없음 | NOT_APPLICABLE | 5개 |
| 302 | 양고기 | NO_CONFLICT_DETECTED | 5개 |
| 302 | 닭고기·연어·생선 각각 | SAFETY_BLOCKED / ALLERGY_CONFLICT | 5개 |

전체 322개가 Mock 대상이다. 동일한 DOG/ADULT/KNOWN_NONE 프로필로 전체를 재현하면 NOT_APPLICABLE 155개, 종 불일치 84개, 생애주기 불일치 31개, 상품 생애주기 근거 부족 52개다. 이는 전 상품이 해당 성견에게 적합하다는 뜻이 아니며, 모든 보류를 없애기 위한 수정도 아니다. 다른 원재료 미해결 사례는 별도이며 이 변경의 명시적 네 원료 범위를 넘어서 추정하지 않는다.

검증: Nutrition pytest 632개 통과, 로컬 HTTP 13요청·58개 assertion 통과, `git diff --check` 통과. 기존 테스트의 특정 생선 보류 검증은 임의 상품의 누락 데이터에 의존하지 않고 명시적 generic fish 입력으로 고정했다. [HTTP 검증 결과](service_mock_http_validation_20261006.json)는 실제 DB·Gateway 검증이 아니다.

운영 DB 변경·Git 게시·배포는 하지 않았다. 인프라 종료 상태에서 로컬 수정과 검증까지 진행했다.
