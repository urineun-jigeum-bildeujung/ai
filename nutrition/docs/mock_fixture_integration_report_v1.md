# Mock Nutrition Integration Fixture — 작업 보고 v1

## 1. 수정 전 실패 경로

`nutrition/scripts/nutrition/service_db_adapter.py`의 기존 `bridge_sku()`는 strict GTIN만 허용했다.

```text
MOCK SKU
→ INVALID_SKU
→ nutrition_source = null
→ nutrition_items = []
→ nutrition_comparison_status = UNKNOWN
```

## 2. 변경한 lookup 구조

### 신규 파일

- `data/integration/mock_nutrition_profiles_v1.json`
  - production/reference artifact와 분리된 고정 nutrient profile
- `scripts/nutrition/mock_integration_fixture.py`
  - `MOCK-*` identifier 탐지
  - deterministic profile/fixture completeness 할당
  - `MOCK_INTEGRATION_FIXTURE` provenance 생성
- `scripts/nutrition/audit_mock_fixture_coverage.py`
  - Service DB 286개 read-only coverage/Golden Set audit
- `tests/test_mock_integration_fixture.py`
  - lookup, fail-close, determinism, Rule Engine, Safety 분리 회귀

### 수정 파일

- `scripts/nutrition/service_db_adapter.py`
  - GTIN 분기를 유지하면서 Mock integration namespace를 별도 분기
- `scripts/nutrition/service_repository.py`
  - 286개 audit용 read-only `list_active_products()` 추가
- `scripts/api_nutrition.py`
  - Mock fixture에서만 기존 Pet `AGE_RULE` 사용 가능하도록 경계 정리
  - `/health` endpoint 목록에 `by-service-id` 반영
  - Service readiness에 mock fixture artifact 확인 추가

Rule Engine, comparison core, NIAS reference selection, allergy core는 변경하지 않았다.

## 3. Golden Set / 로컬 회귀 결과

로컬 contract fixture에서 다음 경로를 확인했다.

| case | fixture | Rule Engine | nutrition comparison | Safety |
|---|---|---|---|---|
| DOG / food / adult / dry | READY | `pipeline_p1c_v1` | 생성됨 | Pet profile에 따라 독립 판정 |
| CAT / food / adult / wet | READY | `pipeline_p1c_v1` | 생성됨 | 독립 판정 |
| deterministic partial slot | PARTIAL | `pipeline_p1c_v1` | UNKNOWN | 기존 fail-close |
| deterministic unavailable slot | UNAVAILABLE | `pipeline_p1c_v1` | UNKNOWN | 기존 fail-close |
| non-food | UNAVAILABLE | category engine | NOT_APPLICABLE | 기존 정책 |

검증:
- Mock fixture + 기존 Service integration boundary: **63/63 PASS**
- Nutrition 전체 회귀: **252 PASS, 1 deselected, 4 subtests PASS**
- strict GTIN 기존 경로 회귀 없음
- 동일 입력 반복 / module reload 재현성 PASS

실제 dev Service DB에서 선택한 Golden Set 10~20건의 ID/SKU별 결과는 배포 후 `audit_mock_fixture_coverage.py` 결과로 확정한다. 로컬 synthetic contract fixture를 실제 dev 상품 결과라고 표현하지 않는다.

## 4. 286개 전체 결과

현재 PR 생성 환경에서는 실제 dev Product DB 286개를 읽지 않았으므로 실제 coverage 수치를 임의 작성하지 않는다.

배포/DB 환경에서 다음 명령으로 확정한다.

```bash
python scripts/nutrition/audit_mock_fixture_coverage.py \
  --expected-total 286 \
  --output /tmp/mock_fixture_coverage_v1.json
```

보고 분모는 항상 286으로 명시한다.

## 5. E2E 상태

기존 배포 환경에서 Gateway → Nutrition `/health`, `/ready`, Service DB 조회 및 인증 route는 확인됐다.

이번 Mock fixture 변경의 실제 Gateway E2E는 **배포 전이므로 PENDING**이다. PR 병합/배포 이후 아래를 별도로 확인한다.

- `/ready`의 `mock_integration_fixture=UP`
- Golden Set 10~20건 실제 Gateway E2E
- `nutrition_source.source_type=MOCK_INTEGRATION_FIXTURE`
- READY에서 실제 comparison 생성
- PARTIAL/UNAVAILABLE에서 UNKNOWN fail-close
- 286개 전체 read-only coverage
- Pod 재시작 후 동일 결과

## 6. 데이터 경계

- Production/reference evidence: 기존 strict GTIN + versioned evidence path
- Mock integration evidence: `MOCK_SKU` + `MOCK_INTEGRATION_FIXTURE`
- `real_product_identity_claimed=false`
- Mock fixture를 실제 제조사 evidence나 real-product crosswalk로 표현하지 않는다.

## 7. Phase 4 보고 원칙

- 실제 사용자 5~10명 평가는 미실시 상태로 유지한다.
- 실제 Product→Production Nutrition Evidence 연결은 0/286이므로 운영 Accuracy/Precision/Recall을 임의 산출하지 않는다.
- Mock fixture 결과는 integration E2E 검증으로만 보고하며 실제 제조사 영양 정확도로 해석하지 않는다.
