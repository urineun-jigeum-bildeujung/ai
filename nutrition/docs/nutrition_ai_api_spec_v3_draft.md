# 영양성분 분석 AI API 명세 — BE/FE 협의용 (v3 draft → v7, 9/4 16:48 KST)

> **Status: SUPERSEDED**
>
> **Superseded by:** 2026-09-21 FE → Nutrition AI direct integration architecture
> 이 문서는 이전 `FE → BE → AI` 구조를 전제로 한 historical target contract다. `/internal/v1/nutrition/*`는 현재 FastAPI runtime이나 현재 Nutrition 서비스 통합 필수 endpoint가 아니다. 현재 서비스 구조는 `service_integration_architecture.md`를 기준으로 한다.

**문서명**: 영양성분 분석 AI API 명세 — BE/FE 협의용
**버전**: v3 draft → v7 (BE/FE 협의용 확정)
**상태**: pre-PR / 9/8 Phase 2 머지 예정 / BE/FE 협의 진행 가능
**정책**: aafco_pass_P0D current 적용
**코드**: parity-test target 기준 / 실 코드 patch 미적용
**작성**: 2026-09-04 16:48 KST
**v3 갱신 (16:30 KST)**: 사용자 결정 5개 반영
**v5 갱신 (16:40 KST)**: 사용자 v3 간결 구조 채택 + P0/P1 5가지 수정
**v6 갱신 (16:42 KST)**: §7 API별 주요 응답 데이터 + §8 분석 처리 흐름 추가
**v7 갱신 (16:48 KST)**: 제목 "BE/FE 협의용" 추가 (사용자 v4 권고 3개 — `reason_codes` / `base_product` / `compare_product` top-level 추가 안 함 — v6에 이미 반영됨 확인)
- P0: diff/exclusions path prefix 통일 (`/internal/v1/nutrition/...`)
- P0: ID 타입 `string (UUID 권고)`
- P0: `life_stage` = 반려동물 생애주기 (예: "ADULT")
- P0: `product_life_stage` = 상품의 AAFCO 매핑 생애주기 (예: "ADULT_MAINTENANCE")
- P1: `is_match` 의미 명확화 (두 생애주기의 매핑 일치 여부)
- P1: §7 API별 응답 데이터 (analyze 핵심 필드, diff difference 정의 + null 처리, exclusions 핵심 필드)
- P2: §8 분석 처리 흐름 (입력 검증 → 조회 → 정규화 → 비교 → 안전성 → 결과 → 설명 → 검증 → 반환)
**P0-D 평가 데이터 (TRUE/FALSE/UNKNOWN)**: 평가 문서에서 관리 (응답 미포함)
**상위 정합**: `docs/zep_산출물2_영양분석_v3.md` §1
**placeholder**: `단계/3단계_구현/API/nutrition_api_spec.md`

---

## 1. 엔드포인트 목록

| 엔드포인트 | Method | 용도 |
|---|---|---|
| `/internal/v1/nutrition/analyze` | POST | 반려동물 프로필 기준 상품 1개 영양성분 분석 |
| `/internal/v1/nutrition/diff` | GET | 두 상품의 영양성분 및 기준 비교 |
| `/internal/v1/nutrition/exclusions` | GET | 알레르기·대상 종·생애주기 기준 상품 제외 사유 조회 |

---

## 2. 공통 요청 파라미터

### `/internal/v1/nutrition/analyze`

| 파라미터 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `pet_id` | string (UUID 권고) | O | 분석 대상 반려동물 ID |
| `product_id` | string (UUID 권고) | O | 분석 대상 상품 ID |

### `/internal/v1/nutrition/diff`

| 파라미터 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `pet_id` | string (UUID 권고) | O | 비교 기준 반려동물 ID |
| `product_id` | string (UUID 권고) | O | 기준 상품 ID |
| `compare_product_id` | string (UUID 권고) | O | 비교 상품 ID |

### `/internal/v1/nutrition/exclusions`

| 파라미터 | 타입 | 필수 | 설명 |
|---|---|---|---|
| `pet_id` | string (UUID 권고) | O | 안전성 검증 대상 반려동물 ID |
| `product_id` | string (UUID 권고) | O | 검증 대상 상품 ID |

**※ 실제 ID 발급 형식은 BE의 ID 컨벤션에 따라 최종 확정한다.**

---

## 3. 공통 응답 형식

모든 API는 다음 공통 응답 구조를 사용한다.

```json
{
  "request_id": "req_01JABC789",
  "status": "READY",
  "data": {},
  "warnings": []
}
```

### `status` (5값)

| 값 | 설명 |
|---|---|
| `READY` | 정상 분석 완료 |
| `SAFETY_BLOCKED` | 안전성 기준에 의해 상품이 차단됨 |
| `INSUFFICIENT_DATA` | 판단에 필요한 데이터 부족 |
| `SERVICE_DEGRADED` | 일부 기능 제한 상태 |
| `FAILED` | 분석 처리 실패 |

### 영양성분별 판정

- `comparison_status`: `IN_RANGE` / `OUT_OF_RANGE` / `UNKNOWN` / `NOT_APPLICABLE` (4값)
- `display_grade`: `ADEQUATE` / `DEFICIENT` / `EXCESS` / `UNKNOWN` (4값)
- `confidence`: `HIGH` / `MEDIUM` / `LOW` / `INSUFFICIENT_DATA` (4값)

### 전체 판단

- `overall_judgment`: `IN_RANGE` / `CAUTION` / `NOT_RECOMMENDED` / `INSUFFICIENT_DATA` (4값)

### 주요 제외 사유 (`reason_code`)

- `ALLERGY_CONFLICT`
- `SPECIES_MISMATCH`
- `LIFE_STAGE_MISMATCH`
- `SAFETY_DATA_INSUFFICIENT`

### AAFCO 적합성 (`aafco_match` 내부 4필드)

`aafco_match` 내부에서 다음 정보를 제공한다.

```json
{
  "aafco_match": {
    "life_stage": "ADULT",
    "product_life_stage": "ADULT_MAINTENANCE",
    "is_match": true,
    "aafco_pass_per_product": {
      "policy": "P0D",
      "policy_status": "CURRENT",
      "value": "UNKNOWN",
      "note": "필수 영양소 데이터 부족으로 적합성 판정 불가"
    }
  }
}
```

- `life_stage`: **반려동물의 생애주기** (BE가 pet profile에서 조회, AI에 전달)
- `product_life_stage`: **상품의 AAFCO 매핑 생애주기** (예: 반려동물 ADULT → 상품 매핑 ADULT_MAINTENANCE)
- `is_match`: 반려동물 생애주기와 상품 생애주기의 **매핑 규칙상 일치 여부**
- `aafco_pass_per_product`: 4필드 (`policy` / `policy_status` / `value` / `note`)
  - `policy`: `P0D` (current) / `MIN_1_PASS` (legacy, deprecated)
  - `policy_status`: `CURRENT` / `LEGACY` / `DEPRECATED` (3값)
  - `value`: `TRUE` / `FALSE` / `UNKNOWN` (3값)
  - `note`: UNKNOWN 등 판정 불가 사유 1줄 설명

### 유지 결정

다음은 의도적 단순화 — API 응답에 노출하지 않음:

- `canonical_status` (KNOWN / INVALID / UNKNOWN / NOT_APPLICABLE) — 내부 데이터 품질 상태
- `derived_reference_life_stage` — 백엔드 매핑 로직에서 처리
- `reference_version` 단일 필드 (5필드 `model_version` / `pipeline_version` / `feature_version` / `reference_matrix_version` / `reference_policy_version` → `reference_version`)
- `P0-D 평가 데이터` (TRUE/FALSE/UNKNOWN 건수) — 평가 문서 (`docs/BASELINE_DECISION_v3.md` §1.3) 에서 관리
- `generated_at` ISO 8601 KST 유지

---

## 4. 에러 응답

| HTTP Status | 상황 |
|---|---|
| 400 | 잘못된 요청 파라미터 |
| 401 | 인증 실패 |
| 403 | 접근 권한 없음 |
| 404 | 반려동물 또는 상품 정보 없음 |
| 422 | 요청 데이터 형식 또는 유효성 오류 |
| 500 | 서버 내부 오류 |
| 503 | AI 서비스 또는 외부 의존 서비스 이용 불가 |

**※ `SAFETY_BLOCKED`, `INSUFFICIENT_DATA` 는 HTTP 오류가 아니라 HTTP 200 응답 내 `status` 로 전달되는 도메인 상태이다.**

---

## 5. 갱신 주기

### 상품 영양성분 / 원재료 / AAFCO 기준 데이터

- 원천 데이터 변경 또는 기준 버전 변경 시 갱신

### 영양성분 분석 결과

- 상품 및 반려동물 정보가 변경되지 않은 경우 캐시 활용
- 상품 정보, 반려동물 알레르기·생애주기 정보 또는 기준 버전 변경 시 캐시 무효화 후 재분석

### `/internal/v1/nutrition/analyze`

- 분석 결과 캐시 우선 사용
- 관련 데이터 변경 시 캐시 무효화

### `/internal/v1/nutrition/diff`

- 비교 결과 캐시 사용 가능
- 비교 대상 상품 정보 또는 기준 데이터 변경 시 캐시 무효화

### `/internal/v1/nutrition/exclusions`

- 안전성 판단 API 이므로 캐시 미사용
- 장바구니 및 결제 직전 실시간 재검증

### 분석 방식

- 영양성분 및 안전성 판단은 **구조화된 데이터와 사전 정의된 룰 기반**으로 수행
- LLM은 영양성분 수치 및 안전성 판정에 사용하지 않음
- 필요한 경우 분석 결과를 사용자 친화적인 문장으로 변환하는 **설명 계층에 선택적 활용**
- LLM 미사용 또는 장애 시에도 정형화된 템플릿으로 설명 제공

---

## 6. 협의 필요 사항

| # | 항목 | BE/FE 협의 |
|---|---|---|
| 1 | API Base URL 및 인증 방식 | BE와 최종 확정 |
| 2 | 캐시 저장 위치 및 TTL | BE 인프라 기준으로 확정 |
| 3 | 상품·반려동물·기준 데이터 변경 시 관련 캐시 무효화 | BE와 이벤트 연동 방식 협의 |
| 4 | `/internal/v1/nutrition/exclusions` 호출 시점 | 상품 상세 조회 시 기본 검증 + 장바구니·결제 직전 재검증 |
| 5 | `/internal/v1/nutrition/diff` 화면 반영 범위 | 비교할 영양소 및 사용자 노출 항목 FE와 협의 |
| 6 | 데이터 변경 이벤트 전달 방식 | 상품/반려동물/기준 데이터 변경 시 분석 캐시 무효화를 위한 BE 연동 방식 협의 |
| 7 | 분석 결과의 사용자 노출 방식 | `overall_judgment`, 영양소별 `display_grade`, `reason_code` 및 설명 문구의 FE 노출 범위 협의 |
| 8 | 대상 종 및 생애주기 매핑 규칙 | `target_species` 및 `LIFE_STAGE_MISMATCH` 처리 기준 BE/AI 간 최종 확정 |
| 9 | 영양성분 기준 단위 | Dry Matter Basis (DMB) 적용 및 단위 변환 기준 최종 확정 |
| 10 | 처리 방식 | API 동기/비동기 처리 여부 및 응답 시간 기준 BE와 협의 |

---

## 7. API별 주요 응답 데이터

### 7.1 `/internal/v1/nutrition/analyze`

- `overall_judgment`: 전체 영양 적합성 판단 (`IN_RANGE` / `CAUTION` / `NOT_RECOMMENDED` / `INSUFFICIENT_DATA`)
- `confidence`: 분석 신뢰도 (`HIGH` / `MEDIUM` / `LOW` / `INSUFFICIENT_DATA`)
- `aafco_match`: 반려동물 생애주기와 상품 기준 생애주기 매칭 결과
  - `life_stage` / `product_life_stage` / `is_match` / `aafco_pass_per_product` (4필드)
- `allergy_check`: 알레르기 충돌 여부
  - `status` (`OK` / `CONFLICT`) / `matched_allergens[]` (allergen_code + detected_keyword)
- `nutrients`: 영양성분별 기준 비교 결과 배열
  - 각 항목: `nutrient_code` / `nutrient_name` / `actual_value` / `actual_unit` / `reference_min` / `reference_max` / `comparison_status` / `display_grade` / `reason_text`
- `excluded`: 추천/분석 대상에서 제외된 경우 사유 배열 (`reason_code` + `reason_text`)
- `warnings`: 데이터 부족·주의사항 배열 (`warning_code`)
- `persona_friendly`: 보호자 화면용 쉬운 설명 (`summary` + `easy_terms` 객체)
- `reference_version`: 적용된 영양 기준 버전 (단일 필드)
- `generated_at`: 분석 생성 시각 (ISO 8601 KST)

### 7.2 `/internal/v1/nutrition/diff`

- 기준 상품과 비교 상품의 영양성분 비교
- **`difference = compare_value - base_value`** (계산식)
  - 양수: 비교 상품의 수치가 더 높음
  - 음수: 비교 상품의 수치가 더 낮음
  - **한쪽 상품의 수치가 없으면 `difference = null`** (예: `base_value=28.0, compare_value=null, difference=null`)
- `aafco_comparison`: 두 상품의 AAFCO 기준 비교 결과
  - `base_product_pass` / `compare_product_pass` (각각 `TRUE` / `FALSE` / `UNKNOWN`)
- `allergy_check`: 두 상품의 알레르기 안전성 비교
  - `base_product` / `compare_product` (각각 `{status, matched_allergens[]}` 객체)
- `nutrients[]`: 비교 가능한 영양소 목록
  - 각 항목: `nutrient_code` / `nutrient_name` / `base_value` / `compare_value` / `unit` / `difference`
- `comparison_status`: 영양소별 비교 결과 (`IN_RANGE` / `OUT_OF_RANGE` / `UNKNOWN` / `NOT_APPLICABLE`)
- `warnings`: 데이터 부족·주의사항

### 7.3 `/internal/v1/nutrition/exclusions`

- `is_excluded` (boolean): 상품 제외 여부
- `exclusion_reasons[]`: 제외 사유 목록
  - 각 항목: `reason_code` + `reason_text`
  - `reason_code` 4값:
    - `ALLERGY_CONFLICT` — 알레르기 원료 충돌
    - `SPECIES_MISMATCH` — 대상 종 불일치
    - `LIFE_STAGE_MISMATCH` — 생애주기 불일치
    - `SAFETY_DATA_INSUFFICIENT` — 안전성 판단 데이터 부족
- `exclusion_reasons = []` and `is_excluded = false`: 제외 사유 없음
- **결제 직전 재검증**: `/internal/v1/nutrition/exclusions` 는 안전성 판단 API 이므로 캐시 미사용, 장바구니·결제 직전 실시간 재호출

---

## 8. 분석 처리 흐름

```
BE 요청
    ↓
입력값 검증 (파라미터, 필수 필드, enum)
    ↓
반려동물 / 상품 데이터 조회
    ↓
영양성분 및 원재료 정규화 (단위·basis·생애주기 매핑)
    ↓
AAFCO / NIAS 기준 조회 (영양소별 reference_min / reference_max)
    ↓
영양성분 기준 비교 (comparison_status, display_grade)
    ↓
알레르기 / 대상 종 / 생애주기 안전성 검사
    ↓
구조화된 분석 결과 생성 (aafco_match.aafco_pass_per_product, overall_judgment)
    ↓
필요 시 사용자 친화적 설명 생성 (persona_friendly)
    ↓
결과 검증 (정형 데이터 일치 확인, LLM 근거 검증)
    ↓
BE 반환
```

**책임 분리**:

- **BE** (전·후): 입력값 검증, 반려동물/상품 원본 데이터 조회, 응답 캐싱, 인증
- **AI 서버** (중간): 정규화 → 기준 비교 → 안전성 검사 → 구조화된 결과 → LLM 설명 (선택)
- **BE/FE** (후): 응답 wrapper 처리, 화면 표시

**LLM 사용 범위** (선택적):

- **사용 O**: `persona_friendly.summary` / `easy_terms` 생성
- **사용 X**: 영양소 수치 비교 / 안전성 판정 / `aafco_match` / `overall_judgment`
- **장애 시 fallback**: 정형화된 템플릿 문장으로 대체 (예: "조단백질 함량이 기준 범위 내입니다.")

**P0-D 정책**:
- `aafco_pass_per_product.value` 는 상품 단위 (1 product = 1 value)
- 전체 평가 데이터 (T=0/F=1/U=151) 는 API 응답에 미포함, 평가 문서 (`docs/BASELINE_DECISION_v3.md` §1.3) 에서 관리

---

**END of nutrition_ai_api_spec_v3_draft.md (v7, BE/FE 협의용 확정, 2026-09-04 16:48 KST)**
