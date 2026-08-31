# UCI 전처리 검증 결과

## 행 보존 및 사용 범위

| 항목 | 건수 | 비율 |
| --- | ---: | ---: |
| 원본 행 | 1,067,371 | 100.00% |
| 재구매 사용 가능 | 802,713 | 75.20% |
| 격리 | 264,658 | 24.80% |
| 금액 분석 사용 가능 | 802,651 | 75.20% |

## 품질 사유

| 사유 코드 | 기록 수 |
| --- | ---: |
| `MISSING_USER_ID` | 243,007 |
| `SUSPECTED_DUPLICATE` | 23,430 |
| `NONPOSITIVE_QUANTITY` | 22,950 |
| `EXPLICIT_CANCELLATION` | 19,494 |
| `NONPOSITIVE_PRICE` | 6,207 |
| `NON_MERCHANDISE_CODE` | 5,653 |
| `MISSING_PRODUCT_NAME` | 4,382 |

- 전체 사유 기록: 325,123건
- 사유가 2개 이상인 행: 27,550건
- 한 행의 최대 사유 수: 4개

## 불변조건 검증

| 검증 조건 | 결과 |
| --- | --- |
| `row_count_preserved` | 통과 |
| `source_ids_preserved` | 통과 |
| `accepted_plus_quarantined_equals_source` | 통과 |
| `quarantined_equals_exclude_reason_ids` | 통과 |
| `accepted_has_no_exclude_reason` | 통과 |
| `reason_ids_exist_in_source` | 통과 |

## 비상품 코드 정책

문자형 상품 코드를 일괄 제외하지 않고 원본에서 비상품으로 확인된 코드만 명시적으로 관리합니다.

`ADJUST`, `ADJUST2`, `BANK CHARGES`, `C2`, `CRUK`, `D`, `DOT`, `M`, `POST`, `TEST001`, `TEST002`

중복 후보는 자동 삭제하지 않고 경고로 남깁니다. 동일 주문·상품은 후속 구매 사건 생성 단계에서 하나의 사건으로 집계합니다.
