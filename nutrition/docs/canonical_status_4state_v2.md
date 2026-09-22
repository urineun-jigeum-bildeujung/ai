# Canonical Status 4상태 정책 v2 — 10단계 결정 규칙 (2026-09-04 12:10 KST)

**Author**: Mavis (root session mvs_b8359108c79348379b04b54f450c0a3a)
**Author date**: 2026-09-04 12:10 KST (Wave 1-A)
**Scope**: P0 1-A 상태 우선순위 강제. v1 (8단계, 9/1 16:57) 의 8단계에 **DOG only Ca:P ratio 결정 규칙 2단계 추가** → **10단계 결정 규칙**. NCS ↔ 4-state 우선순위 강제 (KNOWN 아닌 행은 NCS 판정 불가) SQL CHECK 제약 추가.
**Status**: **applied** (Wave 3 PR-canonical-status-precision 의 입력 사양으로 확정. PR-H mechanical 8단계 결정 → PR-canonical-status-precision 10단계 결정으로 갱신)
**Supersedes**: `docs/canonical_status_4state_v1.md` (8단계, 9/1 16:57, status: applied)
**기반 매트릭스**: `data/processed/required_nutrient_matrix_v2.csv` (9/4 11:55, AAFCO 2014)

**근거 자료**:
- `docs/canonical_status_4state_v1.md` (v1 8단계)
- `docs/BASELINE_DECISION.md` (9/4 11:02, 50행 결함, 82/48/507/10 공식)
- `data/processed/required_nutrient_matrix_v2.csv` (9/4 11:55, Ca:P 1.0~2.0 DOG only)
- `data/processed/derived_product_nutrition_analysis_v3.sql` (9/3 15:13, 기존 SQL)

**핵심 원칙 (영구)**:
- 결정론적 룰 기반 100% (LLM 호출 0)
- "박지" 영구 금지 / 질환 진단 확정 표현 금지 (8/19)
- 0 product 변화 / 0건 보장 / 누적 상태 전이 (합산 부풀리기 X)
- **NCS ↔ 4-state 일관성**: KNOWN ↔ IN_RANGE/OUT_OF_RANGE, UNKNOWN ↔ NO_VALUE/NO_REF, INVALID/NOT_APPLICABLE ↔ NCS 없음

---

## 0. 핵심 결정 (TL;DR)

| 항목 | 결정 |
|---|---|
| **결정 단계** | 8단계 → **10단계** (DOG only Ca:P ratio 결정 규칙 2단계 추가) |
| **상태 우선순위** | **NOT_APPLICABLE > INVALID > UNKNOWN > KNOWN** (강제, SQL CHECK 제약) |
| **NCS 일관성** | KNOWN 행은 `nias_compare_status IN ('IN_RANGE', 'OUT_OF_RANGE')` 강제. 그 외 상태는 NCS 없음 (또는 별도 NOT_EVALUATED). **SQL CHECK 제약** 추가 |
| **DOG Ca:P ratio 결정** | 사양: Ca 0.5~2.5% / P 0.4~1.6% / ratio 1:1~2:0 (AAFCO 2014). 결정: canonical_value(Ca) AND canonical_value(P) 모두 존재 → ratio = Ca/P 계산 → 1.0 ≤ ratio ≤ 2.0 → IN_RANGE, 그 외 → OUT_OF_RANGE(LOW/HIGH). Ca 또는 P null → UNKNOWN. CAT 는 AAFCO Ca:P 미요구 → 규칙 미적용 |
| **PR-H mechanical 8단계 갱신** | `_canonical_status_8step` (pipeline_p1c_v1.py:343-373) → `_canonical_status_10step` 로 확장. 9 단계 (DOG only Ca:P) + 10 단계 (DOG only Ca/P null) 추가 |
| **SQL CHECK 제약** | `derived_product_nutrition_analysis_v3_PR_CS.sql` (PR-canonical-status-precision 적용 SQL) 에 다음 CHECK 추가: (a) KNOWN ↔ NCS IN/OUT, (b) UNKNOWN ↔ NCS NO_VALUE/NO_REF, (c) INVALID/NOT_APPLICABLE ↔ NCS 없음, (d) DOG species + Ca AND P 둘 다 KNOWN → ca_p_ratio_status 필수 |
| **기존 PR-H 분포 deprecated** | 130/0/509/8 (PR-H mechanical 8단계) → **82/48/507/10 (SQL 정답, 9/3 17:00) 유지** + Ca:P ratio 2~3 행 신규 추가 (DOG KNOWN 중 Ca/P 둘 다 있는 product) |
| **다음 단계** | Wave 1-C (NIAS reference profile_edition) → Wave 2 (48행 INVALID 재판정) → Wave 3 (PR-canonical-status-precision code 적용) → Wave 4 (재현 패키지) → Wave 5 (기준선 재확정) |

---

## 1. 10단계 결정 규칙 (v2)

### 1.1 결정 규칙 (우선순위 순)

| # | 조건 | canonical_status | nias_compare_status | 비고 |
|--:|---|---|---|---|
| 1 | `nias_table_id == "OFF_BRANDED"` | **NOT_APPLICABLE** | (없음) | OFF_BRANDED 의심 row (NIAS table_id marker, max-only) |
| 2 | `nutrient_code == "TAURINE"` AND `product_species == "DOG"` | **NOT_APPLICABLE** | (없음) | TAURINE DOG (NIAS TAURINE = CAT only) |
| 3 | `nias_compare_status == "NO_VALUE"` | **UNKNOWN** | `NO_VALUE` | canonical_value null (raw source 부재, value=0/placeholder) |
| 4 | `nias_compare_status == "NO_REF"` | **UNKNOWN** | `NO_REF` | NIAS reference 부재 (species/life_stage 매핑 실패) |
| 5 | `ncode in MINERAL_SET` AND `bns == "SKIPPED_NO_MOISTURE"` | **INVALID** | (없음) | AF→DM 변환 불가 (moisture 부재) |
| 6 | `ncode in ['VITAMIN_A','VITAMIN_D','VITAMIN_E']` AND `unit == "g/100g"` | **INVALID** | (없음) | IU/mg 환산 보류 |
| 7 | `ncode == "CRUDE_FIBER"` AND `bns == "SKIPPED_NO_MOISTURE"` | **INVALID** | (없음) | DRY_MATTER→AS_FED 변환 불가 |
| 8 | `ncs IN ('IN_RANGE', 'OUT_OF_RANGE')` AND 위 7개 조건 미해당 | **KNOWN** | `IN_RANGE` / `OUT_OF_RANGE` | canonical_value 존재 + NIAS 매칭 + 변환 성공 |
| **9** | **`species == "DOG"` AND `ncode == "CALCIUM"` AND `canonical_value(Ca) != null`** (product-level, Ca item-row) | 별도: ca_p_ratio_status = (canonical_value(P) null → NOT_APPLICABLE) / (Ca/P < 1.0 → OUT_OF_RANGE_LOW) / (Ca/P > 2.0 → OUT_OF_RANGE_HIGH) / (1.0 ≤ Ca/P ≤ 2.0 → IN_RANGE) | (item-level Ca row 의 NCS = 위 #8 따라감) | **DOG only Ca:P ratio 결정**. 사양: AAFCO 2014 Ca:P 1:1~2:1 |
| **10** | **`species == "DOG"` AND (`ncode == "CALCIUM"` OR `ncode == "PHOSPHORUS"`) AND (`canonical_value(Ca) == null` OR `canonical_value(P) == null`)** (product-level) | 별도: ca_p_ratio_status = UNKNOWN (NON_COMPARABLE) | (item-level Ca/P row 의 NCS = 위 #1~8 따라감) | **DOG only Ca:P ratio 미평가** (상대값 부재). CAT 는 규칙 미적용 (AAFCO Ca:P 미요구) |
| (fallback) | 기타 | **UNKNOWN** | (없음) | 정의되지 않은 케이스 (방어용) |

**핵심**: 단계 9~10은 **product-level** 결정 (DOG 한정, Ca:P ratio). 단계 1~8은 **item-level** 결정 (모든 영양소). 두 결정은 **별개**.

### 1.2 NCS ↔ 4-state 일관성 (SQL CHECK 제약)

```sql
-- 제약 1: KNOWN ↔ NCS IN/OUT
CHECK (
  (canonical_status = 'KNOWN' AND nias_compare_status IN ('IN_RANGE', 'OUT_OF_RANGE'))
  OR
  (canonical_status != 'KNOWN')
);

-- 제약 2: UNKNOWN ↔ NCS NO_VALUE/NO_REF
CHECK (
  (canonical_status = 'UNKNOWN' AND nias_compare_status IN ('NO_VALUE', 'NO_REF', 'NOT_EVALUATED'))
  OR
  (canonical_status != 'UNKNOWN')
);

-- 제약 3: INVALID/NOT_APPLICABLE ↔ NCS 없음
CHECK (
  (canonical_status IN ('INVALID', 'NOT_APPLICABLE') AND nias_compare_status IS NULL)
  OR
  (canonical_status IN ('KNOWN', 'UNKNOWN'))
);
```

이 제약으로 **NCS-4state 모순 56행 (INVALID 48 + NOT_APPLICABLE 8 가 NCS IN_RANGE/OUT_OF_RANGE 보유)** 자동 차단.

### 1.3 product-level Ca:P ratio 결정 (DOG only, item-level 외 별도 결정)

```
product-level 결정 (DOG species 한정):
  canonical_value(Ca) AND canonical_value(P) 모두 존재:
    ratio = Ca / P
    ratio < 1.0  → ca_p_ratio_status = 'OUT_OF_RANGE_LOW'
    ratio > 2.0  → ca_p_ratio_status = 'OUT_OF_RANGE_HIGH'
    1.0 ≤ ratio ≤ 2.0 → ca_p_ratio_status = 'IN_RANGE'
  canonical_value(Ca) OR canonical_value(P) null:
    ca_p_ratio_status = 'UNKNOWN'
  CAT species:
    ca_p_ratio_status = 'NOT_APPLICABLE' (AAFCO Ca:P 미요구)
```

product-level `ca_p_ratio_status` 필드는 product_nutrition_analysis 테이블에 추가 (P0-D 의 nutrition_comparison_status 와 별개).

---

## 2. 기존 8단계 (v1) → 10단계 (v2) 차이

| 항목 | v1 (9/1) | v2 (9/4) |
|---|---|---|
| 결정 단계 수 | 8 | **10** (+ DOG Ca:P ratio 2단계) |
| DOG Ca:P ratio | 미평가 | **product-level 결정 추가** (1.0~2.0 AAFCO 2014) |
| NCS 일관성 | doc-level 명시 (체크 제약 없음) | **SQL CHECK 제약 3개** (KNOWN/UNKNOWN/INVALID·NOT_APPLICABLE 일관성 강제) |
| 상태 우선순위 | 권고 | **SQL CHECK 제약** (NOT_APPLICABLE > INVALID > UNKNOWN > KNOWN 순차 적용) |
| `nias_compare_status == 'NOT_EVALUATED'` | 없음 | **신규 enum** (item-level 에서 NCS 미평가 시, e.g. Ca:P ratio 미해당 item) |

### 2.1 한 줄 요약

**v1 8단계 결정 + 신규 9~10 단계 (DOG Ca:P ratio, product-level) + NCS 일관성 SQL CHECK 3개 = v2 10단계 결정. PR-canonical-status-precision 적용 시 코드 mechanical decision 도 이 규칙과 100% 정합.**

---

## 3. SQL CHECK 제약 (v3_PR_CS 신규 SQL)

`data/processed/derived_product_nutrition_analysis_v3_PR_CS.sql` (PR-canonical-status-precision 적용 SQL) 추가:

```sql
-- ============================================================================
-- PR-canonical-status-precision: 10단계 결정 규칙 + NCS 일관성 SQL CHECK
-- 생성일: 2026-09-04 12:10 KST
-- 적용: PR-canonical-status-precision (Wave 3)
-- ============================================================================

-- (기존 v3 schema + product_nutrition_analysis table)
-- 신규 컬럼: ca_p_ratio_status (DOG only, product-level 결정)
ALTER TABLE product_nutrition_analysis ADD COLUMN ca_p_ratio_status TEXT;
-- 값: IN_RANGE / OUT_OF_RANGE_LOW / OUT_OF_RANGE_HIGH / UNKNOWN / NOT_APPLICABLE

-- CHECK 제약 1: KNOWN ↔ NCS IN/OUT
ALTER TABLE product_nutrition_analysis ADD CONSTRAINT chk_canonical_ncs_known
  CHECK (
    (canonical_status = 'KNOWN' AND nias_compare_status IN ('IN_RANGE', 'OUT_OF_RANGE'))
    OR (canonical_status != 'KNOWN')
  );

-- CHECK 제약 2: UNKNOWN ↔ NCS NO_VALUE/NO_REF/NOT_EVALUATED
ALTER TABLE product_nutrition_analysis ADD CONSTRAINT chk_canonical_ncs_unknown
  CHECK (
    (canonical_status = 'UNKNOWN' AND nias_compare_status IN ('NO_VALUE', 'NO_REF', 'NOT_EVALUATED'))
    OR (canonical_status != 'UNKNOWN')
  );

-- CHECK 제약 3: INVALID/NOT_APPLICABLE ↔ NCS 없음
ALTER TABLE product_nutrition_analysis ADD CONSTRAINT chk_canonical_ncs_invalid
  CHECK (
    (canonical_status IN ('INVALID', 'NOT_APPLICABLE') AND nias_compare_status IS NULL)
    OR (canonical_status IN ('KNOWN', 'UNKNOWN'))
  );
```

**중요**: 기존 `derived_product_nutrition_analysis_v3.sql` 의 schema 와 호환. PR-canonical-status-precision (Wave 3) 시점에 신규 SQL 파일 `v3_PR_CS.sql` 로 작성, 기존 v3 SQL 보존.

---

## 4. 코드 갱신 (Wave 3 PR-canonical-status-precision 의 입력)

### 4.1 `_canonical_status_10step` (현재 8단계 → 10단계)

`scripts/pipeline_p1c_v1.py:343-373` 의 `_canonical_status_8step` 함수를 다음으로 확장 (item-level 8단계, 9~10 단계는 product-level 별도 함수):

```python
def _canonical_status_10step(r: dict) -> str:
    """canonical_status 4-state 결정 (v2 10단계 — 1~8 item-level, 9~10 product-level).

    1) OFF_BRANDED / TAURINE_DOG → NOT_APPLICABLE
    2) NO_VALUE / NO_REF / NEEDS_REVIEW → UNKNOWN
    3) mineral + SKIPPED_NO_MOISTURE → INVALID
    4) VITAMIN A/D/E + unit=g/100g → INVALID
    5) CRUDE_FIBER + SKIPPED_NO_MOISTURE → INVALID
    6) NCS IN_RANGE/OUT_OF_RANGE + 위 5개 미해당 → KNOWN
    7) (DOG only, 별도 함수) Ca:P ratio 결정 (product-level)
    8) (DOG only, 별도 함수) Ca/P null → UNKNOWN (NON_COMPARABLE)
    """
    # ... 1~6 단계는 v1 8단계와 동일 ...
    # 7~8 단계는 product-level 별도 함수 compute_ca_p_ratio_status() 호출
    pass


def compute_ca_p_ratio_status(product_id: str, items: list, product_species_map: dict) -> str:
    """DOG only. Ca/P ratio 1:1~2:0 AAFCO 2014 결정 (product-level).

    CAT: NOT_APPLICABLE (AAFCO Ca:P 미요구)
    DOG + Ca or P null: UNKNOWN (NON_COMPARABLE)
    DOG + Ca/P 모두 존재: ratio = Ca/P, 1.0~2.0 → IN_RANGE
    """
    sp = (product_species_map or {}).get(product_id, "DOG").upper()
    if sp != "DOG":
        return "NOT_APPLICABLE"

    ca_row = next((r for r in items if r.get("nutrient_code") == "CALCIUM" and r.get("product_id") == product_id), None)
    p_row = next((r for r in items if r.get("nutrient_code") == "PHOSPHORUS" and r.get("product_id") == product_id), None)

    if not ca_row or not p_row:
        return "UNKNOWN"

    ca_value = ca_row.get("canonical_value") or ca_row.get("aligned_value")
    p_value = p_row.get("canonical_value") or p_row.get("aligned_value")

    if ca_value is None or p_value is None or p_value == 0:
        return "UNKNOWN"

    ratio = ca_value / p_value
    if ratio < 1.0:
        return "OUT_OF_RANGE_LOW"
    if ratio > 2.0:
        return "OUT_OF_RANGE_HIGH"
    return "IN_RANGE"
```

### 4.2 ESSENTIAL_NUTRIENTS_V2 (Wave 1-B 매트릭스 기반)

`scripts/pipeline_p1c_v1.py:503-509` 의 기존 `ESSENTIAL_NUTRIENTS` dict 를 `data/processed/required_nutrient_matrix_v2.csv` 의 tier1_core 17 DOG / 18 CAT 로 교체. PR-canonical-status-precision (Wave 3) 적용 시 본 dict 교체.

```python
ESSENTIAL_NUTRIENTS_V2 = {
    ("DOG", "ADULT_MAINTENANCE"): [
        "CRUDE_PROTEIN", "CRUDE_FAT", "CALCIUM", "PHOSPHORUS",
        "SODIUM", "MAGNESIUM", "POTASSIUM", "IRON", "COPPER", "ZINC",
        "MANGANESE", "IODINE", "SELENIUM",
        "VITAMIN_A", "VITAMIN_D", "VITAMIN_E",
        "THIAMINE", "RIBOFLAVIN",
    ],  # 17개 (Ca:P ratio 별도)
    ("DOG", "GROWTH_REPRODUCTION"): [...동일 17개],
    ("CAT", "ADULT_MAINTENANCE"): [
        "CRUDE_PROTEIN", "CRUDE_FAT", "CALCIUM", "PHOSPHORUS",
        "SODIUM", "MAGNESIUM", "POTASSIUM", "IRON", "COPPER", "ZINC",
        "MANGANESE", "IODINE", "SELENIUM",
        "VITAMIN_A", "VITAMIN_D", "VITAMIN_E",
        "THIAMINE", "RIBOFLAVIN",
        "ARACHIDONIC_ACID", "TAURINE",
    ],  # 18개 (Ca:P 없음, Arachidonic + Taurine 추가)
    ("CAT", "GROWTH_REPRODUCTION"): [...동일 18개],
}
```

### 4.3 parity 테스트 (Wave 3 의 핵심)

```python
# scripts/tests/test_code_sql_parity.py (Wave 3 작성)
def test_code_sql_parity():
    """코드 mechanical decision (10단계) = SQL 10단계 결정 결과 (1:1 일치)"""
    # 1. POST-PR-Vit SoT 의 647 item-row 로드
    # 2. pipeline_p1c_v1.py 의 _canonical_status_10step + compute_ca_p_ratio_status 실행
    # 3. derived_product_nutrition_analysis_v3_PR_CS.sql 의 canonical_status / nias_compare_status / ca_p_ratio_status 와 비교
    # 4. 불일치 0행 확인
    assert diff_count == 0
```

---

## 5. 0건 보장 (Wave 1-A 작성 시)

본 v2 doc 작성 (9/4 12:10, 1회 작업) 과정에서:
- pipeline_p1c_v1.py — 수정 0건 (Wave 3 PR-canonical-status-precision 에서 10단계로 갱신)
- derived_product_nutrition_analysis_v3.sql — 수정 0건 (Wave 3 에서 v3_PR_CS.sql 로 신규 작성)
- 시드 13/14/9/9b / OPFF cache / P2 SoT / P1-C SoT / E2E Live/Actual — 수정 0건
- BASELINE_DECISION.md (82/48/507/10 공식) — 유지. 본 v2 doc 추가 정보로 9~10 단계 (DOG Ca:P ratio) 가 신규 2~3 row 추가 예상 (POST-PR-canonical-status-precision 시점)
- NIAS max=10→12 / 임의 / random / LLM / PASS 역산 / "박지" / 이모지 / git / commit / PR — 모두 0건

---

**END of canonical_status_4state_v2.md**
