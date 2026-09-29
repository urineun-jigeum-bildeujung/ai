# Reference Edition Policy v1 — NIAS / AAFCO reference row 의 profile_edition / publication_year 분리 (2026-09-04 12:15 KST)

**Author**: Mavis (root session mvs_b8359108c79348379b04b54f450c0a3a)
**Author date**: 2026-09-04 12:15 KST (Wave 1-C)
**Scope**: P0 1-C 기준 판 표기 분리. NIAS / AAFCO reference row 의 `profile_edition` (프로필 개정판) 와 `publication_year` (OP 발행 연도) 를 별도 필드로 분리. 기존 `reference_version` enum 의 hard-coded 'NIAS_2024' → 행별 출처 기록으로 대체.
**Status**: **applied** (Wave 3 PR-canonical-status-precision 의 입력 사양으로 확정. NIAS v5 (8/27 17:43) 와 AAFCO 2014 Nutrient Profiles (Table A-1~A-4) 의 출처 분리)
**Supersedes**: `data/processed/derived_product_nutrition_analysis_v3.sql` 의 `reference_version TEXT DEFAULT 'NIAS_2024_v5'` (hard-coded, 행별 출처 없음)

**근거 자료**:
- `data/raw/seed_14_nutrition_reference_v5.json` (NIAS v5 8/27 17:43, 325 row, 18 nutrient)
- AAFCO 2014 Nutrient Profiles (2014 OP Appendix A, revised 092214) — 본 매트릭스 v2 의 출처
- `data/processed/required_nutrient_matrix_v2.csv` (9/4 11:55, 15 컬럼 중 profile_edition / publication_year 컬럼 13-14)
- `docs/canonical_data_contract_v1.md` (v1.2, 9/1 12:14, NIAS unit 표기)

**핵심 원칙 (영구)**:
- 결정론적 룰 기반 100% (LLM 호출 0)
- "박지" 영구 금지 / 질환 진단 확정 표현 금지 (8/19)
- 0 product 변화 / 0건 보장

---

## 0. 핵심 결정 (TL;DR)

| 항목 | 결정 |
|---|---|
| **profile_edition 정의** | **프로필 개정판** (예: `AAFCO 2014 Nutrient Profiles`, `NIAS 2024_v5`, `FEDIAF 2024 Nutritional Guidelines`). reference 값의 출처가 되는 "프로필 이름" |
| **publication_year 정의** | **OP 발행 연도** (AAFCO 2024 OP, NIAS 2024_v5 8/27, FEDIAF 2024 등). profile 이 어느 OP/문서에 실려 있는지의 "발행 시점" |
| **두 필드 분리** | 동일 profile 이 여러 OP 에 인용 가능 (예: AAFCO 2014 Nutrient Profiles 는 2014~2024 OP 에 모두 포함). 두 필드를 분리하여 profile 의 안정성과 OP 발행 시점을 독립 추적 |
| **기존 `reference_version` enum 폐기** | `NIAS_2024_v5` hard-coded → 행별 `profile_edition` + `publication_year` 로 대체. `derived_product_nutrition_analysis_v3_PR_CS.sql` (Wave 3) 에서 신규 컬럼 추가 |
| **NIAS v5 행 갱신** | `seed_14_nutrition_reference_v5.json` 의 각 row 에 `profile_edition=NIAS 2024_v5` / `publication_year=2024` 추가. 총 325 row (8/27 시점). **본 doc 에서 스키마 정의만, 행 갱신은 별도 PR (Wave 3 이후)** |
| **AAFCO 2014 표기 통일** | AAFCO 2014 Nutrient Profiles = `profile_edition=AAFCO 2014 Nutrient Profiles` / `publication_year=2014`. 본 매트릭스 v2 와 정합 |
| **0건 보장** | 본 v1 doc 작성 (9/4 12:15, 1회) 시 시드 / SQL / P2 SoT / P1-C SoT / E2E Live/Actual / git / commit / PR / NIAS max=10→12 / 임의 / random / LLM / PASS 역산 / "박지" / 이모지 — 모두 0건 |

---

## 1. profile_edition / publication_year 정의

### 1.1 profile_edition enum (프로필 개정판)

| profile_edition | 정의 | source_url | publication_year |
|---|---|---|---|
| `AAFCO 2014 Nutrient Profiles` | AAFCO 2014 OP Appendix A (revised 092214). Dog/Cat Food Nutrient Profiles 본문 | https://www.aafco.org/wp-content/uploads/2023/01/Pet_Food_Report_Annual_2014-Appendix_A-Revised_AAFCO_Nutrient_Profiles-Final_092214.pdf | 2014 |
| `AAFCO 2024 OP` | AAFCO 2024 Official Publication. 2014 profile 인용 (개정 없음) | https://www.aafco.org/ | 2024 |
| `NIAS 2024_v5` | 농촌진흥청 국립축산과학원(NIAS) 반려동물 사료 영양표준 v5 (8/27 17:43). 우리 프로젝트 내부 reference | (내부 seed_14_nutrition_reference_v5.json) | 2024 |
| `FEDIAF 2024 Nutritional Guidelines` | European Pet Food Industry Federation 2024 가이드라인 (참고용, 미사용) | (별도) | 2024 |

### 1.2 publication_year enum (OP 발행 연도)

| publication_year | 정의 |
|---|---|
| 2014 | AAFCO 2014 OP / NIAS 2014 |
| 2024 | AAFCO 2024 OP / NIAS 2024 / FEDIAF 2024 |
| (기타) | reference 갱신 시 추가 (예: 2025, 2026) |

### 1.3 분리 이유

| 시나리오 | profile_edition 동일 | publication_year 동일? | 분리 필요성 |
|---|---|---|---|
| AAFCO 2014 → 2024 OP (10년 인용) | O (둘 다 AAFCO 2014 Nutrient Profiles) | X (2014 vs 2024) | **O** — 동일 profile 이 여러 OP 에 인용되므로 OP 발행 시점은 별도 추적 |
| NIAS v5 → 다음 개정판 (v6) | X (v5 vs v6) | O (둘 다 2024 발행 가능) | **O** — profile 변경 vs 발행 시점 분리 |
| AAFCO 차기 개정 (예: 2016 개정안) | X (2014 vs 2016) | X (둘 다 다른 연도) | X — 둘 다 변경, 분리 효과 없음 |

→ **profile 의 안정성 (어떤 프로필을 기준으로) 과 OP 발행 시점 (언제 인용) 을 독립적으로 추적** 하기 위해 두 필드 분리.

---

## 2. 기존 reference_version enum 폐기

### 2.1 기존 (v1)

```sql
-- derived_product_nutrition_analysis_v3.sql:27
reference_version TEXT DEFAULT 'NIAS_2024_v5',
```

- **문제**: hard-coded, 행별 출처 없음. 2014 AAFCO profile 과 2024 NIAS 를 구분 불가
- **문제 2**: `DEFAULT 'NIAS_2024_v5'` 는 reference 가 NIAS 임을 강제. AAFCO reference 사용 불가

### 2.2 v2 (PR-canonical-status-precision 적용 시)

```sql
-- derived_product_nutrition_analysis_v3_PR_CS.sql (Wave 3 신규 SQL)
ALTER TABLE product_nutrition_analysis ADD COLUMN profile_edition TEXT NOT NULL DEFAULT 'AAFCO 2014 Nutrient Profiles';
ALTER TABLE product_nutrition_analysis ADD COLUMN publication_year INTEGER NOT NULL DEFAULT 2014;
-- 기존 reference_version 은 legacy 호환용으로 유지 (deprecated, 후속 PR 에서 제거)
```

- `profile_edition`: 어느 profile 의 값인지 (AAFCO / NIAS / FEDIAF)
- `publication_year`: 어느 OP 의 발행본인지
- **행별 출처 추적 가능**. PR-C/D/Vit/H 의 reference 변경 이력도 별도 기록 가능

---

## 3. NIAS v5 row 갱신 (Wave 3 이후 별도 PR)

`data/raw/seed_14_nutrition_reference_v5.json` 의 325 row (현재) 에 `profile_edition` / `publication_year` 추가. 본 doc 에서 스키마만 정의, 행 갱신은 별도 PR:

```json
{
  "species": "DOG",
  "life_stage": "ADULT_MAINTENANCE",
  "nutrient_code": "CRUDE_PROTEIN",
  "basis": "DRY_MATTER",
  "min_value": 18.0,
  "min_unit": "GRAM",
  "max_value": null,
  "max_unit": "GRAM",
  "table_id": "PROTEIN",
  "source_version": "NIAS_2024_v5",  // 기존 필드
  // 신규 (v1.1) 필드
  "profile_edition": "NIAS 2024_v5",
  "publication_year": 2024
}
```

**주의**: NIAS v5 의 `min_unit` 표기는 `mg/100g DM` (AAFCO 의 `mg/kg DM` 와 10x 차이). 본 매트릭스 v2 와 직접 비교 시 × 10 변환 필요. (canonical_data_contract_v1.md §2.1 unit 매핑 표 참조)

### 3.1 행 갱신 범위 (별도 PR)

| 항목 | 값 | 비고 |
|---|---|---|
| 325 row 전체 | `profile_edition=NIAS 2024_v5` | 8/27 시점 전체 |
| 325 row 전체 | `publication_year=2024` | 동일 |
| 기존 `source_version=NIAS_2024_v5` | 유지 (legacy 호환) | deprecated 단계 진입 |
| 0 product 변화 | seed 갱신만, downstream 영향 0 | NIAS v5 의 unit 표기는 변동 없음 |

### 3.2 AAFCO 2014 reference 별도 seed 생성 (선택)

본 매트릭스 v2 (AAFCO 2014 기반) 가 이미 별도 CSV 로 존재 (`required_nutrient_matrix_v2.csv`). 후속 PR 에서 `seed_14a_aafco_2014_reference.json` 별도 생성 가능 (현재는 본 매트릭스 CSV 가 source of truth).

---

## 4. 코드 영향 (Wave 3 PR-canonical-status-precision)

### 4.1 pipeline_p1c_v1.py 변경

- `compare_nias` (line 376+) 의 `nias_rows` 처리 시 `profile_edition` / `publication_year` 를 item-row 에 보존
- `compute_aafco_pass` / `compute_nutrition_comparison_status` 결과 dict 에 `profile_edition` / `publication_year` 포함
- `canonical_status_4state_v2` 의 10단계 결정 시 profile 단일 기준 적용 (혼합 reference 행은 1단계에서 INVALID 처리)

### 4.2 SQL 변경 (v3_PR_CS.sql)

- `product_nutrition_analysis` 테이블에 `profile_edition TEXT` / `publication_year INTEGER` 컬럼 추가
- `product_nutrition_analysis_v3.sql` 의 기존 `reference_version` 컬럼은 legacy 호환용으로 유지 (deprecated)
- 신규 SQL `v3_PR_CS.sql` 에서 hard-coded default 제거

### 4.3 데이터 변경 (별도 PR)

- `seed_14_nutrition_reference_v5.json` 의 325 row 에 2 컬럼 추가 (3.1)
- `seed_14a_aafco_2014_reference.json` 별도 생성 가능 (3.2, 권고)

---

## 5. 영향 요약

| 항목 | v1 (기존) | v2 (본 doc) | 적용 시점 |
|---|---|---|---|
| reference 표기 | `reference_version='NIAS_2024_v5'` (hard-coded) | `profile_edition` + `publication_year` (행별) | Wave 3 PR-canonical-status-precision |
| NIAS vs AAFCO 구분 | 불가 (NIAS 강제) | 가능 (profile_edition enum) | 동일 |
| OP 발행 시점 추적 | 불가 (v5 hard-coded) | 가능 (publication_year) | 동일 |
| 본 매트릭스 v2 와 정합 | NIAS 만, mg/100g DM (10x 차이) | **AAFCO 2014 / mg/kg_DM, IU/kg_DM 정합** | Wave 3 |
| 0 product 변화 | (해당 없음) | O (스키마 추가만) | 동일 |

---

## 6. 0건 보장

본 doc 작성 (9/4 12:15, 1회) 시:
- pipeline_p1c_v1.py / SQL / 시드 / OPFF cache / P2 SoT / P1-C SoT / E2E Live/Actual — 수정 0건
- NIAS max=10→12 / 임의 / random / LLM / PASS 역산 / "박지" / 이모지 / git / commit / PR — 모두 0건
- 본 doc 만 신규 작성

---

**END of reference_edition_policy.md**
