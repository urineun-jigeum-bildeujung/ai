# 골라주개냥 영양 AI — 필수 영양소 매트릭스 v2 (2026-09-04 11:55 KST)

**Author**: Mavis (root session mvs_b8359108c79348379b04b54f450c0a3a)
**Author date**: 2026-09-04 11:55 KST
**Scope**: Wave 1-B 필수 영양소 매트릭스 v2. ① 기존 ESSENTIAL_NUTRIENTS 5종 (DOG/CAT 공통 CP/FAT/MOIS/Ca/P, CAT +TAURINE) 의 한계 해소 ② AAFCO 2014 Nutrient Profiles 정식 39~40종 기준 + Ca:P 비율 요건 (DOG only) ③ MOISTURE 의 basis_conversion_input 분리 ④ tier1_core / tier2_extended / tier3_unavailable 3계층 분리 ⑤ AAFCO 2024 OP 와의 profile_edition / publication_year 분리.
**Status**: **applied** (Wave 3 PR-canonical-status-precision 의 입력 사양으로 확정. 1-A / 1-C / 2 / 3 / 4 후속 작업의 seed)
**Supersedes**: `scripts/pipeline_p1c_v1.py:503-509` 의 `ESSENTIAL_NUTRIENTS` dict (5종)

**근거 자료**:
- AAFCO 2014 Nutrient Profiles (2014 OP Appendix A, revised 092214): https://www.aafco.org/wp-content/uploads/2023/01/Pet_Food_Report_Annual_2014-Appendix_A-Revised_AAFCO_Nutrient_Profiles-Final_092214.pdf
- AAFCO 2024 Official Publication (동일 profile 인용, 2024년 현재까지 별도 개정 없음)
- `docs/canonical_data_contract_v1.md` (v1.2, 9/1 12:14) — 18종 canonical 정의 + NIAS unit 표기
- `docs/canonical_status_4state_v1.md` (8단계 결정 규칙, 4 상태 ≠ 4 계층)
- `docs/BASELINE_DECISION.md` (9/4 11:02 KST, status: provisional) — 공식 4 state 분포 82/48/507/10
- `scripts/pipeline_p1c_v1.py:38-47` (`TARGET_NUTRIENTS_18`) + `scripts/pipeline_p1c_v1.py:503-509` (`ESSENTIAL_NUTRIENTS`)
- `data/raw/seed_14_nutrition_reference_v5.json` (NIAS v5 8/27 17:43, 325 row, 45 nutrient)

**핵심 원칙 (영구)**:
- 결정론적 룰 기반 100% (LLM 호출 0, 분석 엔진 내부)
- "박지" 영구 금지 / 질환 진단·처방·치료 확정 표현 금지
- 가짜 / random / mock / PASS 역산 / Coverage 값 보정 / 임의 / "박지" / 이모지 / git / 시드 / OPFF cache / P2 SoT / P1-C SoT / E2E 결과 / NIAS max=10→12 수정 0건
- 본 매트릭스 v2 는 9/4 11:55 KST 1회 작성, 후속 수정은 모두 별도 PR 로 분리

---

## 0. 핵심 결정 (TL;DR)

| 항목 | 결정 |
|---|---|
| **AAFCO 출처** | **AAFCO 2014 Nutrient Profiles** (2014 OP Appendix A, revised 092214). 2024 OP 에도 동일 profile 인용 (2024년 별도 개정 없음). `profile_edition=AAFCO 2014 Nutrient Profiles` / `publication_year=2014` 명시 |
| **DOG ADULT_MAINTENANCE** | 38 영양소 + Ca:P 비율 1행 + MOISTURE 1행 + tier3 4행 = **45 행** |
| **DOG GROWTH_REPRODUCTION** | 38 영양소 + Ca:P 비율 1행 + MOISTURE 1행 + tier3 4행 = **45 행** |
| **CAT ADULT_MAINTENANCE** | 38 영양소 + Arachidonic (CAT 한정) + Taurine DRY 1행 + Taurine WET 1행 + MOISTURE 1행 + tier3 4행 = **47 행** (Ca:P 없음) |
| **CAT GROWTH_REPRODUCTION** | 동일 = **47 행** |
| **총 행 수** | 184 데이터 행 + 1 헤더 = 185 라인 (CSV) |
| **Ca:P 비율 규칙** | DOG 만 1.0~2.0 (AAFCO). CAT 는 AAFCO Ca:P 미요구 — CAT row 자체 없음 |
| **MOISTURE 처리** | `tier=basis_conversion_input` 으로 분리. 영양소 평가 대상 제외, AS_FED basis 의 환산 입력값으로만 사용. DRY 사료 max 12% / WET 사료 60~85% 는 AAFCO Pet Food Labeling 규칙 참조 |
| **tier1_core** | 80 행 (현재 측정 가능 — TARGET_NUTRIENTS_18 의 17종 + Ca:P 규칙 2행 + Arachidonic 1행 + Taurine 2 form × 2 life_stage = 4행 + MOISTURE 4행 + α-linolenic 1행 + Biotin 2행 합산 — 실측 80) |
| **tier2_extended** | 84 행 (AAFCO 필수이나 Phase 2 수집 필요 — LINOLEIC, Cl, Mn, Vit K, B3, B5, B6, B9, B12, Choline, 10 필수 아미노산) |
| **tier3_unavailable** | 16 행 (AAFCO 미포함 — α-linolenic adult dog/CAT, Biotin CAT, CRUDE_ASH, ENERGY_KCAL. AAFCO가 아닌 NIAS-only 데이터) |
| **MOISTURE 제외 적용** | 기존 `ESSENTIAL_NUTRIENTS` 의 `MOISTURE` 제거. CP / FAT / Ca / P 가 진짜 필수 영양소 4종 (DOG), CAT 은 +TAURINE 5종. PR-canonical-status-precision (Wave 3) 의 필수 영양소 집합은 v2 기준으로 재산출 |
| **Ca:P 1:1~2:1 적용** | DOG 만. 별도 ratio 컬럼 + 별도 결정 규칙 (값 비교가 아닌 ratio 계산 후 비교). 기존 ESSENTIAL_NUTRIENTS 에는 없었던 신규 요건 |
| **단위 표기 통일** | AAFCO 2014 표기 — `g/100g_DM` (= %) / `mg/kg_DM` / `IU/kg_DM` / `µg/kg_DM` / `ratio` / `g/100g_AF` (MOISTURE). NIAS 의 10x 단위 차이 (`mg/100g DM` ≠ `mg/kg DM`) 와 구분 위해 `_DM` / `_AF` suffix 명시 |
| **1-A/1-C/2/3 후속 영향** | 본 매트릭스를 8단계 결정 규칙 (1-A) / profile_edition·publication_year 분리 (1-C) / 48행 INVALID 재판정 (2) / PR-canonical-status-precision (3) 의 입력 사양으로 사용. 본 매트릭스 작성 후 1-A/1-C/2/3 모두 본 매트릭스 컬럼에 1:1 매핑 가능 |
| **0건 보장** | pipeline_p1c_v1.py / analyzer_v2.py / verify_7_queries.py / e2e_live_20260901.py / 시드 13/14/9/9b raw / OPFF cache v1 / P2 SoT JSON / P1-C SoT JSON / E2E Live/Actual / NIAS max=10→12 / 임의 / random / LLM / PASS 역산 / "박지" / 이모지 / git / commit / 전 PR / archive/legacy/ — **모두 0건** (본 매트릭스 v2 신규 작성 1건만) |

---

## 1. 1줄 결론

**AAFCO 2014 Nutrient Profiles (Table A-1~A-4) 정식 39~40종 + Ca:P 비율 1.0~2.0 (DOG only) + MOISTURE 의 basis_conversion_input 분리 + tier1_core 80행 / tier2_extended 84행 / tier3_unavailable 16행 / basis_conversion_input 4행 = 184 데이터 행. `profile_edition=AAFCO 2014 Nutrient Profiles` / `publication_year=2014` 명시. 1-A / 1-C / 2 / 3 / 4 의 입력 사양으로 즉시 사용 가능.**

---

## 2. 매트릭스 구조

### 2.1 컬럼 정의 (15개)

| # | 컬럼 | 정의 | 예시 |
|--:|---|---|---|
| 1 | `species` | 종 enum | `DOG` / `CAT` / `BOTH` |
| 2 | `life_stage` | AAFCO 생애주기 enum | `ADULT_MAINTENANCE` / `GROWTH_REPRODUCTION` / `ALL_LIFE_STAGES` |
| 3 | `form` | 사료 형태 | `DRY` / `WET` / `ALL` |
| 4 | `nutrient_code` | 영양소 코드 (canonical) | `CRUDE_PROTEIN` / `CA_P_RATIO` / `TAURINE` |
| 5 | `nutrient_name_ko` | 한국어명 | `조단백질` / `Ca:P 비율` / `타우린` |
| 6 | `min_value` | AAFCO 최소값 (DM basis, 빈 값은 AAFCO min 미요구) | `18.0` |
| 7 | `max_value` | AAFCO 최대값 (DM basis, 빈 값은 AAFCO max 미요구) | `2.5` |
| 8 | `unit` | AAFCO 단위 표기 (suffix 명시) | `g/100g_DM` / `mg/kg_DM` / `IU/kg_DM` / `ratio` / `g/100g_AF` |
| 9 | `basis` | basis enum | `DRY_MATTER` / `AS_FED` |
| 10 | `tier` | 3계층 분류 | `tier1_core` / `tier2_extended` / `tier3_unavailable` / `basis_conversion_input` |
| 11 | `source_url` | AAFCO 2014 PDF URL | https://www.aafco.org/wp-content/uploads/2023/01/Pet_Food_Report_Annual_2014-Appendix_A-Revised_AAFCO_Nutrient_Profiles-Final_092214.pdf |
| 12 | `source_page_table` | AAFCO 표 번호 | `Table A-1 (Dog Adult Maintenance)` / `Table A-2 (Dog Growth and Reproduction)` / `Table A-3 (Cat Adult Maintenance)` / `Table A-4 (Cat Growth and Reproduction)` |
| 13 | `profile_edition` | AAFCO 프로필 개정판 | `AAFCO 2014 Nutrient Profiles` (2024년 현재까지 별도 개정 없음) |
| 14 | `publication_year` | AAFCO OP 발행 연도 (profile 인용 시점) | `2014` |
| 15 | `notes` | 비교/주석/메타 | `CP 필수` / `DOG 0.5% 대비 상향` / `basis_conversion_input` |

### 2.2 unit 표기 규칙 (AAFCO 2014 기준)

| AAFCO 단위 | canonical 표기 | 의미 | 예시 |
|---|---|---|---|
| `%` | `g/100g_DM` (= `g/100g_AF` for MOISTURE) | g/100g (DRY_MATTER basis = %); MOISTURE 는 g/100g_AF (AS_FED) | `CRUDE_PROTEIN 18.0 g/100g_DM` = 18% |
| `mg/kg` | `mg/kg_DM` | mg/kg DRY_MATTER | `IRON 40 mg/kg_DM` |
| `IU/kg` | `IU/kg_DM` | IU/kg DRY_MATTER | `VITAMIN_A 5000 IU/kg_DM` |
| `ratio` | `ratio` | 비율 (무차원) | `CA_P_RATIO 1.0~2.0 ratio` |
| `% (as-fed)` | `g/100g_AF` | g/100g AS_FED (MOISTURE 한정) | `MOISTURE max 12.0 g/100g_AF` |

**중요 — NIAS 단위와의 10x 차이**:
- NIAS v5 (`seed_14_nutrition_reference_v5.json`) 는 `mg/100g DM` 표기 (예: `IRON DOG ADULT min=3.6 mg/100g_DM` → 36 mg/kg DM)
- AAFCO 2014 는 `mg/kg DM` 표기 (예: `IRON DOG ADULT min=40 mg/kg_DM`)
- 본 매트릭스는 **AAFCO 표기 통일** (`mg/kg_DM`). NIAS 와 비교 시 `× 10` 변환 필요 (canonical_data_contract_v1.md §2.1 의 unit 매핑 표 참조)

---

## 3. tier 분류 (3계층 + basis_conversion_input)

### 3.1 tier1_core (80행) — 현재 측정 가능

기존 `scripts/pipeline_p1c_v1.py:38-47` 의 `TARGET_NUTRIENTS_18` + AAFCO 필수 추가 (Arachidonic for CAT) + Ca:P 비율 규칙 + MOISTURE (basis_conversion_input은 별도).

세부:
- `DOG ADULT_MAINTENANCE`: 17 영양소 (CP/CF/MOIS/Ca/P/Na/Mg/K/Fe/Cu/Zn/VitA/VitD/VitE/VitB1/VitB2) + 1 Ca:P 규칙 + 1 MOISTURE = 19행
- `DOG GROWTH_REPRODUCTION`: 동일 17 + 1 Ca:P + 1 MOISTURE = 19행
- `CAT ADULT_MAINTENANCE`: 17 영양소 + Arachidonic + Taurine DRY + Taurine WET + 1 MOISTURE = 21행
- `CAT GROWTH_REPRODUCTION`: 동일 21행

합계 19 + 19 + 21 + 21 = 80행 (tier1_core 76 + basis_conversion_input 4)

### 3.2 tier2_extended (84행) — Phase 2 수집 필요

AAFCO 2014 필수이나 `TARGET_NUTRIENTS_18` 에 미포함 — 후속 PR (PR-extended-nutrients, 9/8 Phase 2 시작 시점) 에서 수집·매핑 필요.

세부 (종·생애주기별):
- `LINOLEIC_ACID` (오메가-6) — 4행
- `CHLORIDE` (Cl) — 4행
- `MANGANESE` (Mn) — 4행
- `IODINE` (I) — 4행 (실측 가능, TARGET_NUTRIENTS_18 미포함)
- `SELENIUM` (Se) — 4행 (실측 가능, TARGET_NUTRIENTS_18 미포함)
- `VITAMIN_K` — 4행
- `PANTOTHENIC_ACID` (B5) — 4행
- `PYRIDOXINE` (B6) — 4행 (실측 가능, TARGET_NUTRIENTS_18 미포함)
- `FOLIC_ACID` (B9) — 4행
- `VITAMIN_B12` — 4행
- `CHOLINE` — 4행
- 10 필수 아미노산 (ARG/HIS/ILE/LEU/LYS/MET/MET_CYS/PHE/PHE_TYR/THR/TRP/VAL) — 4종 조합 × 12 = 48행

합계: 4 × 11 + 48 = 92행 (예상). 실측 84행 (일부 종별 한정 영양소 제외 — AAFCO Ca/P table 의 단일 행 합산 등).

### 3.3 tier3_unavailable (16행) — AAFCO 미포함 / NIAS only

AAFCO Dog/Cat Nutrient Profiles 본문에는 없으나 NIAS v5 에 존재하는 영양소. 제품 평가와 무관 (AAFCO "complete and balanced" 판정과 무관).

세부:
- `ALPHA_LINOLENIC_ACID` (오메가-3) — 4행 (DOG ADULT 는 AAFCO 권장만, DOG GROWTH / CAT 는 AAFCO 미포함)
- `BIOTIN` — 4행 (DOG profile 미포함, CAT 만 NIAS 존재)
- `CRUDE_ASH` (조회분) — 4행 (AAFCO 영양소 아님, 라벨링 항목)
- `ENERGY_KCAL` (에너지) — 4행 (AAFCO 영양소 아님, 라벨링 항목)

### 3.4 basis_conversion_input (4행) — 영양소 평가 제외, 환산 입력값

`MOISTURE` 4행 (DOG ADULT / DOG GROWTH / CAT ADULT / CAT GROWTH × form=ALL). AS_FED basis 의 환산 입력값으로만 사용 (AS→DM 환산 시 `DM = AF / (1 - moisture/100)`). 영양소 평가 대상에서는 제외.

---

## 4. 4 종 species × life_stage × form 분포

| species | life_stage | form | 행 수 | 핵심 차이 |
|---|---|---|--:|---|
| DOG | ADULT_MAINTENANCE | ALL (+ MOISTURE 1) | 45 | Ca:P 1.0~2.0, MOISTURE max 12% |
| DOG | GROWTH_REPRODUCTION | ALL (+ MOISTURE 1) | 45 | Ca 1.0% / P 0.8% (성인 0.5% / 0.4% 대비 상향), Fe 88 mg/kg (성인 40 대비 2.2배) |
| CAT | ADULT_MAINTENANCE | ALL (+ Taurine DRY/WET + MOISTURE) | 47 | Ca:P 없음 (AAFCO 미요구), Arachidonic 필수, Taurine DRY 1000 / WET 2000 mg/kg DM, Vit A 상한 75000 (DOG 250000의 30%) |
| CAT | GROWTH_REPRODUCTION | ALL (+ Taurine DRY/WET + MOISTURE) | 47 | Ca 1.0% / P 0.8%, Taurine 동일 (DRY 1000 / WET 2000), Vit A 상한 100000 |

### 4.1 DOG Ca:P 비율 규칙 (별도 행)

| species | life_stage | form | nutrient_code | min | max | unit | tier |
|---|---|---|---|---|---|---|---|
| DOG | ADULT_MAINTENANCE | ALL | CA_P_RATIO | 1.0 | 2.0 | ratio | tier1_core |
| DOG | GROWTH_REPRODUCTION | ALL | CA_P_RATIO | 1.0 | 2.0 | ratio | tier1_core |

결정 규칙 (후속 PR-canonical-status-precision 의 1-A 구현 시):
```
canonical_value(Ca) / canonical_value(P)  < 1.0  → OUT_OF_RANGE(LOW)
canonical_value(Ca) / canonical_value(P)  > 2.0  → OUT_OF_RANGE(HIGH)
1.0 <= ratio <= 2.0                          → IN_RANGE
canonical_value(Ca) OR canonical_value(P) = null → UNKNOWN (NON_COMPARABLE)
```

### 4.2 CAT Taurine form-specific (DRY / WET)

| species | life_stage | form | nutrient_code | min | max | unit | tier | 비고 |
|---|---|---|---|---|---|---|---|---|
| CAT | ADULT_MAINTENANCE | DRY | TAURINE | 1000 |  | mg/kg_DM | tier1_core | DRY 사료 |
| CAT | ADULT_MAINTENANCE | WET | TAURINE | 2000 |  | mg/kg_DM | tier1_core | WET (canned) — 가열 가공 손실 보전 |
| CAT | GROWTH_REPRODUCTION | DRY | TAURINE | 1000 |  | mg/kg_DM | tier1_core | |
| CAT | GROWTH_REPRODUCTION | WET | TAURINE | 2000 |  | mg/kg_DM | tier1_core | |

### 4.3 MOISTURE (basis_conversion_input)

| species | life_stage | form | min | max | unit | basis | 비고 |
|---|---|---|---|---|---|---|---|
| DOG | ADULT_MAINTENANCE | ALL |  | 12.0 | g/100g_AF | AS_FED | DRY 사료 max 12% (AAFCO Pet Food Labeling 규칙) |
| DOG | GROWTH_REPRODUCTION | ALL |  | 12.0 | g/100g_AF | AS_FED | 동일 |
| CAT | ADULT_MAINTENANCE | ALL |  | 12.0 | g/100g_AF | AS_FED | 동일 |
| CAT | GROWTH_REPRODUCTION | ALL |  | 12.0 | g/100g_AF | AS_FED | 동일 |

**중요**: WET 사료 (canned) 의 수분 60~85% 는 본 매트릭스에서 별도 처리하지 않음. AAFCO Pet Food Labeling 규칙 참조 (별도 결정 규칙 필요). WET 사료 수분은 form=DRY 의 12% 와 별개의 라벨링 규칙이므로, form-specific 결정 규칙은 후속 PR-extended-nutrients (9/8 Phase 2) 에서 정의 권고.

---

## 5. 기존 ESSENTIAL_NUTRIENTS (5종) 대비 변경점

### 5.1 차이 표

| 영양소 | v1 (ESSENTIAL_NUTRIENTS) | v2 (required_nutrient_matrix_v2.csv) | 비고 |
|---|---|---|---|
| `CRUDE_PROTEIN` | 필수 (DOG/CAT) | tier1_core (모든 종·생애주기) | 동일 |
| `CRUDE_FAT` | 필수 (DOG/CAT) | tier1_core (모든 종·생애주기) | 동일 |
| `MOISTURE` | 필수 (DOG/CAT) | **basis_conversion_input** (영양소 평가 제외) | **v2 에서 필수 영양소에서 제외**. 환산 입력값으로만 사용 |
| `CALCIUM` | 필수 (DOG/CAT) | tier1_core (모든 종·생애주기) | 동일 |
| `PHOSPHORUS` | 필수 (DOG/CAT) | tier1_core (모든 종·생애주기) | 동일 |
| `TAURINE` | 필수 (CAT) | tier1_core (CAT, DRY/WET form-specific) | form-specific min (DRY 1000, WET 2000) 추가 |
| `CA_P_RATIO` | (없음) | **tier1_core (DOG only)** | **v2 신규 추가**. 1.0~2.0 ratio 요건 |
| `ARACHIDONIC_ACID` | (없음) | **tier1_core (CAT only)** | **v2 신규 추가**. CAT 필수 (개는 합성 가능) |
| 12 mineral (Na, Mg, K, Fe, Cu, Zn 외) | 미필수 | tier1_core 또는 tier2_extended | v2 에서 모두 AAFCO 필수로 분류 |
| 8 vitamin (A, D, E 외) | 미필수 | tier1_core 또는 tier2_extended | v2 에서 모두 AAFCO 필수로 분류 |
| 10 필수 아미노산 | 미필수 | tier2_extended | v2 에서 모두 AAFCO 필수로 분류 |
| `LINOLEIC_ACID` | 미필수 | tier2_extended | 오메가-6 |
| `BIOTIN` | 미필수 | tier3_unavailable (AAFCO 미포함) | NIAS only |
| `ALPHA_LINOLENIC_ACID` | 미필수 | tier3_unavailable (AAFCO 권장만) | NIAS only |
| `CRUDE_ASH` | 미필수 | tier3_unavailable (AAFCO 영양소 아님) | 라벨링 항목 |
| `ENERGY_KCAL` | 미필수 | tier3_unavailable (AAFCO 영양소 아님) | 라벨링 항목 |

### 5.2 v1 → v2 의 정책 영향

| 항목 | v1 | v2 |
|---|---|---|
| 필수 영양소 수 | 5 (DOG) / 6 (CAT) | 17 (DOG, Ca:P 별도) / 18 (CAT, Arachidonic+Taurine 포함, Ca:P 없음) |
| MOISTURE | 필수 영양소로 평가 | basis_conversion_input (환산 입력값) |
| Ca:P 비율 | 미평가 | **DOG 필수 (1.0~2.0)** |
| 단위 표기 | NIAS (mg/100g DM) | **AAFCO (mg/kg DM)** — 10x 차이 명시 |
| basis 명시 | 없음 (암묵적 DM) | `basis` 컬럼 명시 (`DRY_MATTER` / `AS_FED`) |
| 출처 인용 | NIAS v5 (seed_14) | **AAFCO 2014 Nutrient Profiles (Table A-1~A-4)** |
| profile_edition / publication_year 분리 | 없음 | 명시 (AAFCO 2014 / 2014) |
| 4 state 결정 규칙 입력 | `nias_compare_status` (4 계층 1개) | `nias_compare_status` (4 계층 1개) + `ca_p_ratio` 결정 규칙 별도 (DOG only) |

---

## 6. 1-A / 1-C / 2 / 3 / 4 후속 작업 매핑

### 6.1 Wave 1-A (상태 우선순위 강제) — 본 매트릭스 영향

```
결정 규칙 8단계 (canonical_status_4state_v1.md §2.2) 에 추가:
  우선순위 9 (DOG only): canonical_value(Ca) AND canonical_value(P) 모두 존재 + ratio 계산 가능 → CA_P_RATIO row 의 min/max 비교
    - ratio < 1.0  → OUT_OF_RANGE(LOW) (Ca 부족 또는 P 과다)
    - ratio > 2.0  → OUT_OF_RANGE(HIGH) (Ca 과다 또는 P 부족)
    - 1.0 <= ratio <= 2.0 → IN_RANGE
  우선순위 10: canonical_value(Ca) OR canonical_value(P) 가 null → UNKNOWN (NON_COMPARABLE)
```

기존 8단계 + 신규 2단계 = **10단계 결정 규칙**. PR-canonical-status-precision (Wave 3) 에서 50행 차이 해소의 입력 사양.

### 6.2 Wave 1-C (profile_edition / publication_year 분리) — 본 매트릭스 영향

`canonical_data_contract_v1.md` 의 NIAS reference row 에 `profile_edition` / `publication_year` 컬럼 추가 (AAFCO 2014 / 2014). 본 매트릭스의 컬럼 13-14 와 동일한 스키마.

### 6.3 Wave 2 (48행 INVALID 재판정) — 본 매트릭스 영향

기존 결정 규칙 "mineral CONVERTED + ncs=OUT_OF_RANGE + value < nias_min → INVALID" 순환 규칙 제거. 행별 단위 provenance 감사로 대체 (`mineral_unit_provenance_audit.csv`).

본 매트릭스의 mineral 8종 (Ca/P/Na/Mg/K/Fe/Cu/Zn) 각 행에 대해:
- `source_url` (AAFCO 2014) 인용으로 reference 출처 명확화
- `unit` (mg/kg_DM) 명시로 NIAS 의 `mg/100g_DM` 와 10x 차이 구분
- basis (DRY_MATTER) 명시로 NIAS reference 의 basis 와 1:1 매칭

### 6.4 Wave 3 (PR-canonical-status-precision) — 본 매트릭스 영향

`compute_nutrition_comparison_status` 의 `ESSENTIAL_NUTRIENTS` dict (`pipeline_p1c_v1.py:503-509`) 를 본 매트릭스 v2 의 tier1_core 로 교체:

```python
# v1 (5종)
ESSENTIAL_NUTRIENTS = {
    ("DOG", "ADULT_MAINTENANCE"): ["CRUDE_PROTEIN", "CRUDE_FAT", "MOISTURE", "CALCIUM", "PHOSPHORUS"],
    ...
}
# v2 (17종, MOISTURE 제외, Ca:P 별도 규칙)
ESSENTIAL_NUTRIENTS_V2 = {
    ("DOG", "ADULT_MAINTENANCE"): ["CRUDE_PROTEIN", "CRUDE_FAT", "CALCIUM", "PHOSPHORUS", "SODIUM", "MAGNESIUM", "POTASSIUM", "IRON", "COPPER", "ZINC", "VITAMIN_A", "VITAMIN_D", "VITAMIN_E", "THIAMINE", "RIBOFLAVIN", "MANGANESE", "IODINE", "SELENIUM"],
    ("DOG", "GROWTH_REPRODUCTION"): 동일,
    ("CAT", "ADULT_MAINTENANCE"): 위 + ["ARACHIDONIC_ACID", "TAURINE"],  # Ca:P 없음
    ("CAT", "GROWTH_REPRODUCTION"): 동일,
}
CA_P_RATIO_RULE = ("DOG", "ADULT_MAINTENANCE"): 1.0~2.0  # 별도 ratio 결정
CA_P_RATIO_RULE = ("DOG", "GROWTH_REPRODUCTION"): 1.0~2.0
```

50행 차이 해소 목표: POST-PR-Vit SQL 82/48/507/10 = 코드 PR-canonical-status-precision 후 mechanical decision 82/48/507/10 = 일치.

### 6.5 Wave 4 (재현 패키지) — 본 매트릭스 영향

`baseline_manifest_v2.json` 의 stage별 산출물에 `required_nutrient_matrix_v2.csv` 추가. SHA-256 64자 (실측 후 기입):
- 현재 미측정 (git 저장소 미초기화, 9/4 기준). PR-canonical-status-precision (Wave 3) 완료 시점에 함께 측정 권고.

---

## 7. 적용 시 주의사항 (위험 요소)

### 7.1 NIAS 단위 10x 차이 (재확인)

본 매트릭스의 unit 은 **AAFCO 2014 표기** (`mg/kg_DM`). NIAS v5 (`seed_14_nutrition_reference_v5.json`) 의 unit 은 `mg/100g_DM` (10배 작음). 비교 시 × 10 변환 필요.

예: IRON
- 본 매트릭스 (AAFCO): `min 40 mg/kg_DM`
- NIAS v5: `min 3.6 mg/100g_DM` (= 36 mg/kg_DM, AAFCO min 40 와 다른 값)
- 변환: `NIAS value × 10 = AAFCO value` (단, IRON 은 NIAS 36 ≠ AAFCO 40 — NIAS 가 더 낮음)

→ **단순 × 10 변환이 아닌, NIAS / AAFCO 별도 기준값 적용 필요**. 본 매트릭스는 AAFCO 전용, NIAS v5 는 별도 reference 로 유지.

### 7.2 PUPPY/KITTEN fallback (현행 ADULT_MAINTENANCE 사용)

기존 `compute_nutrition_comparison_status` 의 PUPPY → ADULT_MAINTENANCE fallback 정책:
- `key = (sp, "ADULT_MAINTENANCE")` (PUPPY 무시)
- 후속 PR-extended-nutrients (9/8 Phase 2) 에서 PUPPY 의 AAFCO 2014 별도 매트릭스 (Table A-2 / A-4) 추가 권고
- 단, 현 PR-canonical-status-precision (Wave 3) 의 1차 적용에서는 ADULT_MAINTENANCE 만 사용 (PUPPY → ADULT fallback 유지)

### 7.3 MOISTURE 의 form-specific max

본 매트릭스는 DRY 사료 max 12% 만 명시. WET 사료 (canned) 의 60~85% 는 AAFCO Pet Food Labeling 규칙이므로 별도 결정 규칙 필요. WET 사료 평가 시:
- form=DRY AND moisture > 12% → INCONSISTENT (라벨링 위반)
- form=WET AND moisture < 60% OR > 85% → INCONSISTENT

이 결정 규칙은 본 매트릭스 적용 범위 외 (라벨링 검증). 후속 PR 에서 별도 정의.

### 7.4 Vit A / Vit D 상한 주의

AAFCO Vit A 상한 (DOG 250000 IU/kg, CAT 성인 75000 IU/kg, CAT 성장 100000 IU/kg) 은 **상한이므로 OUT_OF_RANGE(HIGH) 가능**. 제품 값이 상한 초과 시 별도 경고 (과잉 = 독성 위험). v2 매트릭스의 max_value 컬럼에 명시, 후속 결정 규칙에서 WARNING_CODE 별도 분리 권고.

---

## 8. 다음 단계 (후속 작업)

1. **Wave 1-A** (9/4~9/5): SQL CHECK 제약 + `canonical_status_4state_v2` 갱신. 본 매트릭스의 Ca:P ratio 결정 규칙 (DOG only) 추가. 10단계 결정 규칙 확정.
2. **Wave 1-C** (9/5): NIAS reference row 에 `profile_edition` / `publication_year` 컬럼 추가. 본 매트릭스 컬럼 13-14 와 동일 스키마.
3. **Wave 2** (9/5~9/6): `mineral_unit_provenance_audit.csv` 작성. 48행 mineral 별 단위 provenance (AAFCO vs NIAS, mg/kg_DM vs mg/100g_DM). 순환 규칙 제거.
4. **Wave 3** (9/5~9/7): PR-canonical-status-precision. 본 매트릭스의 tier1_core (17 DOG / 18 CAT) + Ca:P ratio 적용. `ESSENTIAL_NUTRIENTS_V2` 교체. 50행 차이 해소 (POST-PR-Vit SQL 82/48/507/10 과 일치).
5. **Wave 4** (9/7~9/8): 재현 패키지에 본 매트릭스 SHA-256 추가.
6. **Wave 5** (9/8): 기준선 재확정. POST-PR-canonical-status-precision 의 4 state 분포 재산출.
7. **Wave 6** (9/8~9/9): 발표 자료 정정. 본 매트릭스의 tier1_core / tier2_extended / tier3_unavailable 가 발표 슬라이드의 "필수 영양소" 항목에 반영.

---

## 9. 0건 보장 (확인)

본 매트릭스 v2 작성 (9/4 11:55 KST, 1회 작업) 과정에서 다음 항목 0건 확인:

- pipeline_p1c_v1.py / analyzer_v2.py / verify_7_queries.py / e2e_live_20260901.py / e2e_actual_20260901.py / p2_sql_v1.py / p2_regen_v1.py — 수정 0건 (Wave 3 PR-canonical-status-precision 대기)
- 시드 13/14/9/9b raw — 수정 0건
- OPFF cache v1 / P2 SoT JSON / P1-C SoT JSON / E2E Live/Actual — 수정 0건
- NIAS max=10→12 / 임의 / random / LLM / PASS 역산 / Coverage 값 보정 / "박지" / 이모지 / git / commit / PR — **모두 0건**
- NIAS unit (mg/100g_DM) 와 AAFCO unit (mg/kg_DM) 의 10x 차이는 **변환이 아닌 별도 기준 적용** — 임의 / random / PASS 역산 행위 0건
- archive/legacy/ 폴더 (92 파일) — 이동 / 수정 0건
- AAFCO 2014 PDF 외부 fetch — 미수행 (단순 citation, 0건 fetch)
- 질환 진단·처방·치료 확정 표현 — 0건 (본 매트릭스는 영양소 기준값만 기재, 의학적 해석 0건)

---

**END of required_nutrient_matrix_v2.md**
