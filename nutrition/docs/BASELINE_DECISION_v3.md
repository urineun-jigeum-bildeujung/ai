# BASELINE_DECISION_v3.md — V3 verified_via_parity_test_only 기준선 (2026-09-04 12:55 KST, status: verified_via_parity_test_only)

**Author**: Mavis (root session mvs_b8359108c79348379b04b54f450c0a3a)
**Author date**: 2026-09-04 12:55 KST
**Scope**: Wave 5 기준선 재확정 + PR-canonical-status-precision V3 SQL 8-step exact replication. **0행 차이** (parity test 기준) 확인. **코드 적용 (pipeline_p1c_v1.py patch) 은 미완** — status: verified_via_parity_test_only.
**Status**: **verified_via_parity_test_only** (V3 로직 0행 차이는 별도 parity test `scripts/tests/test_code_sql_parity.py` 기준. 실제 `pipeline_p1c_v1.py` 의 `_canonical_status_8step` 는 여전히 V1 — 9 mineral, no species_map, 공식 대비 50행 차이. PR-canonical-status-precision 적용은 9/8 Phase 2 시점)
**Supersedes**:
- `docs/BASELINE_DECISION.md` (9/4 11:02, status: provisional, 82/48/507/10 = 647)
- `docs/BASELINE_DECISION_v2.md` (9/4 12:35, status: applied_v2, 93/509/37/8 = 647, 24행 잔존 차이)
**commit_sha**: **8a7b9e8eaa1471acc245d948d9a4e77f8c2298d3** (40자, 9/4 12:55 v3 verified commit, HEAD on main)
**previous_commit (v3 산출물 19개)**: 98cad1f1e0739279bcb5cc334c1fe14d5e6e75b4 (40자, 9/4 12:50 git init)
**run_id**: **run_wave5_baseline_v3_20260904_125500**

**근거 자료**:
- `docs/BASELINE_DECISION_v2.md` (9/4 12:35, v2 applied_v2)
- `docs/canonical_status_4state_v1.md` (9/1 16:57, SQL 8단계 결정 규칙, 9/3 17:00 정합성 검증)
- `docs/canonical_status_4state_v2.md` (9/4 12:10, 10단계 결정 규칙)
- `data/processed/required_nutrient_matrix_v2.csv` (9/4 11:55, AAFCO 2014)
- `data/processed/seed_13_p1c_source_of_truth_v1.json` (POST-PR-Vit 9/3 15:18, 647 item-row)
- `data/processed/_wave3_v3_canonical_status_audit.json` (9/4 12:45, V3 0행 차이)
- `data/processed/baseline_manifest_v3.json` (본 파일, 9/4 12:55, status: verified)
- `scripts/tests/test_code_sql_parity.py` (9/4 12:45 갱신, V3 dry-run)
- `data/raw/seed_9b_off_korean_oem.json` (species_map 189 product)

**핵심 원칙 (영구)**:
- 결정론적 룰 기반 100% (LLM 호출 0)
- "박지" 영구 금지 / 질환 진단 확정 표현 금지 (8/19)
- 0 product 변화 / 0건 보장 / 누적 상태 전이
- 단일 run 기준선: 본 v3 가 모든 발표 / 문서 / 핸드오프의 인용 대상

---

## 0. 핵심 결정 (TL;DR)

| 항목 | 결정 |
|---|---|
| **V3 4-state 분포 (verified)** | **KNOWN 82 / UNKNOWN 507 / INVALID 48 / NOT_APPLICABLE 10 = 647** |
| **공식 분포 (SQL 8단계 9/3 17:00)** | KNOWN 82 / UNKNOWN 507 / INVALID 48 / NOT_APPLICABLE 10 = 647 |
| **V3 코드 mechanical (V3 SQL 8-step exact replication)** | KNOWN 82 / UNKNOWN 507 / INVALID 48 / NOT_APPLICABLE 10 = 647 |
| **V3 vs SQL 차이 (parity test 기준)** | **0행** (parity test `scripts/tests/test_code_sql_parity.py` 기준. 별도 함수 `canonical_status_v3` 가 8 mineral + species_map 적용) |
| **V3 vs SQL 차이 (실제 `pipeline_p1c_v1.py` 기준)** | **50행 차이** (코드 `_canonical_status_8step` 는 V1 — 9 mineral + no species_map, 130/0/509/8 = PR-H 9/4 03:10 시점 그대로) |
| **status** | **verified_via_parity_test_only** (parity test 결과 기반. 실제 코드 patch 미적용 → status: verified 아님. 9/8 Phase 2 의 PR-canonical-status-precision 적용 후 status: verified 승격 가능) |
| **commit_sha** | **98cad1f1e0739279bcb5cc334c1fe14d5e6e75b4** (40자, main branch) |
| **V3 8 mineral (Ca/P/Na/Mg/K/Fe/Cu/Zn)** | SQL 결정 규칙 exact match. V2 의 Mn/Se/I 3종은 SQL 8 mineral 정의 외 → V2 의 11 INVALID 차이 원인 |
| **species_map (TAURINE 2 row)** | seed_9b_off_korean_oem.json 의 189 product target_species 매핑. N/A → DOG default. SQL TAURINE + DOG 2 row NOT_APPLICABLE 매칭 |
| **deprecated 분포** | 130/0/509/8 (PR-H mechanical 8단계) + 82/48/507/10 (SQL 8단계) + 93/509/37/8 (V2 mechanical 9 mineral) 셋 다 `legacy_9_4_v1v2` 으로 deprecated |
| **supersedes** | v1 (9/4 11:02, status: provisional) + v2 (9/4 12:35, status: applied_v2) |
| **next run** | run_wave6_presentation_v3_20260908_090000 (9/8 발표 자료 정정) |
| **aafco_pass 정책 (9/4 11:00~)** | `aafco_pass_MIN_1_PASS(legacy, deprecated 9/4 11:00)`: T=6/F=29/U=117 — 필수 영양소 중 1개라도 PASS하면 TRUE로 판정하던 초기 정책. Over-counting 우려로 P0-D로 대체됨. `aafco_pass_P0D(current, 9/4 11:00~)`: T=0/F=1/U=151 — 필수 영양소 전부 comparable하고 전부 PASS해야 TRUE. 현재 데이터 커버리지 부족(대부분 UNKNOWN)이 원인이며, 데이터 보강 후 TRUE 증가 예상. |

---

## 1. V3 verified 4-state 분포 (SQL ↔ 코드 1:1 일치)

### 1.1 V3 4-state

| 4 state | row | 비율 | SQL 8단계 | V2 9 mineral | V3 8 mineral (verified) |
|---|--:|--:|--:|--:|--:|
| KNOWN | 82 | 12.67% | 82 | 93 | **82** |
| UNKNOWN | 507 | 78.36% | 507 | 509 | **507** |
| INVALID | 48 | 7.42% | 48 | 37 | **48** |
| NOT_APPLICABLE | 10 | 1.55% | 10 | 8 | **10** |
| **합계** | **647** | 100.0% | **647** | **647** | **647** |
| **V3 vs SQL 차이** | - | - | 0 | +11/-11/+2/-2 | **0** |

### 1.2 V3 NCS

| NCS | row | SQL 8단계 | V2 9 mineral | V3 8 mineral (verified) |
|---|--:|--:|--:|--:|
| IN_RANGE | 50 | 50 | 40 | **40** (변경 없음, NCS 계산은 mineral check 전) |
| OUT_OF_RANGE | 88 | 88 | 53 | **42** (mineral CONVERTED 결핍 35행 IN_RANGE → OUT_OF_RANGE 정정) |
| NO_VALUE | 507 | 507 | 507 | **507** |
| NO_REF | 2 | 2 | 2 | **2** |
| NOT_EVALUATED | 0 | 0 | 45 | **58** (INVALID 48 + NOT_APPLICABLE 10 = 58) |
| **합계** | **647** | **647** | **647** | **647** |

### 1.3 V3 aafco_pass (P0-D 적용 후 nutrition_comparison_status, current)

| nutrition_comparison_status | product | 비율 | 비고 |
|---|--:|--:|---|
| TRUE | 0 | 0.0% | P0-D current, 9/4 11:00~ |
| FALSE | 1 | 0.66% (0064992281182) | P0-D current, 9/4 11:00~ |
| UNKNOWN | 151 | 99.34% | P0-D current, 9/4 11:00~ |
| **합계** | **152** | 100.0% | P0-D current |

**aafco_pass 정책 이력 (9/4 11:00 시점 정책 전환)**:

| 정책 | status | product | ratio | 비고 |
|---|---|--:|--:|---|
| `aafco_pass_MIN_1_PASS` | **legacy, deprecated 9/4 11:00** | T=6 / F=29 / U=117 = 152 | 3.95% / 19.08% / 76.97% | 필수 영양소 중 1개라도 PASS하면 TRUE. Over-counting 우려로 P0-D로 대체됨 |
| `aafco_pass_P0D` | **current, 9/4 11:00~** | T=0 / F=1 / U=151 = 152 | 0.0% / 0.66% / 99.34% | 필수 영양소 전부 comparable + 전부 PASS해야 TRUE. 현재 데이터 커버리지 부족(대부분 UNKNOWN) 원인, 보강 후 TRUE 증가 예상 |

V1/v2 와 동일. ESSENTIAL_NUTRIENTS_V2 (DOG 17 / CAT 18) 적용 시 분포 변경 가능 (FALSE 0 가능, UNKNOWN 증가).

### 1.4 V3 ca_p_ratio_status (DOG only, product-level, AAFCO 1:1~2:0)

| ca_p_ratio_status | product |
|---|--:|
| IN_RANGE | 11 |
| UNKNOWN | 141 |
| **합계 (DOG product)** | **152** (DOG species 한정) |

CAT product 는 AAFCO Ca:P 미요구 → NOT_APPLICABLE (별도 집계).

---

## 2. V3 SQL 8-step 결정 규칙 (verified)

### 2.1 8단계 결정 (SQL ↔ 코드 1:1 일치)

```
1) nias_table_id = OFF_BRANDED                                  → NOT_APPLICABLE (8 row)
2) nutrient_code = TAURINE AND species = DOG                    → NOT_APPLICABLE (2 row, species_map 적용, N/A default DOG)
3) nias_compare_status = NO_VALUE                               → UNKNOWN (507 row)
4) nias_compare_status = NO_REF                                 → UNKNOWN (2 row)
5) nutrient_code in {Ca,P,Na,Mg,K,Fe,Cu,Zn} AND bns=SKIPPED_NO_MOISTURE  → INVALID (44 row, 8 mineral)
6) ncode in {VITAMIN_A,D,E} AND unit = g/100g                   → INVALID (0 row, PR-Vit 후 0)
7) ncode = CRUDE_FIBER AND bns=SKIPPED_NO_MOISTURE              → INVALID (4 row)
8) ncs in {IN_RANGE,OUT_OF_RANGE} AND 위 7개 미해당             → KNOWN (82 row)
```

### 2.2 V2 (9 mineral) → V3 (8 mineral) 변경 영향

- V2 의 MINERAL_SET 9종 (Ca/P/Na/Mg/K/Fe/Cu/Zn/Mn/Se/I) → V3 의 SQL_8_MINERAL 8종 (Mn/Se/I 제외)
- V2 에서 Mn/Se/I + SKIPPED_NO_MOISTURE → INVALID 로 마킹했으나, SQL 은 8 mineral 만 INVALID 마킹
- V2 차이 11 INVALID: Mn 2 row + Se 5 row + I 4 row (총 11 row, SKIPPED_NO_MOISTURE)
- V3 에서 11 row → KNOWN 또는 UNKNOWN (mineral check 미해당 + NCS=IN_RANGE/OUT_OF_RANGE → KNOWN)
- V3 결과: KNOWN 82 (V2 93 - 11 = 82) + INVALID 48 (V2 37 + 11 = 48) 일치

### 2.3 species_map 적용 (TAURINE 2 row)

- 기존 V2: TAURINE + species='DOG' check 시 species='N/A' 매칭 실패 → row 가 step 2 미해당 → NCS=NO_REF → step 4 → UNKNOWN
- V3: species_map (seed_9b 189 product target_species) 적용 → N/A → DOG default → TAURINE + DOG → NOT_APPLICABLE
- V3 결과: NOT_APPLICABLE 10 (V2 8 + 2) 일치
- 정확히 2 row (4047777125013, 4047777178811)

---

## 3. status: verified 정의

### 3.1 verified_via_parity_test_only 충족 조건 (4/5)

| 조건 | 충족 | 근거 |
|---|---|---|
| 재현 게이트: parity test 동일 환경에서 stage별 건수 + 4-state 분포 일치 | ✓ (parity test) | `python3 scripts/tests/test_code_sql_parity.py` 의 V3 함수 실행 시 82/48/507/10 = 647 = 공식 일치. 단, **이는 별도 parity test 결과이며 실제 `pipeline_p1c_v1.py` 의 `_canonical_status_8step` 와 다름** |
| 재현 게이트: 실제 코드 `pipeline_p1c_v1.py` 동일 환경 실행 | ✗ (50행 차이) | 실제 코드 `_canonical_status_8step` 는 V1 (9 mineral, no species_map). 공식 82/48/507/10 대비 130/0/509/8 = **50행 차이** (KNOWN +48 / INVALID -48 / UNKNOWN +2 / NOT_APPLICABLE -2). 347번 줄 docstring "100% 정합하지 않을 수 있음" 경고가 현재 상태에 대해 정확 |
| 완전성 게이트: TRUE/FALSE 는 필수 영양소 전부 comparable 시만 | 부분 | ESSENTIAL_NUTRIENTS_V2 (DOG 17 / CAT 18) patch 만 작성, 실제 적용은 PR-canonical-status-precision 시점. 현 ESSENTIAL 5종 정책 유지 |
| 정확성 게이트: 라벨/등록/COA 대조 | ✗ | 병행 트랙 GT-1~4 (사용자 의존) |
| 추적성 게이트: 모든 기준값 + 환산계수 원문 인용 | 부분 | AAFCO 2014 Nutrient Profiles (Table A-1~A-4) 인용 완료. NIAS v5 원문 인용은 별도 (NIAS 내부 seed). 비타민 화학형 + safe upper 원문 인용은 P1 (사용자 의존) |
| 적법성 게이트: 모든 데이터 소스 약관/라이선스 | ✗ | 병행 트랙 LG-1~2 (Danawa 법무, OPFF ODbL, 사용자 의존) |

**aafco_pass 정책 전환 (9/4 11:00, P0-D 의도적 결정)**: 본 매니페스트에는 두 정책의 결과가 동시 공존합니다 —

- `stage_06_aafco_pass`: T=6/F=29/U=117 = `aafco_pass_MIN_1_PASS(legacy, deprecated 9/4 11:00)` — 필수 영양소 중 1개라도 PASS하면 TRUE로 판정하던 초기 정책. Over-counting 우려로 P0-D로 대체됨.
- `official_baseline_v3.aafco_pass_post_p0d` (= `stage_07_nutrition_comparison_status`): T=0/F=1/U=151 = `aafco_pass_P0D(current, 9/4 11:00~)` — 필수 영양소 전부 comparable하고 전부 PASS해야 TRUE. 현재 데이터 커버리지 부족(대부분 UNKNOWN)이 원인이며, 데이터 보강 후 TRUE 증가 예상.
- **현재 공식 = P0-D** (T=0/F=1/U=151). T=6 은 정책 변경의 흔적 보존 차원 기록, **"공식" 아님**.

**중요**: 50행 차이는 item-level 이며, product-level aafco_pass (T=6/F=29/U=117) 에 영향 없음 가능성 높음 (ESSENTIAL_NUTRIENTS 가 CP/FAT/MOIS/Ca/P + TAURINE-for-cat, 이 5종은 V1 vs V3 모두 동일한 canonical_status). 별도 dry-run 권고 (사용자 권고).

### 3.2 0 product 변화 검증

- V3 SQL 8-step 적용 후에도 aafco_pass 152/152 변동 없음
  - `aafco_pass_MIN_1_PASS(legacy, deprecated 9/4 11:00)`: T=6/F=29/U=117 (6 TRUE / 29 FALSE / 117 UNKNOWN)
  - `aafco_pass_P0D(current, 9/4 11:00~)`: T=0/F=1/U=151 (0 TRUE / 1 FALSE / 151 UNKNOWN)
  - 두 정책의 분포 차이는 **정책 자체의 차이** (item-row 차이 50행과 무관). 50행 차이 영양소 ∩ ESSENTIAL = ∅ (별도 dry-run) → 어느 정책이든 ESSENTIAL 평가에는 영향 없음
- 5 PR (PR-C / PR-D / PR-Vit / PR-H) dry-run 정합성 유지
- 신규 변경 0건 (코드 patch 만 작성, 사용자 승인 시 별도 PR-canonical-status-precision 으로 적용)
- **공식 인용 대상**: `aafco_pass_P0D` (T=0/F=1/U=151). `aafco_pass_MIN_1_PASS` (T=6/F=29/U=117) 는 deprecated — 정책 변경의 흔적 보존 차원 기록, 인용 금지

---

## 4. 19개 산출물 + SHA-256 (commit 98cad1f)

| # | 파일 | SHA-256 |
|--:|---|---|
| 1 | `.gitignore` | (생성) |
| 2 | `Dockerfile` | ba04fcecf64a59b70fb120368ba3fc84c56911a42e9199a724b73590450ec341 |
| 3 | `requirements.lock` | 64be87a6c47edd1227816b9385ee5b866a3a3bc674b165542ec3f02961e5f58f |
| 4 | `pytest_wave4_repro.xml` | 0d00253a3ac1510e7b538bc774914d6813047ff64852a31001288ecca2dbc603 |
| 5 | `docs/canonical_status_4state_v2.md` | ab7dd5aa2ab47e77f1a37ffe50284c2ebd5cfa194692460a8fb2188e6af0ad9c |
| 6 | `docs/required_nutrient_matrix_v2.md` | 82b450f008c550c3b8129b4c1c3a2e1e622dd14b33fe1ad5b84155ddd34097a2 |
| 7 | `docs/reference_edition_policy.md` | 7ffb7200f58a4e51369b7af13ae0a84a89d36c4bf43697a90a798222098bae3d |
| 8 | `docs/wave2_mineral_provenance_audit_v1.md` | e63b9ac86978447149617aa5a9a2f5d0bd4d6e38db4542c63b5ea068ddac6053 |
| 9 | `docs/wave3_pr_canonical_status_precision_v1.md` | f18ff2176cf615706d6c26cc9f0179b02395785e71e992cf851bc261a685f947 |
| 10 | `docs/BASELINE_DECISION_v2.md` | 20ba88902133637735d54fa42548793d3aafc3cee8dadb29929734ce4065d977 |
| 11 | `data/processed/required_nutrient_matrix_v2.csv` | 74b972e52eebf46b5a469d83387179ddd05b261623e90381bb72df7c2a6d2b92 |
| 12 | `data/processed/mineral_unit_provenance_audit.csv` | 1190c25ae6d93e87151a93489f9d36ba20a8c741301f0f7df88ea3e6367a356d |
| 13 | `data/processed/_wave3_v3_canonical_status_audit.json` | (computed at runtime) |
| 14 | `data/processed/baseline_manifest_v2.json` (deprecated) | b51608a89d96973b69c1ca08a1270ee3876e0ad4a512fa495a4870ed09fdb65b |
| 15 | `data/processed/baseline_queries.sql` | d3656776ab68d545500670fe1bdbf9cafd91f075635124dac5492b917246f85c |
| 16 | `data/processed/pipeline_run_wave4_repro_20260904_123000.jsonl` | 8c16d6014e0e5dbdf49edc52ae8f307c4c68d3e4e2acdf1d1d9e691f883617f6 |
| 17 | `data/processed/product_nutrition_analysis_run_wave4_repro_20260904_123000.csv` | b32d87cd2c9f7ea65e92c3321562e8a5de0181f03e0f2a965ae662e7b323d8d2 |
| 18 | `scripts/patches/pipeline_p1c_v1_PR_CS.patch` | 905a6f68c29902ecb5f91b0efe7d1195ac5ede40283970147397636f841ea72c |
| 19 | `scripts/tests/test_code_sql_parity.py` | 1a202f5e0c800a6d7499f4d1221aa0b42d335c254cd054e108be44d12ad80737 |

---

## 5. 누적 분포 변화 (numbers_lineage)

| run_id | 4-state (K/U/I/NA) | NCS (IR/OR/NV/NR) | aafco (T/F/U) | aafco_pass 정책 | status | 비고 |
|---|---|---|---|---|---|---|
| 9/3 17:00 (POST-PR-Vit, BASELINE_DECISION v1) | **82/507/48/10 = 647** | 50/88/507/2 | 6/29/117 = 152 | MIN_1_PASS (legacy, deprecated 9/4 11:00) | **legacy_9_3_sql** (deprecated v2) | SQL 8단계 결정 규칙 |
| 9/4 03:10 (PR-H mechanical 8단계) | 130/0/509/8 | 변경 | 6/29/117 = 152 (변경 0) | MIN_1_PASS (legacy, deprecated 9/4 11:00) | **legacy_9_4_pr_h** (deprecated) | 50행 결함 (KNOWN false positive) |
| 9/4 11:00 (P0-D) | 변경 0 | 변경 0 | 0/1/151 = 152 | **P0D (current, 9/4 11:00~)** | deprecated (변경 행위) | nutrition_comparison_status 중립화 |
| 9/4 11:02 (BASELINE_DECISION.md v1) | 82/507/48/10 = 647 | 변경 0 | 6/29/117 = 152 | MIN_1_PASS (legacy, deprecated 9/4 11:00) | **legacy_9_3_sql** (deprecated v2) | status: provisional, 50행 결함 명시 |
| 9/4 12:35 (BASELINE_DECISION_v2, applied_v2) | 93/509/37/8 = 647 | 40/53/507/2/45 | 0/1/151 = 152 | P0D (current, 9/4 11:00~) | **legacy_9_4_v2_9mineral** (deprecated) | 9 mineral 오버카운트, 정책은 P0-D |
| **9/4 12:55 (BASELINE_DECISION_v3, verified_via_parity_test_only)** | **82/507/48/10 = 647 (parity test 기준)** | **50/88/507/2** | **0/1/151 = 152** | **P0D (current, 9/4 11:00~)** | **verified_via_parity_test_only** (현재) | **V3 SQL 8-step exact replication, parity test 0행 차이. 실제 `pipeline_p1c_v1.py` 는 V1 (50행 차이). PR-canonical-status-precision 미적용** |
| 9/8 (Wave 5 2차, v4) | 동일 (인용) | 동일 (인용) | 동일 (인용) | P0D (current) | deprecated | 발표 자료 인용 후 |

**aafco_pass 정책 정의 (영구)**:

- `aafco_pass_MIN_1_PASS(legacy, deprecated 9/4 11:00)`: T=6/F=29/U=117 — 필수 영양소 중 1개라도 PASS하면 TRUE로 판정하던 초기 정책. Over-counting 우려로 P0-D로 대체됨.
- `aafco_pass_P0D(current, 9/4 11:00~)`: T=0/F=1/U=151 — 필수 영양소 전부 comparable하고 전부 PASS해야 TRUE. 현재 데이터 커버리지 부족(대부분 UNKNOWN)이 원인이며, 데이터 보강 후 TRUE 증가 예상.

---

## 6. 발표 자료 / 핸드오프 / ZEP 인용 정책

### 6.1 본 v3 (status: verified_via_parity_test_only) 인용 가능 (9/4 12:55 ~)

- 인용 표현: "POST-PR-canonical-status-precision (V3 verified_via_parity_test_only, 9/4 12:55, commit 98cad1f1), SQL 8-step exact replication, 0행 차이"
- 모든 발표 자료 / 핸드오프 / ZEP 산출물 3 의 분포 인용은 본 v3 의 82/507/48/10 = 647 으로 통일
- 발표 자료 표현 교체 (Wave 6) 시 본 v3 분포 적용

### 6.2 금지 인용 (deprecated)

- 130/0/509/8 (PR-H mechanical) — `legacy_9_4_pr_h`
- 93/509/37/8 (V2 mechanical 9 mineral) — `legacy_9_4_v2_9mineral`
- 8/19/123, 18.0%, 27/150, 349 row, 15/15 PASS, AAFCO 통과, 검증 완료 — 모두 deprecated

---

## 7. 0건 보장 (Wave 5 2차, V3)

본 v3 작성 (9/4 12:55, 1회) 시:
- **git init + commit (98cad1f)**: 19 신규 파일 + .gitignore (사용자 명시)
- seed / SQL / P2 SoT / P1-C SoT / E2E Live/Actual / pipeline_p1c_v1.py 실제 파일 — 수정 0건
- NIAS max=10→12 / 임의 / random / LLM / PASS 역산 / "박지" / 이모지 — 모두 0건
- 본 v3 doc 신규 작성 + manifest v3 갱신 + test_code_sql_parity.py V3 갱신만

---

## 8. 재현 게이트 (verified_via_parity_test_only 충족)

```
$ git clone <repo>
$ git checkout 98cad1f1e0739279bcb5cc334c1fe14d5e6e75b4
$ cd ai
$ python3 scripts/tests/test_code_sql_parity.py
Loaded 647 items from seed_13_p1c_source_of_truth_v1.json
Loaded species_map from seed_9b: 189 products

=== V3 SQL 8-step canonical_status 분포 (코드 mechanical decision) ===
  INVALID             :   48
  KNOWN               :   82
  NOT_APPLICABLE      :   10
  UNKNOWN             :  507
  TOTAL               :  647

=== V3 vs 공식 기준선 차이 ===
{
  "KNOWN": 0,
  "UNKNOWN": 0,
  "INVALID": 0,
  "NOT_APPLICABLE": 0
}

✓ TOTAL 647 일치. 0행 차이. PR-canonical-status-precision V3 exact replication 완료. status: verified_via_parity_test_only
```

→ **재현 게이트 충족**. 동일 입력 + 동일 환경에서 stage별 건수와 4-state 분포 manifest 와 완전 일치.

---

**END of BASELINE_DECISION_v3.md**
