# 알레르기 매핑 Human Precision Audit 표본 감사 (P2)

본 문서는 영양성분 분석 AI 전체 중 알레르기 안전성 판단 서브시스템의 검증 문서이다. NIAS 영양 기준 비교, AAFCO 판정, 보증성분 정규화 등 영양성분 전체 기능의 완료를 의미하지 않는다.

## v1 표본 실측

`allergen_mapping_precision_audit_p2.csv`는 60행이며 OPFF/OEM/GLOBAL 각 20행이다. 그러나 unique product는 12개뿐이고 source별 unique product도 각 4개다. 한 product에 최대 13행이 집중되어 있다. unique raw ingredient text는 40개, raw text + allergen pair는 43개이며 duplicate semantic mapping은 2행이다.

| 목적 | v1 판정 | 근거 |
|---|---|---|
| evidence-row spot precision audit | `LIMITED` | source quota는 균형이나 product 집중과 semantic 반복이 있어 선택된 evidence 행에만 한정해 해석해야 한다. |
| product-level representative precision audit | `NOT_SUITABLE` | 60행이 12개 product에 집중되어 product-level 대표성이 없다. |
| multilingual precision audit | `NOT_SUITABLE` | `MULTILINGUAL` 3행, `COMPOUND` 6행이며 FR/KO/OCR 표본이 없다. |

v1 mapping method는 `CANONICAL_ALIAS` 42행, `CANONICAL_EXACT` 18행이다. source × method 분포는 OPFF 18/2, OEM 10/10, GLOBAL 14/6 (`CANONICAL_ALIAS`/`CANONICAL_EXACT`)이다. text class는 EN 51, COMPOUND 6, MULTILINGUAL 3이다.

## v2 재표본

v1은 삭제·수정·덮어쓰기 하지 않았다. 사람이 아직 0/60이므로 별도 파일 `allergen_mapping_precision_audit_p2_v2.csv`를 생성했다. sampling seed는 `20260915`이며 자세한 계약은 `allergen_precision_audit_sampling_manifest_p2.json`에 기록했다.

| 항목 | v1 | v2 |
|---|---:|---:|
| rows | 60 | 60 |
| OPFF/OEM/GLOBAL | 20/20/20 | 20/20/20 |
| unique products | 12 | 60 |
| 최대 행/product | 13 | 1 |
| duplicate semantic mapping | 2 | 0 |
| `CANONICAL_ALIAS` / `CANONICAL_EXACT` | 42 / 18 | 50 / 10 |
| EN / NON_ENGLISH / COMPOUND / OCR_NOISY | 51 / 3 / 6 / 0 | 28 / 21 / 10 / 1 |

v2는 source별 quota를 유지하면서 product 다양성을 먼저 최대화하고, 이후 edge case, text 다양성, `CANONICAL_EXACT`를 deterministic하게 우선했다. `corn gluten meal`, `farine de gluten de maïs`, poultry/by-product, flavor, sweet potato, `dinde` 관련 표현은 문제가 있다는 이유만으로 제거하지 않았고 표본에 포함될 수 있게 유지했다.

`MULTILINGUAL`은 한글과 라틴 문자가 함께 있을 때만 사용한다. 프랑스어·이탈리아어처럼 비ASCII 문자가 포함된 단일 언어 텍스트를 특정 언어로 deterministic하게 확정하지 못할 때는 `NON_ENGLISH`로 기록한다. 따라서 v2의 과거 `MULTILINGUAL` 21행 표기는 `NON_ENGLISH` 21행으로 정정했다. 이 정정은 audit row의 mapping 값이나 human label을 변경하지 않는다.

## Canonical parity와 freeze

`allergen_precision_audit_v2_parity_p2.json`에서 current SQLite `product_allergen_refs`를 우선 기준으로 전수 대조했다. 60행 모두 `exact_semantic_match`이며 changed mapping 0, missing 0, ambiguous multiple match 0으로 `AUDIT_CANONICAL_PARITY = PASS`다. 각 행의 `evidence_id`와 `component_occurrence_id`는 trace 목적으로 artifact에 기록했다.

v2는 population-random sample이 아니라 `CANONICAL_ALIAS` 및 edge case를 의도적으로 과대표집한 위험 기반 정밀도 audit이다. 따라서 결과를 전체 catalog의 population-weighted precision으로 직접 일반화하지 않는다. v2는 risk-based evidence precision audit 및 source-stratified safety mapping audit에는 `SUITABLE`하지만, product-level population representative precision과 multilingual 전체 모집단 precision에는 `LIMITED`다.

## Human reviewer용 최종 표본

Human review에는 v2를 사용한다. `review_label`과 `reviewer_note`는 모두 공란이며, 사람이 직접 입력할 때까지 자동으로 생성하지 않는다. Human Review 시작 이후에는 60/60 완료 전까지 row 교체, mapping 수정, dictionary 변경을 하지 않는다.
