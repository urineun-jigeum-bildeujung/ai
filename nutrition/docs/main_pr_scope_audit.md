# 영양성분 main 반영 범위와 PR #99 선별 감사

## 기준과 담당자

- 원격 main 기준: `dac80459d9777c1dc822ec90d56a9b99960d1a55`.
- 조사한 develop: `2cc0377901630789c1db564d9b834208ab287525`.
- #99, #100 담당자는 GitHub 조회에서 모두 `Aku-Korea`로 확인했다.
- #100/#101의 최신 Nutrition 기준점: `adae4c8cdaec2067127a1e827cddc27639342f21`.
- #99 추가 자료 기준점: `67f69fdbe83f99fa3eda4c1a331d122f00f63e00`.

develop 전체 history를 main에 병합하지 않는다. main에서 만든 별도 브랜치로
#100/#101 소유 경로 57개를 blob 단위 그대로 옮겼다. 모든 57개 blob이 adae4c8과
일치하는지 검사했다. 이후 #99에서 아래 9개만 추가 복원했다.
main에 이미 존재하는 다른 파트 코드는 삭제하거나 변경하지 않는다.

## #99에서 추가 복원한 내용

| 영역 | 복원 이유 | 검증 범위 |
|---|---|---|
| P0/P1 Gold evidence intake | 기존 Gold gate/adapter가 받는 사람 검토 evidence의 typed intake·분리 저장 함수와 그 회귀가 #100에 누락됨 | 합성 fixture로 identity/provenance/누락값/충돌 거부 및 artifact → adapter를 검증. 실제 Gold 승인/수집은 수행하지 않음 |
| P2 SQLite catalog | 기존 runtime이 조회하는 catalog 내용이 #100에 빠져 clean checkout에서 빈 DB를 새로 생성함 | #99 원본 SQLite를 byte 동일하게 복원. 실제 current get_refs로 전체 상품 조회 |
| P1.1 lineage 원천 | catalog의 parent와 evidence 의미를 추적하고 교차 검증하는 근거 | component/refs JSON과 DB semantic 비교 및 orphan/버전/상태 회귀 |
| P2 lineage 정책·회귀 | 현재 repository의 parent·stale·serialization fail-close 계약을 보호 | 기존 정책 문서와 test_p2_component_lineage.py를 원본 그대로 복원 |

복원 파일 9개:

- `nutrition/scripts/nutrition/gold_evidence_intake.py`
- `nutrition/tests/test_gold_evidence_intake.py`
- `nutrition/tests/test_gold_operational_adapter.py`
- `nutrition/data/processed/allergen_evidence_catalog_p2.db`
- `nutrition/data/processed/product_allergen_refs_p1_1.json`
- `nutrition/data/processed/product_ingredient_components_p1_1.json`
- `nutrition/data/processed/allergen_lineage_catalog_migration_p2.json`
- `nutrition/tests/test_p2_component_lineage.py`
- `nutrition/docs/allergen_lineage_integrity_policy_p2.md`

## 복원하지 않은 항목

- #99의 구형 API/매처/문서로 #100/#101의 최신 수정사항을 덮어쓰지 않음.
- #99의 P1 pre-lineage evidence, historical metrics/report, collection queue,
  Human Review workbook 및 자동 sample 생성 도구는 현재 실행에 필요한 최소 복원에
  포함하지 않음. 기존 PR/브랜치에 남아 있으며 삭제하지 않음.
- batch 전체 재생성 스크립트는 이번 복원 대상이 아님. 특히 구형
  `build_product_allergen_refs.py`는 parent lineage 없이 `replace_all_refs`를
  호출하므로 현재 P2 catalog 갱신기로 무조건 복원/실행하지 않음.
- 후보 데이터·Gold 값을 새로 만들거나 human label·dictionary/alias를 승인하지 않음.
- 다른 팀원 #113의 `nutrition/Dockerfile`, 공용 `Jenkinsfile` 제외.
- 로컬 Service DB 준비 commit `cbbc7e6`, `693a27f`, `995c268` 제외.
  따라서 그 후속 입력 안전·Docker 보강이 이번 main PR에 들어갔다고 주장하지 않음.
- Recommendation/Repurchase, 공용 인프라, 루트 README/.gitignore/.gitattributes 변경 없음.
- `.github/workflows/nutrition-ci.yml`만 사용자가 허용한 Nutrition 전용 CI 예외로 포함.
  repository 폴더상 nutrition/ 밖 파일은 1개지만, 허용된 Nutrition 소유 범위 밖 변경은 0개.

## 실제 검증

- #100/#101 선택본: `python -m pytest nutrition/tests -q` → 82 passed / 1 warning.
- #99 9개 복원 후: 같은 명령 → 120 passed / 1 warning. 추가 회귀 38건.
- warning: Starlette TestClient의 AnyIO BlockingPortal alias deprecation.
- Python 3.11.15; compile check 통과.
- SQLite `PRAGMA integrity_check` → ok.
- components 2,814 / P1.1 evidence 2,856 / catalog evidence 2,852.
- source evidence identity 집합과 DB evidence_id 집합 일치.
- 공유 필드 전체 비교: catalog 2,852행 semantic 일치; orphan 0.
- current get_refs로 119개 상품 모두 PRECOMPUTED 확인.
- SQLite와 P1.1 JSON 2개는 검사/테스트 전후 #99 원본 SHA-256과 동일.
- API/Rule Engine/current dictionary/raw/Gold gate는 adae4c8과 동일.
- JSON artifact에서 credential 패턴 검색 결과 없음. SQLite는 상품·원재료 lineage
  테이블이며 사용자 계정/credential table을 포함하지 않음을 확인.
- 실제 persisted Gold 또는 AWS DB E2E는 수행하지 않았다. 120개 회귀는
  영양학적 적합성·전체 catalog precision·운영 인증의 검증 완료를 의미하지 않는다.

## 복원으로 달라지는 실행 범위와 남은 한계

catalog가 포함되므로 해당 상품은 빈 catalog의 runtime parsing 대신 PRECOMPUTED
lineage를 사용한다. 숫자·매핑을 새로 생성한 것이 아니라 기존 P2 snapshot 복원이다.
이는 관찰 가능한 실행 경로 변경이므로 단순 문서 추가로 취급하지 않는다.
Human Precision 결과의 재판정, turkey/dinde 승인, population precision 주장은 없다.

현재 개발 브랜치의 후속 입력 hardening과 AWS DB/auth/result persistence는 별도
작업이다. 이번 PR은 이미 develop에 반영된 공개 runtime과 선별된 기존 P0/P1/P2
검증 근거를 main으로 옮기는 범위이며 Production Ready 승인이 아니다.

## 전체 changed files

총 67개: Nutrition 폴더 66개와 Nutrition 전용 CI 1개.
PR 생성 직전 `git diff --name-only main...HEAD`와 이 목록을 정확히 대조한다.

```text
.github/workflows/nutrition-ci.yml
nutrition/API/error_response.json
nutrition/API/nutrition_request.json
nutrition/API/nutrition_response.json
nutrition/README.md
nutrition/data/eval/verified_evidence_gold_v1.json
nutrition/data/processed/allergen_evidence_catalog_p2.db
nutrition/data/processed/allergen_lineage_catalog_migration_p2.json
nutrition/data/processed/baseline_manifest_v3.json
nutrition/data/processed/nutrition_reference_nias_2024_parity_p0_v1.json
nutrition/data/processed/opff_api_cache_v1.json
nutrition/data/processed/product_allergen_refs_p1_1.json
nutrition/data/processed/product_ingredient_components_p1_1.json
nutrition/data/processed/required_nutrient_matrix_v2.csv
nutrition/data/raw/seed_11_allergen_ingredient_map.json
nutrition/data/raw/seed_11_allergen_ingredient_map_v3.json
nutrition/data/raw/seed_13_guaranteed_analysis.json
nutrition/data/raw/seed_14_nutrition_reference_v5.json
nutrition/data/raw/seed_46_nias_2024_nutrient_tables.json
nutrition/data/raw/seed_9_global_brands_v2.json
nutrition/data/raw/seed_9_placeholder_feed_opff.json
nutrition/data/raw/seed_9b_off_korean_oem.json
nutrition/data/raw/seed_feed_codes.json
nutrition/data/raw/seed_nutrition_standard.json
nutrition/data/raw/seed_persona5_nutrient_map.json
nutrition/docs/BASELINE_DECISION_v3.md
nutrition/docs/allergen_lineage_integrity_policy_p2.md
nutrition/docs/canonical_input_contract.md
nutrition/docs/canonical_status_4state_v2.md
nutrition/docs/environment_contract.md
nutrition/docs/main_pr_scope_audit.md
nutrition/docs/nutrition_ai_api_spec_v3_draft.md
nutrition/docs/nutrition_persisted_product_e2e_audit_v1.md
nutrition/docs/reference_edition_policy.md
nutrition/docs/required_nutrient_matrix_v2.md
nutrition/docs/service_db_contract.md
nutrition/docs/service_db_integration_checklist.md
nutrition/docs/service_integration_architecture.md
nutrition/requirements.lock
nutrition/scripts/api_nutrition.py
nutrition/scripts/match_v1.py
nutrition/scripts/nutrition/__init__.py
nutrition/scripts/nutrition/allergen_catalog_versions.py
nutrition/scripts/nutrition/allergen_repository.py
nutrition/scripts/nutrition/allergen_service.py
nutrition/scripts/nutrition/danawa_adapter.py
nutrition/scripts/nutrition/gold_evidence_gate.py
nutrition/scripts/nutrition/gold_evidence_intake.py
nutrition/scripts/nutrition/gtin_validation.py
nutrition/scripts/nutrition/match_v1_1_category.py
nutrition/scripts/nutrition/normalizer.py
nutrition/scripts/nutrition/nutrition_readiness.py
nutrition/scripts/nutrition/product_input_adapter.py
nutrition/scripts/nutrition/reference_parity.py
nutrition/scripts/pipeline_p1c_v1.py
nutrition/tests/test_danawa_adapter.py
nutrition/tests/test_gold_evidence_gate.py
nutrition/tests/test_gold_evidence_intake.py
nutrition/tests/test_gold_operational_adapter.py
nutrition/tests/test_nutrition_readiness.py
nutrition/tests/test_nutrition_reference_parity_p0.py
nutrition/tests/test_p0_contracts.py
nutrition/tests/test_p0_nutrition_safety.py
nutrition/tests/test_p2_component_lineage.py
nutrition/tests/test_persisted_product_adapter.py
nutrition/tests/test_review_runtime_regressions.py
nutrition/tests/test_runtime_e2e.py
```
