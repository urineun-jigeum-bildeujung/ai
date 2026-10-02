# Nutrition 최종 계약 v3 구현 검증 — 2026-10-03

### Validation provenance

This document records a local validation run for the Nutrition v3 change set.
The change set was originally produced from the historical Nutrition source
baseline and then ported, without cross-part changes, onto the latest develop
integration worktree used for final mechanical validation.

Machine-specific worktree paths and local user paths are intentionally omitted
from the repository document. Full machine-local audit artifacts remain outside
the repository under the local audit directory and are not commit evidence.


## REPOSITORY — TEST_DONE

- Root: `<source-worktree>`.
- Remote: `https://github.com/urineun-jigeum-bildeujung/ai.git`.
- Branch: `codex/nutrition-profile-birth-contract`.
- HEAD: `bf2470e56464d6b02c79ddafa9d792fcff0756bb` (변경 전후 동일).
- 시작 시 clean. HEAD는 develop 기반의 병합 전 기능 커밋이다. 병합 SHA `f0c70a6cc0d1055e37e51898f3890ff8476ff5fc`는 조상이 아니지만 `git diff <baseline> -- nutrition`이 비어 있어 Nutrition 기준선 파일은 동일했다. reset하지 않았다.
- Source of Truth: 사용자가 지정한 `nutrition_master_implementation_contract_v3.md`, Section 42/43. 설계를 재제안하지 않았고 지정 순서로 구현했다.
- DB write/DDL, commit/push/PR/merge, 배포 및 다른 파트 수정 없음.

## SOURCE FIELDS — TEST_DONE

- Pet SELECT/return에 `target_breed_size` 추가.
- 단건 Product와 active product 목록에 `target_breed_size`, `feeding_target`, `feeding_method` 추가.
- 컬럼 근거: 저장된 Backend member `V1__baseline_schema.sql`, `PetJpaEntity.java`, product `domain/product/Product.java`의 명시적 column mapping.
- SQL ownership/active/deleted 조건, SELECT-only·repeatable-read·timeout·오류 정보 제한을 유지했다. 단건/목록 새 필드 일치 테스트 PASS.
- 실제 Service DB URL은 현재 실행 환경에 미설정이다. 새 컬럼의 live SELECT 검증으로 주장하지 않는다.

## IMPLEMENTATION

| 기능 | 상태 | 검증된 동작 |
|---|---|---|
| PRODUCT TARGET | RUNTIME_DONE | 완료 개월 <12 GROWTH / 12..83 ADULT / 84+ SENIOR. 기존 Nutrition reference stage와 분리. 12 exact label, NFKC+outer strip, Service 우선·충돌 fail-close. |
| SIZE | RUNTIME_DONE | DOG membership/unknown/mismatch, CAT NOT_APPLICABLE. Overall CONFLICT > MISMATCH > UNKNOWN > MATCHED. |
| PRESENTATION | RUNTIME_DONE | 실제 엔진 original/normalized 값·NIAS min/max·canonical status·source/provenance·counts·정렬. 선택된 reference 단위는 curated artifact에서 확인한다. UI geometry·기능성 주장 없음. |
| SUITABILITY | RUNTIME_DONE | 기존 essential_status/counts만 사용. 모든 required comparable일 때 정수 ratio. Safety·target·unknown·conflict gate에서 null. 가중치 없음. |
| FEEDING | RUNTIME_DONE | healthy_merck_v1, RER/MER/grams 계산 유지. 정상 Mock energy는 READY, energy 부족은 ENERGY_REQUIREMENT_READY, Safety 제외는 BLOCKED/MER·grams null. |
| PRODUCT LABEL | RUNTIME_DONE | Service 원문 급여 target/method와 age/size를 표시 metadata로만 제공. 계산에 parsing하지 않는다. |
| COMPARE | RUNTIME_DONE | 동일 auth/ownership 경계, Pet 한 번·상품 두 번 조회, analyze_service_records 두 번 재사용. score와 동등 basis/unit nutrient 비교. 501 제거, health에 availability 반영. |

위 RUNTIME_DONE은 **로컬 실제 HTTP 서버 + 명시적 synthetic Service source 경계**에서 검증했다는 뜻이다. 원격 배포 상태가 아니다.

## TEST — TEST_DONE

- 변경 전 전체 **441 PASS**. 첫 실행은 기존 환경의 psycopg2 누락으로 440 PASS/1 dependency failure였으며, 기존 `<local-path-redacted>`를 연결한 재실행으로 441 PASS를 확인하고 구현을 시작했다. 설치·코드 수정·skip 없음.
- 변경 후 전체 **581 PASS**, dependency deprecation warning 1개.
- 기존 테스트 삭제 0, skip 증가 0. Feeding 정책이 계약에 의해 확정되면서 이전 미확정 기대값과 compare 501 기대값만 갱신했다. Middleware 단위 테스트의 stub 경계는 direct endpoint wrapper로 옮겼다.
- 12 label, 12/84 month boundary, precedence/conflict, strict fixture identity, score 100/80/60/0 및 null gates, MER species/month/neuter, supported dry/wet feeding, validation/auth/error/privacy, compare READY/PARTIAL/UNAVAILABLE/tie/basis/unit, optional direct inputs, legacy response axes를 검증했다.

## REGRESSION — TEST_DONE

- 이전 실행 기록에 있는 exact focused 파일 집합을 복원했다: `test_service_toxic_safety.py`, `test_service_namespace_mapping.py`, `test_p0_contracts.py`, `test_p2_component_lineage.py`, `test_feeding.py`, `test_feeding_service_contract.py`.
- historical SOURCE_HEAD의 동일 focused 집합은 historical implementation에서 **145 PASS**를 재현했다. v3에 의해 의미가 바뀌지 않은 historical subset은 v3 구현에서 **133 PASS**였고, historical Service Feeding 계약은 **6 PASS / 6 intentional FAIL**로 승인된 계약 변경 6건만 차이가 났다. 현재 v3 focused Toxic/Allergen/Feeding 집합은 **151 PASS**다. 이를 '기존 145개가 v3에서도 그대로 145 PASS'로 해석하지 않는다.
- 저장된 286개 Service Mock source와 기존 `nutrition-profile-develop-replay.json`의 legacy hashes 대조 **286/286 동일**.
- 기존 projection 제외 feeding/product_allergen_refs를 유지하고 신규 target/presentation/suitability/label 및 additive provenance key만 제거하여 기존 필드 전체를 대조했다. 기존 Safety/reference/coverage 판정을 제외하지 않았다.
- Fixture 상태 READY/PARTIAL/UNAVAILABLE **107/13/166** 그대로.
- Python scripts/tests **58파일 compile PASS**. `git diff --check` PASS. 신규 파일 whitespace 검사 PASS.

## API CONTRACT — RUNTIME_DONE

- PetIn/ProductIn의 새 필드는 optional이며 기존 direct caller는 그대로 작동한다. Service 분석은 target_compatibility/suitability/presentation/product_label을 additive하게 제공한다.
- Compare: `{pet_id: positive strict int, product_ids: [distinct positive strict int, distinct positive strict int], allergy_profile_status?: UNKNOWN|KNOWN_NONE|KNOWN_LIST}`. extra field 금지.
- validation 422, auth 401/기존 configuration 503, ownership/product missing 404, DB unavailable 503, invalid source 422 유지.
- READY는 두 numeric scores, PARTIAL은 score 불완전 + 공통 비교 영양소, UNAVAILABLE은 둘 다 없음. tie의 higher id는 null. Winner/better/BEST/HEALTHIEST claim 없음.

## PROVENANCE — RUNTIME_DONE

- MOCK_INTEGRATION_FIXTURE 이름 유지, `data_generation_type=SCHEMA_DRIVEN_SYNTHETIC`, `production_evidence=false`, `schema_contract=SERVICE_DB_COMPATIBLE` 추가.
- Derived integration provenance: DERIVED_RULE_RESULT / SCHEMA_DRIVEN_SYNTHETIC_DATA.
- target fixture **86개**: 저장된 read-only source에서 연령 target이 비어 있는 실제 Mock ID/SKU만 수록했다. artifact에 source SHA256과 deterministic schema-driven label 생성 규칙을 기록했다. 상품명 추론·새 Service 상품 생성 없음.
- Service 값이 다르면 CONFLICT. target fixture로 AAFCO stage나 historical safety blockers를 해제하지 않는다.
- Energy 3850/750 kcal/kg는 synthetic이며 제조사 evidence가 아니다. Mock namespace·exact ID/SKU·resolved nutrition fixture/profile·FOOD·supported form gate 유지. non-Mock fallback 없음.

## RUNTIME — RUNTIME_DONE

- Uvicorn을 127.0.0.1 임시 포트로 실행한 실제 HTTP **13요청 / 58 assertion PASS**.
- `/health`, `/ready`, `/metrics` 모두 200. `/ready`는 **local_artifact mode**다. live service dependency 17/17 검증으로 주장하지 않는다.
- MATCHED target/size, numeric score 100, actual presentation rows, feeding READY, Safety BLOCKED와 null amounts, 내부 compare READY 및 unauthenticated 401 검증.
- 검증 서버는 종료했다. 영구 runtime 설정·배포·Pod 수정 없음.

## E2E — BLOCKED_EXTERNAL

- 로컬 source-boundary HTTP에서 KNOWN_NONE, KNOWN_LIST, growth/adult reference, senior/reference 분리, Product 141 양고기→lamb/귀리→oat/당근·비트 UNRESOLVED, SALMON/TUNA specific fail-close 검증 PASS.
- 실제 Service DB/Gateway E2E는 현재 credential/접속 환경과 새 배포가 없어 수행하지 않았다. 이전 Gateway 19요청·46 assertion의 신규 실접속 재검증으로 주장하지 않는다.

## EXTERNAL HANDOFF — BLOCKED_EXTERNAL

- `GATEWAY_COMPARE_ROUTE_REQUIRED`: public compare Gateway route.
- FE Product Detail mock removal.
- FE Compare actual API hookup.

## UNRESOLVED — BLOCKED_EXTERNAL

원격 새 배포와 실제 Service DB/Gateway 검증, 위 external handoff. 실제 제조사 정확도·의료적 처방은 계약상 평가 범위 밖이다. 실제 schema 충돌 또는 nutrition 밖 필수 코드 변경은 관찰되지 않았다.

## CHANGED FILES — TEST_DONE

변경 파일은 모두 아래 Nutrition 경계 안에 있다.

- `nutrition/README.md`
- `nutrition/data/integration/mock_energy_v1.json`
- `nutrition/data/integration/mock_product_target_v1.json`
- `nutrition/docs/contract_v3_validation_20261003.md`
- `nutrition/scripts/api_nutrition.py`
- `nutrition/scripts/nutrition/feeding.py`
- `nutrition/scripts/nutrition/mer_coefficient_policy_v2.py`
- `nutrition/scripts/nutrition/mock_feeding_fixture.py`
- `nutrition/scripts/nutrition/mock_integration_fixture.py`
- `nutrition/scripts/nutrition/presentation.py`
- `nutrition/scripts/nutrition/product_target_contract.py`
- `nutrition/scripts/nutrition/service_compare.py`
- `nutrition/scripts/nutrition/service_db_adapter.py`
- `nutrition/scripts/nutrition/service_repository.py`
- `nutrition/scripts/nutrition/suitability.py`
- `nutrition/tests/test_compare_service_contract.py`
- `nutrition/tests/test_feeding_service_contract.py`
- `nutrition/tests/test_mer_coefficient_policy_v2.py`
- `nutrition/tests/test_observability_safety_contract.py`
- `nutrition/tests/test_presentation.py`
- `nutrition/tests/test_product_target_contract.py`
- `nutrition/tests/test_runtime_e2e.py`
- `nutrition/tests/test_service_profile_birth_policy.py`
- `nutrition/tests/test_service_repository.py`
- `nutrition/tests/test_suitability.py`
- `nutrition/tests/validate_contract_v3.py`

## FINAL VERDICT — RUNTIME_DONE

Nutrition v3 구현·새 테스트·전체 회귀·legacy projection·로컬 실제 HTTP 검증 완료. live DB/Gateway/배포는 BLOCKED_EXTERNAL로 분리한다. commit/push/PR/merge 없음.

재현: 기존 curated Python 3.11에서 `PYTHONPATH=<local-path-redacted> python -B -m pytest nutrition/tests -q -p no:cacheprovider`. Offline parity + 실제 로컬 HTTP는 `nutrition/tests/validate_contract_v3.py --source <local-path-redacted> --baseline <local-path-redacted> --output <local-path-redacted>`을 같은 Python으로 실행한다. 이 validator는 실제 DB에 접근하지 않고 loopback bind 권한만 필요하다.

### v3 regression interpretation

The v3 validation numbers use separate baselines and must not be collapsed into
one "unchanged 145" claim:

- historical SOURCE_HEAD focused baseline against the historical implementation: **145 PASS**
- historical focused tests unaffected by the v3 Feeding contract, executed against v3: **133 PASS**
- historical Service Feeding contract executed against v3: **6 PASS / 6 intentional FAIL**
- current v3 focused Toxic/Allergen/Feeding suite: **151 PASS**
- current v3 full Nutrition suite: **581 PASS**
- legacy Service Mock projection: **286/286 identical**
- local loopback HTTP validation: **13 requests / 58 assertions PASS**

The six historical Feeding failures are intentional contract deltas caused by
the approved `healthy_merck_v1` baseline policy and schema-driven synthetic
integration energy. They are not described as unchanged historical regression.

These results are local validation evidence. They do **not** claim live Service
DB verification, deployed Gateway E2E, or remote deployment success.
