# Nutrition Gold Fallback Candidate & Evidence Intake Gate P1

## 범위

이 문서는 P0에서 BLOCKED된 후보를 자동으로 VERIFIED로 변경하지 않는다. raw source metadata와 사람이 검증한 operational evidence는 별도 계층이다. `LOCAL_IDENTITY_CONSISTENT`는 repository 내부 source consistency만 뜻하며 manufacturer formulation identity verified가 아니다.

## 이번 실행 실측

- GOLD_NEAR_READY 후보: 12
- CAT shortlist: 3
- DOG shortlist: 3
- human intake rows: 0
- accepted evidence rows: 0
- rejected evidence rows: 0
- VERIFIED CAT candidates: 0
- VERIFIED DOG candidates: 0
- runtime eligible candidates: 0
- real persisted-product Gold E2E: executed=false

## Intake와 Operational Artifact

`human_gold_evidence_intake_v1.csv`가 없거나 비어 있으면 입력은 0건이며, `verified_evidence_gold_v1.csv` 및 `verified_evidence_gold_v1.json`은 정상적인 0-row artifact가 된다. 이 경우 raw PRODUCT_LABEL / seed_13 데이터가 verified evidence로 승격되지 않는다.

검증은 valid canonical GTIN, exact local product-ID membership, GTIN이 보이는 identity evidence, internally consistent product/variant/market/formula binding, authoritative life-stage/product-form provenance, guaranteed nutrient provenance, numeric percent/basis, required nutrient completeness를 요구한다. 제품명 키워드나 raw metadata는 life stage/form evidence가 될 수 없다.

## E2E 구분

테스트의 synthetic row는 fixture-based plumbing integration일 뿐 real persisted-product Gold E2E가 아니다. `REAL_PERSISTED_PRODUCT_GOLD_E2E`는 strict human validation과 runtime eligibility가 모두 true인 실제 artifact에만 실행한다.
