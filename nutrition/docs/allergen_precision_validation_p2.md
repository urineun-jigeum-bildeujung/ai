# 알레르기 매핑 정밀도 검증 현황 (P2)

본 문서는 영양성분 분석 AI 전체 중 알레르기 안전성 판단 서브시스템의 검증 문서이다. NIAS 영양 기준 비교, AAFCO 판정, 보증성분 정규화 등 영양성분 전체 기능의 완료를 의미하지 않는다.

## P2 자동 검증 기준선

`data/processed/allergen_p2_baseline_manifest.json`에 다음 기준선을 고정했다.

| 항목 | 현재 값 |
|---|---:|
| component rows | 2,814 |
| raw evidence | 2,856 |
| catalog evidence | 2,852 |
| semantic duplicate | 4 |
| orphan evidence | 0 |
| automated regression | 75 passed / 0 failed |

향후 P3에서 값이 변경되면 `before`, `after`, `delta`, `reason`을 함께 보고해야 한다.

## Human Precision Audit

감사 표본은 OPFF, OEM, GLOBAL에서 각 20행씩 총 60행으로 구성되어 있다. `review_label`은 사람이 `CORRECT`, `INCORRECT`, `AMBIGUOUS` 중 하나를 입력하기 전까지 공란으로 유지한다.

현재 상태는 다음과 같다.

| 항목 | 상태 |
|---|---|
| reviewed / total | 60 / 60 |
| remaining | 0 |
| `CORRECT` / `INCORRECT` / `AMBIGUOUS` | 50 / 2 / 8 |
| precision report | `COMPLETE` |
| strict precision | 83.33% |
| determinate precision | 96.15% |
| incorrect / ambiguous rate | 3.33% / 13.33% |
| Wilson 95% CI | 71.97% ~ 90.69% |
| precision gate | `PASS` (완료 상태) |

공란 라벨이 하나라도 있으면 evaluator는 `INCOMPLETE`와 `precision: null`을 반환하며 최종 PASS를 허용하지 않는다. 이번 60/60 완료 후 evaluator가 관측 지표를 계산했다. `GATE_F_HUMAN_PRECISION = PASS`의 정확한 의미는 **"Human review 60건 완료 및 evaluator 계산 정상 완료"**다. 별도 성능 합격 임계값은 없으므로 quality-pass, 전체 catalog precision threshold 통과, 전체 알레르기 안전성 검증 완료, 전체 Nutrition AI 완료를 뜻하지 않는다.

strict precision 83.33%는 전체 60건에서 `CORRECT` 비율(50/60)이다. determinate precision 96.15%는 `AMBIGUOUS` 8건을 제외한 확정 판정 52건에서의 `CORRECT` 비율(50/(50+2))이다. 따라서 `AMBIGUOUS` 안에 실제 false positive가 포함됐을 경우 determinate precision은 실제 오류 위험을 과소반영할 수 있다. 표본이 risk-based 표본이라는 한계도 유지하며, 결과를 전체 catalog precision으로 직접 일반화하지 않는다.

현재 상태는 `Human Review = COMPLETE`, `Evaluation Computation = COMPLETE`, `Reviewer Consistency QA = PENDING`, `P2 Precision Validation = PENDING_FINAL_QA`, `P2 allergen subsystem = FUNCTIONALLY COMPLETE`다. consistency QA가 사람의 재판정으로 끝나기 전에는 dictionary, mapping, runtime, safety policy를 변경하지 않는다.

## Coverage와 precision의 분리

Coverage는 precision과 합산하지 않는다. 현재 product-level 기준은 전체 394개, 원료 데이터 보유 119개, `FULLY_RESOLVABLE` 2개, `PARTIALLY_RESOLVED` 82개, `UNRESOLVED` 35개다. KNOWN_LIST allergy profile에서 `SAFETY_EVALUABLE`은 2개, `SAFETY_DATA_INSUFFICIENT`는 392개로 측정된다.

`turkey` / `dinde`는 `PROPOSED` 상태이며, 사람 승인 전 dictionary SoT와 runtime policy는 변경하지 않는다.

## 현재 알레르기 서브파트 상태

영양성분 분석 AI의 알레르기 안전성 서브시스템에서 운영 저장·조회, component-to-evidence lineage, fail-close, 자동 회귀 검증 및 60건 risk-based 사람 매핑 정밀도 audit을 완료했다. `INCORRECT`와 `AMBIGUOUS` 10건은 P3 root-cause backlog로만 기록했으며 runtime 정책은 변경하지 않았다. 대규모 alias/dictionary 확장 P3는 즉시 시작하지 않고, 다음 우선순위는 전체 Nutrition AI E2E 검증이다.

## 영양성분 전체 파트에서 아직 남은 작업

1. 알레르기 P2/P2.5 결과 및 P3 root-cause backlog 검토
2. 영양성분 전체 pipeline 현황 통합
3. guaranteed analysis, 단위, basis, NIAS, AAFCO, species, life-stage, allergen을 하나의 E2E 흐름으로 검증
4. 실제 상품 기준 product-level evaluability 재측정
5. FastAPI 입출력 계약 정리
6. 그 이후 데이터 보강/P3 진행
