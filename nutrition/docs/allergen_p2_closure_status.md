# 알레르기 Evidence P2 완료 상태

본 문서는 영양성분 분석 AI 전체 중 알레르기 안전성 판단 서브시스템의 검증 문서이다. NIAS 영양 기준 비교, AAFCO 판정, 보증성분 정규화 등 영양성분 전체 기능의 완료를 의미하지 않는다.

## 자동 closure 범위

P2에서 운영 evidence 저장·조회, component lineage, runtime fail-close, API 안전 상태 집계, fresh rebuild 및 자동 회귀 검증을 완료했다. canonical schema는 `component_occurrence_id`를 저장하며, semantic evidence identity는 `created_at`을 제외한다. 손상된 evidence, `STALE_EVIDENCE`, 또는 `LINEAGE_INTEGRITY_ERROR`는 `PRECOMPUTED` evidence로 반환되지 않는다.

| 검증 항목 | 결과 |
|---|---|
| component rows | 2,814 |
| raw evidence | 2,856 |
| catalog evidence | 2,852 |
| semantic duplicate | 4 |
| orphan evidence | 0 |
| automated regression | 75 passed / 0 failed |

## closure의 한계

이 closure는 dictionary 후보를 승인하지 않는다. Human Precision Audit은 사람 라벨 60/60을 완료했고, evaluator가 `CORRECT` 50건, `INCORRECT` 2건, `AMBIGUOUS` 8건을 산출했다. strict precision은 83.33%, determinate precision은 96.15%, Wilson 95% CI는 71.97%~90.69%이며 precision 상태는 `COMPLETE`다. `GATE_F_HUMAN_PRECISION = PASS`는 Human review 60건 및 evaluator 계산의 완료 상태일 뿐, 성능 threshold 통과가 아니다. 현재 별도 성능 threshold는 정의되어 있지 않다. 표본은 risk-based audit이므로 전체 catalog precision으로 일반화하지 않는다. `turkey` / `dinde`는 여전히 `PROPOSED`이며 자동 승인은 없다.

현재 정확한 표현은 다음과 같다.

> 영양성분 분석 AI의 알레르기 안전성 서브시스템에서 운영 저장·조회, component-to-evidence lineage, fail-close, 자동 회귀 검증 및 60건 risk-based 사람 매핑 정밀도 audit을 완료했다. 이 결과는 영양성분 분석 전체 기능 또는 전체 catalog의 population-weighted precision 완료를 의미하지 않는다.

현재 운영 상태는 `Human Review = COMPLETE`, `Evaluation Computation = COMPLETE`, `Reviewer Consistency QA = PENDING`, `P2 Precision Validation = PENDING_FINAL_QA`, `P2 allergen subsystem = FUNCTIONALLY COMPLETE`다. consistency QA가 끝나기 전에는 dictionary·mapping·runtime을 변경하지 않는다.
