# 알레르기 매핑 정밀도 검토 가이드 (P2)

본 문서는 영양성분 분석 AI 전체 중 알레르기 안전성 판단 서브시스템의 검증 문서이다. NIAS 영양 기준 비교, AAFCO 판정, 보증성분 정규화 등 영양성분 전체 기능의 완료를 의미하지 않는다.

## 검토 대상과 입력 원칙

검토 대상은 `data/eval/알레르기_Human_Precision_Audit_AI사전라벨_한글번역 (1).xlsx`의 `검토하기` 시트 60행이다. 사람 검토 전에는 `review_label`과 `reviewer_note`를 모두 공란으로 유지한다. 규칙이나 모델은 `review_label`을 생성하거나 변경해서는 안 된다.

사용 가능한 `review_label` 값은 다음과 같다.

- `CORRECT`: 원문 원료와 분할된 원료 문맥이 예측된 allergen을 명확히 의미한다.
- `INCORRECT`: 원문 문맥 기준으로 allergen 매핑이 의미적으로 틀렸다.
- `AMBIGUOUS`: 원문만으로 특정 allergen을 확정할 수 없다.

`poultry fat`처럼 종을 특정할 수 없는 표현, OCR 손상으로 원료명이 판독되지 않는 표현은 특정 종 allergen으로 단정하지 않고 `AMBIGUOUS`로 판단한다. 단순히 사전 alias가 일치했다는 사실만으로 `CORRECT`로 판정하지 않는다.

## 권장 검토 순서

1. `raw_ingredient_text`의 실제 원료 문맥을 읽는다.
2. `segmented_text`가 원문 문맥을 적절히 분리했는지 확인한다.
3. `matched_text`와 `matched_alias`가 어떤 근거로 사용됐는지 확인한다.
4. 예측된 `allergen_code`를 확인한다.
5. 위 근거를 바탕으로 `CORRECT`, `INCORRECT`, `AMBIGUOUS` 중 하나를 사람이 선택한다.

`INCORRECT` 또는 `AMBIGUOUS`인 경우에는 판단 근거를 `reviewer_note`에 남긴다. `error_reason`이 제공된 경우에는 사람이 선택적으로 기록할 수 있으며, 공란으로 둘 수 있다.

## 현재 상태

Human Precision Audit은 60/60 완료됐으며 evaluator는 `COMPLETE`를 반환했다. 그러나 이 표본은 `CANONICAL_ALIAS`와 edge case를 과대표집한 risk-based audit이다. 따라서 알레르기 필터 전체 완성, 영양성분 분석 완성, 전체 안전성 검증 완료 또는 전체 catalog precision 검증 완료라고 표현하지 않는다.
