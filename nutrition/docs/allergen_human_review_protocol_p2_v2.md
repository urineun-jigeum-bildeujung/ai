# 알레르기 Human Review 진행 규약 (P2 v2)

본 문서는 영양성분 분석 AI 전체 중 알레르기 안전성 판단 서브시스템의 검증 문서이다. NIAS 영양 기준 비교, AAFCO 판정, 보증성분 정규화 등 영양성분 전체 기능의 완료를 의미하지 않는다.

## 입력 순서와 bias 완화

검토자는 `row_id`, `product_id`, `raw_ingredient_text`, `segmented_text`, `matched_text`, `matched_alias`, `allergen_code`만 먼저 본다. `confidence`, `mapping_method`, `source_dataset`, `dictionary_version`은 anchoring을 줄이기 위해 별도 참조 시트에 둔다.

판단 순서는 원문 원료 문맥 → 분할 문맥 → 매칭 텍스트/alias → 예측 allergen 순서다. `review_label`은 사람이 `CORRECT`, `INCORRECT`, `AMBIGUOUS` 중 하나를 직접 선택한다. `reviewer_note`는 optional이며 `INCORRECT` 또는 `AMBIGUOUS`에서 가능한 경우 짧은 근거를 기록한다.

## Frozen dataset 규칙

v2는 60행, unique product 60개, OPFF/OEM/GLOBAL 각 20행, semantic duplicate 0행이고 canonical parity 60/60 exact match를 확인했다. Human Review 시작 이후 60/60 완료 전까지 row 교체, dictionary/alias 변경, segmentation 변경, mapping method 변경, safety policy 변경을 금지한다. 발견된 오류는 수정하지 않고 issue candidate로만 기록한다.

이 audit은 `CANONICAL_ALIAS`와 edge case를 의도적으로 과대표집한 risk-based audit이다. 결과를 2,852 catalog evidence 전체의 population-weighted precision으로 직접 일반화하지 않는다.

## 완료 상태

사람이 입력한 완료본 `data/eval/알레르기_Human_Precision_Audit_AI사전라벨_한글번역 (1).xlsx`의 `검토하기` 시트에서 60/60 유효 라벨을 읽었다. 분포는 `CORRECT` 50건, `INCORRECT` 2건, `AMBIGUOUS` 8건이며, evaluator 결과는 `COMPLETE`다. strict precision은 83.33%, determinate precision은 96.15%, Wilson 95% CI는 71.97%~90.69%다.

현재 evaluator는 별도 합격 임계값을 자동 적용하지 않는다. 따라서 `GATE_F_HUMAN_PRECISION = PASS`는 **60건의 사람 라벨을 모두 읽어 지표를 산출했다는 완료 상태**이며, risk-based 표본 결과를 전체 catalog의 population-weighted precision으로 일반화하거나 알레르기 안전성 전체를 검증했다는 뜻은 아니다. `INCORRECT`와 `AMBIGUOUS` 행은 P3 root-cause backlog로만 기록했고, dictionary·mapping·runtime·safety policy는 변경하지 않았다.

`turkey` / `dinde` candidate review는 별도 dataset으로 유지하며 active precision audit과 혼합하지 않는다.
