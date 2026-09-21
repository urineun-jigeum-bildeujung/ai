# UCI XGBoost AFT 후보 쌍 사용자 Bootstrap

- 기준: `normal`, `scale=1.0`
- 후보: `logistic`, `scale=2.0`
- 개선량 방향: `기준 normal Brier - 후보 logistic Brier`

- 사용자 수: `1,926`명
- 반복 수 / seed: `1,000` / `42`
- normal 점 Brier: `0.080934`
- logistic 점 Brier: `0.087991`
- 점 개선량: `-0.007057`
- 평균 개선량: `-0.007074`
- 95% 구간: `[-0.009647, -0.004931]`
- 양수 개선 비율: `0.00%`

normal 기준의 Brier 우위가 사용자 재표본에서도 일관됐습니다.

고정된 두 모델의 동일 Validation 행·정답·IPCW 가중치에서 사용자를 함께 복원추출했습니다. Test는 사용하지 않았습니다. 이 구간은 사용자 구성의 불확실성만 반영하며 모델 재학습과 같은 Validation 후보 선택의 낙관성은 포함하지 않습니다.
