# UCI XGBoost AFT·LightGBM 동일 모집단 비교

- AFT 설정: `normal`, `scale=1.0`, `20 rounds`
- 평가 시점: `30`일
- Bootstrap 개선량: `AFT Brier - LightGBM Brier`

| 모델 | 실제 학습 | 평가 표본 | Brier | C-index | ECE | MCE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| xgboost_aft | 494,213 | 97,329 | 0.080934 | 0.786037 | 0.023814 | 0.156555 |
| lightgbm_probability | 473,544 | 97,329 | 0.082145 | 0.796883 | 0.036780 | 0.205882 |

## 사용자 단위 paired Bootstrap

- 사용자: `1,926`명
- 반복 수 / seed: `1,000` / `42`
- 점 개선량: `-0.001211`
- 평균 개선량: `-0.001211`
- 95% 구간: `[-0.004706, +0.001325]`
- LightGBM 개선 비율: `24.30%`

95% 구간이 0을 포함해 두 모델의 안정적인 우위를 확정하지 않습니다.

## 이력량 구간별 비교

구간 개선량은 `AFT Brier - LightGBM Brier`이며 양수면 LightGBM, 0 이하면 AFT가 낮습니다. 이 열은 Validation에서 Brier가 낮았던 모델을 표시할 뿐 운영 라우팅 규칙이 아니며 rolling cutoff 재검증이 필요합니다.

| 기준 | 구간 | 표본 | 사용자 | AFT Brier | LightGBM Brier | 개선량 | AFT ECE | LightGBM ECE | Validation 최저 Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| history_interval_count | 0 | 52,133 | 2,129 | 0.047486 | 0.048642 | -0.001156 | 0.027893 | 0.030533 | xgboost_aft |
| history_interval_count | 1 | 16,309 | 1,679 | 0.083203 | 0.083292 | -0.000089 | 0.029539 | 0.036742 | xgboost_aft |
| history_interval_count | 2-3 | 13,765 | 1,335 | 0.101813 | 0.104670 | -0.002857 | 0.012463 | 0.044165 | xgboost_aft |
| history_interval_count | 4-7 | 9,153 | 801 | 0.150330 | 0.152327 | -0.001997 | 0.025362 | 0.054701 | xgboost_aft |
| history_interval_count | 8-15 | 4,408 | 294 | 0.228398 | 0.232160 | -0.003761 | 0.065707 | 0.066991 | xgboost_aft |
| history_interval_count | 16-31 | 1,340 | 63 | 0.224383 | 0.222917 | +0.001467 | 0.070558 | 0.064357 | lightgbm_probability |
| history_interval_count | 32-63 | 216 | 10 | 0.131874 | 0.057270 | +0.074604 | 0.183916 | 0.073176 | lightgbm_probability |
| history_interval_count | 64-127 | 5 | 1 | 0.367799 | 0.001457 | +0.366342 | 0.606344 | 0.038170 | lightgbm_probability |
| user_prior_order_count | 0 | 9,279 | 389 | 0.040937 | 0.039517 | +0.001420 | 0.037695 | 0.000814 | lightgbm_probability |
| user_prior_order_count | 1 | 6,321 | 378 | 0.027946 | 0.027297 | +0.000649 | 0.020570 | 0.024968 | lightgbm_probability |
| user_prior_order_count | 2-3 | 11,393 | 512 | 0.031091 | 0.031257 | -0.000166 | 0.018369 | 0.023238 | xgboost_aft |
| user_prior_order_count | 4-7 | 18,985 | 648 | 0.038399 | 0.038972 | -0.000572 | 0.009469 | 0.037001 | xgboost_aft |
| user_prior_order_count | 8-15 | 21,199 | 504 | 0.074399 | 0.073261 | +0.001138 | 0.019217 | 0.034669 | lightgbm_probability |
| user_prior_order_count | 16-31 | 15,687 | 233 | 0.114471 | 0.114475 | -0.000004 | 0.033157 | 0.056593 | xgboost_aft |
| user_prior_order_count | 32-63 | 4,506 | 57 | 0.162942 | 0.161447 | +0.001495 | 0.052841 | 0.065745 | lightgbm_probability |
| user_prior_order_count | 64-127 | 3,580 | 18 | 0.199365 | 0.194594 | +0.004771 | 0.060722 | 0.054497 | lightgbm_probability |
| user_prior_order_count | 128+ | 6,379 | 6 | 0.230353 | 0.256505 | -0.026152 | 0.122565 | 0.140342 | xgboost_aft |

AFT가 사용한 0일 제외 Train 원본·Validation·네 피처를 LightGBM과 공유했습니다. LightGBM은 고정 30일 정답을 확인할 수 있는 Train 행만 실제 학습하며, AFT는 우측검열 행도 사용합니다. LightGBM의 C-index는 1-확률을 순위 점수로 사용한 결과이며 예상 일수 해석이 아닙니다. 사용자 Bootstrap은 고정된 두 모델의 Validation 사용자 구성 불확실성만 측정하며 Test는 사용하지 않았습니다.
