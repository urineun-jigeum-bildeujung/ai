# UCI LightGBM 피처 단계별 비교

- 평가: Validation, 30일 내 동일 상품 재구매 확률
- 동일 학습·평가 표본, IPCW 가중치, 모델 설정을 사용합니다.
- Brier·ECE·MCE는 낮을수록 좋고, 이전 후보 대비 개선량은 양수일수록 좋습니다.

| 후보 | 피처 | 학습 표본 | 평가 전체 / 정답 확인 | Brier | ECE | MCE | 이전 대비 Brier 개선 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A_counts | history_interval_count, user_prior_order_count | 473,572 | 97,340 / 73,542 | 0.092965 | 0.065420 | 0.335764 | — |
| B_counts_median | history_interval_count, history_median_days, user_prior_order_count | 473,572 | 97,340 / 73,542 | 0.082855 | 0.037150 | 0.269432 | +0.010109 |
| C_counts_median_variability | history_interval_count, history_median_days, history_relative_mad, user_prior_order_count | 473,572 | 97,340 / 73,542 | 0.082266 | 0.037183 | 0.203643 | +0.000590 |

표본과 설정을 고정한 단계적 피처 추가 비교입니다. 각 차이는 앞선 피처가 주어진 조건에서의 효과이며, 독립적인 인과 효과나 개선의 통계적 확정을 의미하지 않습니다. 결측값도 입력 정보로 보존합니다.
