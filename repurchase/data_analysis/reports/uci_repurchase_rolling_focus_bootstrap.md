# 재구매 Rolling 집중 구간 사용자 Bootstrap

- 차이 방향: `AFT Brier - LightGBM Brier` (양수면 LightGBM 유리)
- 각 구간 안에서 사용자를 복원추출합니다. 구간 Brier 차이의 구간이지 fold 전체 점수 기여량의 구간은 아닙니다.
- 구간별 반복 원자료는 `uci_repurchase_rolling_focus_bootstrap_trials.json.gz`에 압축해 보관합니다. 요약값과 반복 원자료는 같은 시드로 재현됩니다.

| Fold | 진단 구간 | 정답 확인 사용자 | 정답 확인 행 | 점 차이 | 사용자 Bootstrap 95% | 양수 반복 비율 | 상태 |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| fold_1 | product_train_sample_count=128+ | 2,665 | 66,124 | -0.002491 | -0.008249~+0.001787 | 19.70% | evaluated |
| fold_1 | history_interval_count=8-15 | 149 | 2,658 | -0.029043 | -0.053039~-0.008890 | 0.00% | evaluated |
| fold_2 | product_train_sample_count=128+ | 1,714 | 43,038 | +0.003755 | +0.001029~+0.007173 | 100.00% | evaluated |
| fold_2 | history_interval_count=8-15 | 201 | 2,421 | +0.032268 | +0.017378~+0.049770 | 100.00% | evaluated |
| fold_3 | product_train_sample_count=128+ | 1,884 | 46,397 | -0.001855 | -0.005350~+0.000783 | 11.60% | evaluated |
| fold_3 | history_interval_count=8-15 | 262 | 3,405 | -0.003761 | -0.016705~+0.006473 | 29.80% | evaluated |

각 구간 안에서 사용자를 복원추출하고 같은 사용자의 구매 행을 함께 유지해 구간 내부 AFT-LightGBM IPCW Brier 차이를 비교합니다. 이는 fold 전체 Brier 기여량의 신뢰구간이나 선정된 구간의 사전 검정이 아닙니다. 구간은 앞선 Validation에서 사후 선택했고 여러 fold가 사용자와 Train 이력을 공유하므로 결과를 독립 반복으로 보지 않습니다. 원래 Test는 사용하지 않았습니다.
