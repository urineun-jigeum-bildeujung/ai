"""선택된 AFT 모델의 시점별 조건부 재구매 확률을 Validation에서 검증합니다.

구매 후 0·7·14·30일 시점에 아직 관찰되는 미재구매 표본을 추려
앞으로 30일 확률을 평가합니다. 모델 재선택이나 Test 평가는 하지 않습니다.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Final

import pandas as pd

from .loaders import load_uci_online_retail_ii
from .modeling.evaluation import (
    bootstrap_ipcw_brier_difference_by_user,
    evaluate_ipcw_brier_score,
    summarize_ipcw_calibration,
)
from .modeling.landmark_validation import build_split_landmark_cohort
from .modeling.maturity_analysis import add_split_ipcw_weights
from .modeling.probability_baseline import fit_global_event_probability_baseline
from .modeling.xgboost_aft import (
    build_xgboost_aft_prediction_data,
    calculate_xgboost_aft_conditional_probability,
    predict_xgboost_aft_duration,
    train_xgboost_aft_model,
)
from .paths import REPORT_DIR
from .preprocessing.events import build_uci_purchase_events
from .preprocessing.labels import build_same_product_repurchase_labels
from .preprocessing.uci import classify_uci_rows
from .reporting import dataframe_to_nullable_records, write_text_atomically
from .run_uci_xgboost_aft import (
    XGBoostAFTPreparedExperiment,
    prepare_xgboost_aft_experiment,
)

LANDMARK_DAYS: Final[tuple[int, ...]] = (0, 7, 14, 30)
WINDOW_DAYS: Final[int] = 30
JSON_REPORT_PATH = REPORT_DIR / "uci_aft_conditional_landmark_validation.json"
MARKDOWN_REPORT_PATH = REPORT_DIR / "uci_aft_conditional_landmark_validation.md"


def evaluate_conditional_landmarks(
    prepared: XGBoostAFTPreparedExperiment,
    *,
    landmark_days: tuple[int, ...] = LANDMARK_DAYS,
    window_days: int = WINDOW_DAYS,
    bootstrap_replicates: int | None = None,
    bootstrap_random_seed: int = 42,
) -> dict[str, object]:
    """Train 기준 확률과 고정 AFT 후보를 사용해 Validation 시점별 성능을 봅니다."""
    if not landmark_days or len(set(landmark_days)) != len(landmark_days):
        raise ValueError("평가 시점에는 중복 없는 하나 이상의 일수가 필요합니다.")
    if window_days != prepared.horizon_days:
        raise ValueError("조건부 평가 기간은 준비한 학습 기준 기간과 같아야 합니다.")

    # 이미 선택된 설정을 한 번만 학습하며, 시점별 Validation으로 재선택하지 않습니다.
    model = train_xgboost_aft_model(
        prepared.training_data,
        loss_distribution="normal",
        loss_distribution_scale=1.0,
        num_boost_round=20,
    )
    # 0일 표본 제외 기준도 AFT 학습 행과 맞춰 Train 기준 모집단을 고정합니다.
    training = prepared.training.loc[prepared.training_data.row_index]
    results = []
    for elapsed_days in landmark_days:
        train_cohort = build_split_landmark_cohort(
            training, elapsed_days=elapsed_days, split_name="train"
        )
        validation_cohort = build_split_landmark_cohort(
            prepared.validation, elapsed_days=elapsed_days, split_name="validation"
        )
        train_weighted = add_split_ipcw_weights(
            train_cohort.rows, horizon_days=window_days
        )
        reference = fit_global_event_probability_baseline(train_weighted)
        validation_weighted = add_split_ipcw_weights(
            validation_cohort.rows, horizon_days=window_days
        )

        # 모델 입력은 구매 당시의 과거 피처만 선택하고, 경과 일수는 확률식에만 넣습니다.
        prediction_data = build_xgboost_aft_prediction_data(
            validation_cohort.rows, feature_columns=model.feature_columns
        )
        duration = predict_xgboost_aft_duration(model, prediction_data)
        probability = calculate_xgboost_aft_conditional_probability(
            model,
            duration,
            validation_cohort.rows["elapsed_days"],
            window_days=window_days,
        )
        if not probability.index.equals(validation_weighted.index):
            raise ValueError("조건부 확률과 평가 표본의 행 인덱스·순서가 다릅니다.")
        validation_weighted["predicted_event_probability"] = probability
        brier = evaluate_ipcw_brier_score(
            validation_weighted,
            reference_probability=reference.global_event_probability,
        )
        calibration = summarize_ipcw_calibration(validation_weighted)
        landmark_result: dict[str, object] = {
            "elapsed_days": elapsed_days,
            "train_at_risk_count": len(train_cohort.rows),
            "train_outcome_known_count": reference.global_outcome_known_count,
            "source_validation_count": validation_cohort.source_sample_count,
            "excluded_prior_event_count": validation_cohort.excluded_prior_event_count,
            "excluded_prior_censor_count": validation_cohort.excluded_prior_censor_count,
            "at_risk_count": len(validation_cohort.rows),
            "outcome_unknown_count": int(
                (~validation_weighted["ipcw_outcome_known"]).sum()
            ),
            "observed_event_count": int(
                validation_weighted["ipcw_event_within_horizon"].fillna(False).sum()
            ),
            "brier": brier,
            "expected_calibration_error": float(
                calibration["weighted_absolute_gap_contribution"].sum()
            ),
            "calibration": dataframe_to_nullable_records(calibration),
        }
        # 추가 실험에서만 사용자 재표집을 실행해 기존 기본 평가를 그대로 유지합니다.
        if bootstrap_replicates is not None:
            bootstrap = bootstrap_ipcw_brier_difference_by_user(
                validation_weighted,
                reference_probability=reference.global_event_probability,
                bootstrap_replicates=bootstrap_replicates,
                random_seed=bootstrap_random_seed,
            )
            landmark_result["at_risk_user_count"] = int(
                validation_weighted["user_id"].nunique()
            )
            landmark_result["user_bootstrap"] = bootstrap.summary
            landmark_result["bootstrap_trials"] = dataframe_to_nullable_records(
                bootstrap.trials
            )
        results.append(landmark_result)

    return {
        "dataset": "uci_online_retail_ii",
        "split": "validation",
        "split_boundaries": {
            key: value.isoformat() if isinstance(value, pd.Timestamp) else value
            for key, value in asdict(prepared.split).items()
        },
        "model": {"distribution": "normal", "scale": 1.0, "boost_rounds": 20},
        "window_days": window_days,
        "interpretation": (
            "동일 AFT 모델의 시점별 기술적 검증입니다. 시점에 따라 위험집단·검열률이 "
            "변하므로 각 행의 지표 차이를 모델 개선 효과나 인과 효과로 해석하지 않습니다. "
            "시점 후보·모델 설정을 이 Validation 결과로 다시 고르지 않으며 Test는 열지 않았습니다."
        ),
        "landmarks": results,
    }


def render_conditional_landmarks(report: dict[str, object]) -> str:
    """시점별 모집단과 확률 오차를 한눈에 비교하도록 보고서를 만듭니다."""
    lines = [
        "# UCI AFT 조건부 확률 시점별 Validation",
        "",
        "- 기준: 구매 후 해당 일수까지 재구매하지 않고 관찰된 표본에서 앞으로 30일",
        "- AFT: normal, scale=1.0, boosting 20회; 기준선은 시점별 Train만 사용",
        "- Test는 평가하지 않았습니다.",
        "",
        "| 경과 일수 | 위험집단 | 정답 확인 | 사건 | 검열로 불명 | Brier | Train 기준 Brier | ECE |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report["landmarks"]:
        brier = item["brier"]
        lines.append(
            f"| {item['elapsed_days']} | {item['at_risk_count']:,} | "
            f"{brier['outcome_known_count']:,} | {item['observed_event_count']:,} | "
            f"{item['outcome_unknown_count']:,} | {brier['ipcw_brier_score']:.6f} | "
            f"{brier['ipcw_reference_brier_score']:.6f} | "
            f"{item['expected_calibration_error']:.6f} |"
        )
    lines.extend(["", report["interpretation"], ""])
    return "\n".join(lines)


def main() -> None:
    """UCI 원본을 읽고 재현 가능한 JSON·Markdown 평가 결과를 저장합니다."""
    classified = classify_uci_rows(load_uci_online_retail_ii())
    events = build_uci_purchase_events(classified.rows)
    labels = build_same_product_repurchase_labels(
        events, observation_end_at=pd.Timestamp(events["ordered_at"].max())
    )
    prepared = prepare_xgboost_aft_experiment(labels)
    report = evaluate_conditional_landmarks(prepared)
    markdown = render_conditional_landmarks(report)
    write_text_atomically(
        JSON_REPORT_PATH,
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    write_text_atomically(MARKDOWN_REPORT_PATH, markdown)
    print(markdown)


if __name__ == "__main__":
    main()
