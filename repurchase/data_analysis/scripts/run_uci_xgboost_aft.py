"""UCI Train으로 XGBoost AFT를 학습하고 Validation에서 평가합니다.

공통 전처리·시간 분할을 재사용하되 기존 베이스라인 E2E와 실행 책임을
분리합니다. Test는 모델 선택이 끝나기 전까지 열어보지 않습니다.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from typing import Final

import pandas as pd

from .loaders import load_uci_online_retail_ii
from .modeling.maturity_analysis import (
    add_split_ipcw_weights,
    add_split_survival_observation,
)
from .modeling.model_selection import (
    XGBoostAFTProbabilityEvaluation,
    evaluate_xgboost_aft_ipcw_concordance,
    evaluate_xgboost_aft_ipcw_probability,
)
from .modeling.probability_baseline import fit_global_event_probability_baseline
from .modeling.samples import (
    TemporalSplit,
    assign_temporal_splits,
    build_historical_interval_features,
    make_temporal_split,
)
from .modeling.xgboost_aft import (
    build_xgboost_aft_prediction_data,
    build_xgboost_aft_training_data,
    predict_xgboost_aft_duration,
    train_xgboost_aft_model,
)
from .paths import REPORT_DIR
from .preprocessing.events import build_uci_purchase_events
from .preprocessing.labels import build_same_product_repurchase_labels
from .preprocessing.uci import classify_uci_rows
from .reporting import dataframe_to_nullable_records, write_text_atomically

HORIZON_DAYS: Final[int] = 30
CALIBRATION_BIN_COUNT: Final[int] = 10
BOOTSTRAP_REPLICATES: Final[int] = 1_000
BOOTSTRAP_RANDOM_SEED: Final[int] = 42
AFT_NUM_BOOST_ROUND: Final[int] = 5
JSON_REPORT_PATH = REPORT_DIR / "uci_xgboost_aft_evaluation.json"
MARKDOWN_REPORT_PATH = REPORT_DIR / "uci_xgboost_aft_evaluation.md"
BOOTSTRAP_TRIALS_REPORT_PATH = REPORT_DIR / "uci_xgboost_aft_bootstrap_trials.json"


@dataclass(frozen=True)
class XGBoostAFTExperimentResult:
    """AFT 학습 근거와 Validation 평가 결과를 한 실행 단위로 보관합니다."""

    split: TemporalSplit
    training_summary: dict[str, object]
    validation_predictions: pd.Series
    concordance: dict[str, float | int]
    probability: XGBoostAFTProbabilityEvaluation


def run_xgboost_aft_experiment(
    labels: pd.DataFrame,
    *,
    horizon_days: int = HORIZON_DAYS,
    calibration_bin_count: int = CALIBRATION_BIN_COUNT,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_random_seed: int = BOOTSTRAP_RANDOM_SEED,
    loss_distribution: str = "normal",
    loss_distribution_scale: float = 1.0,
    num_boost_round: int = AFT_NUM_BOOST_ROUND,
) -> XGBoostAFTExperimentResult:
    """같은 시간 분할에서 AFT를 학습하고 Validation 성능만 계산합니다."""
    samples = build_historical_interval_features(labels)
    split = make_temporal_split(samples)
    samples = assign_temporal_splits(samples, split)
    training = samples.loc[samples["split"].eq("train")].copy()
    validation = samples.loc[samples["split"].eq("validation")].copy()

    # AFT 학습은 고정 시점 IPCW와 분리해 생존시간·검열 정보만 사용합니다.
    survival_training = add_split_survival_observation(training)
    training_data = build_xgboost_aft_training_data(survival_training)
    training_result = train_xgboost_aft_model(
        training_data,
        loss_distribution=loss_distribution,
        loss_distribution_scale=loss_distribution_scale,
        num_boost_round=num_boost_round,
    )

    # 학습 때 확정한 피처 이름과 순서를 그대로 사용해 Validation을 예측합니다.
    prediction_data = build_xgboost_aft_prediction_data(
        validation,
        feature_columns=training_result.feature_columns,
    )
    validation_predictions = predict_xgboost_aft_duration(
        training_result,
        prediction_data,
    )

    concordance = evaluate_xgboost_aft_ipcw_concordance(
        validation,
        validation_predictions,
        horizon_days=horizon_days,
    )
    # AFT와 기준선이 같은 Train 모집단을 보도록 0일 제외 후 IPCW를 재계산합니다.
    reference_training = training.loc[training_data.row_index].copy()
    weighted_reference_training = add_split_ipcw_weights(
        reference_training,
        horizon_days=horizon_days,
    )
    global_probability_model = fit_global_event_probability_baseline(
        weighted_reference_training
    )
    probability = evaluate_xgboost_aft_ipcw_probability(
        validation,
        training_result,
        validation_predictions,
        horizon_days=horizon_days,
        training_reference_probability=(
            global_probability_model.global_event_probability
        ),
        calibration_bin_count=calibration_bin_count,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_random_seed=bootstrap_random_seed,
    )

    training_summary: dict[str, object] = {
        "training_split": "train",
        "evaluation_split": "validation",
        "trained_until": split.train_end_at,
        "source_sample_count": training_data.source_sample_count,
        "excluded_zero_duration_count": (training_data.excluded_zero_duration_count),
        "included_sample_count": training_data.included_sample_count,
        "reference_population_sample_count": len(weighted_reference_training),
        "feature_columns": list(training_result.feature_columns),
        "loss_distribution": training_result.loss_distribution,
        "loss_distribution_scale": training_result.loss_distribution_scale,
        "num_boost_round": training_result.num_boost_round,
        "training_aft_nloglik": list(training_result.training_aft_nloglik),
        "training_reference_probability": (
            global_probability_model.global_event_probability
        ),
    }
    return XGBoostAFTExperimentResult(
        split=split,
        training_summary=training_summary,
        validation_predictions=validation_predictions,
        concordance=concordance,
        probability=probability,
    )


def build_xgboost_aft_report(
    result: XGBoostAFTExperimentResult,
) -> dict[str, object]:
    """실험 결과를 표준 JSON으로 저장 가능한 요약 구조로 변환합니다."""
    bootstrap = result.probability.user_bootstrap
    if bootstrap is None:
        raise ValueError("XGBoost AFT 보고서에는 사용자 Bootstrap 결과가 필요합니다.")

    split = {
        key: value.isoformat() if isinstance(value, pd.Timestamp) else value
        for key, value in asdict(result.split).items()
    }
    training = result.training_summary.copy()
    trained_until = training["trained_until"]
    if not isinstance(trained_until, pd.Timestamp):
        raise TypeError("AFT 학습 종료 시각은 pandas Timestamp여야 합니다.")
    training["trained_until"] = trained_until.isoformat()

    lower = float(bootstrap.summary["bootstrap_lower_95_brier_improvement"])
    upper = float(bootstrap.summary["bootstrap_upper_95_brier_improvement"])
    if lower > 0:
        decision = "AFT의 Brier 개선이 사용자 구성 변화에서도 일관되게 양수였습니다."
    elif upper < 0:
        decision = "AFT의 Brier Score가 기준선보다 일관되게 악화됐습니다."
    else:
        decision = (
            "95% Bootstrap 구간이 0을 포함해 AFT의 안정적인 우위를 확정하지 않습니다."
        )

    return {
        "dataset": "uci_online_retail_ii",
        "experiment_version": "xgboost_aft_validation_v1",
        "evaluation_split": "validation",
        "horizon_days": int(result.probability.summary["horizon_days"]),
        "split": split,
        "runtime": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("xgboost", "pandas", "numpy")},
        },
        "training": training,
        "validation_concordance": result.concordance,
        "validation_probability": result.probability.summary,
        "validation_calibration": dataframe_to_nullable_records(
            result.probability.calibration
        ),
        "validation_user_bootstrap": bootstrap.summary,
        "decision": decision,
        "scope": (
            "Train으로 모델과 전체 확률 기준선을 학습하고 Validation에서만 "
            "평가했습니다. Test는 모델 선택이 끝나기 전까지 사용하지 않습니다. "
            "Bootstrap은 고정된 모델·확률·IPCW 가중치에서 사용자 구성의 "
            "불확실성을 추정하며 재학습 불확실성은 포함하지 않습니다."
        ),
    }


def build_xgboost_aft_bootstrap_trials_report(
    result: XGBoostAFTExperimentResult,
) -> dict[str, object]:
    """사용자 Bootstrap 반복 원자료를 요약 보고서와 분리해 반환합니다."""
    bootstrap = result.probability.user_bootstrap
    if bootstrap is None:
        raise ValueError("저장할 XGBoost AFT 사용자 Bootstrap 결과가 없습니다.")
    return {
        "dataset": "uci_online_retail_ii",
        "experiment_version": "xgboost_aft_validation_v1",
        "evaluation_split": "validation",
        "bootstrap_replicates": bootstrap.summary["bootstrap_replicates"],
        "random_seed": bootstrap.summary["random_seed"],
        "summary": bootstrap.summary,
        "trials": dataframe_to_nullable_records(bootstrap.trials),
    }


def render_xgboost_aft_report(report: dict[str, object]) -> str:
    """AFT 핵심 지표와 Bootstrap 판단을 사람이 검토할 Markdown으로 만듭니다."""
    training = report["training"]
    concordance = report["validation_concordance"]
    probability = report["validation_probability"]
    bootstrap = report["validation_user_bootstrap"]
    calibration = report["validation_calibration"]

    calibration_lines = [
        (
            f"| {row['bin_lower_bound']:.0%}~{row['bin_upper_bound']:.0%} | "
            f"{row['sample_count']:,} | {row['mean_predicted_probability']:.2%} | "
            f"{row['observed_event_rate']:.2%} | {row['calibration_gap']:+.2%} |"
        )
        for row in calibration
    ]
    maximum_gap_row = max(
        calibration,
        key=lambda row: row["absolute_calibration_gap"],
    )
    lines = [
        "# UCI XGBoost AFT Validation 평가",
        "",
        "## 학습 설정",
        "",
        f"- 학습 표본: `{training['included_sample_count']:,}`건 "
        f"(0일 제외 `{training['excluded_zero_duration_count']:,}`건)",
        f"- 피처: `{', '.join(training['feature_columns'])}`",
        f"- AFT 분포 / scale: `{training['loss_distribution']}` / "
        f"`{training['loss_distribution_scale']}`",
        f"- 부스팅 반복: `{training['num_boost_round']:,}`회",
        "",
        "## Validation 결과",
        "",
        f"- 평가 시점: `{report['horizon_days']}`일 내 동일 상품 재구매",
        f"- 평가 모집단: 원본 `{probability['source_validation_sample_count']:,}`건 "
        f"→ 0일 제외 `{probability['excluded_zero_duration_count']:,}`건 "
        f"→ AFT 평가 `{probability['aft_evaluation_sample_count']:,}`건",
        f"- 정답 확인 표본: `{probability['outcome_known_count']:,}`건",
        f"- Train 상수확률 기준선: `{training['training_reference_probability']:.2%}`",
        f"- IPCW 가중 평균 예측확률 / 실제 사건률: "
        f"`{probability['weighted_mean_predicted_probability']:.2%}` / "
        f"`{probability['weighted_observed_event_rate']:.2%}`",
        "",
        "| C-index | AFT Brier | 기준 Brier | Brier Skill | ECE | MCE |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {concordance['ipcw_concordance_index']:.6f} | "
            f"{probability['ipcw_brier_score']:.6f} | "
            f"{probability['ipcw_reference_brier_score']:.6f} | "
            f"{probability['brier_skill_score']:.6f} | "
            f"{probability['expected_calibration_error']:.6f} | "
            f"{probability['maximum_calibration_error']:.6f} |"
        ),
        "",
        "## Calibration",
        "",
        "평균 예측·실제 사건률·차이는 IPCW 가중 값이며, 표본은 정답을 확인한 "
        "원시 행 수입니다.",
        "",
        "| 확률 구간 | 표본 | 평균 예측 | 실제 사건률 | 차이(%p) |",
        "| --- | ---: | ---: | ---: | ---: |",
        *calibration_lines,
        "",
        f"MCE 구간의 표본은 `{maximum_gap_row['sample_count']:,}`건이므로 "
        "MCE만으로 전체 확률 성능을 판단하지 않습니다.",
        "",
        "## 사용자 단위 Bootstrap",
        "",
        f"- 정답 확인 행을 가진 Bootstrap 대상 사용자: `{bootstrap['user_count']:,}`명",
        f"- 반복 수 / seed: `{bootstrap['bootstrap_replicates']:,}` / "
        f"`{bootstrap['random_seed']}`",
        f"- 점 개선량: `{bootstrap['point_brier_improvement']:+.6f}`",
        f"- 평균 개선량: `{bootstrap['bootstrap_mean_brier_improvement']:+.6f}`",
        f"- 95% 구간: `[{bootstrap['bootstrap_lower_95_brier_improvement']:+.6f}, "
        f"{bootstrap['bootstrap_upper_95_brier_improvement']:+.6f}]`",
        f"- 양수 개선 비율: `{bootstrap['bootstrap_positive_improvement_rate']:.2%}`",
        "",
        f"현재 `{training['loss_distribution']}`, "
        f"`scale={training['loss_distribution_scale']}`, "
        f"`{training['num_boost_round']}`회 기준 설정에서 {report['decision']}",
        "",
        str(report["scope"]),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """실제 UCI 원본으로 AFT를 실행하고 요약·반복 원자료를 저장합니다."""
    source = load_uci_online_retail_ii()
    classified = classify_uci_rows(source)
    events = build_uci_purchase_events(classified.rows)
    labels = build_same_product_repurchase_labels(
        events,
        observation_end_at=pd.Timestamp(events["ordered_at"].max()),
    )
    result = run_xgboost_aft_experiment(labels)
    report = build_xgboost_aft_report(result)
    bootstrap_trials = build_xgboost_aft_bootstrap_trials_report(result)
    write_text_atomically(
        JSON_REPORT_PATH,
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    write_text_atomically(
        BOOTSTRAP_TRIALS_REPORT_PATH,
        json.dumps(bootstrap_trials, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
    )
    markdown = render_xgboost_aft_report(report)
    write_text_atomically(MARKDOWN_REPORT_PATH, markdown)
    print(markdown)


if __name__ == "__main__":
    main()
