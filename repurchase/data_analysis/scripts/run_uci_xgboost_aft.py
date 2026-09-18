"""UCI Train으로 XGBoost AFT를 학습하고 Validation에서 평가합니다.

공통 전처리·시간 분할을 재사용하되 기존 베이스라인 E2E와 실행 책임을
분리합니다. Test는 모델 선택이 끝나기 전까지 열어보지 않습니다.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from numbers import Integral, Real
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
    XGBoostAFTPredictionData,
    XGBoostAFTTrainingData,
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
AFT_BOOST_ROUND_CANDIDATES: Final[tuple[int, ...]] = (5, 20, 50, 100)
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


@dataclass(frozen=True)
class XGBoostAFTPreparedExperiment:
    """후보 모델들이 공통으로 사용할 시간 분할·학습·평가 데이터를 보관합니다.

    frozen은 필드 교체를 막지만 내부 DataFrame까지 불변으로 만들지는 않으므로,
    후보 평가 함수는 전달받은 공통 데이터를 직접 수정하지 않아야 합니다.
    """

    horizon_days: int
    split: TemporalSplit
    training_data: XGBoostAFTTrainingData
    validation: pd.DataFrame
    prediction_data: XGBoostAFTPredictionData
    training_reference_probability: float
    reference_population_sample_count: int


def prepare_xgboost_aft_experiment(
    labels: pd.DataFrame,
    *,
    horizon_days: int = HORIZON_DAYS,
) -> XGBoostAFTPreparedExperiment:
    """후보마다 반복할 필요가 없는 피처·분할·평가 기준을 한 번만 준비합니다."""
    samples = build_historical_interval_features(labels)
    split = make_temporal_split(samples)
    samples = assign_temporal_splits(samples, split)
    training = samples.loc[samples["split"].eq("train")].copy()
    validation = samples.loc[samples["split"].eq("validation")].copy()

    # AFT 학습은 고정 시점 IPCW와 분리해 생존시간·검열 정보만 사용합니다.
    survival_training = add_split_survival_observation(training)
    training_data = build_xgboost_aft_training_data(survival_training)

    # 모든 후보가 동일한 Validation 행과 피처 이름·순서를 사용하게 고정합니다.
    prediction_data = build_xgboost_aft_prediction_data(
        validation,
        feature_columns=training_data.feature_columns,
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
    return XGBoostAFTPreparedExperiment(
        horizon_days=horizon_days,
        split=split,
        training_data=training_data,
        validation=validation,
        prediction_data=prediction_data,
        training_reference_probability=(
            global_probability_model.global_event_probability
        ),
        reference_population_sample_count=len(weighted_reference_training),
    )


def run_xgboost_aft_experiment(
    labels: pd.DataFrame,
    *,
    horizon_days: int = HORIZON_DAYS,
    calibration_bin_count: int = CALIBRATION_BIN_COUNT,
    bootstrap_replicates: int | None = BOOTSTRAP_REPLICATES,
    bootstrap_random_seed: int = BOOTSTRAP_RANDOM_SEED,
    loss_distribution: str = "normal",
    loss_distribution_scale: float = 1.0,
    num_boost_round: int = AFT_NUM_BOOST_ROUND,
) -> XGBoostAFTExperimentResult:
    """같은 시간 분할에서 AFT를 학습하고 Validation 성능만 계산합니다."""
    prepared = prepare_xgboost_aft_experiment(
        labels,
        horizon_days=horizon_days,
    )
    return evaluate_xgboost_aft_candidate(
        prepared,
        calibration_bin_count=calibration_bin_count,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_random_seed=bootstrap_random_seed,
        loss_distribution=loss_distribution,
        loss_distribution_scale=loss_distribution_scale,
        num_boost_round=num_boost_round,
    )


def evaluate_xgboost_aft_candidate(
    prepared: XGBoostAFTPreparedExperiment,
    *,
    calibration_bin_count: int = CALIBRATION_BIN_COUNT,
    bootstrap_replicates: int | None = BOOTSTRAP_REPLICATES,
    bootstrap_random_seed: int = BOOTSTRAP_RANDOM_SEED,
    loss_distribution: str = "normal",
    loss_distribution_scale: float = 1.0,
    num_boost_round: int = AFT_NUM_BOOST_ROUND,
) -> XGBoostAFTExperimentResult:
    """공통 준비 데이터를 사용해 AFT 후보 하나를 학습하고 평가합니다."""
    training_result = train_xgboost_aft_model(
        prepared.training_data,
        loss_distribution=loss_distribution,
        loss_distribution_scale=loss_distribution_scale,
        num_boost_round=num_boost_round,
    )

    validation_predictions = predict_xgboost_aft_duration(
        training_result,
        prepared.prediction_data,
    )

    concordance = evaluate_xgboost_aft_ipcw_concordance(
        prepared.validation,
        validation_predictions,
        horizon_days=prepared.horizon_days,
    )
    probability = evaluate_xgboost_aft_ipcw_probability(
        prepared.validation,
        training_result,
        validation_predictions,
        horizon_days=prepared.horizon_days,
        training_reference_probability=prepared.training_reference_probability,
        calibration_bin_count=calibration_bin_count,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_random_seed=bootstrap_random_seed,
    )

    training_summary: dict[str, object] = {
        "training_split": "train",
        "evaluation_split": "validation",
        "trained_until": prepared.split.train_end_at,
        "source_sample_count": prepared.training_data.source_sample_count,
        "excluded_zero_duration_count": (
            prepared.training_data.excluded_zero_duration_count
        ),
        "included_sample_count": prepared.training_data.included_sample_count,
        "reference_population_sample_count": (
            prepared.reference_population_sample_count
        ),
        "feature_columns": list(training_result.feature_columns),
        "loss_distribution": training_result.loss_distribution,
        "loss_distribution_scale": training_result.loss_distribution_scale,
        "num_boost_round": training_result.num_boost_round,
        "training_aft_nloglik": list(training_result.training_aft_nloglik),
        "training_reference_probability": prepared.training_reference_probability,
    }
    return XGBoostAFTExperimentResult(
        split=prepared.split,
        training_summary=training_summary,
        validation_predictions=validation_predictions,
        concordance=concordance,
        probability=probability,
    )


def compare_xgboost_aft_boosting_rounds(
    prepared: XGBoostAFTPreparedExperiment,
    *,
    round_candidates: Sequence[int] = AFT_BOOST_ROUND_CANDIDATES,
    calibration_bin_count: int = CALIBRATION_BIN_COUNT,
    loss_distribution: str = "normal",
    loss_distribution_scale: float = 1.0,
) -> pd.DataFrame:
    """동일한 공통 데이터에서 부스팅 반복 횟수 후보의 Validation 지표를 비교합니다."""
    candidates = tuple(round_candidates)
    if not candidates:
        raise ValueError("비교할 부스팅 반복 횟수 후보가 하나 이상 필요합니다.")
    if any(
        isinstance(candidate, bool)
        or not isinstance(candidate, Integral)
        or candidate <= 0
        for candidate in candidates
    ):
        raise ValueError("부스팅 반복 횟수 후보는 0보다 큰 정수여야 합니다.")
    if len(set(candidates)) != len(candidates):
        raise ValueError("부스팅 반복 횟수 후보에는 중복된 값을 사용할 수 없습니다.")

    comparison_rows: list[dict[str, float | int | None]] = []
    for round_count in candidates:
        result = evaluate_xgboost_aft_candidate(
            prepared,
            calibration_bin_count=calibration_bin_count,
            bootstrap_replicates=None,
            loss_distribution=loss_distribution,
            loss_distribution_scale=loss_distribution_scale,
            num_boost_round=round_count,
        )
        training_loss = result.training_summary["training_aft_nloglik"]
        if not isinstance(training_loss, list) or not training_loss:
            raise ValueError("후보 모델의 AFT 학습 손실 이력이 비어 있습니다.")
        probability = result.probability.summary
        comparison_rows.append(
            {
                "num_boost_round": int(round_count),
                "final_training_aft_nloglik": float(training_loss[-1]),
                "minimum_training_aft_nloglik": float(min(training_loss)),
                "ipcw_concordance_index": float(
                    result.concordance["ipcw_concordance_index"]
                ),
                "ipcw_brier_score": float(probability["ipcw_brier_score"]),
                "ipcw_reference_brier_score": float(
                    probability["ipcw_reference_brier_score"]
                ),
                "brier_skill_score": probability["brier_skill_score"],
                "expected_calibration_error": float(
                    probability["expected_calibration_error"]
                ),
                "maximum_calibration_error": float(
                    probability["maximum_calibration_error"]
                ),
                "weighted_calibration_gap": float(
                    probability["weighted_calibration_gap"]
                ),
                "horizon_days": int(probability["horizon_days"]),
                "validation_sample_count": int(probability["validation_sample_count"]),
                "outcome_known_count": int(probability["outcome_known_count"]),
                "ipcw_weight_sum": float(probability["ipcw_weight_sum"]),
                "reference_probability": float(probability["reference_probability"]),
                "aft_evaluation_sample_count": int(
                    probability["aft_evaluation_sample_count"]
                ),
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    for fixed_column in (
        "ipcw_reference_brier_score",
        "aft_evaluation_sample_count",
        "horizon_days",
        "validation_sample_count",
        "outcome_known_count",
        "ipcw_weight_sum",
        "reference_probability",
    ):
        if comparison[fixed_column].nunique(dropna=False) != 1:
            raise RuntimeError(
                f"AFT 반복 횟수 후보의 공통 평가 조건이 달라졌습니다: {fixed_column}"
            )
    return comparison


def select_xgboost_aft_boosting_round(comparison: pd.DataFrame) -> int:
    """Brier를 우선하고 순위·확률 신뢰도·비용 순으로 최종 후보를 선택합니다."""
    required_columns = (
        "num_boost_round",
        "ipcw_brier_score",
        "ipcw_concordance_index",
        "expected_calibration_error",
        "weighted_calibration_gap",
    )
    if comparison.columns.duplicated().any():
        raise ValueError("AFT 후보 선택 결과에 중복된 컬럼 이름이 있습니다.")
    missing_columns = [
        column for column in required_columns if column not in comparison.columns
    ]
    if missing_columns:
        raise ValueError(f"AFT 후보 선택 필수 컬럼이 누락됐습니다: {missing_columns}")
    if comparison.empty:
        raise ValueError("선택할 AFT 반복 횟수 후보 결과가 없습니다.")
    round_values = comparison["num_boost_round"].tolist()
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0
        for value in round_values
    ):
        raise ValueError("AFT 후보 반복 횟수는 0보다 큰 정수여야 합니다.")
    if comparison["num_boost_round"].duplicated().any():
        raise ValueError("AFT 후보 선택 결과에 중복된 반복 횟수가 있습니다.")

    metric_ranges = {
        "ipcw_brier_score": (0.0, 1.0),
        "ipcw_concordance_index": (0.0, 1.0),
        "expected_calibration_error": (0.0, 1.0),
        "weighted_calibration_gap": (-1.0, 1.0),
    }
    for column, (lower_bound, upper_bound) in metric_ranges.items():
        values = comparison[column].tolist()
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not lower_bound <= float(value) <= upper_bound
            for value in values
        ):
            raise ValueError(
                f"AFT 후보 선택 지표 {column}은 "
                f"{lower_bound}부터 {upper_bound} 사이의 유한한 실수여야 합니다."
            )

    ranked = comparison.copy()
    ranked["absolute_weighted_calibration_gap"] = ranked[
        "weighted_calibration_gap"
    ].abs()
    ranked = ranked.sort_values(
        by=[
            "ipcw_brier_score",
            "ipcw_concordance_index",
            "expected_calibration_error",
            "absolute_weighted_calibration_gap",
            "num_boost_round",
        ],
        ascending=[True, False, True, True, True],
        kind="stable",
    )
    return int(ranked.iloc[0]["num_boost_round"])


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
