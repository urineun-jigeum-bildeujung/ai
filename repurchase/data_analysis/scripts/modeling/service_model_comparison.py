"""동일한 서비스 Validation 집단에서 AFT와 LightGBM을 비교합니다."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from .evaluation import (
    IPCWUserBootstrapResult,
    bootstrap_ipcw_brier_pair_difference_by_user,
    evaluate_ipcw_brier_score,
    evaluate_ipcw_concordance_index,
    summarize_ipcw_calibration,
)
from .lightgbm_baseline import (
    build_lightgbm_training_data,
    predict_lightgbm_repurchase_probability,
    train_lightgbm_classifier,
)
from .maturity_analysis import add_split_ipcw_weights
from .operational_aft_input import build_service_aft_training_rows
from .operational_orders import OperationalOrderError
from .operational_temporal_split import ServiceTemporalSplit
from .probability_baseline import (
    fit_global_event_probability_baseline,
    fit_hierarchical_event_probability_baseline,
    predict_hierarchical_event_probability_baseline,
)
from .service_landmark_validation import build_service_landmark_cohort
from .xgboost_aft import (
    XGBoostAFTTrainingData,
    build_xgboost_aft_prediction_data,
    build_xgboost_aft_training_data,
    calculate_xgboost_aft_conditional_probability,
    calculate_xgboost_aft_event_probability,
    predict_xgboost_aft_duration,
    train_xgboost_aft_model,
)


@dataclass(frozen=True)
class ServiceModelComparison:
    """동일 평가 행에서 계산한 지표와 사용자 단위 쌍 비교를 보관합니다."""

    summary: pd.DataFrame
    calibration: pd.DataFrame
    paired_bootstrap: IPCWUserBootstrapResult
    brier_attribution: pd.DataFrame
    product_group_aft_bootstrap: IPCWUserBootstrapResult | None = None
    conditional_landmarks: pd.DataFrame | None = None
    conditional_calibration: pd.DataFrame | None = None


@dataclass(frozen=True)
class AFTRoundSelection:
    """Train 내부 시간 구간에서만 선택한 AFT 반복 횟수와 후보 지표입니다."""

    selected_rounds: int
    candidates: pd.DataFrame


@dataclass(frozen=True)
class AFTScaleSelection:
    """Train 내부 시간 구간에서만 선택한 normal AFT scale과 후보 지표입니다."""

    selected_scale: float
    candidates: pd.DataFrame


@dataclass(frozen=True)
class ProductGroupSmoothingSelection:
    """Train 내부 평가로 고른 상품군 확률 prior 수축 강도입니다."""

    selected_strength: float
    candidates: pd.DataFrame


def _service_product_group_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """서비스 상품군 키를 기존 계층형 확률 기준선의 키로 명시적으로 연결합니다."""
    if "target_id" not in rows or rows["target_id"].isna().any():
        raise OperationalOrderError("서비스 상품군 키가 누락됐습니다.")
    result = rows.copy()
    result["product_id"] = result["target_id"]
    return result


def select_service_product_group_smoothing(
    inner_split: ServiceTemporalSplit,
    *,
    candidate_strengths: tuple[float, ...],
    horizon_days: int = 30,
) -> ProductGroupSmoothingSelection:
    """바깥 Validation을 보지 않고 내부 IPCW Brier로 prior 강도를 선택합니다."""
    if (
        len(candidate_strengths) < 2
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0
            for value in candidate_strengths
        )
        or len(set(candidate_strengths)) != len(candidate_strengths)
    ):
        raise OperationalOrderError(
            "상품군 확률 수축 강도 후보는 서로 다른 양의 유한한 숫자 2개 이상이어야 합니다."
        )
    _validate_aft_inner_split(inner_split)
    train = add_split_ipcw_weights(inner_split.train, horizon_days=horizon_days)
    validation = add_split_ipcw_weights(
        inner_split.validation, horizon_days=horizon_days
    )
    train_groups = _service_product_group_rows(train)
    validation_groups = _service_product_group_rows(inner_split.validation)
    results: list[dict[str, int | float]] = []
    for strength in sorted(float(value) for value in candidate_strengths):
        model = fit_hierarchical_event_probability_baseline(
            train_groups, product_smoothing_strength=strength
        )
        probability = predict_hierarchical_event_probability_baseline(
            model, validation_groups
        )["predicted_event_probability"]
        if not probability.index.equals(validation.index):
            raise OperationalOrderError(
                "상품군 확률 기준선 예측 행 순서가 Validation과 다릅니다."
            )
        evaluation = validation.copy()
        evaluation["predicted_event_probability"] = probability
        brier = evaluate_ipcw_brier_score(
            evaluation, reference_probability=model.global_event_probability
        )
        if not np.isfinite(float(brier["ipcw_brier_score"])):
            raise OperationalOrderError(
                "상품군 확률 기준선 내부 Brier가 유한하지 않습니다."
            )
        results.append(
            {
                "product_group_smoothing_strength": strength,
                "ipcw_brier_score": float(brier["ipcw_brier_score"]),
                "validation_sample_count": int(brier["validation_sample_count"]),
                "outcome_known_count": int(brier["outcome_known_count"]),
            }
        )
    selected = min(
        results,
        key=lambda row: (
            row["ipcw_brier_score"],
            row["product_group_smoothing_strength"],
        ),
    )
    return ProductGroupSmoothingSelection(
        selected_strength=float(selected["product_group_smoothing_strength"]),
        candidates=pd.DataFrame(results),
    )


def select_service_aft_scale(
    inner_split: ServiceTemporalSplit,
    *,
    candidate_scales: tuple[float, ...],
    horizon_days: int = 30,
) -> AFTScaleSelection:
    """반복 20회를 고정하고 normal AFT의 scale만 내부 Brier로 선택합니다."""
    if (
        len(candidate_scales) < 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0
            for value in candidate_scales
        )
        or len(set(candidate_scales)) != len(candidate_scales)
    ):
        raise OperationalOrderError(
            "AFT scale 후보는 서로 다른 양의 유한한 숫자 2개 이상이어야 합니다."
        )
    _validate_aft_inner_split(inner_split)
    train = _canonical_service_train_order(inner_split.train)
    aft_train = build_xgboost_aft_training_data(build_service_aft_training_rows(train))
    weighted_validation = add_split_ipcw_weights(
        inner_split.validation, horizon_days=horizon_days
    )
    results: list[dict[str, int | float]] = []
    for scale in sorted(float(value) for value in candidate_scales):
        brier = _score_aft_inner_candidate(
            aft_train,
            inner_split.validation,
            weighted_validation,
            rounds=20,
            scale=scale,
            horizon_days=horizon_days,
        )
        results.append(
            {
                "loss_distribution_scale": scale,
                "ipcw_brier_score": float(brier["ipcw_brier_score"]),
                "validation_sample_count": int(brier["validation_sample_count"]),
                "outcome_known_count": int(brier["outcome_known_count"]),
            }
        )
    selected = min(
        results,
        key=lambda row: (row["ipcw_brier_score"], row["loss_distribution_scale"]),
    )
    return AFTScaleSelection(
        selected_scale=float(selected["loss_distribution_scale"]),
        candidates=pd.DataFrame(results),
    )


def select_service_aft_boost_rounds(
    inner_split: ServiceTemporalSplit,
    *,
    candidate_rounds: tuple[int, ...],
    horizon_days: int = 30,
) -> AFTRoundSelection:
    """내부 Train으로 학습하고 내부 Validation IPCW Brier만으로 선택합니다.

    두 표본의 생존 라벨은 각자의 종료 컷에서 생성돼 있어야 합니다. 동일 점수면
    더 작은 반복 횟수를 선택하며, 바깥 Validation은 이 함수에 전달하지 않습니다.
    """
    if (
        len(candidate_rounds) < 2
        or any(type(value) is not int or value < 1 for value in candidate_rounds)
        or len(set(candidate_rounds)) != len(candidate_rounds)
    ):
        raise OperationalOrderError(
            "AFT 반복 횟수 후보는 서로 다른 양의 정수 2개 이상이어야 합니다."
        )
    _validate_aft_inner_split(inner_split)

    train = _canonical_service_train_order(inner_split.train)
    aft_train = build_xgboost_aft_training_data(build_service_aft_training_rows(train))
    weighted_validation = add_split_ipcw_weights(
        inner_split.validation, horizon_days=horizon_days
    )
    results: list[dict[str, int | float]] = []
    for rounds in sorted(candidate_rounds):
        brier = _score_aft_inner_candidate(
            aft_train,
            inner_split.validation,
            weighted_validation,
            rounds=rounds,
            scale=1.0,
            horizon_days=horizon_days,
        )
        results.append(
            {
                "num_boost_round": rounds,
                "ipcw_brier_score": float(brier["ipcw_brier_score"]),
                "validation_sample_count": int(brier["validation_sample_count"]),
                "outcome_known_count": int(brier["outcome_known_count"]),
            }
        )
    candidates = pd.DataFrame(results)
    selected = min(
        results, key=lambda row: (row["ipcw_brier_score"], row["num_boost_round"])
    )
    return AFTRoundSelection(
        selected_rounds=int(selected["num_boost_round"]), candidates=candidates
    )


def _score_aft_inner_candidate(
    aft_train: XGBoostAFTTrainingData,
    validation: pd.DataFrame,
    weighted_validation: pd.DataFrame,
    *,
    rounds: int,
    scale: float,
    horizon_days: int,
) -> dict[str, int | float]:
    """같은 내부 평가행·IPCW 가중치로 한 설정의 Brier를 계산합니다."""
    # scale은 학습 손실과 예측 CDF 모두에 쓰이므로 후보마다 재학습합니다.
    model = train_xgboost_aft_model(
        aft_train, loss_distribution_scale=scale, num_boost_round=rounds
    )
    prediction_input = build_xgboost_aft_prediction_data(
        validation, feature_columns=model.feature_columns
    )
    duration = predict_xgboost_aft_duration(model, prediction_input)
    probability = calculate_xgboost_aft_event_probability(
        model, duration, horizon_days=horizon_days
    )
    if not probability.index.equals(weighted_validation.index):
        raise OperationalOrderError("AFT 내부 예측 행 순서가 Validation과 다릅니다.")
    evaluation = weighted_validation.copy()
    evaluation["predicted_event_probability"] = probability
    brier = evaluate_ipcw_brier_score(evaluation, reference_probability=0.5)
    if not np.isfinite(float(brier["ipcw_brier_score"])):
        raise OperationalOrderError("AFT 내부 Brier가 유한하지 않습니다.")
    return brier


def _validate_aft_inner_split(inner_split: ServiceTemporalSplit) -> None:
    """내부 모델 선택 전에 시간 컷과 앵커가 겹치지 않는지 검사합니다."""
    if inner_split.train.empty or inner_split.validation.empty:
        raise OperationalOrderError(
            "AFT 내부 Train·Validation은 비어 있을 수 없습니다."
        )
    train_end = pd.Timestamp(inner_split.train["split_end_at"].iloc[0])
    validation_end = pd.Timestamp(inner_split.validation["split_end_at"].iloc[0])
    if (
        train_end.tzinfo is None
        or validation_end.tzinfo is None
        or train_end >= validation_end
        or not inner_split.train["split_end_at"].eq(train_end).all()
        or not inner_split.validation["split_end_at"].eq(validation_end).all()
        or inner_split.train["anchor_at"].gt(train_end).any()
        or not inner_split.validation["anchor_at"].gt(train_end).all()
        or inner_split.validation["anchor_at"].gt(validation_end).any()
    ):
        raise OperationalOrderError(
            "AFT 내부 시간 분할의 종료 컷·앵커가 유효하지 않습니다."
        )


def summarize_service_brier_attribution(rows: pd.DataFrame) -> pd.DataFrame:
    """같은 정답 확인 행의 Brier 차이를 이력량·주문 수·상품군별로 분해합니다.

    각 기여도의 분모는 Validation 정답 확인 행의 **전체 IPCW 가중치**입니다.
    따라서 같은 축의 구간 기여도를 합하면 전체 AFT−LightGBM Brier와
    일치합니다. 구간 자체의 평균 오차 차이와 혼동하지 않도록 둘 다 기록합니다.
    """
    required = {
        "user_id",
        "target_id",
        "history_interval_count",
        "user_prior_order_count",
        "reference_predicted_event_probability",
        "candidate_predicted_event_probability",
    }
    missing = required - set(rows.columns)
    if missing:
        raise OperationalOrderError(f"Brier 기여도 필수 컬럼 누락: {sorted(missing)}")
    if rows[["user_id", "target_id"]].isna().any().any():
        raise OperationalOrderError(
            "Brier 기여도 사용자·상품군 키에 결측값이 있습니다."
        )
    for column, label in (
        ("history_interval_count", "과거 구매 간격 수"),
        ("user_prior_order_count", "사용자 과거 주문 수"),
    ):
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or value < 0
            or value > np.iinfo(np.int64).max
            for value in rows[column].array
        ):
            raise OperationalOrderError(
                f"{label}는 int64 범위의 0 이상의 정수여야 합니다."
            )

    # 기존 평가 계약으로 두 확률·가중치·정답을 먼저 검증합니다.
    for column in (
        "reference_predicted_event_probability",
        "candidate_predicted_event_probability",
    ):
        evaluation = rows.copy()
        evaluation["predicted_event_probability"] = evaluation[column]
        evaluate_ipcw_brier_score(evaluation, reference_probability=0.5)

    known = rows.loc[rows["ipcw_outcome_known"]].copy()
    actual = known["ipcw_event_within_horizon"].astype("float64")
    weights = known["ipcw_weight"].astype("float64")
    known["aft_weighted_error"] = (
        known["reference_predicted_event_probability"].sub(actual).pow(2).mul(weights)
    )
    known["lightgbm_weighted_error"] = (
        known["candidate_predicted_event_probability"].sub(actual).pow(2).mul(weights)
    )
    known["history_bucket"] = pd.cut(
        known["history_interval_count"],
        bins=[-0.5, 0.5, 1.5, float("inf")],
        labels=["0", "1", "2+"],
    )
    # 임의의 빈도 임계값 없이 실제 과거 주문 수 각각의 오차를 남깁니다.
    known["user_order_count"] = known["user_prior_order_count"].astype("int64")
    known["product_group"] = known["target_id"].astype("string")
    total_weight = float(weights.sum())
    results: list[pd.DataFrame] = []
    for segment_kind, column in (
        ("history_interval_count", "history_bucket"),
        ("user_prior_order_count", "user_order_count"),
        ("product_group", "product_group"),
    ):
        grouped = (
            known.groupby(column, observed=True, sort=True)
            .agg(
                outcome_known_count=(column, "size"),
                user_count=("user_id", "nunique"),
                ipcw_weight_sum=("ipcw_weight", "sum"),
                aft_weighted_error_sum=("aft_weighted_error", "sum"),
                lightgbm_weighted_error_sum=("lightgbm_weighted_error", "sum"),
            )
            .reset_index()
            .rename(columns={column: "segment_value"})
        )
        grouped["segment_value"] = grouped["segment_value"].astype("string")
        grouped.insert(0, "segment_kind", segment_kind)
        grouped["ipcw_weight_share"] = grouped["ipcw_weight_sum"].div(total_weight)
        error_difference = grouped["aft_weighted_error_sum"].sub(
            grouped["lightgbm_weighted_error_sum"]
        )
        grouped["within_segment_brier_improvement"] = error_difference.div(
            grouped["ipcw_weight_sum"]
        )
        grouped["global_brier_contribution"] = error_difference.div(total_weight)
        results.append(grouped)

    attribution = pd.concat(results, ignore_index=True)
    global_improvement = float(
        (known["aft_weighted_error"] - known["lightgbm_weighted_error"]).sum()
        / total_weight
    )
    for segment_kind in (
        "history_interval_count",
        "user_prior_order_count",
        "product_group",
    ):
        contribution = float(
            attribution.loc[
                attribution["segment_kind"].eq(segment_kind),
                "global_brier_contribution",
            ].sum()
        )
        if not np.isclose(contribution, global_improvement, rtol=1e-10, atol=1e-12):
            raise OperationalOrderError(
                "구간별 Brier 기여도 합계가 전체 차이와 다릅니다."
            )
    return attribution


def _canonical_service_train_order(rows: pd.DataFrame) -> pd.DataFrame:
    """ID의 pandas 자료형과 원천 조회 순서에 무관하게 학습 행을 정렬합니다.

    히스토그램 기반 트리의 동점 처리에는 입력 행 순서가 영향을 줄 수 있으므로
    시각과 사건 키를 명시한다. ID는 숫자로 계산하지 않고 정확한 문자열로만
    정렬해 bigint ID의 부동소수점 변환도 피합니다.
    """
    key_columns = ("anchor_at", "user_id", "pet_id", "target_id", "order_id")
    missing = set(key_columns) - set(rows.columns)
    if missing:
        raise OperationalOrderError(f"서비스 학습 정렬 키 누락: {sorted(missing)}")
    if rows.index.has_duplicates or rows.loc[:, list(key_columns)].isna().any().any():
        raise OperationalOrderError("서비스 학습 정렬 키가 중복되거나 비었습니다.")
    keys = rows.loc[:, list(key_columns)].copy()
    for column in key_columns[1:]:
        keys[column] = keys[column].astype("string")
    if keys.duplicated(subset=list(key_columns)).any():
        raise OperationalOrderError("서비스 학습 사건 키가 중복됐습니다.")
    return rows.loc[keys.sort_values(list(key_columns), kind="stable").index].copy()


def _evaluate_candidate(
    rows: pd.DataFrame,
    probability: pd.Series,
    *,
    model_name: str,
    reference_probability: float,
    calibration_bin_count: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """동일 IPCW 행의 확률·순위·보정 오차를 한 번에 평가합니다."""
    if not probability.index.equals(rows.index):
        raise OperationalOrderError(
            "모델 예측 행 순서가 Validation 원본과 일치하지 않습니다."
        )
    evaluation = rows.copy()
    evaluation["predicted_event_probability"] = probability
    # C-index는 일수 예측이 아니라 30일 재구매 위험의 역순을 비교합니다.
    evaluation["predicted_duration_days"] = 1.0 - probability
    brier = evaluate_ipcw_brier_score(
        evaluation, reference_probability=reference_probability
    )
    concordance = evaluate_ipcw_concordance_index(evaluation)
    calibration = summarize_ipcw_calibration(
        evaluation, bin_count=calibration_bin_count
    )
    calibration.insert(0, "model", model_name)
    return (
        {
            "model": model_name,
            "validation_sample_count": int(brier["validation_sample_count"]),
            "outcome_known_count": int(brier["outcome_known_count"]),
            "ipcw_brier_score": float(brier["ipcw_brier_score"]),
            "ipcw_reference_brier_score": float(brier["ipcw_reference_brier_score"]),
            "ipcw_c_index": float(concordance["ipcw_concordance_index"]),
            "comparable_pair_count": int(concordance["comparable_pair_count"]),
            "expected_calibration_error": float(
                calibration["weighted_absolute_gap_contribution"].sum()
            ),
            "weighted_mean_predicted_probability": float(
                calibration["mean_predicted_probability"]
                .mul(calibration["ipcw_weight_share"])
                .sum()
            ),
            "weighted_observed_event_rate": float(
                calibration["observed_event_rate"]
                .mul(calibration["ipcw_weight_share"])
                .sum()
            ),
        },
        calibration,
    )


def compare_service_aft_lightgbm(
    split: ServiceTemporalSplit,
    *,
    horizon_days: int = 30,
    aft_boost_rounds: int = 20,
    aft_loss_distribution_scale: float = 1.0,
    calibration_bin_count: int = 10,
    bootstrap_replicates: int = 1_000,
    bootstrap_random_seed: int = 42,
    product_group_smoothing_strength: float | None = None,
    conditional_landmark_days: tuple[int, ...] | None = None,
) -> ServiceModelComparison:
    """Train에서만 학습하고 두 후보를 같은 Validation 행에서 비교합니다.

    반환 지표는 입력된 시간 분할의 탐색 결과이며 독립 Test 성능이 아닙니다.
    AFT는 전체 Train의 검열 기간을, LightGBM은 정답 확인 Train을 학습합니다.
    """
    if not split.train.index.is_unique or not split.validation.index.is_unique:
        raise OperationalOrderError("서비스 비교 표본의 행 인덱스가 중복됐습니다.")
    if split.train.empty or split.validation.empty:
        raise OperationalOrderError(
            "서비스 비교에는 Train·Validation이 모두 필요합니다."
        )
    if conditional_landmark_days is not None and (
        not conditional_landmark_days
        or len(set(conditional_landmark_days)) != len(conditional_landmark_days)
        or any(
            isinstance(day, bool) or not isinstance(day, Integral) or day < 0
            for day in conditional_landmark_days
        )
    ):
        raise OperationalOrderError(
            "조건부 평가 시점은 중복 없는 0 이상의 정수 일수여야 합니다."
        )

    train = _canonical_service_train_order(split.train)
    weighted_train = add_split_ipcw_weights(train, horizon_days=horizon_days)
    weighted_validation = add_split_ipcw_weights(
        split.validation, horizon_days=horizon_days
    )
    known_train = weighted_train.loc[weighted_train["ipcw_outcome_known"]]
    reference_probability = float(
        known_train["ipcw_event_within_horizon"]
        .astype("float64")
        .mul(known_train["ipcw_weight"])
        .sum()
        / known_train["ipcw_weight"].sum()
    )

    aft_train = build_xgboost_aft_training_data(build_service_aft_training_rows(train))
    aft_model = train_xgboost_aft_model(
        aft_train,
        loss_distribution_scale=aft_loss_distribution_scale,
        num_boost_round=aft_boost_rounds,
    )
    aft_prediction_input = build_xgboost_aft_prediction_data(
        split.validation, feature_columns=aft_model.feature_columns
    )
    aft_duration = predict_xgboost_aft_duration(aft_model, aft_prediction_input)
    aft_probability = calculate_xgboost_aft_event_probability(
        aft_model, aft_duration, horizon_days=horizon_days
    )

    lightgbm_train = build_lightgbm_training_data(weighted_train)
    lightgbm_model = train_lightgbm_classifier(lightgbm_train)
    lightgbm_probability = predict_lightgbm_repurchase_probability(
        lightgbm_model, split.validation
    )
    for probability in (aft_probability, lightgbm_probability):
        if not probability.index.equals(weighted_validation.index):
            raise OperationalOrderError(
                "두 모델의 예측 행 순서가 Validation 원본과 일치하지 않습니다."
            )

    results = []
    calibrations = []
    for name, probability in (
        ("xgboost_aft", aft_probability),
        ("lightgbm", lightgbm_probability),
    ):
        metrics, calibration = _evaluate_candidate(
            weighted_validation,
            probability,
            model_name=name,
            reference_probability=reference_probability,
            calibration_bin_count=calibration_bin_count,
        )
        metrics.update(
            {
                "horizon_days": horizon_days,
                "train_sample_count": len(split.train),
                "aft_train_sample_count": aft_train.included_sample_count,
                "lightgbm_train_sample_count": len(lightgbm_train.target),
                "aft_boost_rounds": aft_boost_rounds,
            }
        )
        results.append(metrics)
        calibrations.append(calibration)

    product_group_aft_bootstrap = None
    if product_group_smoothing_strength is not None:
        baseline = fit_hierarchical_event_probability_baseline(
            _service_product_group_rows(weighted_train),
            product_smoothing_strength=product_group_smoothing_strength,
        )
        baseline_probability = predict_hierarchical_event_probability_baseline(
            baseline, _service_product_group_rows(split.validation)
        )["predicted_event_probability"]
        metrics, calibration = _evaluate_candidate(
            weighted_validation,
            baseline_probability,
            model_name="product_group_probability_baseline",
            reference_probability=reference_probability,
            calibration_bin_count=calibration_bin_count,
        )
        metrics.update(
            {
                "horizon_days": horizon_days,
                "train_sample_count": len(split.train),
                "aft_train_sample_count": aft_train.included_sample_count,
                "lightgbm_train_sample_count": len(lightgbm_train.target),
                "aft_boost_rounds": aft_boost_rounds,
            }
        )
        results.append(metrics)
        calibrations.append(calibration)
        baseline_pair_rows = weighted_validation.copy()
        baseline_pair_rows["reference_predicted_event_probability"] = (
            baseline_probability
        )
        baseline_pair_rows["candidate_predicted_event_probability"] = aft_probability
        product_group_aft_bootstrap = bootstrap_ipcw_brier_pair_difference_by_user(
            baseline_pair_rows,
            bootstrap_replicates=bootstrap_replicates,
            random_seed=bootstrap_random_seed,
        )

    paired_rows = weighted_validation.copy()
    paired_rows["reference_predicted_event_probability"] = aft_probability
    paired_rows["candidate_predicted_event_probability"] = lightgbm_probability
    paired_bootstrap = bootstrap_ipcw_brier_pair_difference_by_user(
        paired_rows,
        bootstrap_replicates=bootstrap_replicates,
        random_seed=bootstrap_random_seed,
    )
    brier_attribution = summarize_service_brier_attribution(paired_rows)
    point_improvement = paired_bootstrap.summary["point_brier_improvement"]
    attribution_improvement = brier_attribution.loc[
        brier_attribution["segment_kind"].eq("history_interval_count"),
        "global_brier_contribution",
    ].sum()
    if not np.isclose(
        attribution_improvement, point_improvement, rtol=1e-10, atol=1e-12
    ):
        raise OperationalOrderError("Brier 기여도와 사용자 쌍 비교 결과가 다릅니다.")
    conditional_landmarks = None
    conditional_calibration = None
    if conditional_landmark_days is not None:
        landmark_results = []
        landmark_calibrations = []
        for elapsed_days in conditional_landmark_days:
            train_cohort = build_service_landmark_cohort(
                train, elapsed_days=elapsed_days, split_name="train"
            )
            validation_cohort = build_service_landmark_cohort(
                split.validation,
                elapsed_days=elapsed_days,
                split_name="validation",
            )
            landmark_train = add_split_ipcw_weights(
                train_cohort.rows, horizon_days=horizon_days
            )
            reference = fit_global_event_probability_baseline(landmark_train)
            landmark_validation = add_split_ipcw_weights(
                validation_cohort.rows, horizon_days=horizon_days
            )
            prediction_input = build_xgboost_aft_prediction_data(
                validation_cohort.rows, feature_columns=aft_model.feature_columns
            )
            duration = predict_xgboost_aft_duration(aft_model, prediction_input)
            probability = calculate_xgboost_aft_conditional_probability(
                aft_model,
                duration,
                validation_cohort.rows["elapsed_days"],
                window_days=horizon_days,
            )
            if not probability.index.equals(landmark_validation.index):
                raise OperationalOrderError(
                    "시점별 조건부 예측 행이 Validation 위험집단과 다릅니다."
                )
            landmark_validation["predicted_event_probability"] = probability
            brier = evaluate_ipcw_brier_score(
                landmark_validation,
                reference_probability=reference.global_event_probability,
            )
            calibration = summarize_ipcw_calibration(
                landmark_validation, bin_count=calibration_bin_count
            )
            calibration["elapsed_days"] = elapsed_days
            landmark_results.append(
                {
                    "elapsed_days": elapsed_days,
                    "train_at_risk_count": len(train_cohort.rows),
                    "source_validation_count": validation_cohort.source_sample_count,
                    "excluded_prior_event_count": (
                        validation_cohort.excluded_prior_event_count
                    ),
                    "excluded_prior_censor_count": (
                        validation_cohort.excluded_prior_censor_count
                    ),
                    "at_risk_count": len(validation_cohort.rows),
                    "at_risk_user_count": int(landmark_validation["user_id"].nunique()),
                    "ipcw_brier_score": float(brier["ipcw_brier_score"]),
                    "ipcw_reference_brier_score": float(
                        brier["ipcw_reference_brier_score"]
                    ),
                    "outcome_known_count": int(brier["outcome_known_count"]),
                    "expected_calibration_error": float(
                        calibration["weighted_absolute_gap_contribution"].sum()
                    ),
                }
            )
            landmark_calibrations.append(calibration)
        conditional_landmarks = pd.DataFrame(landmark_results)
        conditional_calibration = pd.concat(landmark_calibrations, ignore_index=True)
    return ServiceModelComparison(
        summary=pd.DataFrame(results),
        calibration=pd.concat(calibrations, ignore_index=True),
        paired_bootstrap=paired_bootstrap,
        brier_attribution=brier_attribution,
        product_group_aft_bootstrap=product_group_aft_bootstrap,
        conditional_landmarks=conditional_landmarks,
        conditional_calibration=conditional_calibration,
    )
