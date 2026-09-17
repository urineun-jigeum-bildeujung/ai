"""재구매 기간 예측의 전체·fallback 단계별 오차 지표를 계산합니다."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


class RepurchaseEvaluationError(ValueError):
    """평가 입력에 정답이나 예측값이 없을 때 발생합니다."""


EVALUATION_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "target_duration_days",
    "predicted_duration_days",
    "prediction_source",
)

HIT_WINDOWS_DAYS: Final[tuple[int, ...]] = (3, 7)

IPCW_BINARY_EVALUATION_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "predicted_duration_days",
        "ipcw_event_within_horizon",
        "ipcw_horizon_days",
        "ipcw_outcome_known",
        "ipcw_weight",
    }
)

IPCW_CONCORDANCE_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "predicted_duration_days",
        "survival_observed_duration_days",
        "survival_event_observed",
        "ipcw_horizon_days",
        "ipcw_censoring_survival_probability",
    }
)

IPCW_BRIER_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "predicted_event_probability",
        "ipcw_event_within_horizon",
        "ipcw_horizon_days",
        "ipcw_outcome_known",
        "ipcw_weight",
    }
)


class _FenwickCountTree:
    """예측 순위별 누적 표본 수를 로그 시간에 저장하고 조회합니다."""

    def __init__(self, size: int) -> None:
        """조회할 서로 다른 예측 순위의 개수만큼 내부 배열을 준비합니다."""
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise RepurchaseEvaluationError("Fenwick Tree 크기는 양의 정수여야 합니다.")
        self._size = size
        # 0번을 비워두면 마지막 이진 비트를 이용해 부모 구간으로 이동할 수 있습니다.
        self._tree = np.zeros(size + 1, dtype="int64")

    def add(self, index: int) -> None:
        """0부터 시작하는 한 예측 순위의 표본 수를 1만큼 증가시킵니다."""
        if isinstance(index, bool) or not isinstance(index, int):
            raise RepurchaseEvaluationError("Fenwick Tree 순위는 정수여야 합니다.")
        if index < 0 or index >= self._size:
            raise RepurchaseEvaluationError("Fenwick Tree 순위가 범위를 벗어났습니다.")

        tree_index = index + 1
        while tree_index <= self._size:
            self._tree[tree_index] += 1
            tree_index += tree_index & -tree_index

    def prefix_count(self, end: int) -> int:
        """0번부터 end 직전 순위까지 저장된 표본 수를 반환합니다."""
        if isinstance(end, bool) or not isinstance(end, int):
            raise RepurchaseEvaluationError("Fenwick Tree 조회 경계는 정수여야 합니다.")
        if end < 0 or end > self._size:
            raise RepurchaseEvaluationError(
                "Fenwick Tree 조회 경계가 범위를 벗어났습니다."
            )

        count = 0
        tree_index = end
        while tree_index > 0:
            count += int(self._tree[tree_index])
            tree_index -= tree_index & -tree_index
        return count


def calculate_regression_metrics(rows: pd.DataFrame) -> dict[str, float | int]:
    """관측된 재구매 기간의 MAE·중앙 절대오차·허용일 내 적중률을 계산합니다."""
    missing_columns = set(EVALUATION_REQUIRED_COLUMNS) - set(rows.columns)
    if missing_columns:
        raise RepurchaseEvaluationError(
            f"재구매 평가 필수 컬럼이 누락됐습니다: {sorted(missing_columns)}"
        )
    evaluation_rows = rows.dropna(
        subset=["target_duration_days", "predicted_duration_days"]
    ).copy()
    if evaluation_rows.empty:
        raise RepurchaseEvaluationError("평가할 관측 재구매 표본이 없습니다.")

    absolute_error = (
        evaluation_rows["target_duration_days"]
        - evaluation_rows["predicted_duration_days"]
    ).abs()
    metrics: dict[str, float | int] = {
        "sample_count": int(len(evaluation_rows)),
        "mae_days": float(absolute_error.mean()),
        "median_absolute_error_days": float(absolute_error.median()),
    }
    for window_days in HIT_WINDOWS_DAYS:
        metrics[f"within_{window_days}_days_rate"] = float(
            absolute_error.le(window_days).mean()
        )
    return metrics


def evaluate_predictions(rows: pd.DataFrame) -> dict[str, object]:
    """전체 성능과 개인·상품·전역 fallback별 성능 및 사용 비율을 반환합니다."""
    overall = calculate_regression_metrics(rows)
    total_count = int(overall["sample_count"])
    by_source: dict[str, object] = {}
    for source, source_rows in rows.groupby(
        "prediction_source",
        observed=True,
        sort=True,
    ):
        source_metrics = calculate_regression_metrics(source_rows)
        source_metrics["sample_rate"] = float(
            int(source_metrics["sample_count"]) / total_count
        )
        by_source[str(source)] = source_metrics
    return {"overall": overall, "by_prediction_source": by_source}


def _divide_or_none(numerator: float, denominator: float) -> float | None:
    """분모가 0인 비율을 성능 0으로 오해하지 않도록 계산 불가로 남깁니다."""
    if denominator == 0:
        return None
    return numerator / denominator


def calculate_pair_concordance_credit(
    earlier_prediction_days: float,
    later_prediction_days: float,
) -> float:
    """실제 사건 순서가 정해진 한 쌍의 예측 순서에 일치 점수를 부여합니다."""
    if earlier_prediction_days < later_prediction_days:
        return 1.0
    if earlier_prediction_days == later_prediction_days:
        return 0.5
    return 0.0


def is_ipcw_concordance_pair_comparable(
    event_duration_days: float,
    event_observed: bool,
    comparison_duration_days: float,
    *,
    horizon_days: int,
) -> bool:
    """기준 표본의 재구매가 먼저 확인되어 순서를 비교할 수 있는지 판단합니다."""
    return bool(
        event_observed
        and event_duration_days <= horizon_days
        and event_duration_days < comparison_duration_days
    )


def calculate_ipcw_concordance_pair_weight(
    censoring_survival_probability: float,
) -> float:
    """먼저 발생한 사건 시점의 검열 생존확률로 비교 쌍의 IPCW를 계산합니다."""
    if (
        not np.isfinite(censoring_survival_probability)
        or censoring_survival_probability <= 0
        or censoring_survival_probability > 1
    ):
        raise RepurchaseEvaluationError(
            "C-index의 검열 생존확률은 0보다 크고 1 이하여야 합니다."
        )
    return 1.0 / censoring_survival_probability**2


def evaluate_ipcw_concordance_index(
    rows: pd.DataFrame,
) -> dict[str, float | int]:
    """검열을 보정해 고정 시점 안의 재구매 순위 일치도를 계산합니다."""
    missing_columns = IPCW_CONCORDANCE_REQUIRED_COLUMNS - set(rows.columns)
    if missing_columns:
        raise RepurchaseEvaluationError(
            f"IPCW C-index 필수 컬럼이 누락됐습니다: {sorted(missing_columns)}"
        )
    if rows.empty:
        raise RepurchaseEvaluationError("IPCW C-index를 계산할 표본이 없습니다.")

    predicted_duration = rows["predicted_duration_days"]
    observed_duration = rows["survival_observed_duration_days"]
    event_observed = rows["survival_event_observed"]
    if (
        not is_numeric_dtype(predicted_duration.dtype)
        or not np.isfinite(
            predicted_duration.to_numpy(dtype="float64", copy=False)
        ).all()
        or predicted_duration.lt(0).any()
    ):
        raise RepurchaseEvaluationError(
            "C-index의 예상 재구매 일수는 0 이상의 유한한 숫자여야 합니다."
        )
    if (
        not is_numeric_dtype(observed_duration.dtype)
        or not np.isfinite(
            observed_duration.to_numpy(dtype="float64", copy=False)
        ).all()
        or observed_duration.lt(0).any()
    ):
        raise RepurchaseEvaluationError(
            "C-index의 관측 일수는 0 이상의 유한한 숫자여야 합니다."
        )
    if event_observed.isna().any() or not is_bool_dtype(event_observed.dtype):
        raise RepurchaseEvaluationError(
            "C-index의 사건 관측 여부에는 결측값 없는 boolean만 사용할 수 있습니다."
        )

    horizons = rows["ipcw_horizon_days"].drop_duplicates()
    if len(horizons) != 1:
        raise RepurchaseEvaluationError(
            "한 번의 IPCW C-index 평가는 하나의 고정 시점만 사용합니다."
        )
    horizon_value = horizons.iloc[0]
    if (
        isinstance(horizon_value, (bool, np.bool_))
        or not isinstance(horizon_value, (int, float, np.integer, np.floating))
        or not np.isfinite(horizon_value)
        or float(horizon_value) <= 0
        or not float(horizon_value).is_integer()
    ):
        raise RepurchaseEvaluationError(
            "IPCW C-index의 고정 시점은 양의 정수 일수여야 합니다."
        )
    horizon_days = int(horizon_value)

    evaluation_rows = rows.loc[
        :,
        [
            "predicted_duration_days",
            "survival_observed_duration_days",
            "survival_event_observed",
            "ipcw_censoring_survival_probability",
        ],
    ].copy()
    unique_predictions = np.sort(predicted_duration.unique())
    evaluation_rows["prediction_rank_for_concordance"] = np.searchsorted(
        unique_predictions,
        predicted_duration,
    )
    evaluation_rows = evaluation_rows.sort_values(
        "survival_observed_duration_days",
        ascending=False,
        kind="mergesort",
    )

    rank_counts = _FenwickCountTree(size=len(unique_predictions))
    stored_later_count = 0
    event_reference_count = 0
    contributing_event_count = 0
    comparable_pair_count = 0
    concordant_pair_count = 0
    tied_pair_count = 0
    discordant_pair_count = 0
    concordance_credit = 0.0
    weighted_comparable_pair_mass = 0.0
    weighted_concordance_credit = 0.0

    for observed_days, duration_rows in evaluation_rows.groupby(
        "survival_observed_duration_days",
        observed=True,
        sort=False,
    ):
        reference_rows = duration_rows.loc[
            duration_rows["survival_event_observed"]
            & (float(observed_days) <= horizon_days)
        ]
        event_reference_count += len(reference_rows)

        for rank, censoring_probability in zip(
            reference_rows["prediction_rank_for_concordance"],
            reference_rows["ipcw_censoring_survival_probability"],
            strict=True,
        ):
            if stored_later_count == 0:
                continue
            pair_weight = calculate_ipcw_concordance_pair_weight(
                float(censoring_probability)
            )
            rank = int(rank)
            smaller_count = rank_counts.prefix_count(rank)
            same_or_smaller_count = rank_counts.prefix_count(rank + 1)
            equal_count = same_or_smaller_count - smaller_count
            greater_count = stored_later_count - same_or_smaller_count

            contributing_event_count += 1
            comparable_pair_count += stored_later_count
            concordant_pair_count += greater_count
            tied_pair_count += equal_count
            discordant_pair_count += smaller_count
            pair_credit = greater_count + 0.5 * equal_count
            concordance_credit += pair_credit
            weighted_comparable_pair_mass += pair_weight * stored_later_count
            weighted_concordance_credit += pair_weight * pair_credit

        # 같은 관측 시간끼리는 순서를 정할 수 없으므로 현재 그룹 평가는 먼저 끝냅니다.
        for rank in duration_rows["prediction_rank_for_concordance"]:
            rank_counts.add(int(rank))
            stored_later_count += 1

    if comparable_pair_count == 0 or weighted_comparable_pair_mass == 0:
        raise RepurchaseEvaluationError(
            "IPCW C-index를 계산할 비교 가능한 표본 쌍이 없습니다."
        )

    return {
        "horizon_days": horizon_days,
        "validation_sample_count": int(len(rows)),
        "event_reference_count": int(event_reference_count),
        "contributing_event_count": int(contributing_event_count),
        "comparable_pair_count": int(comparable_pair_count),
        "concordant_pair_count": int(concordant_pair_count),
        "tied_pair_count": int(tied_pair_count),
        "discordant_pair_count": int(discordant_pair_count),
        "unweighted_concordance_index": float(
            concordance_credit / comparable_pair_count
        ),
        "ipcw_weighted_comparable_pair_mass": float(weighted_comparable_pair_mass),
        "ipcw_weighted_concordance_credit": float(weighted_concordance_credit),
        "ipcw_concordance_index": float(
            weighted_concordance_credit / weighted_comparable_pair_mass
        ),
    }


def evaluate_ipcw_binary_predictions(
    rows: pd.DataFrame,
) -> dict[str, float | int | None]:
    """예상 일수를 고정 시점 이진 예측으로 바꿔 IPCW 오류율을 계산합니다."""
    missing_columns = IPCW_BINARY_EVALUATION_REQUIRED_COLUMNS - set(rows.columns)
    if missing_columns:
        raise RepurchaseEvaluationError(
            f"IPCW 이진 평가 필수 컬럼이 누락됐습니다: {sorted(missing_columns)}"
        )
    if rows.empty:
        raise RepurchaseEvaluationError("IPCW 이진 평가에 사용할 표본이 없습니다.")

    outcome_known = rows["ipcw_outcome_known"]
    if outcome_known.isna().any() or not is_bool_dtype(outcome_known.dtype):
        raise RepurchaseEvaluationError(
            "IPCW 정답 확인 여부에는 결측값 없는 boolean만 사용할 수 있습니다."
        )
    known_rows = rows.loc[outcome_known].copy()
    if known_rows.empty:
        raise RepurchaseEvaluationError("IPCW 이진 평가의 정답 확인 표본이 없습니다.")
    if known_rows["ipcw_event_within_horizon"].isna().any():
        raise RepurchaseEvaluationError(
            "정답 확인 표본의 기간 내 재구매 여부가 비었습니다."
        )

    predicted_duration = known_rows["predicted_duration_days"]
    weights = known_rows["ipcw_weight"]
    if (
        not is_numeric_dtype(predicted_duration.dtype)
        or not np.isfinite(
            predicted_duration.to_numpy(dtype="float64", copy=False)
        ).all()
    ):
        raise RepurchaseEvaluationError("예상 재구매 일수는 유한한 숫자여야 합니다.")
    if predicted_duration.lt(0).any():
        raise RepurchaseEvaluationError("예상 재구매 일수는 음수일 수 없습니다.")
    if (
        not is_numeric_dtype(weights.dtype)
        or not np.isfinite(weights.to_numpy(dtype="float64", copy=False)).all()
    ):
        raise RepurchaseEvaluationError("IPCW 가중치는 유한한 숫자여야 합니다.")
    if weights.le(0).any():
        raise RepurchaseEvaluationError(
            "정답 확인 표본의 IPCW 가중치는 0보다 커야 합니다."
        )

    horizons = known_rows["ipcw_horizon_days"].drop_duplicates()
    if len(horizons) != 1:
        raise RepurchaseEvaluationError(
            "한 번의 IPCW 평가는 하나의 고정 시점만 사용합니다."
        )
    horizon_days = int(horizons.iloc[0])

    actual_event = known_rows["ipcw_event_within_horizon"].astype(bool)
    predicted_event = predicted_duration.le(horizon_days)
    incorrect = predicted_event.ne(actual_event)
    true_positive = predicted_event & actual_event
    true_negative = ~predicted_event & ~actual_event
    false_positive = predicted_event & ~actual_event
    false_negative = ~predicted_event & actual_event

    weight_sum = float(weights.sum())
    weighted_true_positive_mass = float(weights.loc[true_positive].sum())
    weighted_true_negative_mass = float(weights.loc[true_negative].sum())
    weighted_error_mass = float(weights.loc[incorrect].sum())
    weighted_false_positive_mass = float(weights.loc[false_positive].sum())
    weighted_false_negative_mass = float(weights.loc[false_negative].sum())
    weighted_event_mass = weighted_true_positive_mass + weighted_false_negative_mass
    weighted_no_event_mass = weighted_true_negative_mass + weighted_false_positive_mass
    weighted_predicted_event_mass = (
        weighted_true_positive_mass + weighted_false_positive_mass
    )
    weighted_precision = _divide_or_none(
        weighted_true_positive_mass,
        weighted_predicted_event_mass,
    )
    weighted_recall = _divide_or_none(
        weighted_true_positive_mass,
        weighted_event_mass,
    )
    weighted_specificity = _divide_or_none(
        weighted_true_negative_mass,
        weighted_no_event_mass,
    )
    weighted_balanced_accuracy = (
        None
        if weighted_recall is None or weighted_specificity is None
        else (weighted_recall + weighted_specificity) / 2
    )
    weighted_f1 = (
        None
        if weighted_precision is None
        or weighted_recall is None
        or weighted_precision + weighted_recall == 0
        else 2
        * weighted_precision
        * weighted_recall
        / (weighted_precision + weighted_recall)
    )
    weighted_error_rate = weighted_error_mass / weight_sum
    always_no_event_error_rate = weighted_event_mass / weight_sum
    always_no_event_accuracy = 1.0 - always_no_event_error_rate
    weighted_accuracy = 1.0 - weighted_error_rate

    return {
        "horizon_days": horizon_days,
        "validation_sample_count": int(len(rows)),
        "outcome_known_count": int(len(known_rows)),
        "predicted_event_count": int(predicted_event.sum()),
        "unweighted_binary_error_rate": float(incorrect.mean()),
        "ipcw_weight_sum": weight_sum,
        "ipcw_weighted_true_positive_mass": weighted_true_positive_mass,
        "ipcw_weighted_true_negative_mass": weighted_true_negative_mass,
        "ipcw_weighted_error_mass": weighted_error_mass,
        "ipcw_weighted_binary_error_rate": weighted_error_rate,
        "ipcw_weighted_binary_accuracy": weighted_accuracy,
        "ipcw_weighted_false_positive_mass": weighted_false_positive_mass,
        "ipcw_weighted_false_negative_mass": weighted_false_negative_mass,
        "ipcw_weighted_precision": weighted_precision,
        "ipcw_weighted_recall": weighted_recall,
        "ipcw_weighted_specificity": weighted_specificity,
        "ipcw_weighted_balanced_accuracy": weighted_balanced_accuracy,
        "ipcw_weighted_f1": weighted_f1,
        "always_no_event_error_rate": always_no_event_error_rate,
        "always_no_event_accuracy": always_no_event_accuracy,
        "accuracy_difference_vs_always_no_event": weighted_accuracy
        - always_no_event_accuracy,
    }


def evaluate_ipcw_brier_score(
    rows: pd.DataFrame,
    *,
    reference_probability: float,
) -> dict[str, float | int | None]:
    """확률 예측의 제곱 오차를 IPCW로 보정하고 전체 확률 기준선과 비교합니다."""
    missing_columns = IPCW_BRIER_REQUIRED_COLUMNS - set(rows.columns)
    if missing_columns:
        raise RepurchaseEvaluationError(
            f"IPCW Brier Score 필수 컬럼이 누락됐습니다: {sorted(missing_columns)}"
        )
    if rows.empty:
        raise RepurchaseEvaluationError("IPCW Brier Score를 계산할 표본이 없습니다.")

    outcome_known = rows["ipcw_outcome_known"]
    if outcome_known.isna().any() or not is_bool_dtype(outcome_known.dtype):
        raise RepurchaseEvaluationError(
            "Brier Score 정답 확인 여부에는 결측값 없는 boolean만 사용할 수 있습니다."
        )
    known_rows = rows.loc[outcome_known].copy()
    if known_rows.empty:
        raise RepurchaseEvaluationError("Brier Score의 정답 확인 표본이 없습니다.")

    actual_event = known_rows["ipcw_event_within_horizon"]
    if actual_event.isna().any() or not is_bool_dtype(actual_event.dtype):
        raise RepurchaseEvaluationError(
            "Brier Score의 실제 사건 여부에는 결측값 없는 boolean만 사용할 수 있습니다."
        )
    predicted_probability = known_rows["predicted_event_probability"]
    if (
        not is_numeric_dtype(predicted_probability.dtype)
        or not np.isfinite(
            predicted_probability.to_numpy(dtype="float64", copy=False)
        ).all()
        or not predicted_probability.between(0, 1).all()
    ):
        raise RepurchaseEvaluationError(
            "예측 사건 확률은 0부터 1 사이의 유한한 숫자여야 합니다."
        )
    if (
        isinstance(reference_probability, (bool, np.bool_))
        or not isinstance(
            reference_probability,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(reference_probability)
        or not 0 <= float(reference_probability) <= 1
    ):
        raise RepurchaseEvaluationError(
            "Brier Score 기준 확률은 0부터 1 사이의 유한한 숫자여야 합니다."
        )

    weights = known_rows["ipcw_weight"]
    if (
        not is_numeric_dtype(weights.dtype)
        or not np.isfinite(weights.to_numpy(dtype="float64", copy=False)).all()
        or weights.le(0).any()
    ):
        raise RepurchaseEvaluationError(
            "Brier Score의 IPCW 가중치는 0보다 큰 유한한 숫자여야 합니다."
        )
    horizons = known_rows["ipcw_horizon_days"].drop_duplicates()
    if len(horizons) != 1:
        raise RepurchaseEvaluationError(
            "한 번의 IPCW Brier Score 평가는 하나의 고정 시점만 사용합니다."
        )
    horizon_value = horizons.iloc[0]
    if (
        isinstance(horizon_value, (bool, np.bool_))
        or not isinstance(horizon_value, (int, np.integer))
        or int(horizon_value) <= 0
    ):
        raise RepurchaseEvaluationError(
            "IPCW Brier Score의 고정 시점은 양의 정수 일수여야 합니다."
        )

    actual_value = actual_event.astype("float64")
    squared_error = predicted_probability.sub(actual_value).pow(2)
    normalized_reference_probability = float(reference_probability)
    reference_squared_error = actual_value.sub(normalized_reference_probability).pow(2)
    weight_sum = float(weights.sum())
    weighted_brier_score = float(squared_error.mul(weights).sum() / weight_sum)
    weighted_reference_brier_score = float(
        reference_squared_error.mul(weights).sum() / weight_sum
    )
    brier_skill_score = (
        None
        if weighted_reference_brier_score == 0
        else 1.0 - weighted_brier_score / weighted_reference_brier_score
    )

    return {
        "horizon_days": int(horizon_value),
        "validation_sample_count": int(len(rows)),
        "outcome_known_count": int(len(known_rows)),
        "ipcw_weight_sum": weight_sum,
        "unweighted_brier_score": float(squared_error.mean()),
        "ipcw_brier_score": weighted_brier_score,
        "reference_probability": normalized_reference_probability,
        "ipcw_reference_brier_score": weighted_reference_brier_score,
        "brier_skill_score": brier_skill_score,
    }
