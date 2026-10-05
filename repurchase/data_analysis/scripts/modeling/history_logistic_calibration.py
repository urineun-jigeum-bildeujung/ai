"""Development-only regularized calibration with purchase-history interactions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def history_logistic_features(
    probability: pd.Series, history_interval_count: pd.Series
) -> pd.DataFrame:
    """Use the raw log-odds and two low-history group interactions."""
    if not probability.index.equals(history_interval_count.index):
        raise ValueError("확률과 구매 이력의 행 순서가 다릅니다.")
    raw = probability.to_numpy(dtype="float64")
    count = pd.to_numeric(history_interval_count, errors="coerce").to_numpy(
        dtype="float64"
    )
    if (
        not np.isfinite(raw).all()
        or ((raw < 0) | (raw > 1)).any()
        or not np.isfinite(count).all()
        or (count < 0).any()
        or (count != np.floor(count)).any()
    ):
        raise ValueError("보정 입력 확률 또는 구매 간격 수가 유효하지 않습니다.")
    log_odds = np.log(np.clip(raw, 1e-6, 1 - 1e-6)) - np.log1p(
        -np.clip(raw, 1e-6, 1 - 1e-6)
    )
    zero = (count == 0).astype("float64")
    one_or_two = ((count >= 1) & (count <= 2)).astype("float64")
    return pd.DataFrame(
        {
            "raw_log_odds": log_odds,
            "history_zero": zero,
            "history_one_or_two": one_or_two,
            "raw_log_odds_x_zero": log_odds * zero,
            "raw_log_odds_x_one_or_two": log_odds * one_or_two,
        },
        index=probability.index,
    )


def fit_history_logistic(
    probability: pd.Series,
    history_interval_count: pd.Series,
    outcome: pd.Series,
    weight: pd.Series,
) -> LogisticRegression:
    """Fit a fixed L2 candidate on known inner-calibration outcomes only."""
    features = history_logistic_features(probability, history_interval_count)
    if not features.index.equals(outcome.index) or not features.index.equals(
        weight.index
    ):
        raise ValueError("보정 입력·정답·가중치의 행 순서가 다릅니다.")
    actual = outcome.to_numpy(dtype="float64")
    weights = weight.to_numpy(dtype="float64")
    if (
        len(actual) == 0
        or not np.isin(actual, [0, 1]).all()
        or np.unique(actual).size != 2
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("보정 정답은 두 부류를 포함하고 가중치는 양수여야 합니다.")
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(features, actual.astype("int64"), sample_weight=weights / weights.mean())
    return model


def predict_history_logistic(
    model: LogisticRegression,
    probability: pd.Series,
    history_interval_count: pd.Series,
) -> pd.Series:
    """Preserve row identity and reject invalid fitted-model probabilities."""
    features = history_logistic_features(probability, history_interval_count)
    values = model.predict_proba(features)[:, 1]
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("이력별 보정 확률이 유효하지 않습니다.")
    return pd.Series(
        values, index=probability.index, name="history_logistic_probability"
    )
