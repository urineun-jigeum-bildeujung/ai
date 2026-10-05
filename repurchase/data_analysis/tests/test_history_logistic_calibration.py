"""History-aware candidate uses only aligned, valid calibration inputs."""

import numpy as np
import pandas as pd
import pytest

from scripts.modeling.history_logistic_calibration import (
    fit_history_logistic,
    history_logistic_features,
    predict_history_logistic,
)


def test_history_features_preserve_order_and_interactions():
    probability = pd.Series([0.2, 0.4, 0.6], index=[8, 3, 5])
    count = pd.Series([0, 2, 4], index=probability.index)
    features = history_logistic_features(probability, count)
    assert features.index.tolist() == [8, 3, 5]
    assert features["history_zero"].tolist() == [1, 0, 0]
    assert features["history_one_or_two"].tolist() == [0, 1, 0]
    assert features.loc[5, "raw_log_odds_x_zero"] == 0


def test_history_logistic_learns_group_difference_and_preserves_index():
    index = pd.Index(range(200))
    probability = pd.Series(0.4, index=index)
    count = pd.Series([0] * 100 + [3] * 100, index=index)
    outcome = pd.Series([0] * 90 + [1] * 10 + [0] * 10 + [1] * 90, index=index)
    weight = pd.Series(1.0, index=index)
    model = fit_history_logistic(probability, count, outcome, weight)
    predicted = predict_history_logistic(model, probability, count)
    assert predicted.index.equals(index)
    assert predicted.iloc[:100].mean() < predicted.iloc[100:].mean()
    assert np.isfinite(predicted).all()


@pytest.mark.parametrize("invalid", [float("nan"), -1, 1.1])
def test_invalid_probability_is_rejected(invalid):
    probability = pd.Series([0.3, invalid])
    count = pd.Series([0, 1])
    with pytest.raises(ValueError, match="보정 입력"):
        history_logistic_features(probability, count)


def test_invalid_history_and_misalignment_are_rejected():
    probability = pd.Series([0.3, 0.4], index=[1, 2])
    with pytest.raises(ValueError, match="행 순서"):
        history_logistic_features(probability, pd.Series([0, 1], index=[2, 1]))
    with pytest.raises(ValueError, match="보정 입력"):
        history_logistic_features(probability, pd.Series([0, -1], index=[1, 2]))


def test_fit_requires_two_outcomes_and_valid_weights():
    probability = pd.Series([0.3, 0.4])
    count = pd.Series([0, 3])
    with pytest.raises(ValueError, match="두 부류"):
        fit_history_logistic(
            probability, count, pd.Series([0, 0]), pd.Series([1.0, 1.0])
        )
    with pytest.raises(ValueError, match="가중치"):
        fit_history_logistic(
            probability, count, pd.Series([0, 1]), pd.Series([1.0, 0.0])
        )
