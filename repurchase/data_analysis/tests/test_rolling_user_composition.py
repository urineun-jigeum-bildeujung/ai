"""Rolling fold 사용자 구성 분석의 집계 보존과 시간 창 분리를 검증합니다."""

import pandas as pd
import pytest

from scripts.modeling.rolling_user_composition import (
    bootstrap_fold_difference_by_union_user,
    build_fold_user_composition,
    summarize_fold_user_composition,
)


def _rows(users: list[str], order_prefix: str) -> pd.DataFrame:
    """정답 확인 행과 검열로 정답이 불명인 행을 함께 만듭니다."""
    return pd.DataFrame(
        {
            "user_id": users,
            "order_id": [f"{order_prefix}-{index}" for index in range(len(users))],
            "product_id": ["p1"] * len(users),
            "ipcw_outcome_known": [True, True, False][: len(users)],
            "ipcw_event_within_horizon": pd.array(
                [True, False, pd.NA][: len(users)], dtype="boolean"
            ),
            "ipcw_weight": [2.0, 1.0, 0.0][: len(users)],
            "reference_predicted_event_probability": [0.8, 0.4, 0.5][: len(users)],
            "candidate_predicted_event_probability": [0.7, 0.2, 0.5][: len(users)],
        }
    )


def _fold(
    fold_id: str,
    users: list[str],
    source_train_users: list[str],
    common_train_users: list[str],
    lightgbm_train_users: list[str],
):
    """학습 노출을 서로 다르게 지정한 작은 fold를 만듭니다."""

    def train(ids: list[str]) -> pd.DataFrame:
        return pd.DataFrame({"user_id": ids})

    return build_fold_user_composition(
        fold_id,
        _rows(users, fold_id),
        source_train=train(source_train_users),
        common_train=train(common_train_users),
        lightgbm_train=train(lightgbm_train_users),
    )


def test_user_groups_preserve_rows_weights_and_brier() -> None:
    """이전 Validation·Train만·새 사용자 그룹이 원래 fold를 재구성합니다."""
    first = _fold("fold_1", ["a", "x", "z"], ["a"], ["a"], ["a"])
    second = _fold("fold_2", ["a", "b", "c"], ["a", "b", "c"], ["a", "b"], ["a"])
    groups, diagnostics, overlaps = summarize_fold_user_composition([first, second])
    current = groups.loc[groups["fold_id"].eq("fold_2")]
    assert dict(zip(current["group"], current["user_count"], strict=True)) == {
        "previous_validation": 1,
        "common_train_only": 1,
        "source_train_only_excluded": 1,
        "new_to_source_train": 0,
    }
    assert current["sample_count"].sum() == 3
    assert current["known_count"].sum() == 2
    assert current["unknown_count"].sum() == 1
    assert current["event_count"].sum() == 1
    assert current["no_event_count"].sum() == 1
    assert current["known_weight"].sum() == pytest.approx(3.0)
    assert current["fold_contribution"].sum() == pytest.approx(
        diagnostics.loc[1, "row_weighted_brier_difference"]
    )
    assert diagnostics.loc[1, "evaluation_users_absent_source_train"] == 0
    assert diagnostics.loc[1, "evaluation_users_absent_common_train"] == 1
    assert diagnostics.loc[1, "evaluation_users_absent_lightgbm_train"] == 2
    assert overlaps.loc[0, "shared_user_count"] == 1
    assert overlaps.loc[0, "current_shared_sample_share"] == pytest.approx(1 / 3)
    assert diagnostics.loc[1, "known_user_count"] == 2
    assert diagnostics.loc[1, "row_weighted_brier_difference"] == pytest.approx(
        (0.24 - 0.22) / 3
    )


def test_duplicate_purchase_across_folds_is_rejected() -> None:
    """두 창이 같은 구매 사건을 공유하면 사용자 중복과 다르게 오류로 봅니다."""
    first = _fold("fold_1", ["a", "b"], ["a"], ["a"], ["a"])
    repeated = _rows(["a", "b"], "fold_1")
    second = build_fold_user_composition(
        "fold_2",
        repeated,
        source_train=pd.DataFrame({"user_id": ["a"]}),
        common_train=pd.DataFrame({"user_id": ["a"]}),
        lightgbm_train=pd.DataFrame({"user_id": ["a"]}),
    )
    with pytest.raises(ValueError, match="같은 구매 사건"):
        summarize_fold_user_composition([first, second])


def test_unknown_outcome_does_not_become_no_event() -> None:
    """검열로 불명인 행은 평가 분모와 미사건 수에 넣지 않습니다."""
    fold = _fold("fold_1", ["a", "b", "c"], ["a"], ["a"], ["a"])
    assert fold.users.loc["c", "known_count"] == 0
    assert fold.users.loc["c", "unknown_count"] == 1
    assert fold.users.loc["c", "no_event_count"] == 0
    assert fold.users.loc["c", "known_weight"] == 0


def test_true_new_user_is_not_the_same_as_excluded_training_user() -> None:
    """원천 Train에 없던 사용자만 진짜 신규로 분류합니다."""
    first = _fold("fold_1", ["a", "x", "z"], ["a"], ["a"], ["a"])
    second = _fold("fold_2", ["a", "b", "c"], ["a", "b"], ["a", "b"], ["a"])
    groups, _, _ = summarize_fold_user_composition([first, second])
    current = groups.loc[groups["fold_id"].eq("fold_2")].set_index("group")
    assert current.loc["source_train_only_excluded", "user_count"] == 0
    assert current.loc["new_to_source_train", "user_count"] == 1


def test_invalid_probability_and_duplicate_key_fail_before_summary() -> None:
    """확률 범위 위반과 구매 키 중복을 묵시적으로 집계하지 않습니다."""
    rows = _rows(["a", "b"], "fold_1")
    train = pd.DataFrame({"user_id": ["a"]})
    rows.loc[0, "reference_predicted_event_probability"] = 1.2
    with pytest.raises(ValueError, match="유효하지"):
        build_fold_user_composition(
            "fold_1",
            rows,
            source_train=train,
            common_train=train,
            lightgbm_train=train,
        )
    rows.loc[0, "reference_predicted_event_probability"] = 0.8
    rows.loc[1, ["user_id", "order_id", "product_id"]] = rows.loc[
        0, ["user_id", "order_id", "product_id"]
    ].to_numpy()
    with pytest.raises(ValueError, match="중복"):
        build_fold_user_composition(
            "fold_1",
            rows,
            source_train=train,
            common_train=train,
            lightgbm_train=train,
        )


def test_training_user_sets_must_be_nested() -> None:
    """LightGBM과 공통 Train은 원천 Train의 부분집합이어야 합니다."""
    rows = _rows(["a", "b"], "fold_1")
    with pytest.raises(ValueError, match="부분집합"):
        build_fold_user_composition(
            "fold_1",
            rows,
            source_train=pd.DataFrame({"user_id": ["a"]}),
            common_train=pd.DataFrame({"user_id": ["a", "b"]}),
            lightgbm_train=pd.DataFrame({"user_id": ["a"]}),
        )


def test_union_user_bootstrap_is_reproducible() -> None:
    """사용자 합집합을 공통 재표집하면 seed 고정 결과가 반복됩니다."""
    first = _fold("fold_1", ["a", "b", "c"], ["a"], ["a"], ["a"])
    second = _fold("fold_2", ["a", "b", "d"], ["a"], ["a"], ["a"])
    first_result = bootstrap_fold_difference_by_union_user(
        first, second, replicates=100, seed=42
    )
    second_result = bootstrap_fold_difference_by_union_user(
        first, second, replicates=100, seed=42
    )
    assert first_result == second_result
    assert first_result["status"] in {"evaluated", "incomplete_resampling"}


def test_union_user_bootstrap_reports_missing_fold_in_resample() -> None:
    """작은 합집합에서 한 fold의 정답이 빠진 반복을 성공으로 세지 않습니다."""
    first = _fold("fold_1", ["a", "c"], ["a"], ["a"], ["a"])
    second = _fold("fold_2", ["b", "d"], ["b"], ["b"], ["b"])
    result = bootstrap_fold_difference_by_union_user(
        first, second, replicates=100, seed=42
    )
    assert result["status"] == "incomplete_resampling"
    assert 0 < result["valid_replicates"] < 100
