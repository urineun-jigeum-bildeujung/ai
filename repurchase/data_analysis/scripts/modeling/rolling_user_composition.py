"""Rolling Validation의 사용자 중복과 사용자별 확률 오차를 진단합니다.

원본 사용자 ID는 실행 중 집계에만 사용하며 보고서에는 집계값만 남깁니다.
각 fold의 모델 설정과 IPCW 가중치는 기존 실험에서 고정한 값을 재사용합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FoldUserComposition:
    """한 fold의 사용자별 충분통계와 시간 구간별 식별자 집합입니다."""

    fold_id: str
    users: pd.DataFrame
    source_train_users: frozenset[object]
    common_train_users: frozenset[object]
    lightgbm_train_users: frozenset[object]
    user_product_pairs: frozenset[tuple[object, object]]
    purchase_keys: frozenset[tuple[object, object, object]]


def build_fold_user_composition(
    fold_id: str,
    paired_rows: pd.DataFrame,
    *,
    source_train: pd.DataFrame,
    common_train: pd.DataFrame,
    lightgbm_train: pd.DataFrame,
) -> FoldUserComposition:
    """같은 평가 행에서 사용자별 IPCW 가중 오차 합계를 보존합니다."""
    required = {
        "user_id",
        "order_id",
        "product_id",
        "ipcw_outcome_known",
        "ipcw_event_within_horizon",
        "ipcw_weight",
        "reference_predicted_event_probability",
        "candidate_predicted_event_probability",
    }
    missing = required - set(paired_rows.columns)
    if missing or paired_rows.empty:
        raise ValueError(f"평가 행이 비었거나 필수 열이 없습니다: {sorted(missing)}")
    if paired_rows[["user_id", "order_id", "product_id"]].isna().any().any():
        raise ValueError(
            "평가 행의 사용자·주문·상품 ID에는 결측값을 허용하지 않습니다."
        )
    key_columns = ["user_id", "order_id", "product_id"]
    if paired_rows.duplicated(key_columns).any():
        raise ValueError("같은 fold에서 구매 사건 키가 중복됐습니다.")
    known = paired_rows["ipcw_outcome_known"]
    if known.isna().any() or not pd.api.types.is_bool_dtype(known.dtype):
        raise ValueError("정답 확인 여부는 결측 없는 불리언이어야 합니다.")
    checked = paired_rows.loc[known].copy()
    if checked.empty:
        raise ValueError("정답을 확인할 수 있는 평가 행이 없습니다.")
    events = checked["ipcw_event_within_horizon"]
    weights = checked["ipcw_weight"].astype("float64")
    probabilities = checked[
        [
            "reference_predicted_event_probability",
            "candidate_predicted_event_probability",
        ]
    ].astype("float64")
    if (
        events.isna().any()
        or not pd.api.types.is_bool_dtype(events.dtype)
        or not np.isfinite(weights.to_numpy()).all()
        or weights.le(0).any()
        or not np.isfinite(probabilities.to_numpy()).all()
        or probabilities.lt(0).any().any()
        or probabilities.gt(1).any().any()
    ):
        raise ValueError(
            "정답 확인 행의 사건·IPCW 가중치·예측 확률이 유효하지 않습니다."
        )

    # 미확인 행은 '미재구매'가 아닙니다. 수와 오차를 분리해 집계합니다.
    user_rows = paired_rows.groupby("user_id", sort=True, observed=True).size()
    known_rows = checked.groupby("user_id", sort=True, observed=True).size()
    checked["weighted_event"] = weights.mul(events.astype("float64"))
    checked["aft_error_sum"] = weights.mul(
        probabilities["reference_predicted_event_probability"]
        .sub(events.astype("float64"))
        .pow(2)
    )
    checked["lightgbm_error_sum"] = weights.mul(
        probabilities["candidate_predicted_event_probability"]
        .sub(events.astype("float64"))
        .pow(2)
    )
    checked["known_weight"] = weights
    checked["event_count"] = events.astype("int64")
    totals = checked.groupby("user_id", sort=True, observed=True)[
        [
            "known_weight",
            "weighted_event",
            "aft_error_sum",
            "lightgbm_error_sum",
            "event_count",
        ]
    ].sum()
    users = pd.DataFrame(index=user_rows.index)
    users["sample_count"] = user_rows
    users["known_count"] = known_rows.reindex(users.index, fill_value=0)
    users["unknown_count"] = users["sample_count"] - users["known_count"]
    for column in totals.columns:
        users[column] = totals[column].reindex(users.index, fill_value=0.0)
    users["no_event_count"] = users["known_count"] - users["event_count"]
    if int(users["sample_count"].sum()) != len(paired_rows):
        raise ValueError("사용자별 평가 행의 합계가 원본과 다릅니다.")

    def training_users(rows: pd.DataFrame) -> frozenset[object]:
        """학습 모집단 ID 결측을 거부하고 사용자 집합을 만듭니다."""
        if "user_id" not in rows or rows["user_id"].isna().any():
            raise ValueError("학습 모집단 사용자 ID가 누락됐습니다.")
        return frozenset(rows["user_id"])

    source_users = training_users(source_train)
    common_users = training_users(common_train)
    lightgbm_users = training_users(lightgbm_train)
    if not lightgbm_users <= common_users <= source_users:
        raise ValueError("실제 학습 사용자 집합이 원천 Train의 부분집합이 아닙니다.")
    return FoldUserComposition(
        fold_id=fold_id,
        users=users,
        source_train_users=source_users,
        common_train_users=common_users,
        lightgbm_train_users=lightgbm_users,
        user_product_pairs=frozenset(
            paired_rows[["user_id", "product_id"]].itertuples(index=False, name=None)
        ),
        purchase_keys=frozenset(
            paired_rows[key_columns].itertuples(index=False, name=None)
        ),
    )


def _score(
    users: pd.DataFrame, denominator: float | None = None
) -> dict[str, float | None]:
    """같은 행의 사용자 통계에서 두 모델 Brier와 fold 기여량을 계산합니다."""
    weight = float(users["known_weight"].sum())
    if weight <= 0:
        return {
            "aft_brier": None,
            "lightgbm_brier": None,
            "brier_difference": None,
            "fold_contribution": None,
            "weighted_event_rate": None,
        }
    aft = float(users["aft_error_sum"].sum()) / weight
    lightgbm = float(users["lightgbm_error_sum"].sum()) / weight
    return {
        "aft_brier": aft,
        "lightgbm_brier": lightgbm,
        "brier_difference": aft - lightgbm,
        "fold_contribution": (
            float(users["aft_error_sum"].sum() - users["lightgbm_error_sum"].sum())
            / denominator
            if denominator is not None
            else None
        ),
        "weighted_event_rate": float(users["weighted_event"].sum()) / weight,
    }


def _group_record(
    fold: FoldUserComposition, group: str, subset: pd.DataFrame
) -> dict[str, object]:
    """그룹 내부 Brier와 전체 fold에 대한 기여량을 구분해 반환합니다."""
    users = fold.users
    total_weight = float(users["known_weight"].sum())
    result: dict[str, object] = {
        "fold_id": fold.fold_id,
        "group": group,
        "user_count": len(subset),
        "user_share": len(subset) / len(users),
        "sample_count": int(subset["sample_count"].sum()),
        "sample_share": int(subset["sample_count"].sum())
        / int(users["sample_count"].sum()),
        "known_count": int(subset["known_count"].sum()),
        "unknown_count": int(subset["unknown_count"].sum()),
        "event_count": int(subset["event_count"].sum()),
        "no_event_count": int(subset["no_event_count"].sum()),
        "known_weight": float(subset["known_weight"].sum()),
        "known_weight_share": float(subset["known_weight"].sum()) / total_weight,
    }
    result.update(_score(subset, total_weight))
    result["status"] = "evaluated" if result["known_count"] else "no_known_outcomes"
    return result


def summarize_fold_user_composition(
    folds: list[FoldUserComposition],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """이전 Validation 노출, 현재 Train 노출, fold 간 사용자 중복을 비교합니다."""
    if not folds or len({fold.fold_id for fold in folds}) != len(folds):
        raise ValueError("fold가 비었거나 fold ID가 중복됐습니다.")
    groups: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    overlaps: list[dict[str, object]] = []
    for index, fold in enumerate(folds):
        users = fold.users
        previous_users = frozenset().union(
            *(set(previous.users.index) for previous in folds[:index])
        )
        if index:
            masks = {
                "previous_validation": users.index.isin(previous_users),
                "common_train_only": users.index.isin(
                    fold.common_train_users - previous_users
                ),
                "source_train_only_excluded": users.index.isin(
                    fold.source_train_users - fold.common_train_users - previous_users
                ),
                "new_to_source_train": ~users.index.isin(
                    previous_users | fold.source_train_users
                ),
            }
            memberships = np.stack(list(masks.values())).sum(axis=0)
            if (memberships != 1).any():
                raise ValueError("사용자 노출 그룹이 중복되거나 빠졌습니다.")
            group_rows = [
                _group_record(fold, name, users.loc[mask])
                for name, mask in masks.items()
            ]
            for column in (
                "user_count",
                "sample_count",
                "known_count",
                "unknown_count",
                "event_count",
                "no_event_count",
                "known_weight",
            ):
                expected = (
                    len(users)
                    if column == "user_count"
                    else float(
                        users[
                            "sample_count"
                            if column == "sample_count"
                            else "known_weight"
                            if column == "known_weight"
                            else column
                        ].sum()
                    )
                )
                actual = sum(float(row[column]) for row in group_rows)
                if not isclose(actual, expected, abs_tol=1e-8):
                    raise ValueError(
                        f"{fold.fold_id}: 그룹 {column} 합계가 맞지 않습니다."
                    )
            total_difference = _score(users)["brier_difference"]
            contribution = sum(
                float(row["fold_contribution"] or 0) for row in group_rows
            )
            if not isclose(contribution, float(total_difference), abs_tol=1e-10):
                raise ValueError("그룹 Brier 기여량 합계가 fold 전체 차이와 다릅니다.")
            groups.extend(group_rows)

        count = users["sample_count"].astype("float64")
        share = count.div(count.sum())
        user_brier = users.loc[users["known_weight"].gt(0)]
        known_weight_share = user_brier["known_weight"].div(
            user_brier["known_weight"].sum()
        )
        # 사용자 안에서 두 모델의 행별 차이를 먼저 상쇄한 뒤 절댓값의
        # 집중도를 봅니다. 부호가 다른 사용자들의 기여는 서로 더하지 않습니다.
        absolute_net_contribution = (
            users["aft_error_sum"].sub(users["lightgbm_error_sum"]).abs()
        )
        contribution_total = float(absolute_net_contribution.sum())
        contribution_share = (
            absolute_net_contribution.div(contribution_total)
            if contribution_total > 0
            else None
        )
        equal_user_aft = (
            user_brier["aft_error_sum"].div(user_brier["known_weight"]).mean()
        )
        equal_user_lightgbm = (
            user_brier["lightgbm_error_sum"].div(user_brier["known_weight"]).mean()
        )
        score = _score(users)
        diagnostics.append(
            {
                "fold_id": fold.fold_id,
                "evaluation_user_count": len(users),
                "known_user_count": len(user_brier),
                "known_user_share": len(user_brier) / len(users),
                "source_train_user_count": len(fold.source_train_users),
                "common_train_user_count": len(fold.common_train_users),
                "lightgbm_train_user_count": len(fold.lightgbm_train_users),
                "evaluation_users_absent_source_train": len(
                    set(users.index) - fold.source_train_users
                ),
                "evaluation_users_absent_common_train": len(
                    set(users.index) - fold.common_train_users
                ),
                "evaluation_users_absent_lightgbm_train": len(
                    set(users.index) - fold.lightgbm_train_users
                ),
                "user_row_count_p50": float(count.quantile(0.5)),
                "user_row_count_p95": float(count.quantile(0.95)),
                "user_row_count_max": int(count.max()),
                "row_top1_share": float(share.nlargest(1).sum()),
                "row_top5_share": float(share.nlargest(5).sum()),
                "row_hhi": float(share.pow(2).sum()),
                "row_effective_user_count": float(1 / share.pow(2).sum()),
                "known_weight_top1_share": float(known_weight_share.nlargest(1).sum()),
                "known_weight_top5_share": float(known_weight_share.nlargest(5).sum()),
                "known_weight_hhi": float(known_weight_share.pow(2).sum()),
                "absolute_net_contribution_top1_share": (
                    float(contribution_share.nlargest(1).sum())
                    if contribution_share is not None
                    else None
                ),
                "absolute_net_contribution_top5_share": (
                    float(contribution_share.nlargest(5).sum())
                    if contribution_share is not None
                    else None
                ),
                "absolute_net_contribution_hhi": (
                    float(contribution_share.pow(2).sum())
                    if contribution_share is not None
                    else None
                ),
                "equal_user_aft_brier": float(equal_user_aft),
                "equal_user_lightgbm_brier": float(equal_user_lightgbm),
                "equal_user_brier_difference": float(
                    equal_user_aft - equal_user_lightgbm
                ),
                "row_weighted_brier_difference": score["brier_difference"],
            }
        )

        for previous in folds[:index]:
            prior_users = set(previous.users.index)
            current_users = set(users.index)
            shared = prior_users & current_users
            union = prior_users | current_users
            if previous.purchase_keys & fold.purchase_keys:
                raise ValueError("서로 다른 Validation 창에 같은 구매 사건이 있습니다.")
            previous_shared = previous.users.loc[previous.users.index.isin(shared)]
            current_shared = users.loc[users.index.isin(shared)]
            overlaps.append(
                {
                    "previous_fold_id": previous.fold_id,
                    "current_fold_id": fold.fold_id,
                    "shared_user_count": len(shared),
                    "union_user_count": len(union),
                    "user_jaccard": len(shared) / len(union),
                    "previous_user_retention": len(shared) / len(prior_users),
                    "current_shared_user_share": len(shared) / len(current_users),
                    "previous_shared_sample_count": int(
                        previous_shared["sample_count"].sum()
                    ),
                    "current_shared_sample_count": int(
                        current_shared["sample_count"].sum()
                    ),
                    "current_shared_sample_share": float(
                        current_shared["sample_count"].sum()
                        / users["sample_count"].sum()
                    ),
                    "shared_user_product_pair_count": len(
                        previous.user_product_pairs & fold.user_product_pairs
                    ),
                    "user_product_pair_jaccard": len(
                        previous.user_product_pairs & fold.user_product_pairs
                    )
                    / len(previous.user_product_pairs | fold.user_product_pairs),
                    "previous_shared_brier_difference": _score(previous_shared)[
                        "brier_difference"
                    ],
                    "current_shared_brier_difference": _score(current_shared)[
                        "brier_difference"
                    ],
                }
            )
    return pd.DataFrame(groups), pd.DataFrame(diagnostics), pd.DataFrame(overlaps)


def bootstrap_fold_difference_by_union_user(
    previous: FoldUserComposition,
    current: FoldUserComposition,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """두 fold의 사용자 합집합을 한 번에 재표집해 상대 성능 변화를 평가합니다.

    학습 모델과 IPCW 가중치는 고정되므로 재학습의 불확실성은 포함하지 않습니다.
    """
    if replicates < 2:
        raise ValueError("Bootstrap 반복 수는 2 이상이어야 합니다.")
    union = pd.Index(sorted(set(previous.users.index) | set(current.users.index)))
    if len(union) < 2:
        return {"status": "insufficient_users", "valid_replicates": 0}
    columns = ["known_weight", "aft_error_sum", "lightgbm_error_sum"]
    first = previous.users.reindex(union, fill_value=0)[columns].to_numpy(
        dtype="float64"
    )
    second = current.users.reindex(union, fill_value=0)[columns].to_numpy(
        dtype="float64"
    )
    point = float(_score(current.users)["brier_difference"]) - float(
        _score(previous.users)["brier_difference"]
    )
    random = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        selected = random.integers(0, len(union), size=len(union))
        sampled_first = first[selected].sum(axis=0)
        sampled_second = second[selected].sum(axis=0)
        if sampled_first[0] <= 0 or sampled_second[0] <= 0:
            continue
        first_difference = (sampled_first[1] - sampled_first[2]) / sampled_first[0]
        second_difference = (sampled_second[1] - sampled_second[2]) / sampled_second[0]
        values.append(float(second_difference - first_difference))
    if len(values) != replicates:
        return {
            "status": "incomplete_resampling",
            "valid_replicates": len(values),
            "requested_replicates": replicates,
            "point_change": point,
        }
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "status": "evaluated",
        "valid_replicates": replicates,
        "requested_replicates": replicates,
        "point_change": point,
        "lower_95": float(lower),
        "upper_95": float(upper),
        "positive_change_rate": float(np.mean(np.asarray(values) > 0)),
    }
