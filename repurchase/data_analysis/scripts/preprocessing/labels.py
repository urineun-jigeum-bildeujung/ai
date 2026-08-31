"""구매 사건에서 동일 상품 재구매와 우측검열 학습 라벨을 생성합니다.

UCI에는 신뢰할 수 있는 상품군·카테고리 정보가 없으므로, 이 모듈은 같은
사용자의 동일 ``product_id`` 구매만 재구매로 정의합니다. 각 사용자·상품
시퀀스의 마지막 구매는 부정 라벨로 단정하지 않고 관측 종료 시점까지의
우측검열 표본으로 보존합니다.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from .events import EVENT_KEY_COLUMNS


class RepurchaseLabelBuildError(ValueError):
    """구매 사건이나 라벨이 정의한 시간·키 계약을 위반할 때 발생합니다."""


LABEL_REQUIRED_EVENT_COLUMNS: Final[tuple[str, ...]] = (
    *EVENT_KEY_COLUMNS,
    "ordered_at",
)

LABEL_REQUIRED_RESULT_COLUMNS: Final[tuple[str, ...]] = (
    *EVENT_KEY_COLUMNS,
    "anchor_at",
    "next_order_id",
    "next_same_product_at",
    "observation_end_at",
    "duration_days",
    "event_observed",
    "is_right_censored",
    "has_zero_day_followup",
)

# 시간순 정렬 결과를 재현하기 위해 동률일 때 주문 ID를 최종 정렬 키로 사용합니다.
SEQUENCE_SORT_COLUMNS: Final[tuple[str, ...]] = (
    "user_id",
    "product_id",
    "ordered_at",
    "order_id",
)


def _validate_label_input(
    events: pd.DataFrame,
    observation_end_at: pd.Timestamp,
) -> pd.DataFrame:
    """라벨 생성 전에 필수 컬럼·사건 키·관측 종료 시각을 검증합니다."""
    missing_columns = set(LABEL_REQUIRED_EVENT_COLUMNS) - set(events.columns)
    if missing_columns:
        raise RepurchaseLabelBuildError(
            f"재구매 라벨 필수 컬럼이 누락됐습니다: {sorted(missing_columns)}"
        )
    if events.empty:
        raise RepurchaseLabelBuildError("재구매 라벨을 생성할 구매 사건이 없습니다.")

    rows = events.loc[:, list(LABEL_REQUIRED_EVENT_COLUMNS)].copy()
    if rows[list(EVENT_KEY_COLUMNS)].isna().any(axis=None):
        raise RepurchaseLabelBuildError("구매 사건 키에 결측값이 있습니다.")
    if rows["ordered_at"].isna().any():
        raise RepurchaseLabelBuildError("구매 사건 시각에 결측값이 있습니다.")
    if rows.duplicated(subset=list(EVENT_KEY_COLUMNS)).any():
        raise RepurchaseLabelBuildError("중복된 사용자·주문·상품 사건 키가 있습니다.")

    rows["ordered_at"] = pd.to_datetime(rows["ordered_at"], errors="raise")
    try:
        latest_event_at = rows["ordered_at"].max()
        if observation_end_at < latest_event_at:
            raise RepurchaseLabelBuildError(
                "관측 종료 시각이 실제 마지막 구매 사건보다 빠릅니다: "
                f"observation_end_at={observation_end_at}, "
                f"latest_event_at={latest_event_at}"
            )
    except TypeError as error:
        raise RepurchaseLabelBuildError(
            "구매 사건 시각과 관측 종료 시각의 시간대 설정이 다릅니다."
        ) from error
    return rows


def build_same_product_repurchase_labels(
    events: pd.DataFrame,
    observation_end_at: pd.Timestamp,
) -> pd.DataFrame:
    """각 구매 사건의 다음 동일 상품 구매 또는 우측검열 기간을 계산합니다."""
    normalized_observation_end = pd.Timestamp(observation_end_at)
    if pd.isna(normalized_observation_end):
        raise RepurchaseLabelBuildError("관측 종료 시각이 비어 있습니다.")

    rows = _validate_label_input(events, normalized_observation_end)
    rows = rows.sort_values(
        list(SEQUENCE_SORT_COLUMNS),
        kind="stable",
        ignore_index=True,
    )

    # 같은 사용자·상품 안에서 한 행 뒤의 주문과 시각을 현재 사건으로 당깁니다.
    sequence = rows.groupby(["user_id", "product_id"], observed=True, sort=False)
    rows["next_order_id"] = sequence["order_id"].shift(-1)
    rows["next_same_product_at"] = sequence["ordered_at"].shift(-1)
    rows["anchor_at"] = rows["ordered_at"]
    rows["observation_end_at"] = normalized_observation_end

    # 다음 구매가 있으면 그 시각, 없으면 데이터셋 관측 종료 시각까지 잽니다.
    rows["event_observed"] = rows["next_same_product_at"].notna()
    rows["is_right_censored"] = ~rows["event_observed"]
    label_end_at = rows["next_same_product_at"].fillna(normalized_observation_end)
    rows["duration_days"] = (
        label_end_at - rows["anchor_at"]
    ).dt.total_seconds() / 86_400
    rows["has_zero_day_followup"] = rows["event_observed"] & rows["duration_days"].eq(0)

    if rows["duration_days"].lt(0).any():
        raise RepurchaseLabelBuildError("음수 재구매·검열 기간이 생성됐습니다.")

    return rows.loc[:, list(LABEL_REQUIRED_RESULT_COLUMNS)].sort_values(
        ["user_id", "anchor_at", "order_id", "product_id"],
        kind="stable",
        ignore_index=True,
    )


def validate_same_product_repurchase_labels(
    events: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict[str, object]:
    """사건 보존·시퀀스 종결·기간 관계의 불변조건과 라벨 분포를 검사합니다."""
    missing_columns = set(LABEL_REQUIRED_RESULT_COLUMNS) - set(labels.columns)
    if missing_columns:
        raise RepurchaseLabelBuildError(
            f"재구매 라벨 결과 컬럼이 누락됐습니다: {sorted(missing_columns)}"
        )

    source_keys = pd.MultiIndex.from_frame(
        events.loc[:, list(EVENT_KEY_COLUMNS)].drop_duplicates()
    )
    label_keys = pd.MultiIndex.from_frame(labels.loc[:, list(EVENT_KEY_COLUMNS)])
    pair_count = int(
        events.loc[:, ["user_id", "product_id"]].drop_duplicates().shape[0]
    )
    censored_per_pair = labels.groupby(["user_id", "product_id"], observed=True)[
        "is_right_censored"
    ].sum()

    invariants = {
        "one_label_per_event_key": not labels.duplicated(
            subset=list(EVENT_KEY_COLUMNS)
        ).any(),
        "event_key_sets_preserved": set(label_keys) == set(source_keys),
        "one_right_censored_event_per_pair": bool(censored_per_pair.eq(1).all()),
        "observed_and_censored_are_complements": bool(
            labels["event_observed"].eq(~labels["is_right_censored"]).all()
        ),
        "observed_events_have_next_purchase": bool(
            labels.loc[labels["event_observed"], "next_same_product_at"].notna().all()
        ),
        "censored_events_have_no_next_purchase": bool(
            labels.loc[labels["is_right_censored"], "next_same_product_at"].isna().all()
        ),
        "nonnegative_duration": bool(labels["duration_days"].ge(0).all()),
        "next_purchase_not_after_observation_end": bool(
            labels.loc[labels["event_observed"], "next_same_product_at"]
            .le(labels.loc[labels["event_observed"], "observation_end_at"])
            .all()
        ),
    }
    failed_invariants = [name for name, passed in invariants.items() if not passed]
    if failed_invariants:
        raise RepurchaseLabelBuildError(
            f"재구매 라벨 불변조건을 위반했습니다: {failed_invariants}"
        )

    observed = labels["event_observed"]
    censored = labels["is_right_censored"]
    observed_durations = labels.loc[observed, "duration_days"]
    # 재구매 관측값이 하나도 없는 작은 표본에서도 JSON에 NaN을 기록하지 않습니다.
    observed_duration_summary = (
        {"median": None, "p90": None, "p95": None, "p99": None, "maximum": None}
        if observed_durations.empty
        else {
            "median": float(observed_durations.median()),
            "p90": float(observed_durations.quantile(0.90)),
            "p95": float(observed_durations.quantile(0.95)),
            "p99": float(observed_durations.quantile(0.99)),
            "maximum": float(observed_durations.max()),
        }
    )
    return {
        "source_event_count": int(len(events)),
        "label_count": int(len(labels)),
        "user_product_pair_count": pair_count,
        "observed_repurchase_count": int(observed.sum()),
        "observed_repurchase_rate": float(observed.mean()),
        "right_censored_count": int(censored.sum()),
        "right_censored_rate": float(censored.mean()),
        "zero_day_followup_count": int(labels["has_zero_day_followup"].sum()),
        "zero_day_followup_rate": float(labels["has_zero_day_followup"].mean()),
        "observed_duration_days": observed_duration_summary,
        "invariants": invariants,
    }
