"""목데이터 생성 가이드에 필요한 행동 분포와 조건부 관계를 분석합니다.

기존 재구매 요약 보고서와 분리해 사용자 활동성, 주문 구성, 시간 패턴,
상품 반복·전환처럼 합성 데이터 생성 순서에 직접 필요한 통계를 계산합니다.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .loaders import load_complete_journey, load_uci_online_retail_ii
from .paths import REPORT_DIR
from .profile_repurchase import (
    PET_CATEGORIES,
    build_order_events,
    build_scope_events,
    distribution,
    ratio,
)


class ProfileConfigurationError(ValueError):
    """분석 구간 설정이 겹치거나 일부 값을 포함하지 않을 때 발생합니다."""


@dataclass(frozen=True)
class ActivitySegment:
    """사용자 주문 횟수를 해석할 활동 세그먼트의 명시적 경계입니다."""

    name: str
    minimum_orders: int
    maximum_orders: int | None


USER_ACTIVITY_SEGMENTS = (
    ActivitySegment("single", 1, 1),
    ActivitySegment("low_frequency", 2, 3),
    ActivitySegment("repeat", 4, 10),
    ActivitySegment("high_frequency", 11, None),
)


def frequency_distribution(series: pd.Series) -> dict[str, Any]:
    """범주·정수 값별 건수와 비율을 원래 값 손실 없이 계산합니다."""
    observed = series.dropna()
    counts = observed.value_counts(sort=False).sort_index()
    total = int(observed.size)
    values = []
    for value, count in counts.items():
        python_value = value.item() if hasattr(value, "item") else value
        values.append(
            {
                "value": python_value,
                "count": int(count),
                "rate": float(count / total) if total else 0.0,
            }
        )
    return {
        "observed_count": total,
        "missing_count": int(series.isna().sum()),
        "values": values,
    }


def assign_activity_segments(
    order_counts: pd.Series,
    segments: tuple[ActivitySegment, ...] = USER_ACTIVITY_SEGMENTS,
) -> pd.Series:
    """사용자별 주문 횟수를 겹치지 않는 활동 세그먼트로 분류합니다."""
    numeric_counts = pd.to_numeric(order_counts, errors="coerce")
    assigned = pd.Series(pd.NA, index=order_counts.index, dtype="string")

    for segment in segments:
        mask = numeric_counts.ge(segment.minimum_orders)
        if segment.maximum_orders is not None:
            mask &= numeric_counts.le(segment.maximum_orders)
        if assigned.loc[mask].notna().any():
            raise ProfileConfigurationError(
                f"활동 세그먼트 '{segment.name}'의 주문 수 구간이 다른 구간과 겹칩니다."
            )
        assigned.loc[mask] = segment.name

    unassigned = numeric_counts.notna() & assigned.isna()
    if unassigned.any():
        missing_counts = sorted(numeric_counts.loc[unassigned].unique().tolist())
        raise ProfileConfigurationError(
            f"활동 세그먼트에 포함되지 않은 주문 수가 있습니다: {missing_counts}"
        )
    return assigned


def profile_user_activity(order_events: pd.DataFrame) -> dict[str, Any]:
    """사용자별 주문 수와 활동 세그먼트의 건수·비율을 계산합니다."""
    order_counts = order_events.groupby("user_id", observed=True)["order_id"].nunique()
    segment_names = assign_activity_segments(order_counts)
    segment_counts = segment_names.value_counts(sort=False)
    user_count = int(order_counts.size)

    segment_profile = []
    for segment in USER_ACTIVITY_SEGMENTS:
        segment_order_counts = order_counts.loc[segment_names.eq(segment.name)]
        count = int(segment_counts.get(segment.name, 0))
        segment_profile.append(
            {
                "name": segment.name,
                "minimum_orders": segment.minimum_orders,
                "maximum_orders": segment.maximum_orders,
                "user_count": count,
                "user_rate": float(count / user_count) if user_count else 0.0,
                "order_count_distribution": distribution(segment_order_counts),
                "order_count_frequency": frequency_distribution(segment_order_counts),
            }
        )

    return {
        "user_count": user_count,
        "orders_per_user": distribution(order_counts),
        "segments": segment_profile,
    }


def profile_order_composition(valid: pd.DataFrame) -> dict[str, Any]:
    """주문당 상품 구성과 주문 내부 시각 정합성을 생성 기준으로 집계합니다."""
    orders = (
        valid.groupby(["user_id", "order_id"], observed=True, as_index=False)
        .agg(
            first_ordered_at=("ordered_at", "min"),
            last_ordered_at=("ordered_at", "max"),
            line_count=("source_row_id", "size"),
            distinct_products=("product_id", "nunique"),
        )
        .sort_values(["user_id", "first_ordered_at", "order_id"])
    )
    timestamp_span_seconds = (
        orders["last_ordered_at"] - orders["first_ordered_at"]
    ).dt.total_seconds()

    return {
        "order_count": int(len(orders)),
        "line_count_per_order": distribution(orders["line_count"]),
        "line_count_frequency": frequency_distribution(orders["line_count"]),
        "distinct_products_per_order": distribution(orders["distinct_products"]),
        "distinct_product_frequency": frequency_distribution(
            orders["distinct_products"]
        ),
        "duplicate_product_line_order_rate": ratio(
            orders["line_count"].gt(orders["distinct_products"])
        ),
        "inconsistent_timestamp_order_rate": ratio(timestamp_span_seconds.gt(0)),
        "timestamp_span_seconds": distribution(timestamp_span_seconds),
    }


def profile_cycle_heterogeneity(
    valid: pd.DataFrame,
    scope_column: str,
) -> dict[str, Any]:
    """사용자별 대표 소비주기와 주기 변동성을 범위별로 계산합니다."""
    events = build_scope_events(valid, scope_column)
    positive_intervals = events.loc[events["interval_days"].gt(0)].copy()
    pair_columns = ["user_id", scope_column]
    pair_cycles = (
        positive_intervals.groupby(pair_columns, observed=True)["interval_days"]
        .agg(
            interval_count="size",
            mean_interval="mean",
            median_interval="median",
            interval_std="std",
        )
        .reset_index()
    )
    pair_cycles["interval_cv"] = pair_cycles["interval_std"].div(
        pair_cycles["mean_interval"]
    )

    return {
        "scope": scope_column,
        "observed_interval_count": int(len(positive_intervals)),
        "pair_count_with_observed_interval": int(len(pair_cycles)),
        "event_weighted_interval_days": distribution(
            positive_intervals["interval_days"]
        ),
        "pair_weighted_median_interval_days": distribution(
            pair_cycles["median_interval"]
        ),
        "observed_intervals_per_pair": distribution(pair_cycles["interval_count"]),
        "pair_interval_cv": distribution(pair_cycles["interval_cv"]),
    }


def profile_temporal_behavior(order_events: pd.DataFrame) -> dict[str, Any]:
    """사용자 주문 간격과 월·요일·시간대별 주문 비율을 계산합니다."""
    events = order_events.sort_values(["user_id", "ordered_at", "order_id"]).copy()
    events["user_order_interval_days"] = (
        events.groupby("user_id", observed=True)["ordered_at"]
        .diff()
        .dt.total_seconds()
        .div(86_400)
    )
    positive_intervals = events.loc[events["user_order_interval_days"].gt(0)].copy()
    user_cycles = (
        positive_intervals.groupby("user_id", observed=True)["user_order_interval_days"]
        .agg(
            interval_count="size",
            mean_interval="mean",
            median_interval="median",
            interval_std="std",
        )
        .reset_index()
    )
    user_cycles["interval_cv"] = user_cycles["interval_std"].div(
        user_cycles["mean_interval"]
    )

    return {
        "event_weighted_order_interval_days": distribution(
            positive_intervals["user_order_interval_days"]
        ),
        "user_weighted_median_order_interval_days": distribution(
            user_cycles["median_interval"]
        ),
        "user_order_interval_cv": distribution(user_cycles["interval_cv"]),
        "month_of_year_frequency": frequency_distribution(
            events["ordered_at"].dt.month
        ),
        "weekday_frequency": frequency_distribution(events["ordered_at"].dt.dayofweek),
        "hour_frequency": frequency_distribution(events["ordered_at"].dt.hour),
    }


def spearman_rank_correlation(left: pd.Series, right: pd.Series) -> float | None:
    """두 수치의 단조 관계를 순위로 변환해 계산하고 비교 불가하면 None을 반환합니다."""
    pair = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(pair) < 2 or pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return None
    left_rank = pair["left"].rank(method="average")
    right_rank = pair["right"].rank(method="average")
    return float(left_rank.corr(right_rank))


def profile_quantity_interval_relationship(
    valid: pd.DataFrame,
    scope_column: str,
) -> dict[str, Any]:
    """이전 구매 수량과 그다음 구매까지의 간격 관계를 범위별로 계산합니다."""
    events = build_scope_events(valid, scope_column)
    pair_columns = ["user_id", scope_column]
    events["previous_quantity"] = events.groupby(pair_columns, observed=True)[
        "quantity"
    ].shift(1)
    eligible = events.loc[
        events["interval_days"].gt(0) & events["previous_quantity"].gt(0)
    ].copy()

    return {
        "scope": scope_column,
        "eligible_transition_count": int(len(eligible)),
        "previous_quantity": distribution(eligible["previous_quantity"]),
        "next_interval_days": distribution(eligible["interval_days"]),
        "spearman_quantity_to_next_interval": spearman_rank_correlation(
            eligible["previous_quantity"], eligible["interval_days"]
        ),
    }


def profile_product_switching(valid: pd.DataFrame) -> dict[str, Any]:
    """동일 카테고리의 연속 단일상품 주문에서 전환·정착·복귀 행동을 계산합니다."""
    category_orders = (
        valid.dropna(subset=["category_id", "product_id"])
        .groupby(
            ["user_id", "category_id", "order_id"],
            observed=True,
            as_index=False,
        )
        .agg(
            ordered_at=("ordered_at", "min"),
            product_id=("product_id", "first"),
            distinct_products=("product_id", "nunique"),
        )
        .sort_values(["user_id", "category_id", "ordered_at", "order_id"])
    )
    pair_columns = ["user_id", "category_id"]
    grouped = category_orders.groupby(pair_columns, observed=True)
    category_orders["category_order_number"] = grouped.cumcount() + 1
    category_orders["previous_product_id"] = grouped["product_id"].shift(1)
    category_orders["previous_distinct_products"] = grouped["distinct_products"].shift(
        1
    )
    category_orders["previous_ordered_at"] = grouped["ordered_at"].shift(1)
    category_orders["next_product_id"] = grouped["product_id"].shift(-1)
    category_orders["next_distinct_products"] = grouped["distinct_products"].shift(-1)

    eligible = category_orders.loc[
        category_orders["distinct_products"].eq(1)
        & category_orders["previous_distinct_products"].eq(1)
    ].copy()
    eligible["interval_days"] = (
        (eligible["ordered_at"] - eligible["previous_ordered_at"])
        .dt.total_seconds()
        .div(86_400)
    )
    eligible["is_switch"] = eligible["product_id"].ne(eligible["previous_product_id"])
    switches = eligible.loc[eligible["is_switch"]].copy()
    repeat_intervals = eligible.loc[~eligible["is_switch"], "interval_days"]
    early_switch_threshold = (
        float(repeat_intervals.quantile(0.25)) if not repeat_intervals.empty else None
    )
    if early_switch_threshold is None:
        switches["is_early_switch"] = False
    else:
        switches["is_early_switch"] = switches["interval_days"].lt(
            early_switch_threshold
        )
    first_followup = eligible.loc[eligible["category_order_number"].eq(2)]
    evaluable_followups = switches.loc[switches["next_distinct_products"].eq(1)].copy()
    evaluable_followups["is_retained_next"] = evaluable_followups["next_product_id"].eq(
        evaluable_followups["product_id"]
    )
    evaluable_followups["is_switched_back_next"] = evaluable_followups[
        "next_product_id"
    ].eq(evaluable_followups["previous_product_id"])

    return {
        "category_order_count": int(len(category_orders)),
        "multi_product_category_order_rate": ratio(
            category_orders["distinct_products"].gt(1)
        ),
        "eligible_transition_count": int(len(eligible)),
        "switch_count": int(len(switches)),
        "switch_rate": ratio(eligible["is_switch"]),
        "first_followup_transition_count": int(len(first_followup)),
        "first_followup_switch_rate": ratio(first_followup["is_switch"]),
        "early_switch_threshold_days": early_switch_threshold,
        "early_switch_count": int(switches["is_early_switch"].sum()),
        "early_switch_rate": ratio(switches["is_early_switch"]),
        "repeat_interval_days": distribution(repeat_intervals),
        "switch_interval_days": distribution(switches["interval_days"]),
        "evaluable_switch_followup_count": int(len(evaluable_followups)),
        "next_order_same_product_rate": ratio(evaluable_followups["is_retained_next"]),
        "next_order_switch_back_rate": ratio(
            evaluable_followups["is_switched_back_next"]
        ),
    }


def build_mock_generation_profile(
    frame: pd.DataFrame,
    include_category_breakdown: bool = False,
) -> dict[str, Any]:
    """한 데이터셋에서 목데이터 생성·검증에 필요한 통계를 한 구조로 묶습니다."""
    valid = frame.loc[frame["is_valid_repurchase_candidate"]].copy()
    order_events = build_order_events(valid)
    profile: dict[str, Any] = {
        "dataset": str(frame["dataset"].iloc[0]),
        "source_line_count": int(len(frame)),
        "valid_line_count": int(len(valid)),
        "user_activity": profile_user_activity(order_events),
        "order_composition": profile_order_composition(valid),
        "temporal_behavior": profile_temporal_behavior(order_events),
        "product_cycle": profile_cycle_heterogeneity(valid, "product_id"),
        "category_cycle": profile_cycle_heterogeneity(valid, "category_id"),
        "quantity_to_product_interval": profile_quantity_interval_relationship(
            valid, "product_id"
        ),
        "product_switching": profile_product_switching(valid),
    }
    if include_category_breakdown:
        profile["category_breakdown"] = [
            {
                "category_id": str(category_id),
                "line_count": int(len(category_frame)),
                "product_cycle": profile_cycle_heterogeneity(
                    category_frame, "product_id"
                ),
                "quantity_to_product_interval": (
                    profile_quantity_interval_relationship(category_frame, "product_id")
                ),
                "product_switching": profile_product_switching(category_frame),
            }
            for category_id, category_frame in valid.dropna(
                subset=["category_id"]
            ).groupby("category_id", observed=True)
        ]
    return profile


def write_profile(dataset: str, frame: pd.DataFrame) -> None:
    """이미 적재된 데이터프레임을 분석해 생성 기준 프로파일을 저장합니다."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    profile = build_mock_generation_profile(
        frame,
        include_category_breakdown=dataset == "complete_journey_pet",
    )
    output_path = REPORT_DIR / f"{dataset}_mock_generation_profile.json"
    output_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[완료] {dataset}: {output_path.name}")


def run(dataset: str) -> None:
    """단일 데이터셋을 적재해 목데이터 생성 기준 프로파일을 저장합니다."""
    if dataset == "uci_online_retail_ii":
        frame = load_uci_online_retail_ii()
    elif dataset == "complete_journey":
        frame = load_complete_journey()
    elif dataset == "complete_journey_pet":
        frame = load_complete_journey()
        frame = frame.loc[frame["category_id"].isin(PET_CATEGORIES)].copy()
        frame["dataset"] = "complete_journey_pet"
    else:
        raise ValueError(f"지원하지 않는 데이터셋입니다: {dataset}")
    write_profile(dataset, frame)


def parse_args() -> argparse.Namespace:
    """명령줄에서 생성 가이드 프로파일 대상 데이터셋을 읽습니다."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=(
            "all",
            "uci_online_retail_ii",
            "complete_journey",
            "complete_journey_pet",
        ),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    """선택한 공개 데이터셋의 생성 가이드 프로파일을 순서대로 만듭니다."""
    args = parse_args()
    if args.dataset != "all":
        run(args.dataset)
        return

    run("uci_online_retail_ii")
    complete_journey = load_complete_journey()
    write_profile("complete_journey", complete_journey)
    pet = complete_journey.loc[
        complete_journey["category_id"].isin(PET_CATEGORIES)
    ].copy()
    pet["dataset"] = "complete_journey_pet"
    write_profile("complete_journey_pet", pet)


if __name__ == "__main__":
    main()
