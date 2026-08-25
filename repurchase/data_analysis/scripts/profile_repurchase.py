"""공통 거래 스키마에서 데이터 품질과 재구매 행동 분포를 분석합니다.

주문 상품 행을 사용자·주문 단위 구매 사건으로 합친 뒤 동일 상품 및
동일 카테고리의 구매 간격, 반복 구매율, 중도절단 비율을 계산합니다.
결과는 백엔드 목데이터 생성 기준에 사용할 JSON·Markdown·CSV로 저장합니다.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from .loaders import load_complete_journey, load_uci_online_retail_ii
from .paths import REPORT_DIR

PET_CATEGORIES = ("CAT FOOD", "DOG FOODS", "PET CARE SUPPLIES")


def distribution(series: pd.Series) -> dict[str, float | int | None]:
    """수치형 값의 개수·평균·표준편차와 주요 분위수를 계산합니다."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0}
    quantiles = values.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()) if values.size > 1 else None,
        "min": float(values.min()),
        "p01": float(quantiles.loc[0.01]),
        "p05": float(quantiles.loc[0.05]),
        "p25": float(quantiles.loc[0.25]),
        "p50": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
        "max": float(values.max()),
    }


def ratio(mask: pd.Series) -> float:
    """불리언 조건을 만족하는 행의 비율을 계산합니다."""
    return float(mask.mean()) if len(mask) else 0.0


def build_order_events(valid: pd.DataFrame) -> pd.DataFrame:
    """주문 상품 행을 사용자·주문별 하나의 구매 사건으로 집계합니다."""
    return (
        valid.groupby(["user_id", "order_id"], observed=True, as_index=False)
        .agg(
            ordered_at=("ordered_at", "min"),
            line_count=("source_row_id", "size"),
            distinct_products=("product_id", "nunique"),
            quantity=("quantity", "sum"),
            order_amount=("line_amount", "sum"),
        )
        .sort_values(["user_id", "ordered_at", "order_id"])
    )


def build_scope_events(valid: pd.DataFrame, scope_column: str) -> pd.DataFrame:
    """동일 주문의 상품·카테고리 행을 구매 사건으로 합치고 이전 간격을 계산합니다."""
    events = (
        valid.dropna(subset=[scope_column])
        .groupby(["user_id", scope_column, "order_id"], observed=True, as_index=False)
        .agg(
            ordered_at=("ordered_at", "min"),
            quantity=("quantity", "sum"),
            line_amount=("line_amount", "sum"),
        )
        .sort_values(["user_id", scope_column, "ordered_at", "order_id"])
    )
    pair_columns = ["user_id", scope_column]
    events["interval_days"] = (
        events.groupby(pair_columns, observed=True)["ordered_at"]
        .diff()
        .dt.total_seconds()
        .div(86_400)
    )
    return events


def profile_repurchase_scope(
    valid: pd.DataFrame, scope_column: str
) -> dict[str, Any] | None:
    """상품 또는 카테고리 범위의 반복 구매율과 구매 간격을 분석합니다."""
    scoped = valid.dropna(subset=[scope_column]).copy()
    if scoped.empty:
        return None

    events = build_scope_events(scoped, scope_column)
    pair_columns = ["user_id", scope_column]

    pair_event_counts = events.groupby(pair_columns, observed=True).size()
    repeated_pairs = pair_event_counts.ge(2)
    repeat_users = pair_event_counts[repeated_pairs].reset_index()["user_id"].nunique()
    intervals = events["interval_days"].dropna()
    positive_intervals = intervals[intervals.gt(0)]

    return {
        "scope": scope_column,
        "event_count": int(len(events)),
        "pair_count": int(len(pair_event_counts)),
        "repeated_pair_count": int(repeated_pairs.sum()),
        "repeated_pair_rate": float(repeated_pairs.mean()),
        "users_with_repeat_count": int(repeat_users),
        "users_with_repeat_rate": float(repeat_users / events["user_id"].nunique()),
        "observed_interval_count": int(len(intervals)),
        "zero_day_interval_rate": ratio(intervals.eq(0)),
        "positive_interval_days": distribution(positive_intervals),
        "event_level_right_censoring_rate": float(len(pair_event_counts) / len(events)),
        "events_per_pair": distribution(pair_event_counts),
    }


def profile_category_breakdown(valid: pd.DataFrame) -> list[dict[str, Any]]:
    """카테고리별 사용자·주문·재구매 간격 지표를 비교할 표로 만듭니다."""
    rows: list[dict[str, Any]] = []
    for category_id, category_frame in valid.dropna(subset=["category_id"]).groupby(
        "category_id", observed=True
    ):
        orders = build_order_events(category_frame)
        user_order_counts = orders.groupby("user_id", observed=True)[
            "order_id"
        ].nunique()
        product_scope = profile_repurchase_scope(category_frame, "product_id")
        if product_scope is None:
            continue
        interval = product_scope["positive_interval_days"]
        rows.append(
            {
                "category_id": str(category_id),
                "line_count": int(len(category_frame)),
                "user_count": int(orders["user_id"].nunique()),
                "order_count": int(len(orders)),
                "repeat_category_user_rate": float(user_order_counts.ge(2).mean()),
                "repeat_product_user_rate": product_scope["users_with_repeat_rate"],
                "quantity_per_order_p50": orders["quantity"].quantile(0.5),
                "quantity_per_order_p95": orders["quantity"].quantile(0.95),
                "product_interval_p25": interval.get("p25"),
                "product_interval_p50": interval.get("p50"),
                "product_interval_p75": interval.get("p75"),
                "product_interval_p95": interval.get("p95"),
            }
        )
    return rows


def profile_dataset(
    frame: pd.DataFrame, include_category_breakdown: bool = False
) -> tuple[dict[str, Any], pd.DataFrame]:
    """한 데이터셋의 품질·주문·재구매 지표와 월별 주문량을 계산합니다."""
    dataset = str(frame["dataset"].iloc[0])
    valid = frame.loc[frame["is_valid_repurchase_candidate"]].copy()
    monetary_valid = frame.loc[frame["is_valid_monetary_candidate"]].copy()
    orders = build_order_events(valid)
    monetary_orders = build_order_events(monetary_valid)
    user_order_counts = orders.groupby("user_id", observed=True)["order_id"].nunique()

    monthly_orders = (
        orders.assign(month=orders["ordered_at"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(order_count=("order_id", "nunique"), user_count=("user_id", "nunique"))
    )

    profile = {
        "dataset": dataset,
        "observation": {
            "start": frame["ordered_at"].min().isoformat(),
            "end": frame["ordered_at"].max().isoformat(),
            "days": int((frame["ordered_at"].max() - frame["ordered_at"].min()).days),
        },
        "raw": {
            "line_count": int(len(frame)),
            "user_count_including_missing": int(frame["user_id"].nunique(dropna=False)),
            "order_count": int(frame["order_id"].nunique()),
            "product_count": int(frame["product_id"].nunique()),
            "category_count": int(frame["category_id"].nunique()),
        },
        "quality": {
            "valid_repurchase_candidate_count": int(len(valid)),
            "valid_repurchase_candidate_rate": ratio(
                frame["is_valid_repurchase_candidate"]
            ),
            "valid_monetary_candidate_count": int(len(monetary_valid)),
            "valid_monetary_candidate_rate": ratio(
                frame["is_valid_monetary_candidate"]
            ),
            "missing_user_rate": ratio(frame["is_user_missing"]),
            "explicit_cancellation_rate": ratio(frame["is_explicit_cancellation"]),
            "nonpositive_quantity_rate": ratio(frame["is_nonpositive_quantity"]),
            "nonpositive_amount_rate": ratio(frame["is_nonpositive_amount"]),
            "missing_category_rate": ratio(frame["category_id"].isna()),
        },
        "orders": {
            "valid_order_count": int(len(orders)),
            "valid_user_count": int(orders["user_id"].nunique()),
            "repeat_order_user_count": int(user_order_counts.ge(2).sum()),
            "repeat_order_user_rate": float(user_order_counts.ge(2).mean()),
            "orders_per_user": distribution(user_order_counts),
            "line_count_per_order": distribution(orders["line_count"]),
            "distinct_products_per_order": distribution(orders["distinct_products"]),
            "quantity_per_order": distribution(orders["quantity"]),
            "amount_per_order": distribution(monetary_orders["order_amount"]),
        },
        "product_repurchase": profile_repurchase_scope(valid, "product_id"),
        "category_repurchase": profile_repurchase_scope(valid, "category_id"),
    }
    if include_category_breakdown:
        profile["category_breakdown"] = profile_category_breakdown(valid)
    return profile, monthly_orders


def fmt_number(value: Any) -> str:
    """보고서 표에 사용할 값을 천 단위 구분 형식으로 변환합니다."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.4f}"
    return f"{value:,}"


def render_markdown(profile: dict[str, Any]) -> str:
    """분석 결과 딕셔너리를 사람이 읽기 쉬운 Markdown 보고서로 변환합니다."""
    quality = profile["quality"]
    orders = profile["orders"]
    lines = [
        f"# {profile['dataset']} 재구매 데이터 프로파일",
        "",
        "## 관측 범위",
        "",
        f"- 기간: `{profile['observation']['start']}` ~ `{profile['observation']['end']}`",
        f"- 관측 일수: {profile['observation']['days']:,}일",
        f"- 원본 행: {profile['raw']['line_count']:,}건",
        "",
        "## 데이터 품질",
        "",
        "| 항목 | 비율 |",
        "| --- | ---: |",
        f"| 유효 재구매 사건 후보 | {quality['valid_repurchase_candidate_rate']:.2%} |",
        f"| 유효 금액 분석 후보 | {quality['valid_monetary_candidate_rate']:.2%} |",
        f"| 사용자 ID 누락 | {quality['missing_user_rate']:.2%} |",
        f"| 명시적 취소 | {quality['explicit_cancellation_rate']:.2%} |",
        f"| 0 이하 수량 | {quality['nonpositive_quantity_rate']:.2%} |",
        f"| 0 이하 결제금액 | {quality['nonpositive_amount_rate']:.2%} |",
        f"| 카테고리 누락 | {quality['missing_category_rate']:.2%} |",
        "",
        "## 주문 행동",
        "",
        f"- 유효 사용자: {orders['valid_user_count']:,}명",
        f"- 유효 주문: {orders['valid_order_count']:,}건",
        f"- 2회 이상 주문 사용자: {orders['repeat_order_user_rate']:.2%}",
        f"- 사용자당 주문 수 중앙값: {orders['orders_per_user']['p50']:.1f}건",
        f"- 사용자당 주문 수 95분위: {orders['orders_per_user']['p95']:.1f}건",
        "",
    ]

    for key, title in (
        ("product_repurchase", "동일 상품 재구매"),
        ("category_repurchase", "동일 카테고리 재구매"),
    ):
        scope = profile[key]
        if scope is None:
            continue
        intervals = scope["positive_interval_days"]
        lines.extend(
            [
                f"## {title}",
                "",
                f"- 반복 구매 쌍 비율: {scope['repeated_pair_rate']:.2%}",
                f"- 반복 구매 경험 사용자 비율: {scope['users_with_repeat_rate']:.2%}",
                f"- 0일 간격 비율: {scope['zero_day_interval_rate']:.2%}",
                f"- 마지막 사건 중도절단 비율: {scope['event_level_right_censoring_rate']:.2%}",
                "",
                "| 양수 구매 간격 | 일수 |",
                "| --- | ---: |",
                f"| 5분위 | {fmt_number(intervals.get('p05'))} |",
                f"| 25분위 | {fmt_number(intervals.get('p25'))} |",
                f"| 중앙값 | {fmt_number(intervals.get('p50'))} |",
                f"| 75분위 | {fmt_number(intervals.get('p75'))} |",
                f"| 95분위 | {fmt_number(intervals.get('p95'))} |",
                "",
            ]
        )
    if profile.get("category_breakdown"):
        lines.extend(
            [
                "## 카테고리별 비교",
                "",
                "| 카테고리 | 사용자 | 주문 | 카테고리 반복 사용자 | 동일 상품 반복 사용자 | 상품 간격 중앙값 | 상품 간격 95분위 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in profile["category_breakdown"]:
            lines.append(
                "| {category_id} | {user_count:,} | {order_count:,} | "
                "{repeat_category_user_rate:.2%} | {repeat_product_user_rate:.2%} | "
                "{product_interval_p50:.2f}일 | {product_interval_p95:.2f}일 |".format(
                    **row
                )
            )
        lines.append("")
    return "\n".join(lines)


def run(dataset: str) -> None:
    """지정한 데이터셋을 분석하고 JSON·Markdown·CSV 보고서를 저장합니다."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
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

    profile, monthly_orders = profile_dataset(
        frame, include_category_breakdown=dataset == "complete_journey_pet"
    )
    (REPORT_DIR / f"{dataset}_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORT_DIR / f"{dataset}_profile.md").write_text(
        render_markdown(profile), encoding="utf-8"
    )
    monthly_orders.to_csv(REPORT_DIR / f"{dataset}_monthly_orders.csv", index=False)
    print(f"[완료] {dataset}: 원본 {len(frame):,}행")


def parse_args() -> argparse.Namespace:
    """명령줄에서 분석할 데이터셋 선택값을 읽습니다."""
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
    """선택된 데이터셋을 순서대로 분석하는 명령줄 진입점입니다."""
    args = parse_args()
    datasets = (
        ("uci_online_retail_ii", "complete_journey", "complete_journey_pet")
        if args.dataset == "all"
        else (args.dataset,)
    )
    for dataset in datasets:
        run(dataset)


if __name__ == "__main__":
    main()
