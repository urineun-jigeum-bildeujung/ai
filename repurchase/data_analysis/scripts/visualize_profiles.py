"""재구매 데이터 분석 결과를 비교 가능한 정적 그래프로 생성합니다.

프로파일링 단계에서 생성한 JSON·CSV 보고서만 읽어 데이터 품질, 반복 구매,
재구매 간격, 반려동물 카테고리와 월별 주문 추이를 시각화합니다. 원본 데이터가
없는 환경에서도 같은 그래프를 재생성할 수 있도록 집계 결과만 사용합니다.
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from .paths import FIGURE_DIR, REPORT_DIR
from .plotting import configure_korean_font, save_figure

DATASETS = {
    "uci_online_retail_ii": "UCI 전체",
    "complete_journey": "Complete Journey 전체",
    "complete_journey_pet": "반려동물 부분집합",
}

PET_CATEGORY_LABELS = {
    "CAT FOOD": "고양이 사료",
    "DOG FOODS": "강아지 사료",
    "PET CARE SUPPLIES": "반려동물 용품",
}

COLORS = ("#3366CC", "#DC3912", "#109618", "#FF9900", "#990099")


def load_profiles() -> dict[str, dict[str, Any]]:
    """데이터셋별 JSON 프로파일을 읽어 데이터셋 ID 기준으로 반환합니다."""
    profiles: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        path = REPORT_DIR / f"{dataset}_profile.json"
        profiles[dataset] = json.loads(path.read_text(encoding="utf-8"))
    return profiles


def add_bar_labels(axis: plt.Axes, bars: Any, percent: bool = True) -> None:
    """막대 위에 비율 또는 수치 레이블을 표시합니다."""
    labels = [
        f"{bar.get_height():.1%}" if percent else f"{bar.get_height():,.0f}"
        for bar in bars
    ]
    axis.bar_label(bars, labels=labels, padding=3, fontsize=8)


def filter_complete_months(
    frame: pd.DataFrame, observation: dict[str, Any]
) -> pd.DataFrame:
    """관측 시작·종료 때문에 일부 날짜만 포함된 경계 월을 제외합니다."""
    start = pd.Timestamp(observation["start"])
    end = pd.Timestamp(observation["end"])
    result = frame.copy()
    if start.day != 1:
        result = result.loc[result["month"] != start.strftime("%Y-%m")]
    if end.normalize() < end.normalize() + pd.offsets.MonthEnd(0):
        result = result.loc[result["month"] != end.strftime("%Y-%m")]
    return result.reset_index(drop=True)


def plot_data_quality(profiles: dict[str, dict[str, Any]]) -> None:
    """데이터셋별 재구매 사건 유효 비율과 주요 품질 이슈를 비교합니다."""
    labels = list(DATASETS.values())
    keys = list(DATASETS)
    valid_rates = [
        profiles[key]["quality"]["valid_repurchase_candidate_rate"] for key in keys
    ]

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    valid_bars = axes[0].bar(labels, valid_rates, color=COLORS[:3], width=0.65)
    axes[0].set_title("유효 재구매 사건 후보 비율")
    axes[0].set_ylabel("전체 원본 행 대비 비율")
    axes[0].set_ylim(0, 1.12)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].tick_params(axis="x", rotation=18)
    add_bar_labels(axes[0], valid_bars)

    issue_names = ("사용자 ID 누락", "명시적 취소", "0 이하 수량", "0 이하 금액")
    issue_keys = (
        "missing_user_rate",
        "explicit_cancellation_rate",
        "nonpositive_quantity_rate",
        "nonpositive_amount_rate",
    )
    positions = np.arange(len(issue_names))
    width = 0.24
    for index, (dataset, label) in enumerate(DATASETS.items()):
        values = [profiles[dataset]["quality"][key] for key in issue_keys]
        axes[1].bar(
            positions + (index - 1) * width,
            values,
            width,
            label=label,
            color=COLORS[index],
        )
    axes[1].set_title("품질 이슈 비율")
    axes[1].set_ylabel("전체 원본 행 대비 비율")
    axes[1].set_xticks(positions, issue_names, rotation=18)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)

    figure.suptitle("공개 데이터셋 품질 비교", fontsize=15)
    save_figure(figure, "dataset_quality_comparison.png")


def plot_repurchase_behavior(profiles: dict[str, dict[str, Any]]) -> None:
    """사용자 반복 주문, 동일 상품 반복 구매와 중도절단 비율을 비교합니다."""
    metric_names = (
        "2회 이상 주문 사용자",
        "동일 상품 반복 사용자",
        "상품 사건 중도절단",
    )
    positions = np.arange(len(metric_names))
    width = 0.24
    figure, axis = plt.subplots(figsize=(10.5, 5.8))

    for index, (dataset, label) in enumerate(DATASETS.items()):
        profile = profiles[dataset]
        values = (
            profile["orders"]["repeat_order_user_rate"],
            profile["product_repurchase"]["users_with_repeat_rate"],
            profile["product_repurchase"]["event_level_right_censoring_rate"],
        )
        bars = axis.bar(
            positions + (index - 1) * width,
            values,
            width,
            label=label,
            color=COLORS[index],
        )
        add_bar_labels(axis, bars)

    axis.set_title("재구매 행동과 관측 한계 비교", fontsize=15)
    axis.set_ylabel("사용자 또는 구매 사건 비율")
    axis.set_xticks(positions, metric_names)
    axis.set_ylim(0, 1.12)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    save_figure(figure, "repurchase_behavior_comparison.png")


def draw_interval_ranges(
    axis: plt.Axes,
    profiles: dict[str, dict[str, Any]],
    scope_key: str,
    title: str,
) -> None:
    """재구매 간격의 25~75분위 범위, 중앙값과 95분위를 한 축에 표시합니다."""
    for index, dataset in enumerate(DATASETS):
        interval = profiles[dataset][scope_key]
        if interval is None:
            axis.text(2, index, "카테고리 정보 없음", va="center", fontsize=9)
            continue
        values = interval["positive_interval_days"]
        p25 = values["p25"]
        p50 = values["p50"]
        p75 = values["p75"]
        p95 = values["p95"]
        axis.hlines(index, p25, p75, color=COLORS[index], linewidth=8, alpha=0.55)
        axis.scatter(p50, index, color=COLORS[index], s=70, zorder=3)
        axis.scatter(p95, index, color=COLORS[index], marker="D", s=45, zorder=3)
        axis.text(p95 + 3, index, f"{p95:.1f}일", va="center", fontsize=8)

    axis.set_title(title)
    axis.set_xlabel("재구매 간격(일)")
    axis.set_yticks(range(len(DATASETS)), DATASETS.values())
    axis.grid(axis="x", alpha=0.2)
    axis.legend(
        handles=(
            Line2D(
                [], [], color=COLORS[0], marker="o", linestyle="None", label="중앙값"
            ),
            Line2D(
                [], [], color=COLORS[0], marker="D", linestyle="None", label="95분위"
            ),
        ),
        frameon=False,
        fontsize=8,
    )


def plot_interval_quantiles(profiles: dict[str, dict[str, Any]]) -> None:
    """동일 상품과 동일 카테고리의 재구매 간격 분위수를 나란히 비교합니다."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    draw_interval_ranges(axes[0], profiles, "product_repurchase", "동일 상품")
    draw_interval_ranges(axes[1], profiles, "category_repurchase", "동일 카테고리")
    figure.suptitle("재구매 간격 분포 비교", fontsize=15)
    save_figure(figure, "repurchase_interval_quantiles.png")


def plot_pet_category_profile(profile: dict[str, Any]) -> None:
    """반려동물 세부 카테고리별 반복 구매율과 상품 구매 간격을 비교합니다."""
    rows = profile["category_breakdown"]
    labels = [
        PET_CATEGORY_LABELS.get(row["category_id"], row["category_id"]) for row in rows
    ]
    positions = np.arange(len(rows))
    width = 0.34
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    category_rates = [row["repeat_category_user_rate"] for row in rows]
    product_rates = [row["repeat_product_user_rate"] for row in rows]
    category_bars = axes[0].bar(
        positions - width / 2,
        category_rates,
        width,
        label="동일 카테고리 반복",
        color=COLORS[0],
    )
    product_bars = axes[0].bar(
        positions + width / 2,
        product_rates,
        width,
        label="동일 상품 반복",
        color=COLORS[2],
    )
    axes[0].set_title("반복 구매 사용자 비율")
    axes[0].set_ylabel("카테고리 사용자 대비 비율")
    axes[0].set_xticks(positions, labels, rotation=12)
    axes[0].set_ylim(0, 0.85)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend(frameon=False, fontsize=8)
    add_bar_labels(axes[0], category_bars)
    add_bar_labels(axes[0], product_bars)

    for index, row in enumerate(rows):
        p25 = row["product_interval_p25"]
        p50 = row["product_interval_p50"]
        p75 = row["product_interval_p75"]
        p95 = row["product_interval_p95"]
        axes[1].hlines(index, p25, p75, color=COLORS[index], linewidth=9, alpha=0.55)
        axes[1].scatter(p50, index, color=COLORS[index], s=70, zorder=3)
        axes[1].scatter(p95, index, color=COLORS[index], marker="D", s=45, zorder=3)
        axes[1].text(p95 + 3, index, f"{p95:.1f}일", va="center", fontsize=8)
    axes[1].set_title("동일 상품 재구매 간격")
    axes[1].set_xlabel("재구매 간격(일)")
    axes[1].set_yticks(positions, labels)
    axes[1].grid(axis="x", alpha=0.2)

    figure.suptitle("반려동물 카테고리 비교", fontsize=15)
    save_figure(figure, "pet_category_repurchase_profile.png")


def plot_monthly_orders() -> None:
    """데이터셋별 월간 주문 수와 사용자 수의 시간 흐름을 표시합니다."""
    profiles = load_profiles()
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    for axis, (dataset, label) in zip(axes, DATASETS.items(), strict=True):
        frame = pd.read_csv(REPORT_DIR / f"{dataset}_monthly_orders.csv")
        frame = filter_complete_months(frame, profiles[dataset]["observation"])
        positions = np.arange(len(frame))
        axis.plot(
            positions,
            frame["order_count"],
            color=COLORS[0],
            marker="o",
            markersize=3,
            label="주문 수",
        )
        axis.plot(
            positions,
            frame["user_count"],
            color=COLORS[1],
            marker="o",
            markersize=3,
            label="사용자 수",
        )
        tick_step = max(1, len(frame) // 8)
        ticks = positions[::tick_step]
        axis.set_xticks(ticks, frame.loc[ticks, "month"], rotation=25)
        axis.set_title(label)
        axis.set_ylabel("건·명")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(frameon=False, fontsize=8, ncol=2)

    figure.suptitle("월별 주문 및 구매 사용자 추이", fontsize=15)
    save_figure(figure, "monthly_order_trends.png")


def main() -> None:
    """모든 프로파일을 읽어 다섯 종류의 분석 그래프를 생성합니다."""
    configure_korean_font()
    profiles = load_profiles()
    plot_data_quality(profiles)
    plot_repurchase_behavior(profiles)
    plot_interval_quantiles(profiles)
    plot_pet_category_profile(profiles["complete_journey_pet"])
    plot_monthly_orders()
    print(f"[완료] 시각화 결과: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
