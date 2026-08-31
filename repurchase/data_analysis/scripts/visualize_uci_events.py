"""UCI 구매 사건의 중복 민감도와 수량 차이 분포를 시각화합니다.

전체 원본을 다시 읽지 않고 검증 단계에서 생성한 JSON 보고서만 사용합니다.
따라서 데이터 파일이 없는 환경에서도 동일한 집계 그래프를 재생성할 수 있습니다.
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from .paths import FIGURE_DIR, REPORT_DIR
from .plotting import configure_korean_font, save_figure

REPORT_PATH = REPORT_DIR / "uci_purchase_event_validation.json"
FIGURE_NAME = "uci_purchase_event_duplicate_sensitivity.png"


def load_event_summary() -> dict[str, Any]:
    """구매 사건 검증 JSON을 읽어 시각화 입력으로 반환합니다."""
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def plot_duplicate_sensitivity(summary: dict[str, Any]) -> None:
    """전체 영향률과 영향 사건 내부 수량 차이 분위수를 함께 표시합니다."""
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    rate_labels = ("영향 사건", "수량 차이", "금액 차이")
    rates = (
        summary["duplicate_affected_event_rate"],
        summary["quantity_difference_rate"],
        summary["line_amount_difference_rate"],
    )
    bars = axes[0].bar(rate_labels, rates, color=("#3366CC", "#109618", "#FF9900"))
    axes[0].bar_label(bars, labels=[f"{value:.2%}" for value in rates], padding=3)
    axes[0].set_title("전체 구매 사건에서 중복 후보의 영향")
    axes[0].set_ylabel("전체 대비 비율")
    axes[0].set_ylim(0, max(rates) * 1.35)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].grid(axis="y", alpha=0.2)

    quantiles = summary["quantity_difference_quantiles_on_affected_events"]
    quantile_labels = tuple(name.upper() for name in quantiles)
    values = tuple(quantiles.values())
    bars = axes[1].bar(quantile_labels, values, color="#8B5CF6")
    axes[1].bar_label(bars, labels=[f"{value:,.0f}" for value in values], padding=3)
    axes[1].set_title("중복 영향 사건의 수량 차이 분위수")
    axes[1].set_ylabel("원본 수량 - 완전 중복 제거 수량")
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", alpha=0.2)

    figure.suptitle("UCI 구매 사건 중복 민감도", fontsize=15)
    save_figure(figure, FIGURE_NAME)


def main() -> None:
    """검증 보고서를 읽어 중복 민감도 그래프를 생성합니다."""
    configure_korean_font()
    plot_duplicate_sensitivity(load_event_summary())
    print(f"[완료] 시각화 결과: {FIGURE_DIR / FIGURE_NAME}")


if __name__ == "__main__":
    main()
