"""저장된 UCI 베이스라인 평가 결과로 오차 원인 그래프를 생성합니다.

원본 XLSX와 모델을 다시 실행하지 않고 JSON 보고서만 읽습니다. 따라서 모델
평가와 시각화 표현을 분리하고, 분석 그래프를 빠르게 반복 수정할 수 있습니다.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from matplotlib.ticker import PercentFormatter

from .paths import FIGURE_DIR, REPORT_DIR
from .plotting import configure_korean_font, plt, save_figure

REPORT_PATH = REPORT_DIR / "uci_baseline_e2e_evaluation.json"
FIGURE_NAME = "uci_repurchase_error_diagnostics.png"


def load_error_summary() -> dict[str, Any]:
    """E2E JSON 보고서를 읽어 시각화 입력으로 반환합니다."""
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def build_error_diagnostic_figure(summary: dict[str, Any]) -> plt.Figure:
    """수축 후보·이력 수·월·상품 관점의 오차 진단 그래프를 만듭니다."""
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    candidates = pd.DataFrame(summary["validation_shrinkage_candidates"])
    axes[0, 0].plot(
        candidates["shrinkage_strength"],
        candidates["mae_days"],
        marker="o",
        label="전체 MAE",
    )
    axes[0, 0].plot(
        candidates["shrinkage_strength"],
        candidates["tail_mae_days"],
        marker="o",
        label="후보별 최악 5% MAE",
    )
    axes[0, 0].plot(
        candidates["shrinkage_strength"],
        candidates["fixed_cohort_candidate_mae_days"],
        marker="o",
        label="기존 최악 5% MAE",
    )
    axes[0, 0].set_title("Validation 수축 강도별 오차")
    axes[0, 0].set_xlabel("수축 강도 k")
    axes[0, 0].set_ylabel("평균 절대오차(일)")
    axes[0, 0].grid(alpha=0.2)
    axes[0, 0].legend()

    test_evaluation = summary["test_evaluation"]
    history = pd.DataFrame(test_evaluation["user_product_error_by_history_count"])
    max_sample_count = float(history["sample_count"].max())
    # 점 크기는 이력 구간의 표본 수를 상대적으로 표현하며 모델 계산에는 쓰지 않습니다.
    point_sizes = history["sample_count"].div(max_sample_count).mul(260).add(20)
    axes[0, 1].scatter(
        history["history_interval_count"],
        history["mae_days"],
        s=point_sizes,
        alpha=0.65,
        color="#8B5CF6",
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("Test 개인 이력 수와 MAE")
    axes[0, 1].set_xlabel("과거 구매 간격 수(로그 축)")
    axes[0, 1].set_ylabel("평균 절대오차(일)")
    axes[0, 1].grid(alpha=0.2)

    monthly = pd.DataFrame(test_evaluation["user_product_error_by_anchor_month"])
    axes[1, 0].plot(
        monthly["anchor_month"],
        monthly["mae_days"],
        marker="o",
        label="MAE",
    )
    axes[1, 0].plot(
        monthly["anchor_month"],
        monthly["median_absolute_error_days"],
        marker="o",
        label="중앙 절대오차",
    )
    axes[1, 0].set_title("Test 예측 기준 월별 오차")
    axes[1, 0].set_xlabel("예측 기준 월")
    axes[1, 0].set_ylabel("오차(일)")
    axes[1, 0].tick_params(axis="x", rotation=30)
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend()

    products = pd.DataFrame(
        test_evaluation["user_product_error_by_product"]["top_contributors"]
    ).sort_values("absolute_error_share")
    axes[1, 1].barh(
        products["product_id"].astype(str),
        products["absolute_error_share"],
        color="#FF9900",
    )
    axes[1, 1].set_title("Test 절대오차 기여 상위 상품")
    axes[1, 1].set_xlabel("개인 이력 전체 절대오차 기여율")
    axes[1, 1].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1, 1].grid(axis="x", alpha=0.2)

    figure.suptitle("UCI 재구매 예측 오차 진단", fontsize=16)
    return figure


def main() -> None:
    """저장된 E2E 결과를 읽어 오차 진단 PNG를 저장합니다."""
    configure_korean_font()
    figure = build_error_diagnostic_figure(load_error_summary())
    save_figure(figure, FIGURE_NAME)
    print(f"[완료] 시각화 결과: {FIGURE_DIR / FIGURE_NAME}")


if __name__ == "__main__":
    main()
