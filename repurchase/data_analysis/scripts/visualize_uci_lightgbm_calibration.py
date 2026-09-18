"""저장된 UCI LightGBM B/C 보정 통계로 Calibration curve를 생성합니다.

모델을 다시 학습하지 않고 동일 Validation의 JSON 결과만 사용합니다. 곡선 아래에
구간별 표본 수를 함께 표시해 소표본 고확률 구간의 차이를 과대해석하지 않습니다.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .paths import FIGURE_DIR, REPORT_DIR
from .plotting import configure_korean_font, plt, save_figure

REPORT_PATH = REPORT_DIR / "uci_lightgbm_feature_comparison.json"
FIGURE_NAME = "uci_lightgbm_bc_calibration.png"
FEATURE_LABELS = {
    "B_counts_median": "B: 횟수 + 중앙값",
    "C_counts_median_variability": "C: B + 불규칙성",
}


def load_bc_calibration() -> pd.DataFrame:
    """저장된 Validation 보정표에서 B와 C의 행만 선택합니다."""
    report: dict[str, Any] = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    calibration = pd.DataFrame(report["calibration"])
    selected = calibration.loc[calibration["feature_set"].isin(FEATURE_LABELS),].copy()
    if selected.empty or set(selected["feature_set"]) != set(FEATURE_LABELS):
        raise ValueError("Calibration curve에 필요한 B/C 구간이 모두 있어야 합니다.")
    bin_indices = [
        group["calibration_bin_index"].tolist()
        for _, group in selected.groupby("feature_set", observed=True, sort=True)
    ]
    if any(len(indices) != len(set(indices)) for indices in bin_indices):
        raise ValueError("한 후보 안에서 Calibration 구간이 중복될 수 없습니다.")
    if len(bin_indices) != 2 or bin_indices[0] != bin_indices[1]:
        raise ValueError("B와 C는 같은 Calibration 구간 인덱스를 사용해야 합니다.")
    return selected


def build_bc_calibration_figure(calibration: pd.DataFrame) -> plt.Figure:
    """B/C 보정 곡선과 확률 구간별 표본 수를 한 그림에 배치합니다."""
    figure, (curve_axis, count_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 9),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    curve_axis.plot([0, 1], [0, 1], linestyle="--", color="#6B7280", label="완전 보정")
    colors = ("#1D76DB", "#8B5CF6")
    offsets = (-0.018, 0.018)
    bar_width = 0.036
    for (feature_set, label), color, offset in zip(
        FEATURE_LABELS.items(), colors, offsets, strict=True
    ):
        rows = calibration.loc[calibration["feature_set"].eq(feature_set)].sort_values(
            "calibration_bin_index"
        )
        curve_axis.plot(
            rows["mean_predicted_probability"],
            rows["observed_event_rate"],
            marker="o",
            color=color,
            label=label,
        )
        bin_centers = rows[["bin_lower_bound", "bin_upper_bound"]].mean(axis=1)
        count_axis.bar(
            bin_centers + offset,
            rows["sample_count"],
            width=bar_width,
            color=color,
            alpha=0.75,
            label=label,
        )

    curve_axis.set_xlim(0, 1)
    curve_axis.set_ylim(0, 1)
    curve_axis.set_ylabel("IPCW 가중 실제 재구매율")
    curve_axis.set_title("Validation 30일 재구매 확률 Calibration")
    curve_axis.grid(alpha=0.2)
    curve_axis.legend()
    count_axis.set_yscale("log")
    count_axis.set_xlabel("평균 예측 확률")
    count_axis.set_ylabel("표본 수\n(로그 축)")
    count_axis.grid(axis="y", alpha=0.2)
    count_axis.legend()
    figure.tight_layout()
    return figure


def main() -> None:
    """저장된 보정 통계를 읽어 B/C Calibration PNG를 저장합니다."""
    configure_korean_font()
    figure = build_bc_calibration_figure(load_bc_calibration())
    save_figure(figure, FIGURE_NAME)
    print(f"[완료] 시각화 결과: {FIGURE_DIR / FIGURE_NAME}")


if __name__ == "__main__":
    main()
