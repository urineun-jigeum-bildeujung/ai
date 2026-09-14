"""UCI 재구매 오차 진단 보고서의 시각화 연결을 검증합니다."""

from __future__ import annotations

from scripts.plotting import plt
from scripts.visualize_uci_baseline_errors import (
    build_error_diagnostic_figure,
    load_error_summary,
)


def test_error_diagnostic_figure_builds_four_analysis_panels() -> None:
    """저장된 E2E JSON에서 네 오차 분석 패널을 생성합니다."""
    figure = build_error_diagnostic_figure(load_error_summary())

    try:
        assert len(figure.axes) == 4
        assert [axis.get_title() for axis in figure.axes] == [
            "Validation 수축 강도별 오차",
            "Test 개인 이력 수와 MAE",
            "Test 예측 기준 월별 오차",
            "Test 절대오차 기여 상위 상품",
        ]
    finally:
        plt.close(figure)
