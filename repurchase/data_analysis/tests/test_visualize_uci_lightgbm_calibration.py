"""UCI LightGBM B/C Calibration 시각화 연결을 검증합니다."""

from __future__ import annotations

from scripts.plotting import configure_korean_font, plt
from scripts.visualize_uci_lightgbm_calibration import (
    build_bc_calibration_figure,
    load_bc_calibration,
)


def test_bc_calibration_figure_builds_curve_and_sample_count_panels() -> None:
    """B/C 각 10개 구간으로 보정 곡선과 표본 수 패널을 생성합니다."""
    configure_korean_font()
    calibration = load_bc_calibration()
    figure = build_bc_calibration_figure(calibration)

    try:
        assert len(calibration) == 20
        assert calibration.groupby("feature_set").size().eq(10).all()
        assert len(figure.axes) == 2
        assert figure.axes[0].get_title() == "Validation 30일 재구매 확률 Calibration"
        assert figure.axes[1].get_yscale() == "log"
    finally:
        plt.close(figure)
