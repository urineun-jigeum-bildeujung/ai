"""재구매 분석 그래프에서 공통으로 사용하는 백엔드·글꼴·저장 설정입니다."""

import matplotlib

# 화면이 없는 CI·서버에서도 그래프 창을 열지 않고 PNG를 생성합니다.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager

from .paths import FIGURE_DIR


def configure_korean_font() -> None:
    """실행 환경에서 사용할 수 있는 한글 글꼴을 찾아 그래프에 적용합니다."""
    candidates = (
        "AppleGothic",
        "Arial Unicode MS",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Malgun Gothic",
    )
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((font for font in candidates if font in available), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": selected,
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 180,
        }
    )


def save_figure(figure: plt.Figure, filename: str) -> None:
    """그래프 여백을 정리해 공통 figures 폴더에 PNG로 저장합니다."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(figure)
