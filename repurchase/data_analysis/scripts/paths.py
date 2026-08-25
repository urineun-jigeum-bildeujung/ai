"""재구매 데이터 분석에서 공통으로 사용하는 디렉터리 경로를 정의합니다.

프로젝트 구조를 직접 해석하는 코드를 한 파일에 모아, 폴더 구조가 변경될 때
수정해야 하는 위치를 최소화합니다.
"""

from pathlib import Path

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ANALYSIS_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORT_DIR = ANALYSIS_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
