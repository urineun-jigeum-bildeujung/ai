"""분석·검증 보고서를 안전하게 저장하는 공통 기능을 제공합니다.

보고서 생성 도중 프로세스가 중단돼도 기존의 정상 보고서를 훼손하지 않도록
완성된 내용을 임시 파일에 먼저 기록한 뒤 최종 경로로 원자적으로 교체합니다.
"""

from pathlib import Path
from typing import Any

import pandas as pd


def dataframe_to_nullable_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame 결측값을 표준 JSON의 null로 변환할 수 있는 레코드로 만듭니다."""
    # 실수형 열은 None을 다시 NaN으로 바꾸므로 먼저 object형으로 변환합니다.
    nullable_frame = frame.astype(object).where(frame.notna(), None)
    return nullable_frame.to_dict(orient="records")


def write_text_atomically(path: Path, content: str) -> None:
    """문자열 전체를 임시 파일에 쓴 뒤 최종 보고서 경로로 교체합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.part")
    try:
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
