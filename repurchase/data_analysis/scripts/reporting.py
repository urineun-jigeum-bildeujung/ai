"""분석·검증 보고서를 안전하게 저장하는 공통 기능을 제공합니다.

보고서 생성 도중 프로세스가 중단돼도 기존의 정상 보고서를 훼손하지 않도록
완성된 내용을 임시 파일에 먼저 기록한 뒤 최종 경로로 원자적으로 교체합니다.
"""

from pathlib import Path


def write_text_atomically(path: Path, content: str) -> None:
    """문자열 전체를 임시 파일에 쓴 뒤 최종 보고서 경로로 교체합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.part")
    try:
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
