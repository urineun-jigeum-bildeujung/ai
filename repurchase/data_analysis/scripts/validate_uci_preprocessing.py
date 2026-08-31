"""실제 UCI 전체 데이터에 전처리를 적용하고 재현 가능한 검증 보고서를 만듭니다.

단위 테스트와 달리 원본 ZIP의 두 시트를 모두 읽는 통합 검증입니다. 원본과
분류 결과의 행·식별자 보존, 제외 사유 일치 여부를 검사하고 JSON과 Markdown
요약을 reports 디렉터리에 저장합니다.
"""

from __future__ import annotations

import json
from typing import Any

from .loaders import load_uci_online_retail_ii
from .paths import REPORT_DIR
from .preprocessing.uci import (
    UCI_NON_MERCHANDISE_CODES,
    classify_uci_rows,
    validate_uci_classification,
)
from .reporting import write_text_atomically

JSON_REPORT_PATH = REPORT_DIR / "uci_preprocessing_validation.json"
MARKDOWN_REPORT_PATH = REPORT_DIR / "uci_preprocessing_validation.md"


def render_markdown(summary: dict[str, Any]) -> str:
    """검증 요약을 사람이 검토하기 쉬운 Markdown 표로 변환합니다."""
    lines = [
        "# UCI 전처리 검증 결과",
        "",
        "## 행 보존 및 사용 범위",
        "",
        "| 항목 | 건수 | 비율 |",
        "| --- | ---: | ---: |",
        (f"| 원본 행 | {summary['source_rows']:,} | 100.00% |"),
        (
            "| 재구매 사용 가능 | "
            f"{summary['accepted_repurchase_rows']:,} | "
            f"{summary['accepted_repurchase_rate']:.2%} |"
        ),
        (
            f"| 격리 | {summary['quarantined_rows']:,} | "
            f"{summary['quarantined_rate']:.2%} |"
        ),
        (
            "| 금액 분석 사용 가능 | "
            f"{summary['accepted_monetary_rows']:,} | "
            f"{summary['accepted_monetary_rate']:.2%} |"
        ),
        "",
        "## 품질 사유",
        "",
        "| 사유 코드 | 기록 수 |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| `{reason_code}` | {count:,} |"
        for reason_code, count in summary["reason_counts"].items()
    )
    lines.extend(
        [
            "",
            f"- 전체 사유 기록: {summary['reason_record_count']:,}건",
            (f"- 사유가 2개 이상인 행: {summary['rows_with_multiple_reasons']:,}건"),
            (f"- 한 행의 최대 사유 수: {summary['maximum_reasons_on_one_row']:,}개"),
            "",
            "## 불변조건 검증",
            "",
            "| 검증 조건 | 결과 |",
            "| --- | --- |",
        ]
    )
    lines.extend(
        f"| `{name}` | {'통과' if passed else '실패'} |"
        for name, passed in summary["invariants"].items()
    )
    lines.extend(
        [
            "",
            "## 비상품 코드 정책",
            "",
            "문자형 상품 코드를 일괄 제외하지 않고 원본에서 비상품으로 확인된 "
            "코드만 명시적으로 관리합니다.",
            "",
            ", ".join(f"`{code}`" for code in sorted(UCI_NON_MERCHANDISE_CODES)),
            "",
            "중복 후보는 자동 삭제하지 않고 경고로 남깁니다. 동일 주문·상품은 "
            "후속 구매 사건 생성 단계에서 하나의 사건으로 집계합니다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """UCI 전체 데이터를 분류·검증하고 두 형식의 보고서를 저장합니다."""
    source = load_uci_online_retail_ii()
    result = classify_uci_rows(source)
    summary = validate_uci_classification(source, result)

    write_text_atomically(
        JSON_REPORT_PATH,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    write_text_atomically(MARKDOWN_REPORT_PATH, render_markdown(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
