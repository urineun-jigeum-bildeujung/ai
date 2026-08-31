"""UCI 분류 결과를 구매 사건으로 집계하고 전체 데이터 검증 보고서를 만듭니다.

단위 테스트로 확인한 사건 키·행 보존 규칙을 실제 두 시트 전체에 다시 적용합니다.
중복 후보를 유지한 결과와 완전 동일 행만 제거한 결과를 함께 비교해, 이후 모델
실험에서 어떤 수량 표현을 선택했는지 추적할 수 있는 근거를 남깁니다.
"""

from __future__ import annotations

import json
from typing import Any

from .loaders import load_uci_online_retail_ii
from .paths import REPORT_DIR
from .preprocessing.events import (
    build_uci_purchase_events,
    validate_uci_purchase_events,
)
from .preprocessing.uci import classify_uci_rows
from .reporting import write_text_atomically

JSON_REPORT_PATH = REPORT_DIR / "uci_purchase_event_validation.json"
MARKDOWN_REPORT_PATH = REPORT_DIR / "uci_purchase_event_validation.md"


def render_markdown(summary: dict[str, Any]) -> str:
    """구매 사건 검증 결과를 사람이 검토하기 쉬운 Markdown으로 변환합니다."""
    quantiles = summary["quantity_difference_quantiles_on_affected_events"]
    lines = [
        "# UCI 구매 사건 검증 결과",
        "",
        "## 사건 집계",
        "",
        "| 항목 | 값 |",
        "| --- | ---: |",
        f"| 재구매 사용 가능 원본 행 | {summary['accepted_source_line_count']:,} |",
        f"| 사용자·주문·상품 구매 사건 | {summary['purchase_event_count']:,} |",
        (
            "| 중복 후보 영향 사건 | "
            f"{summary['duplicate_affected_event_count']:,} "
            f"({summary['duplicate_affected_event_rate']:.2%}) |"
        ),
        (f"| 시각 변동 사건 | {summary['timestamp_variation_event_count']:,} |"),
        (f"| 최대 사건 기록 범위 | {summary['max_event_duration_seconds']:,.0f}초 |"),
        "",
        "## 중복 민감도",
        "",
        "| 항목 | 원본 합계 | 완전 중복 제거 합계 | 차이 | 차이율 |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            "| 수량 | "
            f"{summary['quantity_raw_total']:,.0f} | "
            f"{summary['quantity_deduplicated_total']:,.0f} | "
            f"{summary['quantity_difference_total']:,.0f} | "
            f"{summary['quantity_difference_rate']:.2%} |"
        ),
        (
            "| 금액 | "
            f"{summary['line_amount_raw_total']:,.2f} | "
            f"{summary['line_amount_deduplicated_total']:,.2f} | "
            f"{summary['line_amount_difference_total']:,.2f} | "
            f"{summary['line_amount_difference_rate']:.2%} |"
        ),
        "",
        "중복 영향을 받은 사건의 수량 차이는 다음과 같습니다.",
        "",
        "| 분위수 | 수량 차이 |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {name.upper()} | {value:,.0f} |" for name, value in quantiles.items()
    )
    lines.extend(
        [
            "",
            "전체 차이율만 보면 영향이 작지만 상위 꼬리에서는 차이가 커질 수 있어, "
            "원본 수량과 완전 중복 제거 수량을 모두 보존합니다.",
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
            "같은 주문번호 안의 시각 차이는 별도 구매로 분리하지 않습니다. "
            "최초 시각을 사건 기준으로 사용하고 최종 시각과 기록 범위를 품질 "
            "확인용으로 함께 보존합니다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """UCI 전체 행을 분류·사건화·검증하고 두 형식으로 저장합니다."""
    source = load_uci_online_retail_ii()
    classified = classify_uci_rows(source).rows
    events = build_uci_purchase_events(classified)
    summary = validate_uci_purchase_events(classified, events)

    write_text_atomically(
        JSON_REPORT_PATH,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    write_text_atomically(MARKDOWN_REPORT_PATH, render_markdown(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
