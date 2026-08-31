"""실제 UCI 입력부터 재구매·우측검열 라벨까지 한 사이클을 검증합니다.

개별 단계의 단위 테스트와 별도로 로더, 행 품질 분류, 구매 사건 집계, 라벨
생성이 실제 연결 순서에서도 호환되는지 확인하고 단계별 건수와 불변조건을
하나의 JSON·Markdown 보고서로 남깁니다.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .loaders import load_uci_online_retail_ii
from .paths import REPORT_DIR
from .preprocessing.events import (
    build_uci_purchase_events,
    validate_uci_purchase_events,
)
from .preprocessing.labels import (
    build_same_product_repurchase_labels,
    validate_same_product_repurchase_labels,
)
from .preprocessing.uci import classify_uci_rows, validate_uci_classification
from .reporting import write_text_atomically

JSON_REPORT_PATH = REPORT_DIR / "uci_preprocessing_e2e_validation.json"
MARKDOWN_REPORT_PATH = REPORT_DIR / "uci_preprocessing_e2e_validation.md"


def run_uci_preprocessing_e2e(source: pd.DataFrame) -> dict[str, Any]:
    """하나의 UCI 데이터프레임을 라벨까지 처리하고 모든 단계를 검증합니다."""
    cleaning_result = classify_uci_rows(source)
    cleaning_summary = validate_uci_classification(source, cleaning_result)

    events = build_uci_purchase_events(cleaning_result.rows)
    event_summary = validate_uci_purchase_events(cleaning_result.rows, events)

    # 실제 배포 파일에서 확인되는 마지막 유효 구매를 이번 실행의 관측 끝으로 둡니다.
    observation_end_at = pd.Timestamp(events["ordered_at"].max())
    labels = build_same_product_repurchase_labels(events, observation_end_at)
    label_summary = validate_same_product_repurchase_labels(events, labels)

    pipeline_invariants = {
        "classification_preserves_source_rows": (
            cleaning_summary["classified_rows"] == cleaning_summary["source_rows"]
        ),
        "event_input_matches_accepted_rows": (
            event_summary["accepted_source_line_count"]
            == cleaning_summary["accepted_repurchase_rows"]
        ),
        "one_label_per_purchase_event": (
            label_summary["label_count"] == event_summary["purchase_event_count"]
        ),
        "all_stage_invariants_passed": all(
            [
                *cleaning_summary["invariants"].values(),
                *event_summary["invariants"].values(),
                *label_summary["invariants"].values(),
            ]
        ),
    }
    failed_invariants = [
        name for name, passed in pipeline_invariants.items() if not passed
    ]
    if failed_invariants:
        raise RuntimeError(f"UCI E2E 불변조건을 위반했습니다: {failed_invariants}")

    return {
        "dataset": "uci_online_retail_ii",
        "label_scope": "same_product",
        "observation_end_at": observation_end_at.isoformat(),
        "cleaning": cleaning_summary,
        "purchase_events": event_summary,
        "repurchase_labels": label_summary,
        "pipeline_invariants": pipeline_invariants,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    """E2E 처리 단계와 라벨 분포를 검토하기 쉬운 Markdown으로 변환합니다."""
    cleaning = summary["cleaning"]
    events = summary["purchase_events"]
    labels = summary["repurchase_labels"]
    durations = labels["observed_duration_days"]
    duration_median = (
        "관측값 없음"
        if durations["median"] is None
        else f"{durations['median']:,.2f}일"
    )
    duration_p99 = (
        "관측값 없음" if durations["p99"] is None else f"{durations['p99']:,.2f}일"
    )
    duration_maximum = (
        "관측값 없음"
        if durations["maximum"] is None
        else f"{durations['maximum']:,.2f}일"
    )
    lines = [
        "# UCI 전처리 E2E 검증 결과",
        "",
        f"- 라벨 범위: `{summary['label_scope']}`",
        f"- 관측 종료 시각: `{summary['observation_end_at']}`",
        "",
        "## 단계별 건수",
        "",
        "| 단계 | 건수 | 앞 단계 대비 의미 |",
        "| --- | ---: | --- |",
        f"| 원본 행 | {cleaning['source_rows']:,} | 로더 입력 |",
        (
            f"| 재구매 사용 가능 행 | {cleaning['accepted_repurchase_rows']:,} | "
            "취소·결측·비상품 등 제외 |"
        ),
        (
            f"| 구매 사건 | {events['purchase_event_count']:,} | "
            "사용자·주문·상품 단위 집계 |"
        ),
        (f"| 라벨 | {labels['label_count']:,} | 모든 구매 사건에 하나씩 생성 |"),
        "",
        "## 라벨 분포",
        "",
        "| 항목 | 건수 | 비율 |",
        "| --- | ---: | ---: |",
        (
            f"| 재구매 관측 | {labels['observed_repurchase_count']:,} | "
            f"{labels['observed_repurchase_rate']:.2%} |"
        ),
        (
            f"| 우측검열 | {labels['right_censored_count']:,} | "
            f"{labels['right_censored_rate']:.2%} |"
        ),
        (
            f"| 0일 후속 구매 | {labels['zero_day_followup_count']:,} | "
            f"{labels['zero_day_followup_rate']:.2%} |"
        ),
        "",
        f"관측된 동일 상품 재구매 간격은 중앙값 **{duration_median}**, "
        f"P99 **{duration_p99}**, 최대 **{duration_maximum}**입니다.",
        "",
        "우측검열률은 고객의 비재구매율이 아닙니다. 사용자·상품 시퀀스마다 "
        "마지막 사건 하나를 검열로 보존하므로 생기는 사건 단위 비율입니다.",
        "",
        "0일 후속 구매는 연속 주문번호가 같은 분 단위 시각으로 기록된 사례입니다. "
        "즉시 재구매·주문 분할·시각 정밀도 한계를 현재 데이터만으로 구분할 수 "
        "없으므로 삭제하지 않고 플래그로 보존하며, 학습 단계에서 포함·제외 "
        "민감도를 비교합니다.",
        "",
        "## 전체 파이프라인 불변조건",
        "",
        "| 검증 조건 | 결과 |",
        "| --- | --- |",
    ]
    lines.extend(
        f"| `{name}` | {'통과' if passed else '실패'} |"
        for name, passed in summary["pipeline_invariants"].items()
    )
    lines.extend(
        [
            "",
            "UCI에는 신뢰할 수 있는 카테고리·상품군 정보가 없어 동일 상품 "
            "재구매만 라벨링했습니다. 상품 전환 라벨은 Complete Journey에서 "
            "별도로 검증합니다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """실제 UCI 배포 파일로 E2E를 실행하고 검증 보고서를 저장합니다."""
    summary = run_uci_preprocessing_e2e(load_uci_online_retail_ii())
    write_text_atomically(
        JSON_REPORT_PATH,
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    write_text_atomically(MARKDOWN_REPORT_PATH, render_markdown(summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
