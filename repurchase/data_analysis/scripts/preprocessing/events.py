"""분류된 UCI 주문 상품 행을 사용자·주문·상품 구매 사건으로 집계합니다.

재구매 간격은 원본 행이 아니라 실제 구매 사건 사이에서 계산해야 합니다.
동일 주문의 동일 상품 행은 하나의 사건으로 합치되, 중복 여부가 불확실한
수량·금액은 원본 합계와 완전 중복 제거 합계를 모두 보존합니다.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

from .uci import DUPLICATE_COMPARISON_COLUMNS


class UciEventBuildError(ValueError):
    """구매 사건 입력이나 집계 결과가 정의된 계약을 위반할 때 발생합니다."""


EVENT_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "user_id",
    "order_id",
    "product_id",
)

EVENT_REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    *EVENT_KEY_COLUMNS,
    *DUPLICATE_COMPARISON_COLUMNS,
    "source_row_id",
    "line_amount",
    "is_accepted_repurchase",
)

# 중복 영향의 일반적인 크기와 긴 꼬리를 함께 확인할 분위수입니다.
DUPLICATE_DIFFERENCE_QUANTILES: Final[tuple[float, ...]] = (
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    1.00,
)


def _validate_event_input(rows: pd.DataFrame) -> None:
    """구매 사건 집계에 필요한 분류 결과 컬럼이 모두 있는지 검사합니다."""
    missing_columns = set(EVENT_REQUIRED_COLUMNS) - set(rows.columns)
    if missing_columns:
        raise UciEventBuildError(
            f"UCI 구매 사건 필수 컬럼이 누락됐습니다: {sorted(missing_columns)}"
        )


def _aggregate_purchase_lines(
    rows: pd.DataFrame,
    suffix: str,
) -> pd.DataFrame:
    """하나의 행 집합을 사용자·주문·상품별 수량·금액·행 수로 집계합니다.

    동일 주문의 상품 행이 여러 시각에 걸쳐 기록될 수 있으므로 최초·최종 시각을
    모두 보존합니다. 최초 시각은 사건의 기준 시각으로 사용하고, 최종 시각은
    기록 시간 범위와 원천 데이터 품질을 점검하는 데 사용합니다.
    """
    return (
        rows.groupby(list(EVENT_KEY_COLUMNS), observed=True, as_index=False)
        .agg(
            ordered_at=("ordered_at", "min"),
            ordered_at_last=("ordered_at", "max"),
            quantity=("quantity", "sum"),
            line_amount=("line_amount", "sum"),
            source_line_count=("source_row_id", "size"),
        )
        .rename(
            columns={
                "quantity": f"quantity_{suffix}",
                "line_amount": f"line_amount_{suffix}",
                "source_line_count": f"source_line_count_{suffix}",
            }
        )
    )


def build_uci_purchase_events(classified_rows: pd.DataFrame) -> pd.DataFrame:
    """재구매 사용 가능 행을 하나의 사용자·주문·상품 구매 사건으로 변환합니다."""
    _validate_event_input(classified_rows)
    accepted = classified_rows.loc[classified_rows["is_accepted_repurchase"]].copy()

    raw_events = _aggregate_purchase_lines(accepted, "raw")
    deduplicated_lines = accepted.drop_duplicates(
        subset=list(DUPLICATE_COMPARISON_COLUMNS),
        keep="first",
    )
    deduplicated_events = _aggregate_purchase_lines(
        deduplicated_lines,
        "deduplicated",
    ).drop(columns=["ordered_at", "ordered_at_last"])

    events = raw_events.merge(
        deduplicated_events,
        on=list(EVENT_KEY_COLUMNS),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not events["_merge"].eq("both").all():
        raise UciEventBuildError("중복 처리 전후 구매 사건 키가 달라졌습니다.")
    events = events.drop(columns="_merge")

    events["quantity_difference"] = (
        events["quantity_raw"] - events["quantity_deduplicated"]
    )
    events["line_amount_difference"] = (
        events["line_amount_raw"] - events["line_amount_deduplicated"]
    )
    events["had_suspected_duplicate"] = events["source_line_count_raw"].gt(
        events["source_line_count_deduplicated"]
    )
    # 같은 주문번호 안의 시각 차이는 별도 구매가 아니라 입력 시간의 변동으로 봅니다.
    events["event_duration_seconds"] = (
        events["ordered_at_last"] - events["ordered_at"]
    ).dt.total_seconds()
    events["has_timestamp_variation"] = events["event_duration_seconds"].gt(0)

    return events.sort_values(
        ["user_id", "ordered_at", "order_id", "product_id"],
        kind="stable",
        ignore_index=True,
    )


def validate_uci_purchase_events(
    classified_rows: pd.DataFrame,
    events: pd.DataFrame,
) -> dict[str, object]:
    """구매 사건 키·행 기여도·수량 관계 불변조건과 분포를 검사합니다."""
    _validate_event_input(classified_rows)
    accepted = classified_rows.loc[classified_rows["is_accepted_repurchase"]]
    expected_keys = pd.MultiIndex.from_frame(
        accepted.loc[:, list(EVENT_KEY_COLUMNS)].drop_duplicates()
    )
    actual_keys = pd.MultiIndex.from_frame(events.loc[:, list(EVENT_KEY_COLUMNS)])
    duplicated_event_keys = events.duplicated(
        subset=list(EVENT_KEY_COLUMNS),
        keep=False,
    )

    invariants = {
        "one_row_per_event_key": not duplicated_event_keys.any(),
        "event_key_sets_preserved": set(actual_keys) == set(expected_keys),
        "raw_lines_conserved": (
            int(events["source_line_count_raw"].sum()) == len(accepted)
        ),
        "deduplicated_lines_not_greater_than_raw": bool(
            events["source_line_count_deduplicated"]
            .le(events["source_line_count_raw"])
            .all()
        ),
        "positive_event_quantity": bool(events["quantity_raw"].gt(0).all()),
        "deduplicated_quantity_not_greater_than_raw": bool(
            events["quantity_deduplicated"].le(events["quantity_raw"]).all()
        ),
        "nonnegative_event_duration": bool(
            events["event_duration_seconds"].ge(0).all()
        ),
    }
    failed_invariants = [name for name, passed in invariants.items() if not passed]
    if failed_invariants:
        raise UciEventBuildError(
            f"UCI 구매 사건 불변조건을 위반했습니다: {failed_invariants}"
        )

    affected = events["had_suspected_duplicate"]
    affected_quantity_difference = events.loc[affected, "quantity_difference"]
    quantity_difference_quantiles = {
        f"p{int(quantile * 100)}": (
            0.0
            if affected_quantity_difference.empty
            else float(affected_quantity_difference.quantile(quantile))
        )
        for quantile in DUPLICATE_DIFFERENCE_QUANTILES
    }
    return {
        "accepted_source_line_count": int(len(accepted)),
        "purchase_event_count": int(len(events)),
        "duplicate_affected_event_count": int(affected.sum()),
        "duplicate_affected_event_rate": float(affected.mean()),
        "timestamp_variation_event_count": int(events["has_timestamp_variation"].sum()),
        "max_event_duration_seconds": float(events["event_duration_seconds"].max()),
        "quantity_raw_total": float(events["quantity_raw"].sum()),
        "quantity_deduplicated_total": float(events["quantity_deduplicated"].sum()),
        "quantity_difference_total": float(events["quantity_difference"].sum()),
        "quantity_difference_rate": float(
            events["quantity_difference"].sum() / events["quantity_raw"].sum()
        ),
        "quantity_difference_quantiles_on_affected_events": (
            quantity_difference_quantiles
        ),
        "line_amount_raw_total": float(events["line_amount_raw"].sum()),
        "line_amount_deduplicated_total": float(
            events["line_amount_deduplicated"].sum()
        ),
        "line_amount_difference_total": float(events["line_amount_difference"].sum()),
        "line_amount_difference_rate": float(
            events["line_amount_difference"].sum() / events["line_amount_raw"].sum()
        ),
        "invariants": invariants,
    }
