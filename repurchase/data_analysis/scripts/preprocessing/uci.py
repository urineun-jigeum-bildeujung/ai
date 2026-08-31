"""UCI Online Retail II 행을 삭제하지 않고 사용 가능 범위별로 분류합니다.

개인별 재구매 학습에서 제외할 행과 금액 분석에서만 제외할 행, 추가 검토가
필요한 경고 행을 구분합니다. 모든 원본 행과 제외 사유를 보존해 전처리 전후
행 수를 검증하고 정책 변경 시 다시 판단할 수 있게 합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import pandas as pd


class UciPreprocessingError(ValueError):
    """입력 스키마나 분류 결과가 전처리 계약을 위반할 때 발생합니다."""


class ReasonCode(StrEnum):
    """UCI 행을 제외하거나 주의해서 사용할 이유를 식별합니다."""

    # 사용자 식별자가 없어 개인별 구매 이력과 연결할 수 없는 행입니다.
    MISSING_USER_ID = "MISSING_USER_ID"
    # 주문·상품·구매시각·수량 중 하나가 없어 구매 사건을 만들 수 없는 행입니다.
    MISSING_REQUIRED_EVENT_FIELD = "MISSING_REQUIRED_EVENT_FIELD"
    # 주문번호가 C로 시작해 원본에서 취소로 명시된 행입니다.
    EXPLICIT_CANCELLATION = "EXPLICIT_CANCELLATION"
    # 수량이 0 이하라 정상 구매보다 반품·파손·재고 조정일 가능성이 높은 행입니다.
    NONPOSITIVE_QUANTITY = "NONPOSITIVE_QUANTITY"
    # 가격이 0 이하라 구매 시점은 남겨도 금액 피처에는 사용할 수 없는 행입니다.
    NONPOSITIVE_PRICE = "NONPOSITIVE_PRICE"
    # 배송비·수수료·할인·수동 조정처럼 실제 소비 상품이 아닌 코드입니다.
    NON_MERCHANDISE_CODE = "NON_MERCHANDISE_CODE"
    # 상품 코드는 있지만 설명이 없어 상품명 기반 확인이 어려운 행입니다.
    MISSING_PRODUCT_NAME = "MISSING_PRODUCT_NAME"
    # 주요 값이 다른 행과 같지만 정상 반복 행인지 적재 중복인지 확정할 수 없는 행입니다.
    SUSPECTED_DUPLICATE = "SUSPECTED_DUPLICATE"


class ReasonLevel(StrEnum):
    """품질 사유가 데이터 사용 범위에 미치는 수준을 나타냅니다."""

    # 개인별 재구매 사건과 금액 분석에서 모두 제외합니다.
    EXCLUDE = "EXCLUDE"
    # 구매 행동은 유지하지만 가격·결제금액 관련 분석에서만 제외합니다.
    MONETARY_EXCLUDE = "MONETARY_EXCLUDE"
    # 자동 제외하지 않고 후속 집계나 민감도 분석에서 다시 확인합니다.
    WARNING = "WARNING"


@dataclass(frozen=True)
class ReasonPolicy:
    """하나의 사유가 재구매 및 금액 분석을 제외하는지 정의합니다."""

    level: ReasonLevel
    excludes_repurchase: bool
    excludes_monetary: bool


@dataclass(frozen=True)
class UciCleaningResult:
    """원본 행을 보존한 분류 결과와 행별 품질 사유를 함께 반환합니다."""

    rows: pd.DataFrame
    reasons: pd.DataFrame


REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "source_row_id",
    "source_period",
    "user_id",
    "order_id",
    "product_id",
    "product_name",
    "ordered_at",
    "quantity",
    "unit_price",
    "country",
    "is_explicit_cancellation",
)

DUPLICATE_COMPARISON_COLUMNS: Final[tuple[str, ...]] = (
    "source_period",
    "user_id",
    "order_id",
    "product_id",
    "product_name",
    "ordered_at",
    "quantity",
    "unit_price",
    "country",
)

# 실제 원본에서 상품이 아닌 배송비·수수료·할인·조정·테스트로 확인된 코드만
# 명시적으로 관리합니다. 문자 코드 전체를 제외하면 정상 상품까지 제거될 수 있습니다.
UCI_NON_MERCHANDISE_CODES: Final[frozenset[str]] = frozenset(
    {
        "ADJUST",
        "ADJUST2",
        "BANK CHARGES",
        "C2",
        "CRUK",
        "D",
        "DOT",
        "M",
        "POST",
        "TEST001",
        "TEST002",
    }
)

REASON_POLICIES: Final[dict[ReasonCode, ReasonPolicy]] = {
    ReasonCode.MISSING_USER_ID: ReasonPolicy(ReasonLevel.EXCLUDE, True, True),
    ReasonCode.MISSING_REQUIRED_EVENT_FIELD: ReasonPolicy(
        ReasonLevel.EXCLUDE,
        True,
        True,
    ),
    ReasonCode.EXPLICIT_CANCELLATION: ReasonPolicy(
        ReasonLevel.EXCLUDE,
        True,
        True,
    ),
    ReasonCode.NONPOSITIVE_QUANTITY: ReasonPolicy(
        ReasonLevel.EXCLUDE,
        True,
        True,
    ),
    ReasonCode.NONPOSITIVE_PRICE: ReasonPolicy(
        ReasonLevel.MONETARY_EXCLUDE,
        False,
        True,
    ),
    ReasonCode.NON_MERCHANDISE_CODE: ReasonPolicy(
        ReasonLevel.EXCLUDE,
        True,
        True,
    ),
    ReasonCode.MISSING_PRODUCT_NAME: ReasonPolicy(
        ReasonLevel.WARNING,
        False,
        False,
    ),
    ReasonCode.SUSPECTED_DUPLICATE: ReasonPolicy(
        ReasonLevel.WARNING,
        False,
        False,
    ),
}


def _validate_input(frame: pd.DataFrame) -> None:
    """필수 컬럼과 원본 행 식별자의 유일성을 분류 전에 검사합니다."""
    missing_columns = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise UciPreprocessingError(
            f"UCI 전처리 필수 컬럼이 누락됐습니다: {sorted(missing_columns)}"
        )

    if frame["source_row_id"].isna().any():
        raise UciPreprocessingError("source_row_id에는 결측값이 없어야 합니다.")
    if frame["source_row_id"].duplicated().any():
        raise UciPreprocessingError("source_row_id는 원본 행마다 고유해야 합니다.")


def _build_reason_rows(
    source_row_ids: pd.Series,
    mask: pd.Series,
    reason_code: ReasonCode,
) -> pd.DataFrame:
    """조건을 만족하는 원본 행 ID에 하나의 품질 사유와 적용 범위를 연결합니다."""
    policy = REASON_POLICIES[reason_code]
    matched_ids = source_row_ids.loc[mask.fillna(False)]
    return pd.DataFrame(
        {
            "source_row_id": matched_ids.astype("string"),
            "reason_code": reason_code.value,
            "reason_level": policy.level.value,
            "excludes_repurchase": policy.excludes_repurchase,
            "excludes_monetary": policy.excludes_monetary,
        }
    )


def classify_uci_rows(frame: pd.DataFrame) -> UciCleaningResult:
    """모든 UCI 행을 보존하면서 재구매·금액 분석 사용 가능 여부를 분류합니다."""
    _validate_input(frame)
    rows = frame.copy()

    required_event_fields = (
        rows["order_id"].notna()
        & rows["product_id"].notna()
        & rows["ordered_at"].notna()
        & rows["quantity"].notna()
    )
    is_suspected_duplicate = rows.duplicated(
        subset=list(DUPLICATE_COMPARISON_COLUMNS),
        keep=False,
    )

    reason_conditions = (
        (ReasonCode.MISSING_USER_ID, rows["user_id"].isna()),
        (ReasonCode.MISSING_REQUIRED_EVENT_FIELD, ~required_event_fields),
        (
            ReasonCode.EXPLICIT_CANCELLATION,
            rows["is_explicit_cancellation"].fillna(False),
        ),
        (ReasonCode.NONPOSITIVE_QUANTITY, rows["quantity"].le(0)),
        (ReasonCode.NONPOSITIVE_PRICE, rows["unit_price"].le(0)),
        (
            ReasonCode.NON_MERCHANDISE_CODE,
            rows["product_id"].isin(UCI_NON_MERCHANDISE_CODES),
        ),
        (ReasonCode.MISSING_PRODUCT_NAME, rows["product_name"].isna()),
        (ReasonCode.SUSPECTED_DUPLICATE, is_suspected_duplicate),
    )
    reason_frames = [
        _build_reason_rows(rows["source_row_id"], mask, reason_code)
        for reason_code, mask in reason_conditions
    ]
    reasons = pd.concat(reason_frames, ignore_index=True)

    repurchase_excluded_ids = reasons.loc[
        reasons["excludes_repurchase"],
        "source_row_id",
    ]
    monetary_excluded_ids = reasons.loc[
        reasons["excludes_monetary"],
        "source_row_id",
    ]
    rows["is_suspected_duplicate"] = is_suspected_duplicate
    rows["is_non_merchandise"] = rows["product_id"].isin(UCI_NON_MERCHANDISE_CODES)
    rows["is_accepted_repurchase"] = ~rows["source_row_id"].isin(
        repurchase_excluded_ids
    )
    rows["is_accepted_monetary"] = ~rows["source_row_id"].isin(monetary_excluded_ids)

    return UciCleaningResult(rows=rows, reasons=reasons)


def validate_uci_classification(
    source: pd.DataFrame,
    result: UciCleaningResult,
) -> dict[str, object]:
    """원본 행 보존과 사유-분류 일치 여부를 검사하고 감사 요약을 반환합니다."""
    rows = result.rows
    reasons = result.reasons
    required_result_columns = {
        "source_row_id",
        "is_accepted_repurchase",
        "is_accepted_monetary",
    }
    missing_result_columns = required_result_columns - set(rows.columns)
    if missing_result_columns:
        raise UciPreprocessingError(
            f"UCI 분류 결과 컬럼이 누락됐습니다: {sorted(missing_result_columns)}"
        )

    source_ids = set(source["source_row_id"])
    result_ids = set(rows["source_row_id"])
    reason_ids = set(reasons["source_row_id"])
    accepted_ids = set(rows.loc[rows["is_accepted_repurchase"], "source_row_id"])
    quarantined_ids = set(rows.loc[~rows["is_accepted_repurchase"], "source_row_id"])
    exclude_ids = set(
        reasons.loc[
            reasons["reason_level"].eq(ReasonLevel.EXCLUDE.value),
            "source_row_id",
        ]
    )

    accepted_count = int(rows["is_accepted_repurchase"].sum())
    quarantined_count = int((~rows["is_accepted_repurchase"]).sum())
    invariants = {
        "row_count_preserved": len(rows) == len(source),
        "source_ids_preserved": result_ids == source_ids,
        "accepted_plus_quarantined_equals_source": (
            accepted_count + quarantined_count == len(source)
        ),
        "quarantined_equals_exclude_reason_ids": quarantined_ids == exclude_ids,
        "accepted_has_no_exclude_reason": accepted_ids.isdisjoint(exclude_ids),
        "reason_ids_exist_in_source": reason_ids.issubset(source_ids),
    }
    failed_invariants = [name for name, passed in invariants.items() if not passed]
    if failed_invariants:
        raise UciPreprocessingError(
            f"UCI 전처리 불변조건을 위반했습니다: {failed_invariants}"
        )

    reasons_per_row = reasons.groupby("source_row_id", observed=True).size()
    return {
        "source_rows": int(len(source)),
        "classified_rows": int(len(rows)),
        "accepted_repurchase_rows": accepted_count,
        "accepted_repurchase_rate": float(rows["is_accepted_repurchase"].mean()),
        "quarantined_rows": quarantined_count,
        "quarantined_rate": float((~rows["is_accepted_repurchase"]).mean()),
        "accepted_monetary_rows": int(rows["is_accepted_monetary"].sum()),
        "accepted_monetary_rate": float(rows["is_accepted_monetary"].mean()),
        "reason_record_count": int(len(reasons)),
        "reason_counts": {
            str(key): int(value)
            for key, value in reasons["reason_code"].value_counts().items()
        },
        "reason_level_counts": {
            str(key): int(value)
            for key, value in reasons["reason_level"].value_counts().items()
        },
        "rows_with_multiple_reasons": int(reasons_per_row.gt(1).sum()),
        "maximum_reasons_on_one_row": (
            int(reasons_per_row.max()) if not reasons_per_row.empty else 0
        ),
        "invariants": invariants,
    }
