"""UCI 전처리가 원본 행과 제외 사유를 손실 없이 보존하는지 검증합니다.

실제 파일 전체를 매번 읽지 않고, 품질 문제가 의도적으로 섞인 작은 표본으로
각 규칙의 동작과 여러 사유의 동시 기록을 빠르게 검사합니다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.preprocessing.uci import (
    ReasonCode,
    ReasonLevel,
    UciCleaningResult,
    UciPreprocessingError,
    classify_uci_rows,
    validate_uci_classification,
)


def make_uci_fixture() -> pd.DataFrame:
    """정상·결측·취소·비상품·0원·중복 사례를 포함한 표본을 만듭니다."""
    return pd.DataFrame(
        {
            "source_row_id": ["r1", "r2", "r3", "r4", "r5", "r6", "r7"],
            "source_period": ["Year 2010-2011"] * 7,
            "user_id": ["u1", pd.NA, "u2", "u3", "u4", "u5", "u5"],
            "order_id": ["o1", "o2", "C3", "o4", "o5", "o6", "o6"],
            "product_id": ["p1", "p2", "p3", "POST", "p5", "p6", "p6"],
            "product_name": [
                "상품 1",
                "상품 2",
                "상품 3",
                "POSTAGE",
                "증정 상품",
                "중복 후보",
                "중복 후보",
            ],
            "ordered_at": pd.to_datetime(["2026-01-01"] * 7),
            "quantity": [1, 1, -1, 1, 1, 2, 2],
            "unit_price": [10.0, 10.0, 10.0, 5.0, 0.0, 3.0, 3.0],
            "line_amount": [10.0, 10.0, -10.0, 5.0, 0.0, 6.0, 6.0],
            "country": ["United Kingdom"] * 7,
            "is_explicit_cancellation": [
                False,
                False,
                True,
                False,
                False,
                False,
                False,
            ],
        }
    )


def test_classification_preserves_every_source_row() -> None:
    """분류 전후 원본 행 수와 행 식별자가 그대로 보존되는지 확인합니다."""
    source = make_uci_fixture()

    result = classify_uci_rows(source)

    assert len(result.rows) == len(source)
    assert set(result.rows["source_row_id"]) == set(source["source_row_id"])


def test_exclusion_and_warning_scopes_are_separated() -> None:
    """재구매 제외·금액만 제외·경고가 서로 다른 범위로 적용되는지 확인합니다."""
    result = classify_uci_rows(make_uci_fixture())
    rows = result.rows.set_index("source_row_id")

    assert bool(rows.loc["r1", "is_accepted_repurchase"])
    assert not bool(rows.loc["r2", "is_accepted_repurchase"])
    assert not bool(rows.loc["r3", "is_accepted_repurchase"])
    assert not bool(rows.loc["r4", "is_accepted_repurchase"])

    assert bool(rows.loc["r5", "is_accepted_repurchase"])
    assert not bool(rows.loc["r5", "is_accepted_monetary"])

    assert bool(rows.loc["r6", "is_accepted_repurchase"])
    assert bool(rows.loc["r7", "is_accepted_repurchase"])
    assert bool(rows.loc["r6", "is_suspected_duplicate"])
    assert bool(rows.loc["r7", "is_suspected_duplicate"])


def test_one_row_can_keep_multiple_quality_reasons() -> None:
    """취소이면서 음수 수량인 한 행에 두 사유가 모두 기록되는지 확인합니다."""
    result = classify_uci_rows(make_uci_fixture())

    reason_codes = set(
        result.reasons.loc[
            result.reasons["source_row_id"].eq("r3"),
            "reason_code",
        ]
    )

    assert reason_codes == {
        ReasonCode.EXPLICIT_CANCELLATION.value,
        ReasonCode.NONPOSITIVE_QUANTITY.value,
    }


def test_zero_price_is_monetary_exclusion_not_repurchase_exclusion() -> None:
    """0원 상품은 시점 행동에는 남기고 금액 분석에서만 제외하는지 확인합니다."""
    result = classify_uci_rows(make_uci_fixture())
    zero_price_reason = result.reasons.loc[
        result.reasons["source_row_id"].eq("r5")
        & result.reasons["reason_code"].eq(ReasonCode.NONPOSITIVE_PRICE.value)
    ].iloc[0]

    assert zero_price_reason["reason_level"] == ReasonLevel.MONETARY_EXCLUDE.value
    assert not bool(zero_price_reason["excludes_repurchase"])
    assert bool(zero_price_reason["excludes_monetary"])


def test_duplicate_source_row_id_is_rejected() -> None:
    """원본 행 ID가 중복되면 사유를 정확히 연결할 수 없으므로 중단합니다."""
    source = make_uci_fixture()
    source.loc[1, "source_row_id"] = "r1"

    with pytest.raises(UciPreprocessingError, match="고유해야"):
        classify_uci_rows(source)


def test_validation_detects_classified_row_loss() -> None:
    """분류 결과에서 원본 행이 사라지면 전체 데이터 검증이 실패하는지 확인합니다."""
    source = make_uci_fixture()
    result = classify_uci_rows(source)
    damaged_result = UciCleaningResult(
        rows=result.rows.iloc[:-1].copy(),
        reasons=result.reasons.copy(),
    )

    with pytest.raises(UciPreprocessingError, match="불변조건"):
        validate_uci_classification(source, damaged_result)
