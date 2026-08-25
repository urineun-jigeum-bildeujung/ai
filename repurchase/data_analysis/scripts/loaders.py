"""서로 다른 공개 거래 데이터셋을 하나의 공통 스키마로 변환합니다.

UCI와 Complete Journey의 필드명을 통일하고 결측 사용자, 취소 주문,
0 이하 수량·금액을 품질 플래그로 표시합니다. 원본 행은 삭제하지 않으며,
실제 분석 단계에서 유효 구매 후보를 명시적으로 선택할 수 있게 합니다.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import pyreadr

from .dataset_registry import (
    COMPLETE_JOURNEY_PRODUCTS,
    COMPLETE_JOURNEY_TRANSACTIONS,
    UCI_ONLINE_RETAIL_II,
)
from .paths import RAW_DIR


class DatasetLoadError(RuntimeError):
    """원천 데이터의 구조가 분석 코드의 기대와 다를 때 발생하는 오류입니다."""


CANONICAL_COLUMNS = (
    "dataset",
    "source_row_id",
    "source_period",
    "user_id",
    "order_id",
    "product_id",
    "category_id",
    "product_name",
    "ordered_at",
    "quantity",
    "unit_price",
    "line_amount",
    "country",
    "is_user_missing",
    "is_explicit_cancellation",
    "is_nonpositive_quantity",
    "is_nonpositive_amount",
    "is_valid_repurchase_candidate",
    "is_valid_monetary_candidate",
)

UCI_REQUIRED_COLUMNS = (
    "Customer ID",
    "Invoice",
    "StockCode",
    "Description",
    "InvoiceDate",
    "Quantity",
    "Price",
    "Country",
)

COMPLETE_JOURNEY_TRANSACTION_COLUMNS = (
    "household_id",
    "basket_id",
    "product_id",
    "transaction_timestamp",
    "quantity",
    "sales_value",
    "week",
)

COMPLETE_JOURNEY_PRODUCT_COLUMNS = (
    "product_id",
    "product_category",
)


def _string_id(series: pd.Series) -> pd.Series:
    """숫자·문자 혼합 식별자를 변환하고 불필요한 '.0' 접미사를 제거합니다."""
    return series.astype("string").str.replace(r"\.0$", "", regex=True)


def _validate_source_columns(
    frame: pd.DataFrame,
    required_columns: tuple[str, ...],
    source_name: str,
) -> None:
    """원천 파일에 분석에 필요한 컬럼이 모두 있는지 변환 전에 검사합니다."""
    missing = set(required_columns) - set(frame.columns)
    if missing:
        raise DatasetLoadError(
            f"{source_name}에 필요한 원천 컬럼이 누락됐습니다: {sorted(missing)}"
        )


def _validate_canonical(frame: pd.DataFrame) -> pd.DataFrame:
    """공통 스키마의 필수 필드를 검사하고 정해진 순서로 반환합니다."""
    missing = set(CANONICAL_COLUMNS) - set(frame.columns)
    if missing:
        raise DatasetLoadError(f"공통 스키마 필드가 누락됐습니다: {sorted(missing)}")
    return frame.loc[:, CANONICAL_COLUMNS]


def _add_candidate_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """재구매 행동과 금액 분석의 유효 후보 조건을 서로 분리해 표시합니다."""
    has_required_event_fields = (
        frame["user_id"].notna()
        & frame["order_id"].notna()
        & frame["product_id"].notna()
        & frame["ordered_at"].notna()
        & frame["quantity"].gt(0)
    )
    frame["is_valid_repurchase_candidate"] = (
        has_required_event_fields & ~frame["is_explicit_cancellation"]
    )
    frame["is_valid_monetary_candidate"] = frame[
        "is_valid_repurchase_candidate"
    ] & frame["line_amount"].gt(0)
    return frame


def _read_uci_sheets_from_archive(archive_path: Path) -> dict[str, pd.DataFrame]:
    """UCI 워크북을 임시로 추출해 읽고 임시 파일은 자동으로 정리합니다."""
    member = UCI_ONLINE_RETAIL_II.archive_member
    if member is None:
        raise DatasetLoadError("UCI 압축 파일의 내부 파일명이 설정되지 않았습니다.")

    with tempfile.TemporaryDirectory(prefix="uci-online-retail-") as temp_dir:
        workbook_path = Path(temp_dir) / Path(member).name
        with zipfile.ZipFile(archive_path) as zip_file:
            with zip_file.open(member) as source:
                with workbook_path.open("wb") as output:
                    shutil.copyfileobj(source, output)
        return pd.read_excel(workbook_path, sheet_name=None)


def load_uci_online_retail_ii(
    workbook_path: Path | None = None,
    archive_path: Path | None = None,
) -> pd.DataFrame:
    """UCI 워크북의 두 시트를 읽고 취소·품질 플래그를 보존합니다."""
    if workbook_path is not None and archive_path is not None:
        raise DatasetLoadError(
            "workbook_path와 archive_path는 동시에 지정할 수 없습니다."
        )

    if workbook_path is not None:
        sheets = pd.read_excel(workbook_path, sheet_name=None)
    else:
        if archive_path is None:
            archive_path = RAW_DIR / UCI_ONLINE_RETAIL_II.filename
        sheets = _read_uci_sheets_from_archive(archive_path)

    if not sheets:
        raise DatasetLoadError("UCI 워크북에 읽을 수 있는 시트가 없습니다.")

    frames: list[pd.DataFrame] = []
    for sheet_name, source in sheets.items():
        _validate_source_columns(
            source,
            UCI_REQUIRED_COLUMNS,
            f"UCI 시트 '{sheet_name}'",
        )
        frame = source.rename(
            columns={
                "Customer ID": "user_id",
                "Invoice": "order_id",
                "StockCode": "product_id",
                "Description": "product_name",
                "InvoiceDate": "ordered_at",
                "Quantity": "quantity",
                "Price": "unit_price",
                "Country": "country",
            }
        ).copy()

        frame["dataset"] = UCI_ONLINE_RETAIL_II.dataset
        frame["source_period"] = sheet_name
        frame["source_row_id"] = [f"{sheet_name}:{index}" for index in frame.index]
        frame["user_id"] = _string_id(frame["user_id"])
        frame["order_id"] = _string_id(frame["order_id"])
        frame["product_id"] = _string_id(frame["product_id"])
        frame["category_id"] = pd.Series(pd.NA, index=frame.index, dtype="string")
        frame["ordered_at"] = pd.to_datetime(frame["ordered_at"], errors="coerce")
        frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce")
        frame["unit_price"] = pd.to_numeric(frame["unit_price"], errors="coerce")
        frame["line_amount"] = frame["quantity"] * frame["unit_price"]

        frame["is_user_missing"] = frame["user_id"].isna()
        frame["is_explicit_cancellation"] = frame["order_id"].str.startswith(
            "C", na=False
        )
        frame["is_nonpositive_quantity"] = frame["quantity"].le(0)
        frame["is_nonpositive_amount"] = frame["line_amount"].le(0)
        frame = _add_candidate_flags(frame)
        frames.append(_validate_canonical(frame))

    return pd.concat(frames, ignore_index=True)


def _read_single_r_frame(path: Path) -> pd.DataFrame:
    """R 데이터 파일에서 단일 데이터프레임을 읽어 반환합니다."""
    objects = pyreadr.read_r(path)
    if len(objects) != 1:
        raise DatasetLoadError(
            f"{path.name}에는 R 객체가 1개 있어야 하지만 "
            f"다음 객체가 발견됐습니다: {list(objects)}"
        )
    frame = next(iter(objects.values()))
    if not isinstance(frame, pd.DataFrame):
        raise DatasetLoadError(
            f"{path.name}의 단일 R 객체가 DataFrame이 아닙니다: {type(frame).__name__}"
        )
    return frame


def load_complete_journey(
    transactions_path: Path | None = None,
    products_path: Path | None = None,
) -> pd.DataFrame:
    """Complete Journey 거래를 읽고 상품 카테고리를 연결합니다."""
    if transactions_path is None:
        transactions_path = RAW_DIR / COMPLETE_JOURNEY_TRANSACTIONS.filename
    if products_path is None:
        products_path = RAW_DIR / COMPLETE_JOURNEY_PRODUCTS.filename

    transactions = _read_single_r_frame(transactions_path)
    product_source = _read_single_r_frame(products_path)
    _validate_source_columns(
        transactions,
        COMPLETE_JOURNEY_TRANSACTION_COLUMNS,
        "Complete Journey 거래",
    )
    _validate_source_columns(
        product_source,
        COMPLETE_JOURNEY_PRODUCT_COLUMNS,
        "Complete Journey 상품 마스터",
    )
    products = product_source.loc[:, list(COMPLETE_JOURNEY_PRODUCT_COLUMNS)].rename(
        columns={"product_category": "category_id"}
    )
    products["product_id"] = _string_id(products["product_id"])

    frame = transactions.rename(
        columns={
            "household_id": "user_id",
            "basket_id": "order_id",
            "transaction_timestamp": "ordered_at",
            "sales_value": "line_amount",
        }
    ).copy()
    frame["user_id"] = _string_id(frame["user_id"])
    frame["order_id"] = _string_id(frame["order_id"])
    frame["product_id"] = _string_id(frame["product_id"])
    frame = frame.merge(products, on="product_id", how="left", validate="many_to_one")

    frame["dataset"] = COMPLETE_JOURNEY_TRANSACTIONS.dataset
    frame["source_period"] = frame["week"].astype("string").radd("week_")
    frame["source_row_id"] = frame.index.map(lambda index: f"transaction:{index}")
    frame["product_name"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame["country"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame["ordered_at"] = pd.to_datetime(frame["ordered_at"], errors="coerce")
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce")
    frame["line_amount"] = pd.to_numeric(frame["line_amount"], errors="coerce")
    frame["unit_price"] = (
        frame["line_amount"].div(frame["quantity"]).where(frame["quantity"].ne(0))
    )

    frame["is_user_missing"] = frame["user_id"].isna()
    frame["is_explicit_cancellation"] = False
    frame["is_nonpositive_quantity"] = frame["quantity"].le(0)
    frame["is_nonpositive_amount"] = frame["line_amount"].le(0)
    frame = _add_candidate_flags(frame)

    return _validate_canonical(frame)
