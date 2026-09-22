"""재구매 배치 컨테이너가 호출하는 단일 실행 진입점입니다.

현재는 실제 클라우드 DB나 서비스 모델을 연결하지 않습니다. 고정된 원천·결과
계약 파일을 외부 경로로 받아 전체 계약을 검증하고 구조화된 JSON 로그를 남깁니다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NoReturn, cast

import pandas as pd

from scripts.modeling.operational_orders import (
    OperationalOrderError,
    build_valid_order_items,
)
from scripts.modeling.prediction_publications import (
    PredictionPublicationError,
    select_latest_published_predictions,
    validate_prediction_publications,
)
from scripts.modeling.purchase_events import build_product_group_purchase_events

SUCCESS_EXIT_CODE = 0
UNEXPECTED_FAILURE_EXIT_CODE = 1
CONTRACT_REJECTION_EXIT_CODE = 2
CONTRACT_CHECK_SCHEMA_VERSION = 1


class BatchRuntimeError(ValueError):
    """배치가 작업을 시작하기 전에 입력·설정 계약을 거절할 때 발생합니다."""


class _BatchArgumentParser(argparse.ArgumentParser):
    """CLI 문법 오류를 구조화된 배치 계약 오류로 변환합니다."""

    def error(self, message: str) -> NoReturn:
        """기본 사용법 출력과 즉시 종료 대신 호출자가 처리할 오류를 발생시킵니다."""
        raise BatchRuntimeError(message)


@dataclass(frozen=True)
class ContractCheckSummary:
    """계약 점검의 입력 지문과 단계별 처리 건수를 기록합니다."""

    schema_version: int
    mode: str
    source_contract_sha256: str
    publication_contract_sha256: str
    source_order_count: int
    source_order_item_count: int
    valid_order_item_count: int
    purchase_event_count: int
    publication_batch_count: int
    publication_result_count: int
    latest_visible_result_count: int


def _sha256(path: Path) -> str:
    """큰 입력도 한 번에 메모리에 올리지 않고 SHA-256 지문을 계산합니다."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise BatchRuntimeError(f"입력 파일을 읽을 수 없습니다: {path}") from error
    return digest.hexdigest()


def _load_json_object(path: Path, *, name: str) -> dict[str, object]:
    """외부 JSON 파일을 읽고 최상위 객체가 아니면 실행 전에 거절합니다."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BatchRuntimeError(f"{name} JSON을 읽을 수 없습니다: {path}") from error
    if not isinstance(payload, dict):
        raise BatchRuntimeError(f"{name} JSON의 최상위 값은 객체여야 합니다.")
    return cast(dict[str, object], payload)


def _require_exact_keys(
    payload: dict[str, object], expected: frozenset[str], *, name: str
) -> None:
    """계약 파일의 필수 영역 누락과 합의되지 않은 영역 유입을 차단합니다."""
    missing = expected - set(payload)
    unexpected = set(payload) - expected
    if missing:
        raise BatchRuntimeError(f"{name} 영역이 누락됐습니다: {sorted(missing)}")
    if unexpected:
        raise BatchRuntimeError(
            f"{name}에 지원하지 않는 영역이 있습니다: {sorted(unexpected)}"
        )


def _as_rows(payload: dict[str, object], name: str) -> pd.DataFrame:
    """JSON 객체 배열을 DataFrame으로 바꾸고 다른 구조는 거절합니다."""
    value = payload.get(name)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise BatchRuntimeError(f"{name}은 JSON 객체 배열이어야 합니다.")
    return pd.DataFrame(cast(list[dict[str, object]], value))


def _normalize_json_records(rows: pd.DataFrame) -> list[dict[str, object]]:
    """pandas 시각·결측 표현을 언어 중립적인 JSON 값으로 정규화합니다."""
    normalized = rows.copy()
    if "paid_at" in normalized:
        normalized["paid_at"] = normalized["paid_at"].map(
            lambda value: value.isoformat().replace("+00:00", "Z")
        )
    normalized = normalized.astype(object).where(normalized.notna(), None)
    return normalized.to_dict(orient="records")


def _require_expected_records(
    actual: pd.DataFrame,
    expected: object,
    *,
    name: str,
) -> None:
    """실제 변환 결과가 팀과 공유한 언어 중립 계약 예제와 같은지 확인합니다."""
    if not isinstance(expected, list) or not all(
        isinstance(row, dict) for row in expected
    ):
        raise BatchRuntimeError(f"{name}은 JSON 객체 배열이어야 합니다.")
    if _normalize_json_records(actual) != expected:
        raise BatchRuntimeError(f"{name}과 실제 변환 결과가 일치하지 않습니다.")


def run_contract_check(
    source_contract_path: Path,
    publication_contract_path: Path,
) -> ContractCheckSummary:
    """외부 계약 예제를 읽어 원천 변환과 결과 발행 경계를 끝까지 검증합니다."""
    source_path = Path(source_contract_path)
    publication_path = Path(publication_contract_path)
    source = _load_json_object(source_path, name="원천 계약")
    publication = _load_json_object(publication_path, name="결과 발행 계약")
    _require_exact_keys(
        source,
        frozenset(
            {
                "orders",
                "order_items",
                "expected_valid_order_items",
                "expected_purchase_events",
            }
        ),
        name="원천 계약",
    )
    _require_exact_keys(
        publication,
        frozenset({"batches", "results"}),
        name="결과 발행 계약",
    )

    orders = _as_rows(source, "orders")
    order_items = _as_rows(source, "order_items")
    valid_items = build_valid_order_items(orders, order_items)
    purchase_events = build_product_group_purchase_events(orders, order_items)
    _require_expected_records(
        valid_items,
        source["expected_valid_order_items"],
        name="expected_valid_order_items",
    )
    _require_expected_records(
        purchase_events,
        source["expected_purchase_events"],
        name="expected_purchase_events",
    )

    batches = _as_rows(publication, "batches")
    results = _as_rows(publication, "results")
    normalized_batches, normalized_results = validate_prediction_publications(
        batches, results
    )
    latest = select_latest_published_predictions(normalized_batches, normalized_results)
    return ContractCheckSummary(
        schema_version=CONTRACT_CHECK_SCHEMA_VERSION,
        mode="contract-check",
        source_contract_sha256=_sha256(source_path),
        publication_contract_sha256=_sha256(publication_path),
        source_order_count=len(orders),
        source_order_item_count=len(order_items),
        valid_order_item_count=len(valid_items),
        purchase_event_count=len(purchase_events),
        publication_batch_count=len(normalized_batches),
        publication_result_count=len(normalized_results),
        latest_visible_result_count=len(latest),
    )


def _build_parser() -> _BatchArgumentParser:
    """현재 지원하는 배치 명령과 필수 외부 파일 인자를 정의합니다."""
    parser = _BatchArgumentParser(prog="repurchase-batch")
    commands = parser.add_subparsers(
        dest="command",
        required=True,
        parser_class=_BatchArgumentParser,
    )
    contract_check = commands.add_parser(
        "contract-check",
        help="실제 DB 쓰기 없이 고정 입력·결과 발행 계약을 검증합니다.",
    )
    contract_check.add_argument(
        "--source-contract",
        type=Path,
        required=True,
        help="orders와 order_items를 포함한 원천 계약 JSON 경로",
    )
    contract_check.add_argument(
        "--publication-contract",
        type=Path,
        required=True,
        help="발행 배치와 결과를 포함한 계약 JSON 경로",
    )
    return parser


def _write_json(payload: dict[str, object], *, stream: object) -> None:
    """로그 수집기가 한 줄씩 읽을 수 있도록 표준 JSON 한 줄을 기록합니다."""
    print(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False),
        file=stream,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """명령을 실행하고 운영 환경이 구분할 수 있는 종료 코드를 반환합니다."""
    try:
        arguments = _build_parser().parse_args(argv)
        if arguments.command != "contract-check":
            raise BatchRuntimeError(f"지원하지 않는 명령입니다: {arguments.command}")
        summary = run_contract_check(
            arguments.source_contract,
            arguments.publication_contract,
        )
    except (
        BatchRuntimeError,
        OperationalOrderError,
        PredictionPublicationError,
    ) as error:
        _write_json(
            {
                "event": "repurchase_batch_rejected",
                "status": "REJECTED",
                "error_type": type(error).__name__,
                "message": str(error),
            },
            stream=sys.stderr,
        )
        return CONTRACT_REJECTION_EXIT_CODE
    except (
        Exception
    ) as error:  # pragma: no cover - 마지막 안전망은 통합 테스트로 검증합니다.
        _write_json(
            {
                "event": "repurchase_batch_failed",
                "status": "FAILED",
                "error_type": type(error).__name__,
                "message": str(error),
            },
            stream=sys.stderr,
        )
        return UNEXPECTED_FAILURE_EXIT_CODE

    _write_json(
        {
            "event": "repurchase_batch_contract_checked",
            "status": "SUCCEEDED",
            **asdict(summary),
        },
        stream=sys.stdout,
    )
    return SUCCESS_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
