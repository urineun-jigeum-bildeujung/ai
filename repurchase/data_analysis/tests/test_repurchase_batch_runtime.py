"""재구매 배치 진입점의 성공 로그와 실행 전 거절 동작을 검증합니다."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_repurchase_batch import (
    CONTRACT_REJECTION_EXIT_CODE,
    SUCCESS_EXIT_CODE,
    main,
    run_contract_check,
)

CONTRACT_DIRECTORY = Path(__file__).parent / "fixtures" / "cloud_contract"
SOURCE_CONTRACT = CONTRACT_DIRECTORY / "source_orders.json"
PUBLICATION_CONTRACT = CONTRACT_DIRECTORY / "prediction_publications.json"


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        [
            "contract-check",
            "--source-contract",
            str(SOURCE_CONTRACT),
        ],
    ],
    ids=["missing-command", "missing-subcommand-option"],
)
def test_cli_rejects_syntax_errors_as_one_json_record(
    arguments: list[str], capsys: object
) -> None:
    """최상위·하위 명령 문법 오류도 동일한 JSON 거절 계약을 따릅니다."""
    exit_code = main(arguments)

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == CONTRACT_REJECTION_EXIT_CODE
    assert captured.out == ""
    assert len(captured.err.splitlines()) == 1
    assert payload["event"] == "repurchase_batch_rejected"
    assert payload["status"] == "REJECTED"
    assert payload["error_type"] == "BatchRuntimeError"


def test_contract_check_reports_reproducible_stage_counts() -> None:
    """원천 변환과 결과 발행 단계가 기대한 행 수와 입력 지문을 남깁니다."""
    summary = run_contract_check(SOURCE_CONTRACT, PUBLICATION_CONTRACT)

    assert summary.schema_version == 1
    assert summary.mode == "contract-check"
    assert len(summary.source_contract_sha256) == 64
    assert len(summary.publication_contract_sha256) == 64
    assert summary.source_order_count == 3
    assert summary.source_order_item_count == 6
    assert summary.valid_order_item_count == 4
    assert summary.purchase_event_count == 3
    assert summary.publication_batch_count == 5
    assert summary.publication_result_count == 8
    assert summary.latest_visible_result_count == 2


def test_cli_writes_one_json_success_record(capsys: object) -> None:
    """로컬과 컨테이너가 같은 성공 JSON 한 줄과 종료 코드 0을 사용합니다."""
    exit_code = main(
        [
            "contract-check",
            "--source-contract",
            str(SOURCE_CONTRACT),
            "--publication-contract",
            str(PUBLICATION_CONTRACT),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == SUCCESS_EXIT_CODE
    assert captured.err == ""
    assert payload["event"] == "repurchase_batch_contract_checked"
    assert payload["status"] == "SUCCEEDED"
    assert payload["latest_visible_result_count"] == 2


def test_cli_rejects_missing_external_input_before_work(capsys: object) -> None:
    """이미지에 입력을 내장하지 않고 누락된 외부 파일을 명확하게 거절합니다."""
    exit_code = main(
        [
            "contract-check",
            "--source-contract",
            "missing-source.json",
            "--publication-contract",
            str(PUBLICATION_CONTRACT),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == CONTRACT_REJECTION_EXIT_CODE
    assert captured.out == ""
    assert payload["event"] == "repurchase_batch_rejected"
    assert payload["status"] == "REJECTED"
    assert payload["error_type"] == "BatchRuntimeError"


def test_cli_rejects_contract_result_mismatch(tmp_path: Path, capsys: object) -> None:
    """예상 결과를 임의로 바꾼 계약 파일은 성공 로그를 만들지 못합니다."""
    source = json.loads(SOURCE_CONTRACT.read_text(encoding="utf-8"))
    source["expected_purchase_events"][0]["net_unit_count"] = 999
    changed_source = tmp_path / "changed-source.json"
    changed_source.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")

    exit_code = main(
        [
            "contract-check",
            "--source-contract",
            str(changed_source),
            "--publication-contract",
            str(PUBLICATION_CONTRACT),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == CONTRACT_REJECTION_EXIT_CODE
    assert captured.out == ""
    assert payload["status"] == "REJECTED"
    assert "일치하지 않습니다" in payload["message"]
