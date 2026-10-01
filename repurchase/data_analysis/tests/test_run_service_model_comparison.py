"""서비스 모델 비교 CLI의 원천 CSV 읽기 계약을 확인합니다."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from pandas.api.types import is_bool_dtype

from scripts import run_service_model_comparison as runner
from scripts.run_service_model_comparison import (
    _file_sha256,
    _read_sources,
    _require_finite_c_index,
)


def test_read_sources_preserves_ids_and_postgres_boolean(tmp_path: Path) -> None:
    """PostgreSQL ID의 문자열과 t/f boolean이 변환 중 손실되지 않습니다."""
    source = tmp_path / "order-items.csv"
    source.write_text(
        "order_item_id,order_id,product_id,product_group_id_snapshot,pet_id,is_replenishable_snapshot\n"
        "001,010,123,456,,t\n"
        "002,011,124,457,999,f\n",
        encoding="utf-8",
    )

    rows = _read_sources({"order_items": source})["order_items"]

    assert rows["order_item_id"].tolist() == ["001", "002"]
    assert rows["order_id"].tolist() == ["010", "011"]
    assert rows["pet_id"].isna().tolist() == [True, False]
    assert is_bool_dtype(rows["is_replenishable_snapshot"])
    assert rows["is_replenishable_snapshot"].tolist() == [True, False]
    assert len(_file_sha256(source)) == 64


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_c_index_is_rejected(score: float) -> None:
    """비유한 C-index는 JSON의 결측값이나 문자열로 저장하지 않습니다."""
    with pytest.raises(ValueError, match="C-index"):
        _require_finite_c_index(pd.DataFrame({"ipcw_c_index": [score]}))


@pytest.mark.parametrize(
    ("observation_end", "replicates"),
    [("2026-09-29T15:44:00", "1000"), ("2026-09-29T15:44:00+09:00", "0")],
)
def test_cli_rejects_invalid_options_before_reading_csv(
    monkeypatch: pytest.MonkeyPatch, observation_end: str, replicates: str
) -> None:
    """시간대와 반복 횟수 오류는 원천 파일 접근 전에 종료됩니다."""
    arguments = ["run_service_model_comparison"]
    for name in (
        "orders",
        "order-items",
        "pets",
        "histories",
        "claims",
        "claim-items",
    ):
        arguments.extend((f"--{name}", "/not-a-real-source.csv"))
    arguments.extend(
        ("--observation-end-at", observation_end, "--bootstrap-replicates", replicates)
    )
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(SystemExit) as exc:
        runner.main()

    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("observation_end", "replicates"),
    [("2026-09-29T15:44:00", 1), ("2026-09-29T15:44:00+09:00", 0)],
)
def test_library_entry_rejects_invalid_options_before_reading_csv(
    observation_end: str, replicates: int
) -> None:
    """직접 함수 호출에서도 잘못된 설정은 같은 단계에서 거절됩니다."""
    with pytest.raises(ValueError):
        runner.run_comparison(
            {},
            observation_end_at=pd.Timestamp(observation_end),
            bootstrap_replicates=replicates,
        )


def test_cli_does_not_write_nonfinite_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """비유한 지표가 들어오면 결과 파일을 만들기 전에 직렬화를 거절합니다."""
    arguments = ["run_service_model_comparison"]
    for name in (
        "orders",
        "order-items",
        "pets",
        "histories",
        "claims",
        "claim-items",
    ):
        arguments.extend((f"--{name}", "/unused.csv"))
    output = tmp_path / "invalid-result.json"
    arguments.extend(
        (
            "--observation-end-at",
            "2026-09-29T15:44:00+09:00",
            "--output",
            str(output),
        )
    )
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(
        runner,
        "run_comparison",
        lambda paths, **kwargs: {"summary": [{"ipcw_c_index": float("nan")}]},
    )

    with pytest.raises(ValueError, match="Out of range float"):
        runner.main()

    assert not output.exists()
