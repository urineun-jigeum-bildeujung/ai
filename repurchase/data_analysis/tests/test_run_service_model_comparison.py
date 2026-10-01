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
    _summarize_validation_population,
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


@pytest.mark.parametrize(
    ("train_fraction", "validation_fraction"),
    [(0, 0.7), (0.7, 0.7), (0.8, 0.7), (0.7, 1), (float("nan"), 0.8)],
)
def test_library_rejects_invalid_time_cut_fractions_before_reading_csv(
    train_fraction: float, validation_fraction: float
) -> None:
    """시간 컷이 역전되거나 유효 범위를 벗어나면 원천 조회 전에 거절합니다."""
    with pytest.raises(ValueError, match="시간 컷 비율"):
        runner.run_comparison(
            {},
            observation_end_at=pd.Timestamp("2026-09-29T15:44:00+09:00"),
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )


def test_cli_rejects_invalid_time_cuts_before_reading_csv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI도 잘못된 시간 컷을 파일 접근 전에 종료 코드 2로 거절합니다."""
    arguments = ["run_service_model_comparison"]
    for name in ("orders", "order-items", "pets", "histories", "claims", "claim-items"):
        arguments.extend((f"--{name}", "/not-a-real-source.csv"))
    arguments.extend(
        (
            "--observation-end-at",
            "2026-09-29T15:44:00+09:00",
            "--train-fraction",
            "0.85",
            "--validation-fraction",
            "0.70",
        )
    )
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(SystemExit) as exc:
        runner.main()

    assert exc.value.code == 2


def test_validation_population_counts_each_history_bucket_once() -> None:
    """구매 간격 0·1·2회 이상 표본 수가 전체 Validation과 일치합니다."""
    rows = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u3"],
            "target_id": ["p1", "p1", "p2", "p3"],
            "history_interval_count": [0, 1, 2, 4],
        }
    )

    assert _summarize_validation_population(rows) == {
        "user_count": 3,
        "product_group_count": 3,
        "history_interval_count_0": 1,
        "history_interval_count_1": 1,
        "history_interval_count_2_or_more": 2,
    }


@pytest.mark.parametrize("invalid", [None, -1, 1.5, float("inf")])
def test_validation_population_rejects_invalid_history_count(invalid: object) -> None:
    """이력량 결측·음수·소수·무한대가 분포에서 조용히 빠지지 않습니다."""
    rows = pd.DataFrame(
        {
            "user_id": ["u1"],
            "target_id": ["p1"],
            "history_interval_count": [invalid],
        }
    )

    with pytest.raises(ValueError, match="과거 구매 간격 수"):
        _summarize_validation_population(rows)
