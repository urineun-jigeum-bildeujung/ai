"""보고서 변환과 저장 경계에서 데이터 의미가 보존되는지 검증합니다."""

from __future__ import annotations

import json
from gzip import compress, decompress
from pathlib import Path

import pandas as pd

from scripts.reporting import dataframe_to_nullable_records, write_bytes_atomically


def test_dataframe_to_nullable_records_converts_nan_to_json_null() -> None:
    """계산 불가 숫자를 0이 아닌 표준 JSON null로 변환합니다."""
    frame = pd.DataFrame(
        {
            "anchor_month": ["2026-07", "2026-08"],
            "mae_days": [6.0, float("nan")],
        }
    )

    records = dataframe_to_nullable_records(frame)
    encoded = json.dumps(records, allow_nan=False)

    assert records == [
        {"anchor_month": "2026-07", "mae_days": 6.0},
        {"anchor_month": "2026-08", "mae_days": None},
    ]
    assert '"mae_days": null' in encoded
    assert pd.isna(frame.loc[1, "mae_days"])


def test_write_bytes_atomically_keeps_compressed_report_readable(
    tmp_path: Path,
) -> None:
    """압축 보고서를 다시 읽을 수 있고 임시 파일이 남지 않습니다."""
    path = tmp_path / "bootstrap_trials.json.gz"
    content = compress(b'{"trials":[{"score":0.1}]}\n', mtime=0)

    write_bytes_atomically(path, content)

    assert decompress(path.read_bytes()) == b'{"trials":[{"score":0.1}]}\n'
    assert not path.with_suffix(".gz.part").exists()
