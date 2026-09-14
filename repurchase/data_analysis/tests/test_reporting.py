"""보고서 변환과 저장 경계에서 데이터 의미가 보존되는지 검증합니다."""

from __future__ import annotations

import json

import pandas as pd

from scripts.reporting import dataframe_to_nullable_records


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
