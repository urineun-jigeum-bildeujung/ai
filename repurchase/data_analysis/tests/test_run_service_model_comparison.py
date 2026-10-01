"""서비스 모델 비교 CLI의 원천 CSV 읽기 계약을 확인합니다."""

from __future__ import annotations

from pathlib import Path

from pandas.api.types import is_bool_dtype

from scripts.run_service_model_comparison import _file_sha256, _read_sources


def test_read_sources_preserves_ids_and_postgres_boolean(tmp_path: Path) -> None:
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
