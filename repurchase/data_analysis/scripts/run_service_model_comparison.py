"""읽기 전용 CSV 추출본에서 서비스 모델의 Validation 비교를 재현합니다."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from scripts.modeling.operational_event_intervals import (
    build_operational_event_intervals,
)
from scripts.modeling.operational_quantity_intervals import (
    build_order_item_quantity_intervals,
)
from scripts.modeling.operational_status_intervals import build_order_status_intervals
from scripts.modeling.operational_temporal_split import (
    build_service_train_validation_split,
)
from scripts.modeling.operational_training_samples import (
    TEMPORAL_SERVICE_FEATURE_GENERATION_VERSION,
)
from scripts.modeling.operational_validity_intervals import (
    build_valid_purchase_item_intervals,
)
from scripts.modeling.service_model_comparison import compare_service_aft_lightgbm


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_sources(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    """ID를 문자열로 읽어 데이터베이스 bigint의 정밀도와 결측을 보존합니다."""
    id_columns = {
        "orders": ("order_id", "user_id"),
        "order_items": (
            "order_item_id",
            "order_id",
            "product_id",
            "product_group_id_snapshot",
            "pet_id",
        ),
        "pets": ("pet_id",),
        "histories": ("history_id", "order_id"),
        "claims": ("claim_id", "order_id"),
        "claim_items": ("claim_item_id", "claim_id", "order_item_id"),
    }
    return {
        name: pd.read_csv(
            path,
            dtype={column: "string" for column in id_columns[name]},
            true_values=["t", "true", "True"],
            false_values=["f", "false", "False"],
        )
        for name, path in paths.items()
    }


def run_comparison(
    paths: dict[str, Path],
    *,
    observation_end_at: pd.Timestamp,
    bootstrap_replicates: int = 1_000,
) -> dict[str, object]:
    """원천 해시, 시간 컷과 평가 수치를 한 실행 결과로 묶습니다."""
    sources = _read_sources(paths)
    orders = sources["orders"]
    items = sources["order_items"]
    status = build_order_status_intervals(sources["histories"])
    quantity = build_order_item_quantity_intervals(
        orders, items, sources["claims"], sources["claim_items"]
    )
    valid = build_valid_purchase_item_intervals(status, quantity)
    events = build_operational_event_intervals(valid, orders, items, sources["pets"])
    first = events.pet_targets["valid_from"].min()
    end = pd.Timestamp(observation_end_at)
    if pd.isna(first) or end.tzinfo is None or end <= first:
        raise ValueError(
            "관측 종료 시각은 최초 유효 구매보다 늦은 timezone-aware 시각이어야 합니다."
        )
    span = end - first
    train_end = first + span * 0.70
    validation_end = first + span * 0.85
    split = build_service_train_validation_split(
        events, orders, train_end_at=train_end, validation_end_at=validation_end
    )
    comparison = compare_service_aft_lightgbm(
        split, bootstrap_replicates=bootstrap_replicates
    )
    return {
        "source_sha256": {name: _file_sha256(path) for name, path in paths.items()},
        "runtime_versions": {
            "python": sys.version.split()[0],
            "pandas": version("pandas"),
            "xgboost": version("xgboost"),
            "lightgbm": version("lightgbm"),
        },
        "feature_generation_version": TEMPORAL_SERVICE_FEATURE_GENERATION_VERSION,
        "observation_end_at_assumption": end.isoformat(),
        "train_end_at": train_end.isoformat(),
        "validation_end_at": validation_end.isoformat(),
        "test_evaluated": False,
        "summary": comparison.summary.to_dict(orient="records"),
        "paired_bootstrap_summary": comparison.paired_bootstrap.summary,
        "paired_bootstrap_trials": comparison.paired_bootstrap.trials.to_dict(
            orient="records"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("orders", "order_items", "pets", "histories", "claims", "claim_items"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True, type=Path)
    parser.add_argument("--observation-end-at", required=True, type=pd.Timestamp)
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = {
        name: getattr(args, name)
        for name in (
            "orders",
            "order_items",
            "pets",
            "histories",
            "claims",
            "claim_items",
        )
    }
    result = run_comparison(
        paths,
        observation_end_at=args.observation_end_at,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.output is None:
        print(rendered)
    else:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "paired_bootstrap_summary": result["paired_bootstrap_summary"],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
