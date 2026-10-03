"""로컬 서비스 원천에서 AFT 조건부 확률의 Train 내부 보정 실험을 실행합니다."""

from __future__ import annotations

import argparse
import json
import math
import sys
from numbers import Integral
from pathlib import Path

import pandas as pd

from scripts.modeling.operational_event_intervals import (
    build_operational_event_intervals,
)
from scripts.modeling.operational_orders import OperationalOrderError
from scripts.modeling.operational_quantity_intervals import (
    build_order_item_quantity_intervals,
)
from scripts.modeling.operational_source_quarantine import (
    quarantine_unrestorable_orders,
)
from scripts.modeling.operational_status_intervals import build_order_status_intervals
from scripts.modeling.operational_temporal_split import (
    build_service_train_validation_split,
)
from scripts.modeling.operational_validity_intervals import (
    build_valid_purchase_item_intervals,
)
from scripts.modeling.service_probability_calibration import (
    compare_service_aft_isotonic_calibration,
)
from scripts.run_service_model_comparison import (
    _file_sha256,
    _read_sources,
    model_code_sha256,
)

HORIZON_DAYS = 30
AFT_BOOST_ROUNDS = 20
AFT_LOSS_DISTRIBUTION_SCALE = 1.0


def run_calibration(
    paths: dict[str, Path],
    *,
    observation_end_at: pd.Timestamp,
    landmark_days: tuple[int, ...],
    train_fraction: float = 0.70,
    validation_fraction: float = 0.85,
    inner_train_ratio: float = 0.80,
    aft_boost_rounds: int = AFT_BOOST_ROUNDS,
    aft_loss_distribution_scale: float = AFT_LOSS_DISTRIBUTION_SCALE,
    bootstrap_replicates: int = 1_000,
) -> dict[str, object]:
    """원천 감사부터 보정 평가까지 실행하고 입력·설정·지표를 반환합니다."""
    end = pd.Timestamp(observation_end_at)
    if pd.isna(end) or end.tzinfo is None:
        raise ValueError("관측 종료 시각에는 시간대가 필요합니다.")
    if not (
        math.isfinite(train_fraction)
        and math.isfinite(validation_fraction)
        and math.isfinite(inner_train_ratio)
        and 0 < train_fraction < validation_fraction < 1
        and 0 < inner_train_ratio < 1
    ):
        raise ValueError("Train·Validation·내부 Train 시간 컷 비율이 잘못됐습니다.")
    if isinstance(bootstrap_replicates, bool) or bootstrap_replicates < 1:
        raise ValueError("Bootstrap 반복 횟수는 양의 정수여야 합니다.")
    if (
        isinstance(aft_boost_rounds, bool)
        or not isinstance(aft_boost_rounds, Integral)
        or aft_boost_rounds < 1
    ):
        raise ValueError("AFT 부스팅 횟수는 양의 정수여야 합니다.")
    if (
        isinstance(aft_loss_distribution_scale, bool)
        or not math.isfinite(aft_loss_distribution_scale)
        or aft_loss_distribution_scale <= 0
    ):
        raise ValueError("AFT scale은 양의 유한값이어야 합니다.")
    code_hashes = model_code_sha256()
    source_hashes = {name: _file_sha256(path) for name, path in paths.items()}
    sources = _read_sources(paths)
    quarantine = quarantine_unrestorable_orders(
        sources["orders"],
        sources["order_items"],
        sources["histories"],
        sources["claims"],
        sources["claim_items"],
    )
    orders = quarantine.orders
    items = quarantine.order_items
    status = build_order_status_intervals(quarantine.status_histories)
    quantity = build_order_item_quantity_intervals(
        orders, items, quarantine.claims, quarantine.claim_items
    )
    valid = build_valid_purchase_item_intervals(status, quantity)
    events = build_operational_event_intervals(valid, orders, items, sources["pets"])
    first = events.pet_targets["valid_from"].min()
    if pd.isna(first) or end <= first:
        raise ValueError("관측 종료 시각은 최초 유효 구매보다 늦어야 합니다.")
    train_end = first + (end - first) * train_fraction
    validation_end = first + (end - first) * validation_fraction
    inner_train_end = first + (train_end - first) * inner_train_ratio
    inner = build_service_train_validation_split(
        events, orders, train_end_at=inner_train_end, validation_end_at=train_end
    )
    outer = build_service_train_validation_split(
        events, orders, train_end_at=train_end, validation_end_at=validation_end
    )
    trial = compare_service_aft_isotonic_calibration(
        inner,
        outer,
        landmark_days=landmark_days,
        horizon_days=HORIZON_DAYS,
        aft_boost_rounds=aft_boost_rounds,
        aft_loss_distribution_scale=aft_loss_distribution_scale,
        bootstrap_replicates=bootstrap_replicates,
    )
    if model_code_sha256() != code_hashes:
        raise ValueError("모델 보정 실행 중 코드 파일이 변경됐습니다.")
    if {name: _file_sha256(path) for name, path in paths.items()} != source_hashes:
        raise ValueError("모델 보정 실행 중 원천 파일이 변경됐습니다.")
    return {
        "source_sha256": source_hashes,
        "code_sha256": {
            **code_hashes,
            Path(__file__).name: _file_sha256(Path(__file__)),
        },
        "observation_end_at_assumption": end.isoformat(),
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "inner_train_ratio": inner_train_ratio,
        "inner_train_end_at": inner_train_end.isoformat(),
        "calibration_end_at": train_end.isoformat(),
        "outer_validation_end_at": validation_end.isoformat(),
        "landmark_days": list(landmark_days),
        "horizon_days": HORIZON_DAYS,
        "aft_configuration": {
            "num_boost_round": aft_boost_rounds,
            "loss_distribution_scale": aft_loss_distribution_scale,
        },
        "same_inner_train_aft_for_both_candidates": True,
        "test_evaluated": False,
        "summary": trial.summary.to_dict(orient="records"),
        "calibration": trial.calibration.to_dict(orient="records"),
        "mappings": trial.mappings,
        "paired_bootstrap_summary": trial.bootstrap_summary.to_dict(orient="records"),
        "paired_bootstrap_trials": trial.bootstrap_trials.to_dict(orient="records"),
        "source_quarantine": {
            "missing_history_order_count": quarantine.missing_history_order_count,
            "status_mismatch_order_count": quarantine.status_mismatch_order_count,
        },
    }


def main() -> int:
    """로컬 CSV와 관측 컷을 받아 기존 결과를 덮어쓰지 않고 저장합니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("orders", "order_items", "pets", "histories", "claims", "claim_items"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True, type=Path)
    parser.add_argument("--observation-end-at", required=True, type=pd.Timestamp)
    parser.add_argument("--landmark-days", nargs="+", type=int, default=[0, 7, 14, 30])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--aft-boost-rounds", type=int, default=AFT_BOOST_ROUNDS)
    parser.add_argument(
        "--aft-loss-distribution-scale",
        type=float,
        default=AFT_LOSS_DISTRIBUTION_SCALE,
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
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
    try:
        if args.output.exists():
            raise FileExistsError("결과 파일이 이미 있어 덮어쓰지 않습니다.")
        if not args.output.parent.is_dir():
            raise ValueError("결과 파일의 상위 폴더가 없습니다.")
        result = run_calibration(
            paths,
            observation_end_at=args.observation_end_at,
            landmark_days=tuple(args.landmark_days),
            aft_boost_rounds=args.aft_boost_rounds,
            aft_loss_distribution_scale=args.aft_loss_distribution_scale,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        with args.output.open("x", encoding="utf-8") as destination:
            destination.write(
                json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            )
    except (OSError, ValueError, OperationalOrderError) as error:
        print(
            json.dumps(
                {
                    "event": "repurchase_calibration_failed",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {"event": "repurchase_calibration_completed", "output": str(args.output)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
