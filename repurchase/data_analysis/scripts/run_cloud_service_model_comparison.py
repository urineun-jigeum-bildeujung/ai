"""공용 DB 원천을 읽기 전용으로 조회해 서비스 모델 Validation 비교를 실행합니다.

원천 행과 DB 자격 증명은 파일이나 로그에 남기지 않습니다. 결과 파일에는
집계 지표와 DB별 추출 시각만 기록합니다. 두 DB의 원자적 스냅샷은 아닙니다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import psycopg

from scripts.audit_cloud_source_reader import (
    audit_snapshots,
    read_cloud_snapshots,
)
from scripts.modeling.cloud_source_reader import OrderSourceSnapshot, PetSourceSnapshot
from scripts.modeling.operational_orders import OperationalOrderError
from scripts.run_service_model_comparison import (
    _validate_aft_round_selection,
    _validate_product_group_smoothing_candidates,
    run_comparison,
)


def compare_snapshots(
    orders: OrderSourceSnapshot,
    pets: PetSourceSnapshot,
    *,
    observation_end_at: pd.Timestamp,
    bootstrap_replicates: int = 1_000,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.85,
    aft_round_candidates: tuple[int, ...] | None = None,
    aft_scale_candidates: tuple[float, ...] | None = None,
    product_group_smoothing_candidates: tuple[float, ...] | None = None,
    inner_train_ratio: float = 0.8,
) -> dict[str, object]:
    """소유 관계와 관측 컷을 먼저 검증한 뒤 같은 모델 비교기에 전달합니다."""
    audit = audit_snapshots(orders, pets, as_of_timestamp=observation_end_at)
    sources = {
        "orders": orders.orders,
        "order_items": orders.order_items,
        "histories": orders.status_histories,
        "claims": orders.claims,
        "claim_items": orders.claim_items,
        "pets": pets.pets,
    }
    return run_comparison(
        None,
        sources=sources,
        source_metadata={"source_snapshots": asdict(audit)},
        observation_end_at=observation_end_at,
        bootstrap_replicates=bootstrap_replicates,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        aft_round_candidates=aft_round_candidates,
        aft_scale_candidates=aft_scale_candidates,
        product_group_smoothing_candidates=product_group_smoothing_candidates,
        inner_train_ratio=inner_train_ratio,
    )


def _parse_observation_end(value: str) -> pd.Timestamp:
    """시간대가 없는 컷이나 NaT를 DB 연결 전에 거절합니다."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "관측 종료 시각은 시간대가 있는 ISO 시각이어야 합니다."
        ) from error
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("관측 종료 시각은 시간대가 있는 ISO 시각이어야 합니다.")
    return timestamp.tz_convert("UTC")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-end-at", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.85)
    parser.add_argument("--aft-round-candidates", type=int, nargs="+")
    parser.add_argument("--aft-scale-candidates", type=float, nargs="+")
    parser.add_argument("--product-group-smoothing-candidates", type=float, nargs="+")
    parser.add_argument("--inner-train-ratio", type=float, default=0.8)
    parser.add_argument("--prompt-password", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        end = _parse_observation_end(args.observation_end_at)
        if args.bootstrap_replicates < 1:
            raise ValueError("Bootstrap 반복 횟수는 1 이상이어야 합니다.")
        if not (
            math.isfinite(args.train_fraction)
            and math.isfinite(args.validation_fraction)
            and 0 < args.train_fraction < args.validation_fraction < 1
        ):
            raise ValueError("시간 컷 비율은 0 < Train < Validation < 1이어야 합니다.")
        candidate_rounds = (
            tuple(args.aft_round_candidates)
            if args.aft_round_candidates is not None
            else None
        )
        candidate_scales = (
            tuple(args.aft_scale_candidates)
            if args.aft_scale_candidates is not None
            else None
        )
        smoothing_candidates = (
            tuple(args.product_group_smoothing_candidates)
            if args.product_group_smoothing_candidates is not None
            else None
        )
        _validate_aft_round_selection(
            candidate_rounds, candidate_scales, args.inner_train_ratio
        )
        _validate_product_group_smoothing_candidates(smoothing_candidates)
        if args.output is not None and not args.output.parent.is_dir():
            raise ValueError("결과 파일의 상위 디렉터리가 없습니다.")
        orders, pets = read_cloud_snapshots(prompt_password=args.prompt_password)
        result = compare_snapshots(
            orders,
            pets,
            observation_end_at=end,
            bootstrap_replicates=args.bootstrap_replicates,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            aft_round_candidates=candidate_rounds,
            aft_scale_candidates=candidate_scales,
            product_group_smoothing_candidates=smoothing_candidates,
            inner_train_ratio=args.inner_train_ratio,
        )
        rendered = json.dumps(
            result, ensure_ascii=False, indent=2, default=str, allow_nan=False
        )
        if args.output is None:
            print(rendered)
        else:
            args.output.write_text(rendered + "\n", encoding="utf-8")
            print(json.dumps({"output": str(args.output)}, ensure_ascii=False))
    except (ValueError, OperationalOrderError) as error:
        print(
            json.dumps(
                {
                    "event": "repurchase_model_comparison_rejected",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    except OSError as error:
        # 파일 쓰기 오류에는 로컬 경로나 권한 정보가 포함될 수 있습니다.
        print(
            json.dumps(
                {
                    "event": "repurchase_model_comparison_failed",
                    "error_type": type(error).__name__,
                }
            ),
            file=sys.stderr,
        )
        return 1
    except psycopg.Error as error:
        # DB 드라이버 메시지에는 호스트·접속 문자열이 포함될 수 있습니다.
        print(
            json.dumps(
                {
                    "event": "repurchase_model_comparison_failed",
                    "error_type": type(error).__name__,
                }
            ),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
