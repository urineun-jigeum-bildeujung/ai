"""읽기 전용 CSV 추출본에서 서비스 모델의 Validation 비교를 재현합니다."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.lightgbm_baseline import create_lightgbm_classifier
from scripts.modeling.operational_event_intervals import (
    build_operational_event_intervals,
)
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
from scripts.modeling.operational_training_samples import (
    TEMPORAL_SERVICE_FEATURE_GENERATION_VERSION,
)
from scripts.modeling.operational_validity_intervals import (
    build_valid_purchase_item_intervals,
)
from scripts.modeling.service_evaluation_population import (
    service_evaluation_population_policy,
)
from scripts.modeling.service_model_comparison import (
    compare_service_aft_lightgbm,
    select_service_aft_boost_rounds,
    select_service_aft_scale,
    select_service_product_group_smoothing,
)
from scripts.modeling.xgboost_aft import create_xgboost_aft_parameters

MODEL_CODE_PATHS = tuple(sorted((Path(__file__).parent / "modeling").glob("*.py"))) + (
    Path(__file__),
)


def _file_sha256(path: Path) -> str:
    """추출 파일을 변경하지 않고 원본 바이트의 SHA-256을 계산합니다."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_code_sha256() -> dict[str, str]:
    """실행 시점의 디스크 모델 코드 지문을 기록합니다.

    이미 import된 메모리 코드와 같은 바이트라는 증명은 아닙니다.
    """
    return {path.name: _file_sha256(path) for path in MODEL_CODE_PATHS}


# 모듈 로드 후 비교 실행 전 파일이 바뀐 경우도 결과 생성 전에 거절합니다.
# 각 모듈이 실제로 로드한 바이트까지 증명하려면 불변 실행 이미지가 필요합니다.
IMPORTED_MODEL_CODE_SHA256 = model_code_sha256()


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


def _require_finite_c_index(summary: pd.DataFrame) -> None:
    """계산 불능 C-index를 정상 JSON 결과처럼 저장하지 않습니다."""
    scores = pd.to_numeric(summary["ipcw_c_index"], errors="coerce")
    if not np.isfinite(scores.to_numpy(dtype="float64")).all():
        raise ValueError("IPCW C-index가 유한하지 않아 비교 결과를 저장할 수 없습니다.")


def _summarize_validation_population(rows: pd.DataFrame) -> dict[str, int]:
    """모델 지표와 별도로 Validation 모집단의 규모·과거 이력량을 기록합니다."""
    required = {"user_id", "target_id", "history_interval_count"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Validation 모집단 키가 누락됐습니다: {sorted(missing)}")
    if rows[["user_id", "target_id"]].isna().any().any():
        raise ValueError("Validation 모집단 키에 결측값이 있습니다.")
    intervals = rows["history_interval_count"]
    values = pd.to_numeric(intervals, errors="coerce").to_numpy(dtype="float64")
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or not np.equal(values, np.floor(values)).all()
    ):
        raise ValueError("Validation 과거 구매 간격 수는 0 이상의 정수여야 합니다.")
    return {
        "user_count": int(rows["user_id"].nunique()),
        "product_group_count": int(rows["target_id"].nunique()),
        "history_interval_count_0": int(intervals.eq(0).sum()),
        "history_interval_count_1": int(intervals.eq(1).sum()),
        "history_interval_count_2_or_more": int(intervals.ge(2).sum()),
    }


def _validate_aft_round_selection(
    candidate_rounds: tuple[int, ...] | None,
    candidate_scales: tuple[float, ...] | None,
    inner_train_ratio: float,
) -> None:
    """학습 후보·내부 컷 오류를 원천 CSV 접근 전에 거절합니다."""
    if not math.isfinite(inner_train_ratio) or not 0 < inner_train_ratio < 1:
        raise ValueError("AFT 내부 Train 비율은 0과 1 사이여야 합니다.")
    if candidate_rounds is not None and (
        len(candidate_rounds) < 2
        or any(type(value) is not int or value < 1 for value in candidate_rounds)
        or len(set(candidate_rounds)) != len(candidate_rounds)
    ):
        raise ValueError(
            "AFT 반복 횟수 후보는 서로 다른 양의 정수 2개 이상이어야 합니다."
        )
    if candidate_rounds is not None and candidate_scales is not None:
        raise ValueError("AFT 반복 횟수와 scale 선택은 별도 실험으로 실행해야 합니다.")
    if candidate_scales is not None and (
        len(candidate_scales) < 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in candidate_scales
        )
        or len(set(candidate_scales)) != len(candidate_scales)
    ):
        raise ValueError(
            "AFT scale 후보는 서로 다른 양의 유한한 숫자 2개 이상이어야 합니다."
        )


def _validate_product_group_smoothing_candidates(
    candidates: tuple[float, ...] | None,
) -> None:
    if candidates is not None and (
        len(candidates) < 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in candidates
        )
        or len(set(candidates)) != len(candidates)
    ):
        raise ValueError(
            "상품군 확률 수축 강도 후보는 서로 다른 양의 유한한 숫자 2개 이상이어야 합니다."
        )


def run_comparison(
    paths: dict[str, Path] | None,
    *,
    observation_end_at: pd.Timestamp,
    bootstrap_replicates: int = 1_000,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.85,
    aft_round_candidates: tuple[int, ...] | None = None,
    aft_scale_candidates: tuple[float, ...] | None = None,
    product_group_smoothing_candidates: tuple[float, ...] | None = None,
    inner_train_ratio: float = 0.8,
    sources: dict[str, pd.DataFrame] | None = None,
    source_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """CSV 또는 메모리 원천의 출처, 시간 컷과 평가 수치를 한 결과로 묶습니다."""
    end = pd.Timestamp(observation_end_at)
    if end.tzinfo is None:
        raise ValueError("관측 종료 시각은 timezone-aware 시각이어야 합니다.")
    if bootstrap_replicates < 1:
        raise ValueError("Bootstrap 반복 횟수는 1 이상이어야 합니다.")
    if not (
        math.isfinite(train_fraction)
        and math.isfinite(validation_fraction)
        and 0 < train_fraction < validation_fraction < 1
    ):
        raise ValueError("시간 컷 비율은 0 < Train < Validation < 1이어야 합니다.")
    _validate_aft_round_selection(
        aft_round_candidates, aft_scale_candidates, inner_train_ratio
    )
    _validate_product_group_smoothing_candidates(product_group_smoothing_candidates)
    code_hashes = IMPORTED_MODEL_CODE_SHA256
    if model_code_sha256() != code_hashes:
        raise ValueError("모델 비교 실행 전에 코드 파일이 변경됐습니다.")
    if sources is None:
        if paths is None or source_metadata is not None:
            raise ValueError("CSV 경로와 메모리 원천 메타데이터가 일치하지 않습니다.")
        sources = _read_sources(paths)
        source_metadata = {
            "source_sha256": {name: _file_sha256(path) for name, path in paths.items()}
        }
    elif (
        paths is not None
        or source_metadata is None
        or set(source_metadata) != {"source_snapshots"}
    ):
        raise ValueError("메모리 원천에는 DB 스냅샷 메타데이터만 필요합니다.")
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
        raise ValueError(
            "관측 종료 시각은 최초 유효 구매보다 늦은 timezone-aware 시각이어야 합니다."
        )
    span = end - first
    train_end = first + span * train_fraction
    validation_end = first + span * validation_fraction
    selected_rounds = 20
    selected_scale = 1.0
    selection_record: dict[str, object] | None = None
    scale_selection_record: dict[str, object] | None = None
    smoothing_selection_record: dict[str, object] | None = None
    selected_smoothing_strength: float | None = None
    if (
        aft_round_candidates is not None
        or aft_scale_candidates is not None
        or product_group_smoothing_candidates is not None
    ):
        # 내부 라벨·검열을 내부 종료 컷으로 다시 만들고, 바깥 Validation은 보지 않습니다.
        inner_train_end = first + (train_end - first) * inner_train_ratio
        inner_split = build_service_train_validation_split(
            events,
            orders,
            train_end_at=inner_train_end,
            validation_end_at=train_end,
        )
        if aft_round_candidates is not None:
            selection = select_service_aft_boost_rounds(
                inner_split, candidate_rounds=aft_round_candidates
            )
            selected_rounds = selection.selected_rounds
            selection_record = {
                "inner_train_end_at": inner_train_end.isoformat(),
                "inner_validation_end_at": train_end.isoformat(),
                "inner_train_ratio": inner_train_ratio,
                "selection_metric": "ipcw_brier_score",
                "tie_break": "lowest_num_boost_round",
                "selected_rounds": selected_rounds,
                "candidate_scores": selection.candidates.to_dict(orient="records"),
            }
        if aft_scale_candidates is not None:
            scale_selection = select_service_aft_scale(
                inner_split, candidate_scales=aft_scale_candidates
            )
            selected_scale = scale_selection.selected_scale
            scale_selection_record = {
                "inner_train_end_at": inner_train_end.isoformat(),
                "inner_validation_end_at": train_end.isoformat(),
                "inner_train_ratio": inner_train_ratio,
                "selection_metric": "ipcw_brier_score",
                "tie_break": "lowest_scale",
                "loss_distribution": "normal",
                "fixed_num_boost_round": 20,
                "selected_scale": selected_scale,
                "candidate_scores": scale_selection.candidates.to_dict(
                    orient="records"
                ),
            }
        if product_group_smoothing_candidates is not None:
            smoothing_selection = select_service_product_group_smoothing(
                inner_split, candidate_strengths=product_group_smoothing_candidates
            )
            selected_smoothing_strength = smoothing_selection.selected_strength
            smoothing_selection_record = {
                "inner_train_end_at": inner_train_end.isoformat(),
                "inner_validation_end_at": train_end.isoformat(),
                "inner_train_ratio": inner_train_ratio,
                "selection_metric": "ipcw_brier_score",
                "tie_break": "lowest_product_group_smoothing_strength",
                "selected_strength": selected_smoothing_strength,
                "candidate_scores": smoothing_selection.candidates.to_dict(
                    orient="records"
                ),
            }
    # 후보 선택이 끝난 다음에만 바깥 Validation의 라벨을 생성합니다.
    split = build_service_train_validation_split(
        events, orders, train_end_at=train_end, validation_end_at=validation_end
    )
    comparison_options: dict[str, object] = {
        "aft_boost_rounds": selected_rounds,
        "bootstrap_replicates": bootstrap_replicates,
    }
    if aft_scale_candidates is not None:
        comparison_options["aft_loss_distribution_scale"] = selected_scale
    if selected_smoothing_strength is not None:
        comparison_options["product_group_smoothing_strength"] = (
            selected_smoothing_strength
        )
    comparison = compare_service_aft_lightgbm(split, **comparison_options)
    _require_finite_c_index(comparison.summary)
    if model_code_sha256() != code_hashes:
        raise ValueError("모델 비교 실행 중 코드 파일이 변경됐습니다.")
    result: dict[str, object] = {
        **source_metadata,
        "code_sha256": code_hashes,
        "source_quarantine": {
            "missing_history_order_count": quarantine.missing_history_order_count,
            "missing_history_paid_order_count": (
                quarantine.missing_history_paid_order_count
            ),
            "missing_history_order_item_count": (
                quarantine.missing_history_order_item_count
            ),
            "status_mismatch_order_count": quarantine.status_mismatch_order_count,
            "status_mismatch_order_item_count": (
                quarantine.status_mismatch_order_item_count
            ),
        },
        "runtime_versions": {
            "python": sys.version.split()[0],
            "pandas": version("pandas"),
            "xgboost": version("xgboost"),
            "lightgbm": version("lightgbm"),
        },
        "feature_generation_version": TEMPORAL_SERVICE_FEATURE_GENERATION_VERSION,
        "observation_end_at_assumption": end.isoformat(),
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "train_end_at": train_end.isoformat(),
        "validation_end_at": validation_end.isoformat(),
        "test_evaluated": False,
        "evaluation_population_policy": service_evaluation_population_policy(),
        # 기본값까지 포함한 실제 학습 팩토리 설정을 남겨 이후 검증자가
        # 코드 지문뿐 아니라 두 후보의 설정 자체를 대조할 수 있게 합니다.
        "model_configuration": {
            "xgboost_aft": {
                "parameters": create_xgboost_aft_parameters(
                    loss_distribution_scale=selected_scale
                ),
                "num_boost_round": selected_rounds,
            },
            "lightgbm": {"parameters": create_lightgbm_classifier().get_params()},
        },
        "validation_population": _summarize_validation_population(split.validation),
        "summary": comparison.summary.to_dict(orient="records"),
        "calibration": comparison.calibration.to_dict(orient="records"),
        "brier_attribution": comparison.brier_attribution.to_dict(orient="records"),
        "paired_bootstrap_summary": comparison.paired_bootstrap.summary,
        "paired_bootstrap_trials": comparison.paired_bootstrap.trials.to_dict(
            orient="records"
        ),
    }
    if selection_record is not None:
        result["aft_round_selection"] = selection_record
    if scale_selection_record is not None:
        result["aft_scale_selection"] = scale_selection_record
    if smoothing_selection_record is not None:
        result["product_group_smoothing_selection"] = smoothing_selection_record
        result["model_configuration"]["product_group_probability_baseline"] = {
            "product_group_smoothing_strength": selected_smoothing_strength,
            "horizon_days": 30,
            "source_key": "target_id",
            "prior": "train_global_ipcw_event_probability",
        }
        if comparison.product_group_aft_bootstrap is None:
            raise ValueError("상품군 기준선과 AFT의 쌍 비교 결과가 없습니다.")
        result["product_group_aft_bootstrap"] = {
            "reference_model": "product_group_probability_baseline",
            "candidate_model": "xgboost_aft",
            "summary": comparison.product_group_aft_bootstrap.summary,
            "trials": comparison.product_group_aft_bootstrap.trials.to_dict(
                orient="records"
            ),
        }
    return result


def main() -> None:
    """잘못된 실행 설정을 CSV 조회나 모델 학습 전에 거절합니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("orders", "order_items", "pets", "histories", "claims", "claim_items"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True, type=Path)
    parser.add_argument("--observation-end-at", required=True, type=pd.Timestamp)
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.85)
    parser.add_argument("--aft-round-candidates", type=int, nargs="+")
    parser.add_argument("--aft-scale-candidates", type=float, nargs="+")
    parser.add_argument("--product-group-smoothing-candidates", type=float, nargs="+")
    parser.add_argument("--inner-train-ratio", type=float, default=0.8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.observation_end_at.tzinfo is None:
        parser.error("--observation-end-at은 timezone-aware 시각이어야 합니다.")
    if args.bootstrap_replicates < 1:
        parser.error("--bootstrap-replicates는 1 이상이어야 합니다.")
    if not (
        math.isfinite(args.train_fraction)
        and math.isfinite(args.validation_fraction)
        and 0 < args.train_fraction < args.validation_fraction < 1
    ):
        parser.error("시간 컷 비율은 0 < Train < Validation < 1이어야 합니다.")
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
    try:
        _validate_aft_round_selection(
            candidate_rounds, candidate_scales, args.inner_train_ratio
        )
        _validate_product_group_smoothing_candidates(smoothing_candidates)
    except ValueError as exc:
        parser.error(str(exc))
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
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "paired_bootstrap_summary": result["paired_bootstrap_summary"],
                },
                ensure_ascii=False,
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    main()
