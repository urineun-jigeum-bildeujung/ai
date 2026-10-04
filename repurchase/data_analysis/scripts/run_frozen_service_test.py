"""고정된 두 모델을 봉인된 Test에서 한 번만 평가합니다.

실행 전 ``--confirm-final-test``가 필요합니다. Test를 열기 직전에 고정 디렉터리
옆에 실행 기록 디렉터리를 독점 생성하며, 실패한 실행도 자동 재시도하지 않습니다.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.freeze_service_models import _eligible_training_sources, build_refit_rows
from scripts.modeling.artifacts import load_model_artifact, predict_artifact_probability
from scripts.modeling.evaluation import bootstrap_ipcw_brier_pair_difference_by_user
from scripts.modeling.maturity_analysis import add_split_ipcw_weights
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
from scripts.modeling.operational_temporal_split import _attach_evaluation_contract
from scripts.modeling.operational_training_samples import (
    build_temporal_service_training_samples,
)
from scripts.modeling.operational_validity_intervals import (
    build_valid_purchase_item_intervals,
)
from scripts.modeling.service_model_comparison import _evaluate_candidate
from scripts.run_service_model_comparison import _file_sha256, _read_sources
from scripts.validate_frozen_service_test_readiness import validate_frozen_inputs
from scripts.validate_service_final_evaluation import SOURCE_NAMES


def _record(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def _reference_probability(
    sources: dict[str, pd.DataFrame], *, validation_end_at: pd.Timestamp, horizon: int
) -> float:
    """Test 이전의 재학습 모집단에서만 고정 확률 기준선을 계산합니다."""
    train = build_refit_rows(sources, validation_end_at=validation_end_at)
    weighted = add_split_ipcw_weights(train, horizon_days=horizon)
    known = weighted.loc[weighted["ipcw_outcome_known"]]
    weights = known["ipcw_weight"].astype("float64")
    return float(
        known["ipcw_event_within_horizon"].astype("float64").mul(weights).sum()
        / weights.sum()
    )


def _test_rows(
    sources: dict[str, pd.DataFrame],
    *,
    validation_end_at: pd.Timestamp,
    observation_end_at: pd.Timestamp,
) -> pd.DataFrame:
    """관측 컷에서 원천을 복원하고 Validation 이후 앵커만 남깁니다."""
    eligible = _eligible_training_sources(sources, validation_end_at=observation_end_at)
    quarantine = quarantine_unrestorable_orders(
        eligible["orders"],
        eligible["order_items"],
        eligible["histories"],
        eligible["claims"],
        eligible["claim_items"],
        as_of_at=observation_end_at,
    )
    status = build_order_status_intervals(quarantine.status_histories)
    quantity = build_order_item_quantity_intervals(
        quarantine.orders,
        quarantine.order_items,
        quarantine.claims,
        quarantine.claim_items,
    )
    valid = build_valid_purchase_item_intervals(status, quantity)
    events = build_operational_event_intervals(
        valid, quarantine.orders, quarantine.order_items, sources["pets"]
    )
    at_end = build_temporal_service_training_samples(
        events, quarantine.orders, observation_end_at=observation_end_at
    )
    rows = at_end.loc[at_end["anchor_at"].gt(validation_end_at)].copy()
    if rows.empty or rows["anchor_at"].gt(observation_end_at).any():
        raise ValueError("최종 Test 표본이 비었거나 관측 컷을 벗어났습니다.")
    return _attach_evaluation_contract(
        rows, split_name="test", split_end_at=observation_end_at
    )


def evaluate_frozen_test(
    frozen_dir: Path,
    validation_result: Path,
    sources: dict[str, Path],
) -> Path:
    """사전검증 뒤 실행권을 독점하고, Test 결과를 한 번만 기록합니다."""
    frozen_dir = frozen_dir.resolve()
    output_dir = frozen_dir.parent / f"{frozen_dir.name}-final-test"
    if output_dir.exists():
        raise ValueError(
            "이 고정 모델의 Test 실행 기록이 이미 있습니다. 재실행할 수 없습니다."
        )
    verified = validate_frozen_inputs(frozen_dir, validation_result, sources)
    manifest = json.loads(
        (frozen_dir / "pretest-manifest.json").read_text(encoding="utf-8")
    )
    validation_end = pd.Timestamp(manifest["validation_end_at"])
    observation_end = pd.Timestamp(manifest["observation_end_at_assumption"])
    if not validation_end < observation_end:
        raise ValueError("Test 평가 시각 컷이 유효하지 않습니다.")
    output_dir.mkdir(exist_ok=False)
    state_path = output_dir / "execution-state.json"
    state = {
        "status": "started",
        "started_at": datetime.now(UTC).isoformat(),
        "validation_result_sha256": _file_sha256(validation_result),
        "evaluator_sha256": _file_sha256(Path(__file__)),
        "artifact_ids": verified["artifact_ids"],
    }
    _record(state_path, state)
    try:
        data = _read_sources(sources)
        for name in SOURCE_NAMES:
            if _file_sha256(sources[name]) != manifest["source_sha256"][name]:
                raise ValueError(f"Test 추출 중 {name} 원천이 변경됐습니다.")
        horizon = int(manifest["evaluation"]["horizon_days"])
        reference = _reference_probability(
            data, validation_end_at=validation_end, horizon=horizon
        )
        test = _test_rows(
            data,
            validation_end_at=validation_end,
            observation_end_at=observation_end,
        )
        weighted = add_split_ipcw_weights(test, horizon_days=horizon)
        predictions = {}
        results = []
        calibrations = []
        for family in ("xgboost_aft", "lightgbm"):
            artifact = load_model_artifact(frozen_dir / family)
            if artifact.artifact_id != verified["artifact_ids"][family]:
                raise ValueError(f"{family} 아티팩트가 실행 중 변경됐습니다.")
            probability = predict_artifact_probability(artifact, test)
            predictions[family] = probability
            metrics, calibration = _evaluate_candidate(
                weighted,
                probability,
                model_name=family,
                reference_probability=reference,
                calibration_bin_count=10,
            )
            metrics["test_sample_count"] = metrics.pop("validation_sample_count")
            if not np.isfinite(metrics["ipcw_c_index"]):
                raise ValueError(f"{family} Test C-index가 유한하지 않습니다.")
            results.append(metrics)
            calibrations.extend(calibration.to_dict(orient="records"))
        paired = weighted.copy()
        paired["reference_predicted_event_probability"] = predictions["xgboost_aft"]
        paired["candidate_predicted_event_probability"] = predictions["lightgbm"]
        bootstrap = bootstrap_ipcw_brier_pair_difference_by_user(
            paired,
            bootstrap_replicates=manifest["evaluation"]["bootstrap_replicates"],
            random_seed=manifest["evaluation"]["bootstrap_random_seed"],
        )
        # 파일과 모델이 평가 중 바뀐 경우 결과를 확정하지 않습니다.
        validate_frozen_inputs(frozen_dir, validation_result, sources)
        if _file_sha256(Path(__file__)) != state["evaluator_sha256"]:
            raise ValueError("Test 평가 코드가 실행 중 변경됐습니다.")
        result = {
            "schema_version": 1,
            "test_evaluated": True,
            "validation_end_at": manifest["validation_end_at"],
            "observation_end_at": manifest["observation_end_at_assumption"],
            "source_sha256": manifest["source_sha256"],
            "artifact_ids": verified["artifact_ids"],
            "validation_result_sha256": state["validation_result_sha256"],
            "evaluator_sha256": state["evaluator_sha256"],
            "horizon_days": horizon,
            "reference_probability_from_pretest_rows": reference,
            "sample_count": len(test),
            "outcome_known_count": int(weighted["ipcw_outcome_known"].sum()),
            "model_metrics": results,
            "calibration_bins": calibrations,
            "paired_user_bootstrap": bootstrap.summary,
            "bootstrap_trials": bootstrap.trials.to_dict(orient="records"),
            "interpretation": "고정 모델의 Test 평가이며 Test 결과로 모델을 재선택하지 않습니다.",
        }
        result_path = output_dir / "result.json"
        _record(result_path, result)
        state["status"] = "completed"
        state["completed_at"] = datetime.now(UTC).isoformat()
        state["result_sha256"] = _file_sha256(result_path)
        _record(state_path, state)
        return result_path
    except BaseException as exc:
        state["status"] = "failed"
        state["failed_at"] = datetime.now(UTC).isoformat()
        state["error_type"] = type(exc).__name__
        _record(state_path, state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--validation-result", type=Path, required=True)
    parser.add_argument("--confirm-final-test", action="store_true")
    for name in SOURCE_NAMES:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args()
    if not args.confirm_final_test:
        parser.error("최종 Test 실행에는 --confirm-final-test가 필요합니다.")
    result = evaluate_frozen_test(
        args.frozen_dir,
        args.validation_result,
        {name: getattr(args, name) for name in SOURCE_NAMES},
    )
    print(json.dumps({"output": str(result)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
