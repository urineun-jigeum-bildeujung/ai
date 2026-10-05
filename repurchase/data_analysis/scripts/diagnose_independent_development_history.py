"""Reproduce development-only calibration and inspect history groups by landmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.diagnose_independent_calibration_history import summarize_history_groups
from scripts.evaluate_independent_synthetic_calibration import (
    _apply_mapping,
    _conditional_probability,
    _events,
    _fit_and_reproduce_development,
)
from scripts.export_local_service_snapshot import verify_snapshot
from scripts.modeling.evaluation import bootstrap_ipcw_brier_pair_difference_by_user
from scripts.modeling.maturity_analysis import add_split_ipcw_weights
from scripts.modeling.operational_temporal_split import (
    build_service_train_validation_split,
)
from scripts.modeling.service_landmark_validation import build_service_landmark_cohort
from scripts.run_service_model_comparison import _file_sha256, model_code_sha256

SOURCE_NAMES = ("orders", "order_items", "histories", "claims", "claim_items", "pets")


def validate_development(result: dict[str, object], directory: Path) -> None:
    """Reject mismatched sources, code, or a non-development snapshot."""
    verify_snapshot(directory)
    plan = json.loads((directory / "evaluation-plan.json").read_text(encoding="utf-8"))
    if plan.get("dataset_role") != "calibration_development":
        raise ValueError("개발용 원천만 진단할 수 있습니다.")
    if result.get("test_evaluated") is not False:
        raise ValueError(
            "이미 최종 평가에 사용한 결과는 개발 진단에 사용할 수 없습니다."
        )
    if result.get("source_sha256") != {
        name: _file_sha256(directory / f"{name}.csv") for name in SOURCE_NAMES
    }:
        raise ValueError("개발 원천 파일 지문이 결과와 다릅니다.")
    code_sha256 = result.get("code_sha256")
    if not isinstance(code_sha256, dict):
        raise ValueError(
            "개발 평가 결과의 모델 코드 지문이 누락됐거나 올바르지 않습니다."
        )
    if any(
        code_sha256.get(name) != digest for name, digest in model_code_sha256().items()
    ):
        raise ValueError("개발 평가 이후 모델 코드가 변경됐습니다.")


def diagnose(
    result_path: Path,
    development_dir: Path,
    *,
    bootstrap_replicates: int = 1_000,
    random_seed: int = 42,
) -> dict[str, object]:
    """Read no final-evaluation directory or labels."""
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validate_development(result, development_dir)
    configuration = result["aft_configuration"]
    freeze = {
        "aft_num_boost_round": configuration["num_boost_round"],
        "aft_loss_distribution_scale": configuration["loss_distribution_scale"],
        "landmark_days": result["landmark_days"],
    }
    model, _, mappings = _fit_and_reproduce_development(freeze, result, development_dir)
    events, orders = _events(development_dir)
    inner = build_service_train_validation_split(
        events,
        orders,
        train_end_at=pd.Timestamp(result["inner_train_end_at"]),
        validation_end_at=pd.Timestamp(result["calibration_end_at"]),
    )
    outer = build_service_train_validation_split(
        events,
        orders,
        train_end_at=pd.Timestamp(result["calibration_end_at"]),
        validation_end_at=pd.Timestamp(result["outer_validation_end_at"]),
    )
    landmarks = []
    for day in result["landmark_days"]:
        segments = {}
        for name, rows in (
            ("inner_calibration_fit_in_sample", inner.validation),
            ("outer_validation", outer.validation),
        ):
            cohort = build_service_landmark_cohort(
                rows, elapsed_days=day, split_name="validation"
            )
            weighted = add_split_ipcw_weights(cohort.rows, horizon_days=30)
            raw = _conditional_probability(model, cohort.rows, 30)
            calibrated = _apply_mapping(raw, mappings[day])
            groups = summarize_history_groups(weighted, raw, calibrated)
            if name == "outer_validation":
                for group in groups:
                    count = weighted["history_interval_count"]
                    label = group["history_interval_count"]
                    mask = (
                        count.eq(0)
                        if label == "0"
                        else count.between(1, 2)
                        if label == "1-2"
                        else count.ge(3)
                    )
                    pair = weighted.loc[mask].copy()
                    pair["reference_predicted_event_probability"] = raw.loc[mask]
                    pair["candidate_predicted_event_probability"] = calibrated.loc[mask]
                    group["user_bootstrap"] = (
                        bootstrap_ipcw_brier_pair_difference_by_user(
                            pair,
                            bootstrap_replicates=bootstrap_replicates,
                            random_seed=random_seed,
                        ).summary
                    )
            segments[name] = groups
        landmarks.append({"elapsed_days": day, "segments": segments})
    return {
        "status": "development_only_history_diagnostic_not_model_selection",
        "development_dataset_run_id": json.loads(
            (development_dir / "evaluation-plan.json").read_text(encoding="utf-8")
        )["dataset_run_id"],
        "development_result_sha256": _file_sha256(result_path),
        "bootstrap_replicates": bootstrap_replicates,
        "random_seed": random_seed,
        "landmarks": landmarks,
        "final_labels_read": False,
        "operational_probability_publication_approved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-result", type=Path, required=True)
    parser.add_argument("--development-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.parent.is_dir():
        parser.error("출력 상위 폴더가 없습니다.")
    result = diagnose(args.development_result, args.development_directory)
    with args.output.open("x", encoding="utf-8") as destination:
        destination.write(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
