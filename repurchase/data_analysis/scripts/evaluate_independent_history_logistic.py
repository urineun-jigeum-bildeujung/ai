"""Evaluate one frozen synthetic history-logistic candidate exactly once.

This is a pipeline test, not evidence of real-user quality or publication approval.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_independent_synthetic_calibration import (
    _apply_mapping,
    _brier_and_ece,
    _conditional_probability,
    _events,
    _fit_and_reproduce_development,
)
from scripts.experiment_independent_history_logistic import (
    HISTORY_GROUPS,
    _group_mask,
    _group_scores,
)
from scripts.export_local_service_snapshot import verify_snapshot
from scripts.modeling.evaluation import (
    bootstrap_ipcw_brier_pair_difference_by_user,
    evaluate_ipcw_concordance_index,
)
from scripts.modeling.history_logistic_calibration import (
    fit_history_logistic,
    predict_history_logistic,
)
from scripts.modeling.maturity_analysis import add_split_ipcw_weights
from scripts.modeling.operational_temporal_split import (
    _attach_evaluation_contract,
    build_service_train_validation_split,
)
from scripts.modeling.operational_training_samples import (
    build_temporal_service_training_samples,
)
from scripts.modeling.service_landmark_validation import build_service_landmark_cohort
from scripts.modeling.xgboost_aft import (
    build_xgboost_aft_prediction_data,
    predict_xgboost_aft_duration,
)
from scripts.run_service_model_comparison import _file_sha256, model_code_sha256
from scripts.validate_independent_dataset_preflight import validate_preflight

LANDMARK_DAYS = [0, 7, 14, 30]


def _run_marker_path(evaluation_dir: Path) -> Path:
    """Bind the single-use marker to the final manifest, not an output path."""
    resolved = evaluation_dir.resolve()
    digest = _file_sha256(resolved / "manifest.json")
    return resolved.parent / f".{resolved.name}-history-logistic-final-{digest}"


def _reserve_final_run(evaluation_dir: Path, freeze_path: Path) -> Path:
    """Reserve final-label access exclusively; retain the marker after failure."""
    marker = _run_marker_path(evaluation_dir)
    try:
        marker.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise ValueError("이 최종 합성 데이터의 평가 기록이 이미 있습니다.") from exc
    (marker / "execution-state.json").write_text(
        json.dumps(
            {
                "status": "started",
                "started_at": datetime.now(UTC).isoformat(),
                "final_manifest_sha256": _file_sha256(evaluation_dir / "manifest.json"),
                "freeze_sha256": _file_sha256(freeze_path),
                "evaluator_sha256": _file_sha256(Path(__file__)),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return marker


def validate_frozen_inputs(
    freeze_path: Path,
    development_result_path: Path,
    development_comparison_path: Path,
    development_dir: Path,
    evaluation_dir: Path,
    baseline_dir: Path,
    baseline_test_result: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    """Check identities and metadata before parsing any final purchase outcome."""
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    development = json.loads(development_result_path.read_text(encoding="utf-8"))
    comparison = json.loads(development_comparison_path.read_text(encoding="utf-8"))
    if (
        freeze.get("schema_version") != 1
        or freeze.get("purpose") != "synthetic_pipeline_evaluation_only"
        or freeze.get("candidate") != "l2_history_group_interaction_logistic"
        or freeze.get("landmark_days") != LANDMARK_DAYS
        or freeze.get("horizon_days") != 30
        or freeze.get("regularization_c") != 1.0
        or freeze.get("bootstrap_replicates") != 1000
        or freeze.get("bootstrap_random_seed") != 42
        or freeze.get("final_decision_rule")
        != "whole_and_both_low_history_groups_have_no_point_brier_worsening_at_every_landmark"
        or freeze.get("final_evaluation_used_for_selection") is not False
        or freeze.get("operational_probability_publication_approved") is not False
    ):
        raise ValueError("이력 보정 최종 합성 평가 계약이 유효하지 않습니다.")
    for key, path in (
        ("development_result_sha256", development_result_path),
        ("development_comparison_sha256", development_comparison_path),
        ("development_manifest_sha256", development_dir / "manifest.json"),
        ("final_manifest_sha256", evaluation_dir / "manifest.json"),
        ("evaluation_code_sha256", Path(__file__)),
        (
            "development_experiment_code_sha256",
            Path(__file__).with_name("experiment_independent_history_logistic.py"),
        ),
    ):
        if freeze.get(key) != _file_sha256(path):
            raise ValueError(f"동결된 {key} 지문이 다릅니다.")
    code_hashes = development.get("code_sha256")
    if (
        development.get("test_evaluated") is not False
        or development.get("landmark_days") != LANDMARK_DAYS
        or development.get("horizon_days") != 30
        or development.get("aft_configuration")
        != {"num_boost_round": 20, "loss_distribution_scale": 1.0}
        or not isinstance(code_hashes, dict)
        or any(
            code_hashes.get(name) != digest
            for name, digest in model_code_sha256().items()
        )
    ):
        raise ValueError("개발 결과의 모델·시점·코드 계약이 다릅니다.")
    screen = comparison.get("development_screen")
    if (
        comparison.get("status")
        != "development_only_history_logistic_experiment_not_model_selection"
        or not isinstance(screen, dict)
        or screen.get("passed") is not True
        or comparison.get("final_labels_read") is not False
        or comparison.get("development_result_sha256")
        != _file_sha256(development_result_path)
        or not freeze.get("development_dataset_run_id")
        or not freeze.get("final_dataset_run_id")
        or freeze.get("development_dataset_run_id")
        != comparison.get("development_dataset_run_id")
    ):
        raise ValueError("개발 후보의 선택 근거가 완전하지 않습니다.")
    if (
        len(comparison.get("landmarks", [])) != len(LANDMARK_DAYS)
        or [row.get("elapsed_days") for row in comparison["landmarks"]] != LANDMARK_DAYS
    ):
        raise ValueError("개발 후보의 경과일 근거가 완전하지 않습니다.")
    dev_plan = json.loads(
        (development_dir / "evaluation-plan.json").read_text(encoding="utf-8")
    )
    final_plan = json.loads(
        (evaluation_dir / "evaluation-plan.json").read_text(encoding="utf-8")
    )
    if (
        dev_plan.get("dataset_role") != "calibration_development"
        or final_plan.get("dataset_role") != "final_evaluation"
        or dev_plan.get("dataset_run_id") != freeze.get("development_dataset_run_id")
        or final_plan.get("dataset_run_id") != freeze.get("final_dataset_run_id")
    ):
        raise ValueError("개발·최종 생성본의 역할 또는 실행 ID가 다릅니다.")
    verify_snapshot(development_dir)
    verify_snapshot(evaluation_dir)
    preflight = validate_preflight(
        baseline_dir, baseline_test_result, development_dir, evaluation_dir
    )
    return freeze, development, comparison, preflight


def _final_point_screen(landmarks: list[dict[str, object]]) -> dict[str, object]:
    """Report the frozen whole/low-history rule without altering the candidate."""
    if [landmark.get("elapsed_days") for landmark in landmarks] != LANDMARK_DAYS:
        raise ValueError("최종 평가 경과일이 동결된 기준과 다릅니다.")
    whole = []
    low = []
    for landmark in landmarks:
        day = landmark["elapsed_days"]
        whole.append(
            {
                "elapsed_days": day,
                "point_brier_improvement": landmark["summary"]["raw"][
                    "ipcw_brier_score"
                ]
                - landmark["summary"]["history_logistic"]["ipcw_brier_score"],
            }
        )
        by_history = {
            row["history_interval_count"]: row for row in landmark["history_groups"]
        }
        for label in HISTORY_GROUPS[:2]:
            row = by_history.get(label)
            if row is None or row["outcome_known_count"] < 1:
                raise ValueError("최종 저이력 평가 표본이 완전하지 않습니다.")
            low.append(
                {
                    "elapsed_days": day,
                    "history_interval_count": label,
                    "point_brier_improvement": row["raw_ipcw_brier"]
                    - row["history_logistic_ipcw_brier"],
                }
            )
    return {
        "rule": "whole_and_both_low_history_groups_have_no_point_brier_worsening_at_every_landmark",
        "passed": all(row["point_brier_improvement"] >= 0 for row in whole + low),
        "whole": whole,
        "low_history": low,
        "interpretation": "synthetic_pipeline_test_not_real_user_or_operational_approval",
    }


def evaluate_frozen(
    *,
    freeze_path: Path,
    development_result_path: Path,
    development_comparison_path: Path,
    development_dir: Path,
    evaluation_dir: Path,
    baseline_dir: Path,
    baseline_test_result: Path,
) -> dict[str, object]:
    """Validate, reproduce the fixed model, then open one synthetic final set."""
    if _run_marker_path(evaluation_dir).exists():
        raise ValueError("이 최종 합성 데이터의 평가 기록이 이미 있습니다.")
    freeze, development, comparison, preflight = validate_frozen_inputs(
        freeze_path,
        development_result_path,
        development_comparison_path,
        development_dir,
        evaluation_dir,
        baseline_dir,
        baseline_test_result,
    )
    model, reference, mappings = _fit_and_reproduce_development(
        {
            "aft_num_boost_round": 20,
            "aft_loss_distribution_scale": 1.0,
            "landmark_days": LANDMARK_DAYS,
        },
        development,
        development_dir,
    )
    dev_events, dev_orders = _events(development_dir)
    inner = build_service_train_validation_split(
        dev_events,
        dev_orders,
        train_end_at=pd.Timestamp(development["inner_train_end_at"]),
        validation_end_at=pd.Timestamp(development["calibration_end_at"]),
    )
    calibrators = {}
    for day, recorded in zip(LANDMARK_DAYS, comparison["landmarks"], strict=True):
        cohort = build_service_landmark_cohort(
            inner.validation, elapsed_days=day, split_name="validation"
        )
        weighted = add_split_ipcw_weights(cohort.rows, horizon_days=30)
        known = weighted["ipcw_outcome_known"]
        raw = _conditional_probability(model, cohort.rows, 30)
        fitted = fit_history_logistic(
            raw.loc[known],
            weighted.loc[known, "history_interval_count"],
            weighted.loc[known, "ipcw_event_within_horizon"],
            weighted.loc[known, "ipcw_weight"],
        )
        if not np.allclose(
            fitted.coef_, recorded["calibrator"]["coefficients"], atol=1e-8, rtol=0
        ) or not np.allclose(
            fitted.intercept_, recorded["calibrator"]["intercept"], atol=1e-8, rtol=0
        ):
            raise ValueError("동결된 개발 보정 계수를 재현하지 못했습니다.")
        calibrators[day] = fitted

    # Reserve exactly once after development reproduction, before final labels.
    _reserve_final_run(evaluation_dir, freeze_path)
    final_events, final_orders = _events(evaluation_dir)
    final_plan = json.loads(
        (evaluation_dir / "evaluation-plan.json").read_text(encoding="utf-8")
    )
    end = pd.Timestamp(final_plan["observation_end_at"])
    final_rows = build_temporal_service_training_samples(
        final_events, final_orders, observation_end_at=end
    )
    final_rows = _attach_evaluation_contract(
        final_rows, split_name="validation", split_end_at=end
    )
    landmarks = []
    for day in LANDMARK_DAYS:
        cohort = build_service_landmark_cohort(
            final_rows, elapsed_days=day, split_name="validation"
        )
        weighted = add_split_ipcw_weights(cohort.rows, horizon_days=30)
        duration = predict_xgboost_aft_duration(
            model,
            build_xgboost_aft_prediction_data(
                cohort.rows, feature_columns=model.feature_columns
            ),
        )
        ranked = weighted.copy()
        ranked["predicted_duration_days"] = duration
        raw = _conditional_probability(model, cohort.rows, 30)
        probabilities = {
            "raw": raw,
            "isotonic": _apply_mapping(raw, mappings[day]),
            "history_logistic": predict_history_logistic(
                calibrators[day], raw, weighted["history_interval_count"]
            ),
        }
        summary = {}
        calibration_bins = {}
        for name, probability in probabilities.items():
            brier, ece, bins = _brier_and_ece(weighted, probability, reference[day])
            summary[name] = {
                "ipcw_brier_score": brier,
                "expected_calibration_error": ece,
            }
            calibration_bins[name] = bins.to_dict(orient="records")
        groups = _group_scores(weighted, probabilities)
        for group in groups:
            if group["history_interval_count"] not in HISTORY_GROUPS[:2]:
                continue
            mask = _group_mask(
                weighted["history_interval_count"], group["history_interval_count"]
            )
            paired = weighted.loc[mask].copy()
            paired["reference_predicted_event_probability"] = raw.loc[mask]
            paired["candidate_predicted_event_probability"] = probabilities[
                "history_logistic"
            ].loc[mask]
            group["user_bootstrap"] = bootstrap_ipcw_brier_pair_difference_by_user(
                paired,
                bootstrap_replicates=freeze["bootstrap_replicates"],
                random_seed=freeze["bootstrap_random_seed"],
            ).summary
        paired = weighted.copy()
        paired["reference_predicted_event_probability"] = raw
        paired["candidate_predicted_event_probability"] = probabilities[
            "history_logistic"
        ]
        landmarks.append(
            {
                "elapsed_days": day,
                "at_risk_count": len(cohort.rows),
                "outcome_known_count": int(weighted["ipcw_outcome_known"].sum()),
                "summary": summary,
                "history_groups": groups,
                "calibration_bins": calibration_bins,
                "aft_ipcw_concordance": evaluate_ipcw_concordance_index(ranked),
                "user_bootstrap": bootstrap_ipcw_brier_pair_difference_by_user(
                    paired,
                    bootstrap_replicates=freeze["bootstrap_replicates"],
                    random_seed=freeze["bootstrap_random_seed"],
                ).summary,
            }
        )
    return {
        "status": "synthetic_final_evaluation_not_real_user_approval",
        "test_evaluated": True,
        "freeze_sha256": _file_sha256(freeze_path),
        "development_result_sha256": _file_sha256(development_result_path),
        "development_comparison_sha256": _file_sha256(development_comparison_path),
        "preflight": preflight,
        "final_source_sha256": {
            name: _file_sha256(evaluation_dir / f"{name}.csv")
            for name in (
                "orders",
                "order_items",
                "histories",
                "claims",
                "claim_items",
                "pets",
            )
        },
        "landmarks": landmarks,
        "final_point_screen": _final_point_screen(landmarks),
        "operational_probability_publication_approved": False,
    }


def main() -> None:
    """Write the one-time synthetic final evaluation to a new JSON file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--development-result", required=True, type=Path)
    parser.add_argument("--development-comparison", required=True, type=Path)
    parser.add_argument("--development-directory", required=True, type=Path)
    parser.add_argument("--evaluation-directory", required=True, type=Path)
    parser.add_argument("--baseline-directory", required=True, type=Path)
    parser.add_argument("--baseline-test-result", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("결과 파일이 이미 있거나 출력 상위 폴더가 없습니다.")
    result = evaluate_frozen(
        freeze_path=args.freeze,
        development_result_path=args.development_result,
        development_comparison_path=args.development_comparison,
        development_dir=args.development_directory,
        evaluation_dir=args.evaluation_directory,
        baseline_dir=args.baseline_directory,
        baseline_test_result=args.baseline_test_result,
    )
    with args.output.open("x", encoding="utf-8") as destination:
        destination.write(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    marker = _run_marker_path(args.evaluation_directory)
    (marker / "execution-state.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "result_sha256": _file_sha256(args.output),
                "completed_at": datetime.now(UTC).isoformat(),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
