"""Apply a frozen development-only AFT/Isotonic candidate to synthetic holdout.

This is a pipeline exercise. Its metrics do not establish real-user model quality.
The holdout is not read for labels until --evaluate is explicitly supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.export_local_service_snapshot import verify_snapshot
from scripts.modeling.evaluation import (
    bootstrap_ipcw_brier_pair_difference_by_user,
    evaluate_ipcw_brier_score,
    evaluate_ipcw_concordance_index,
    summarize_ipcw_calibration,
)
from scripts.modeling.maturity_analysis import add_split_ipcw_weights
from scripts.modeling.operational_aft_input import build_service_aft_training_rows
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
    _attach_evaluation_contract,
    build_service_train_validation_split,
)
from scripts.modeling.operational_training_samples import (
    build_temporal_service_training_samples,
)
from scripts.modeling.operational_validity_intervals import (
    build_valid_purchase_item_intervals,
)
from scripts.modeling.probability_baseline import (
    fit_global_event_probability_baseline,
)
from scripts.modeling.service_landmark_validation import build_service_landmark_cohort
from scripts.modeling.service_probability_calibration import _conditional_probability
from scripts.modeling.xgboost_aft import (
    build_xgboost_aft_prediction_data,
    build_xgboost_aft_training_data,
    predict_xgboost_aft_duration,
    train_xgboost_aft_model,
)
from scripts.run_service_model_comparison import (
    _file_sha256,
    _read_sources,
    model_code_sha256,
)
from scripts.validate_independent_dataset_preflight import validate_preflight

SOURCE_NAMES = ("orders", "order_items", "histories", "claims", "claim_items", "pets")


def _paths(directory: Path) -> dict[str, Path]:
    return {name: directory / f"{name}.csv" for name in SOURCE_NAMES}


def validate_freeze(
    freeze_path: Path,
    development_result_path: Path,
    development_dir: Path,
    evaluation_dir: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Check the decision and source fingerprints without reading holdout labels."""
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    result = json.loads(development_result_path.read_text(encoding="utf-8"))
    if (
        freeze.get("schema_version") != 1
        or freeze.get("purpose") != "synthetic_pipeline_evaluation_only"
    ):
        raise ValueError("합성 평가 동결 계약이 유효하지 않습니다.")
    if freeze.get("development_result_sha256") != _file_sha256(development_result_path):
        raise ValueError("동결된 개발 결과 해시가 다릅니다.")
    if (
        freeze.get("candidate") != "ipcw_weighted_isotonic"
        or freeze.get("final_evaluation_used_for_selection") is not False
        or freeze.get("operational_probability_publication_approved") is not False
    ):
        raise ValueError("후보 선택 또는 운영 노출 계약이 유효하지 않습니다.")
    if (
        freeze.get("decision_rule")
        != "development_validation_brier_lower_at_every_landmark"
    ):
        raise ValueError("후보 선택 규칙이 다릅니다.")
    if freeze.get("horizon_days") != 30 or freeze.get("landmark_days") != [
        0,
        7,
        14,
        30,
    ]:
        raise ValueError("예측 기간 또는 경과 시점이 다릅니다.")
    if (
        freeze.get("aft_num_boost_round") != 20
        or freeze.get("aft_loss_distribution_scale") != 1.0
    ):
        raise ValueError("AFT 설정이 동결값과 다릅니다.")
    if (
        freeze.get("bootstrap_replicates") != 1000
        or freeze.get("bootstrap_random_seed") != 42
    ):
        raise ValueError("Bootstrap 설정이 동결값과 다릅니다.")
    if (
        result.get("test_evaluated") is not False
        or result.get("horizon_days") != 30
        or result.get("landmark_days") != freeze["landmark_days"]
    ):
        raise ValueError("개발 결과의 평가 범위가 다릅니다.")
    if result.get("aft_configuration") != {
        "num_boost_round": 20,
        "loss_distribution_scale": 1.0,
    }:
        raise ValueError("개발 결과의 AFT 설정이 다릅니다.")
    model_hashes = model_code_sha256()
    if any(
        result.get("code_sha256", {}).get(name) != digest
        for name, digest in model_hashes.items()
    ):
        raise ValueError("개발 실험 이후 모델 코드가 변경됐습니다.")
    for role, directory, expected_id, manifest_key in (
        (
            "calibration_development",
            development_dir,
            freeze["development_dataset_run_id"],
            "development_manifest_sha256",
        ),
        (
            "final_evaluation",
            evaluation_dir,
            freeze["final_dataset_run_id"],
            "final_manifest_sha256",
        ),
    ):
        if freeze.get(manifest_key) != _file_sha256(directory / "manifest.json"):
            raise ValueError(f"{role} manifest 지문이 동결값과 다릅니다.")
        verify_snapshot(directory)
        plan = json.loads(
            (directory / "evaluation-plan.json").read_text(encoding="utf-8")
        )
        if (
            plan.get("dataset_role") != role
            or plan.get("dataset_run_id") != expected_id
        ):
            raise ValueError(f"{role} 데이터 실행 ID가 동결값과 다릅니다.")
    if result.get("source_sha256") != {
        name: _file_sha256(path) for name, path in _paths(development_dir).items()
    }:
        raise ValueError("개발 결과의 원천 파일이 다릅니다.")
    summary = {
        (row["elapsed_days"], row["candidate"]): row for row in result["summary"]
    }
    if len(result["summary"]) != 2 * len(freeze["landmark_days"]) or set(summary) != {
        (day, candidate)
        for day in freeze["landmark_days"]
        for candidate in ("raw", "isotonic")
    }:
        raise ValueError("개발 후보별 지표가 완전하지 않습니다.")
    if any(
        summary[day, "isotonic"]["ipcw_brier_score"]
        >= summary[day, "raw"]["ipcw_brier_score"]
        for day in freeze["landmark_days"]
    ):
        raise ValueError("개발 후보 선택 규칙을 충족하지 않습니다.")
    mappings = result.get("mappings")
    if (
        not isinstance(mappings, list)
        or len(mappings) != len(freeze["landmark_days"])
        or {row.get("elapsed_days") for row in mappings} != set(freeze["landmark_days"])
    ):
        raise ValueError("경과 시점별 보정 매핑이 완전하지 않습니다.")
    for mapping in mappings:
        if (
            mapping.get("method") != "ipcw_weighted_isotonic"
            or mapping.get("out_of_bounds") != "clip"
        ):
            raise ValueError("보정 매핑 계약이 다릅니다.")
        x = np.asarray(mapping.get("x_thresholds"), dtype="float64")
        y = np.asarray(mapping.get("y_thresholds"), dtype="float64")
        if (
            x.ndim != 1
            or len(x) < 2
            or x.shape != y.shape
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or not np.diff(x).min() > 0
            or (np.diff(y) < 0).any()
            or (y < 0).any()
            or (y > 1).any()
        ):
            raise ValueError("보정 매핑 임계값이 유효하지 않습니다.")
    return freeze, result


def _events(directory: Path):
    sources = _read_sources(_paths(directory))
    q = quarantine_unrestorable_orders(*(sources[name] for name in SOURCE_NAMES[:-1]))
    if q.missing_history_order_count or q.status_mismatch_order_count:
        raise ValueError("합성 평가 원천에 격리 주문이 있습니다.")
    status = build_order_status_intervals(q.status_histories)
    quantity = build_order_item_quantity_intervals(
        q.orders, q.order_items, q.claims, q.claim_items
    )
    valid = build_valid_purchase_item_intervals(status, quantity)
    return build_operational_event_intervals(
        valid, q.orders, q.order_items, sources["pets"]
    ), q.orders


def _brier_and_ece(
    weighted: pd.DataFrame, probability: pd.Series, reference: float
) -> tuple[float, float, pd.DataFrame]:
    rows = weighted.copy()
    rows["predicted_event_probability"] = probability
    brier = evaluate_ipcw_brier_score(rows, reference_probability=reference)
    calibration = summarize_ipcw_calibration(rows, bin_count=10)
    return (
        float(brier["ipcw_brier_score"]),
        float(calibration["weighted_absolute_gap_contribution"].sum()),
        calibration,
    )


def _subgroup_brier(
    weighted: pd.DataFrame, probability: pd.Series
) -> list[dict[str, object]]:
    """Inspect low-history groups using weights fitted on the full final cohort."""
    rows = weighted.copy()
    rows["prediction"] = probability
    known = rows["ipcw_outcome_known"]
    groups = []
    for feature in ("history_interval_count", "user_prior_order_count"):
        count = pd.to_numeric(rows[feature], errors="coerce")
        for label, mask in (
            ("0", count.eq(0)),
            ("1-2", count.between(1, 2)),
            ("3+", count.ge(3)),
            ("unknown", count.isna()),
        ):
            selected = rows.loc[known & mask]
            if selected.empty:
                continue
            weights = selected["ipcw_weight"].to_numpy(dtype="float64")
            actual = selected["ipcw_event_within_horizon"].to_numpy(dtype="float64")
            predicted = selected["prediction"].to_numpy(dtype="float64")
            groups.append(
                {
                    "feature": feature,
                    "group": label,
                    "outcome_known_count": len(selected),
                    "ipcw_brier_score": float(
                        np.average((predicted - actual) ** 2, weights=weights)
                    ),
                }
            )
    return groups


def _apply_mapping(probability: pd.Series, mapping: dict[str, object]) -> pd.Series:
    x = np.asarray(mapping["x_thresholds"], dtype="float64")
    y = np.asarray(mapping["y_thresholds"], dtype="float64")
    values = np.interp(probability.to_numpy(dtype="float64"), x, y)
    return pd.Series(values, index=probability.index)


def _fit_and_reproduce_development(freeze, development, development_dir):
    """Train only on development and reject a non-reproducible frozen choice."""
    dev_events, dev_orders = _events(development_dir)
    train_end = pd.Timestamp(development["inner_train_end_at"])
    calibration_end = pd.Timestamp(development["calibration_end_at"])
    validation_end = pd.Timestamp(development["outer_validation_end_at"])
    inner = build_service_train_validation_split(
        dev_events,
        dev_orders,
        train_end_at=train_end,
        validation_end_at=calibration_end,
    )
    outer = build_service_train_validation_split(
        dev_events,
        dev_orders,
        train_end_at=calibration_end,
        validation_end_at=validation_end,
    )
    model = train_xgboost_aft_model(
        build_xgboost_aft_training_data(build_service_aft_training_rows(inner.train)),
        num_boost_round=freeze["aft_num_boost_round"],
        loss_distribution_scale=freeze["aft_loss_distribution_scale"],
    )
    reference = {}
    mappings = {row["elapsed_days"]: row for row in development["mappings"]}
    reported = {
        (row["elapsed_days"], row["candidate"]): row for row in development["summary"]
    }
    for day in freeze["landmark_days"]:
        train = build_service_landmark_cohort(
            inner.train, elapsed_days=day, split_name="train"
        )
        reference[day] = fit_global_event_probability_baseline(
            add_split_ipcw_weights(train.rows, horizon_days=30)
        ).global_event_probability
        validation = build_service_landmark_cohort(
            outer.validation, elapsed_days=day, split_name="validation"
        )
        weighted = add_split_ipcw_weights(validation.rows, horizon_days=30)
        raw = _conditional_probability(model, validation.rows, 30)
        calibrated = _apply_mapping(raw, mappings[day])
        for candidate, probability in (("raw", raw), ("isotonic", calibrated)):
            brier, _, _ = _brier_and_ece(weighted, probability, reference[day])
            if not np.isclose(
                brier, reported[day, candidate]["ipcw_brier_score"], atol=1e-8, rtol=0
            ):
                raise ValueError("개발용 모델·보정 매핑 지표를 재현하지 못했습니다.")
    return model, reference, mappings


def evaluate_frozen(
    *,
    freeze_path: Path,
    development_result_path: Path,
    development_dir: Path,
    evaluation_dir: Path,
    baseline_dir: Path,
    baseline_test_result: Path,
) -> dict[str, object]:
    freeze, development = validate_freeze(
        freeze_path, development_result_path, development_dir, evaluation_dir
    )
    preflight = validate_preflight(
        baseline_dir, baseline_test_result, development_dir, evaluation_dir
    )
    model, reference, mappings = _fit_and_reproduce_development(
        freeze, development, development_dir
    )

    # The final sources and labels are first opened only after all frozen-development checks.
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
    summaries = []
    bootstrap = []
    concordance = []
    calibration_bins = []
    subgroups = []
    for day in freeze["landmark_days"]:
        cohort = build_service_landmark_cohort(
            final_rows, elapsed_days=day, split_name="validation"
        )
        weighted = add_split_ipcw_weights(cohort.rows, horizon_days=30)
        prediction_input = build_xgboost_aft_prediction_data(
            cohort.rows, feature_columns=model.feature_columns
        )
        aft_duration = predict_xgboost_aft_duration(model, prediction_input)
        ranked = weighted.copy()
        ranked["predicted_duration_days"] = aft_duration
        concordance.append(
            {
                "elapsed_days": day,
                "scope": "underlying_aft_ordering_not_probability_calibration",
                **evaluate_ipcw_concordance_index(ranked),
            }
        )
        raw = _conditional_probability(model, cohort.rows, 30)
        calibrated = _apply_mapping(raw, mappings[day])
        for candidate, probability in (("raw", raw), ("isotonic", calibrated)):
            brier, ece, bins = _brier_and_ece(weighted, probability, reference[day])
            calibration_bins.extend(
                {"elapsed_days": day, "candidate": candidate, **row}
                for row in bins.to_dict(orient="records")
            )
            subgroups.extend(
                {"elapsed_days": day, "candidate": candidate, **row}
                for row in _subgroup_brier(weighted, probability)
            )
            summaries.append(
                {
                    "elapsed_days": day,
                    "candidate": candidate,
                    "at_risk_count": len(cohort.rows),
                    "outcome_known_count": int(weighted["ipcw_outcome_known"].sum()),
                    "ipcw_brier_score": brier,
                    "expected_calibration_error": ece,
                }
            )
        paired = weighted.copy()
        paired["reference_predicted_event_probability"] = raw
        paired["candidate_predicted_event_probability"] = calibrated
        paired_result = bootstrap_ipcw_brier_pair_difference_by_user(
            paired,
            bootstrap_replicates=freeze["bootstrap_replicates"],
            random_seed=freeze["bootstrap_random_seed"],
        )
        bootstrap.append({"elapsed_days": day, **paired_result.summary})
    return {
        "status": "synthetic_final_evaluation_not_real_user_approval",
        "test_evaluated": True,
        "freeze_sha256": _file_sha256(freeze_path),
        "development_result_sha256": _file_sha256(development_result_path),
        "preflight": preflight,
        "final_source_sha256": {
            name: _file_sha256(path) for name, path in _paths(evaluation_dir).items()
        },
        "summary": summaries,
        "calibration_bins": calibration_bins,
        "low_history_subgroups": subgroups,
        "aft_ipcw_concordance": concordance,
        "paired_bootstrap_summary": bootstrap,
        "operational_probability_publication_approved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--development-result", type=Path, required=True)
    parser.add_argument("--development-directory", type=Path, required=True)
    parser.add_argument("--evaluation-directory", type=Path, required=True)
    parser.add_argument("--baseline-directory", type=Path, required=True)
    parser.add_argument("--baseline-test-result", type=Path, required=True)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    validate_freeze(
        args.freeze,
        args.development_result,
        args.development_directory,
        args.evaluation_directory,
    )
    preflight = validate_preflight(
        args.baseline_directory,
        args.baseline_test_result,
        args.development_directory,
        args.evaluation_directory,
    )
    if not args.evaluate:
        freeze, development = validate_freeze(
            args.freeze,
            args.development_result,
            args.development_directory,
            args.evaluation_directory,
        )
        _fit_and_reproduce_development(freeze, development, args.development_directory)
        print(
            json.dumps(
                {
                    "status": "frozen_development_reproduced_no_final_labels_read",
                    "preflight": preflight,
                },
                ensure_ascii=False,
            )
        )
        return
    if args.output is None or args.output.exists() or not args.output.parent.is_dir():
        parser.error(
            "최종 평가 출력 경로는 존재하지 않는 파일과 기존 상위 폴더여야 합니다."
        )
    result = evaluate_frozen(
        freeze_path=args.freeze,
        development_result_path=args.development_result,
        development_dir=args.development_directory,
        evaluation_dir=args.evaluation_directory,
        baseline_dir=args.baseline_directory,
        baseline_test_result=args.baseline_test_result,
    )
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": result["status"], "output": str(args.output)}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
