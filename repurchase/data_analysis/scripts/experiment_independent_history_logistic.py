"""Compare a fixed history-interaction calibrator on development data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.diagnose_independent_development_history import validate_development
from scripts.evaluate_independent_synthetic_calibration import (
    _apply_mapping,
    _conditional_probability,
    _events,
    _fit_and_reproduce_development,
)
from scripts.modeling.evaluation import (
    bootstrap_ipcw_brier_pair_difference_by_user,
    evaluate_ipcw_brier_score,
    summarize_ipcw_calibration,
)
from scripts.modeling.history_logistic_calibration import (
    fit_history_logistic,
    predict_history_logistic,
)
from scripts.modeling.maturity_analysis import add_split_ipcw_weights
from scripts.modeling.operational_temporal_split import (
    build_service_train_validation_split,
)
from scripts.modeling.service_landmark_validation import build_service_landmark_cohort
from scripts.run_service_model_comparison import _file_sha256

HISTORY_GROUPS = ("0", "1-2", "3+")


def _group_mask(count: pd.Series, label: str) -> pd.Series:
    if label == "0":
        return count.eq(0)
    if label == "1-2":
        return count.between(1, 2)
    return count.ge(3)


def _group_scores(
    weighted: pd.DataFrame, probabilities: dict[str, pd.Series]
) -> list[dict[str, object]]:
    """Score identical known-outcome rows and IPCW weights across candidates."""
    if any(
        not weighted.index.equals(series.index) for series in probabilities.values()
    ):
        raise ValueError("후보 확률과 평가 표본의 행 순서가 다릅니다.")
    count = weighted["history_interval_count"]
    known = weighted["ipcw_outcome_known"]
    scores = []
    for label in HISTORY_GROUPS:
        mask = _group_mask(count, label)
        chosen = weighted.loc[known & mask]
        if chosen.empty:
            continue
        weights = chosen["ipcw_weight"].to_numpy(dtype="float64")
        actual = chosen["ipcw_event_within_horizon"].to_numpy(dtype="float64")
        row = {
            "history_interval_count": label,
            "at_risk_count": int(mask.sum()),
            "outcome_known_count": len(chosen),
            "user_count": int(chosen["user_id"].nunique()),
            "weighted_event_rate": float(np.average(actual, weights=weights)),
        }
        for name, probability in probabilities.items():
            values = probability.loc[chosen.index].to_numpy(dtype="float64")
            row[f"{name}_mean_probability"] = float(np.average(values, weights=weights))
            row[f"{name}_ipcw_brier"] = float(
                np.average((values - actual) ** 2, weights=weights)
            )
        scores.append(row)
    return scores


def _development_screen(landmarks: list[dict[str, object]]) -> dict[str, object]:
    """Require all-landmark whole Brier and both low-history point nonworsening."""
    groups = []
    whole = []
    for landmark in landmarks:
        day = landmark["elapsed_days"]
        summary = landmark["summary"]
        whole.append(
            {
                "elapsed_days": day,
                "point_brier_improvement": summary["raw"]["ipcw_brier_score"]
                - summary["history_logistic"]["ipcw_brier_score"],
            }
        )
        by_history = {
            row["history_interval_count"]: row for row in landmark["history_groups"]
        }
        for label in HISTORY_GROUPS[:2]:
            row = by_history.get(label)
            if row is None or row["outcome_known_count"] < 1:
                raise ValueError("저이력 평가 표본이 완전하지 않습니다.")
            groups.append(
                {
                    "elapsed_days": day,
                    "history_interval_count": label,
                    "point_brier_improvement": row["raw_ipcw_brier"]
                    - row["history_logistic_ipcw_brier"],
                }
            )
    return {
        "rule": "whole_and_both_low_history_groups_have_no_point_brier_worsening_at_every_development_landmark",
        "passed": all(row["point_brier_improvement"] >= 0 for row in whole + groups),
        "whole": whole,
        "low_history": groups,
        "interpretation": "development_screen_only_not_significance_or_operational_approval",
    }


def experiment(
    result_path: Path,
    development_dir: Path,
    *,
    bootstrap_replicates: int = 1_000,
    random_seed: int = 42,
) -> dict[str, object]:
    """Fit only on inner calibration and score only on outer development Validation."""
    if bootstrap_replicates < 1:
        raise ValueError("Bootstrap 반복 횟수는 양수여야 합니다.")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validate_development(result, development_dir)
    configuration = result["aft_configuration"]
    freeze = {
        "aft_num_boost_round": configuration["num_boost_round"],
        "aft_loss_distribution_scale": configuration["loss_distribution_scale"],
        "landmark_days": result["landmark_days"],
    }
    model, references, mappings = _fit_and_reproduce_development(
        freeze, result, development_dir
    )
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
        fit_cohort = build_service_landmark_cohort(
            inner.validation, elapsed_days=day, split_name="validation"
        )
        weighted_fit = add_split_ipcw_weights(fit_cohort.rows, horizon_days=30)
        known_fit = weighted_fit["ipcw_outcome_known"]
        raw_fit = _conditional_probability(model, fit_cohort.rows, 30)
        calibrator = fit_history_logistic(
            raw_fit.loc[known_fit],
            weighted_fit.loc[known_fit, "history_interval_count"],
            weighted_fit.loc[known_fit, "ipcw_event_within_horizon"],
            weighted_fit.loc[known_fit, "ipcw_weight"],
        )
        cohort = build_service_landmark_cohort(
            outer.validation, elapsed_days=day, split_name="validation"
        )
        weighted = add_split_ipcw_weights(cohort.rows, horizon_days=30)
        raw = _conditional_probability(model, cohort.rows, 30)
        probabilities = {
            "raw": raw,
            "isotonic": _apply_mapping(raw, mappings[day]),
            "history_logistic": predict_history_logistic(
                calibrator, raw, weighted["history_interval_count"]
            ),
        }
        summary = {}
        for name, probability in probabilities.items():
            scored = weighted.copy()
            scored["predicted_event_probability"] = probability
            brier = evaluate_ipcw_brier_score(
                scored, reference_probability=references[day]
            )
            bins = summarize_ipcw_calibration(scored, bin_count=10)
            summary[name] = {
                "ipcw_brier_score": float(brier["ipcw_brier_score"]),
                "expected_calibration_error": float(
                    bins["weighted_absolute_gap_contribution"].sum()
                ),
            }
        groups = _group_scores(weighted, probabilities)
        for group in groups:
            if group["history_interval_count"] not in HISTORY_GROUPS[:2]:
                continue
            mask = _group_mask(
                weighted["history_interval_count"],
                group["history_interval_count"],
            )
            paired = weighted.loc[mask].copy()
            paired["reference_predicted_event_probability"] = raw.loc[mask]
            paired["candidate_predicted_event_probability"] = probabilities[
                "history_logistic"
            ].loc[mask]
            group["user_bootstrap"] = bootstrap_ipcw_brier_pair_difference_by_user(
                paired,
                bootstrap_replicates=bootstrap_replicates,
                random_seed=random_seed,
            ).summary
        landmarks.append(
            {
                "elapsed_days": day,
                "inner_calibration_known_count": int(known_fit.sum()),
                "outer_validation_known_count": int(
                    weighted["ipcw_outcome_known"].sum()
                ),
                "calibrator": {
                    "method": "l2_history_group_interaction_logistic",
                    "regularization_c": 1.0,
                    "feature_names": list(calibrator.feature_names_in_),
                    "intercept": calibrator.intercept_.tolist(),
                    "coefficients": calibrator.coef_.tolist(),
                },
                "summary": summary,
                "history_groups": groups,
            }
        )
    return {
        "status": "development_only_history_logistic_experiment_not_model_selection",
        "development_dataset_run_id": json.loads(
            (development_dir / "evaluation-plan.json").read_text(encoding="utf-8")
        )["dataset_run_id"],
        "development_result_sha256": _file_sha256(result_path),
        "bootstrap_replicates": bootstrap_replicates,
        "random_seed": random_seed,
        "landmarks": landmarks,
        "development_screen": _development_screen(landmarks),
        "final_labels_read": False,
        "operational_probability_publication_approved": False,
    }


def main() -> None:
    """Write one immutable development-only result file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-result", type=Path, required=True)
    parser.add_argument("--development-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.parent.is_dir():
        parser.error("출력 상위 폴더가 없습니다.")
    result = experiment(args.development_result, args.development_directory)
    with args.output.open("x", encoding="utf-8") as destination:
        destination.write(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
