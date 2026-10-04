"""Inspect development-only 0-day calibration by purchase-history depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_independent_synthetic_calibration import (
    _apply_mapping,
    _conditional_probability,
    _events,
    _fit_and_reproduce_development,
    validate_freeze,
)
from scripts.modeling.maturity_analysis import add_split_ipcw_weights
from scripts.modeling.operational_temporal_split import (
    build_service_train_validation_split,
)
from scripts.modeling.service_landmark_validation import build_service_landmark_cohort


def summarize_history_groups(
    weighted: pd.DataFrame, raw: pd.Series, calibrated: pd.Series
) -> list[dict[str, object]]:
    """Compute paired weighted scores on the same known-outcome rows."""
    if not weighted.index.equals(raw.index) or not raw.index.equals(calibrated.index):
        raise ValueError("위험집단과 확률의 행 순서가 다릅니다.")
    count = pd.to_numeric(weighted["history_interval_count"], errors="coerce")
    if count.isna().any():
        raise ValueError("구매 간격 수가 누락됐습니다.")
    known = weighted["ipcw_outcome_known"]
    result = []
    for group, mask in (
        ("0", count.eq(0)),
        ("1-2", count.between(1, 2)),
        ("3+", count.ge(3)),
    ):
        chosen = weighted.loc[known & mask]
        if chosen.empty:
            continue
        weights = chosen["ipcw_weight"].to_numpy(dtype="float64")
        actual = chosen["ipcw_event_within_horizon"].to_numpy(dtype="float64")
        raw_values = raw.loc[chosen.index].to_numpy(dtype="float64")
        calibrated_values = calibrated.loc[chosen.index].to_numpy(dtype="float64")
        if (
            not np.isfinite(weights).all()
            or (weights <= 0).any()
            or not np.isfinite(actual).all()
            or not np.isfinite(raw_values).all()
            or not np.isfinite(calibrated_values).all()
        ):
            raise ValueError("저이력 구간의 가중치·결과·확률이 유효하지 않습니다.")
        result.append(
            {
                "history_interval_count": group,
                "at_risk_count": int(mask.sum()),
                "outcome_known_count": len(chosen),
                "weighted_event_rate": float(np.average(actual, weights=weights)),
                "mean_raw_probability": float(np.average(raw_values, weights=weights)),
                "mean_isotonic_probability": float(
                    np.average(calibrated_values, weights=weights)
                ),
                "raw_ipcw_brier": float(
                    np.average((raw_values - actual) ** 2, weights=weights)
                ),
                "isotonic_ipcw_brier": float(
                    np.average((calibrated_values - actual) ** 2, weights=weights)
                ),
            }
        )
    return result


def passes_low_history_guardrail(rows: list[dict[str, object]]) -> bool:
    """Require both low-history groups without point Brier deterioration."""
    low_rows = [row for row in rows if row["history_interval_count"] in ("0", "1-2")]
    selected = {row["history_interval_count"]: row for row in low_rows}
    return (
        len(low_rows) == 2
        and set(selected) == {"0", "1-2"}
        and all(
            row["isotonic_ipcw_brier"] <= row["raw_ipcw_brier"]
            for row in selected.values()
        )
    )


def diagnose(
    freeze_path: Path,
    development_result_path: Path,
    development_dir: Path,
    evaluation_dir: Path,
) -> dict[str, object]:
    """Use final metadata for the freeze check, but no final rows or labels."""
    freeze, development = validate_freeze(
        freeze_path, development_result_path, development_dir, evaluation_dir
    )
    model, _, mappings = _fit_and_reproduce_development(
        freeze, development, development_dir
    )
    events, orders = _events(development_dir)
    inner = build_service_train_validation_split(
        events,
        orders,
        train_end_at=pd.Timestamp(development["inner_train_end_at"]),
        validation_end_at=pd.Timestamp(development["calibration_end_at"]),
    )
    outer = build_service_train_validation_split(
        events,
        orders,
        train_end_at=pd.Timestamp(development["calibration_end_at"]),
        validation_end_at=pd.Timestamp(development["outer_validation_end_at"]),
    )
    segments = {}
    for name, split in (
        ("inner_calibration_fit_in_sample", inner.validation),
        ("outer_validation", outer.validation),
    ):
        cohort = build_service_landmark_cohort(
            split, elapsed_days=0, split_name="validation"
        )
        weighted = add_split_ipcw_weights(cohort.rows, horizon_days=30)
        raw = _conditional_probability(model, cohort.rows, 30)
        calibrated = _apply_mapping(raw, mappings[0])
        segments[name] = summarize_history_groups(weighted, raw, calibrated)
    low_history_guardrail_passed = passes_low_history_guardrail(
        segments["outer_validation"]
    )
    return {
        "status": "development_only_history_diagnostic_not_model_selection",
        "development_dataset_run_id": freeze["development_dataset_run_id"],
        "elapsed_days": 0,
        "horizon_days": 30,
        "segments": segments,
        "low_history_guardrail": {
            "rule": "both_0_and_1_to_2_groups_present_and_no_point_brier_worsening_on_outer_validation",
            "passed": low_history_guardrail_passed,
            "interpretation": "conservative_development_screen_not_significance_or_operational_approval",
        },
        "final_labels_read": False,
        "operational_probability_publication_approved": False,
    }


def main() -> None:
    """Write the diagnostic to a new file without overwriting prior results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--development-result", type=Path, required=True)
    parser.add_argument("--development-directory", type=Path, required=True)
    parser.add_argument("--evaluation-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.parent.is_dir():
        parser.error("출력 상위 폴더가 없습니다.")
    result = diagnose(
        args.freeze,
        args.development_result,
        args.development_directory,
        args.evaluation_directory,
    )
    with args.output.open("x", encoding="utf-8") as destination:
        destination.write(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
