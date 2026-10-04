"""Summarize a sealed synthetic evaluation without selecting or fitting a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LANDMARKS = (0, 7, 14, 30)


def summarize(result: dict) -> dict:
    if (
        result.get("status") != "synthetic_final_evaluation_not_real_user_approval"
        or result.get("test_evaluated") is not True
        or result.get("operational_probability_publication_approved") is not False
    ):
        raise ValueError("동결된 합성 최종 평가 결과가 아닙니다.")
    rows = result.get("summary", [])
    by_key = {(r["elapsed_days"], r["candidate"]): r for r in rows}
    expected = {
        (day, candidate) for day in LANDMARKS for candidate in ("raw", "isotonic")
    }
    if len(rows) != len(expected) or set(by_key) != expected:
        raise ValueError("경과일·후보별 평가 결과가 완전하지 않습니다.")
    bootstrap = result.get("paired_bootstrap_summary", [])
    by_day = {r["elapsed_days"]: r for r in bootstrap}
    if len(bootstrap) != len(LANDMARKS) or set(by_day) != set(LANDMARKS):
        raise ValueError("경과일별 쌍 비교가 완전하지 않습니다.")
    landmarks = []
    for day in LANDMARKS:
        raw, calibrated, pair = by_key[day, "raw"], by_key[day, "isotonic"], by_day[day]
        if (
            raw["outcome_known_count"] != calibrated["outcome_known_count"]
            or pair["outcome_known_count"] != raw["outcome_known_count"]
            or abs(
                pair["point_brier_improvement"]
                - (raw["ipcw_brier_score"] - calibrated["ipcw_brier_score"])
            )
            > 1e-10
        ):
            raise ValueError("후보 간 모집단 또는 개선량이 일치하지 않습니다.")
        landmarks.append(
            {
                "elapsed_days": day,
                "outcome_known_count": raw["outcome_known_count"],
                "raw_ipcw_brier": raw["ipcw_brier_score"],
                "isotonic_ipcw_brier": calibrated["ipcw_brier_score"],
                "point_brier_improvement": pair["point_brier_improvement"],
                "improvement_95_interval": [
                    pair["bootstrap_lower_95_brier_improvement"],
                    pair["bootstrap_upper_95_brier_improvement"],
                ],
                "raw_ece": raw["expected_calibration_error"],
                "isotonic_ece": calibrated["expected_calibration_error"],
            }
        )
    subgroups = result.get("low_history_subgroups", [])
    history = {
        (r["candidate"], r["group"]): r
        for r in subgroups
        if r["elapsed_days"] == 0 and r["feature"] == "history_interval_count"
    }
    low_history = []
    for group in ("0", "1-2"):
        raw, calibrated = history.get(("raw", group)), history.get(("isotonic", group))
        if (
            not raw
            or not calibrated
            or raw["outcome_known_count"] != calibrated["outcome_known_count"]
        ):
            raise ValueError("저이력 구간의 쌍 비교가 완전하지 않습니다.")
        low_history.append(
            {
                "history_interval_count": group,
                "outcome_known_count": raw["outcome_known_count"],
                "raw_ipcw_brier": raw["ipcw_brier_score"],
                "isotonic_ipcw_brier": calibrated["ipcw_brier_score"],
            }
        )
    return {
        "status": "synthetic_posthoc_summary_not_model_selection",
        "freeze_sha256": result["freeze_sha256"],
        "development_result_sha256": result["development_result_sha256"],
        "landmarks": landmarks,
        "elapsed_zero_low_history": low_history,
        "uniform_brier_improvement": all(
            row["point_brier_improvement"] > 0 for row in landmarks
        ),
        "operational_probability_publication_approved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("출력 경로는 존재하지 않는 파일과 기존 상위 폴더여야 합니다.")
    summary = summarize(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": summary["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
