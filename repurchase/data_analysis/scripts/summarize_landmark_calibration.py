"""기존 Validation JSON에서 시점별 조건부 확률의 보정 오차를 진단합니다.

모델을 학습·보정하거나 Test를 평가하지 않습니다. 원본 평가 JSON은 보존합니다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: 유한한 숫자가 필요합니다.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name}: 유한한 숫자가 필요합니다.")
    return result


def _nonnegative_count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name}: 0 이상의 정수가 필요합니다.")
    return value


def summarize_landmark_calibration(report: dict[str, Any]) -> list[dict[str, Any]]:
    """IPCW calibration 구간을 표본 지지와 오차 방향으로 요약합니다."""
    if report.get("test_evaluated") is not False:
        raise ValueError("Test 미평가 Validation 결과만 진단할 수 있습니다.")
    landmarks = report.get("conditional_aft_landmarks")
    if not isinstance(landmarks, dict) or landmarks.get("test_evaluated") is not False:
        raise ValueError("시점별 Validation 결과가 필요합니다.")
    summaries = landmarks.get("summary")
    calibration = landmarks.get("calibration")
    if not isinstance(summaries, list) or not summaries:
        raise ValueError("시점별 평가 요약이 비어 있습니다.")
    if not isinstance(calibration, list) or not calibration:
        raise ValueError("시점별 calibration 구간이 비어 있습니다.")

    by_day: dict[int, list[dict[str, Any]]] = {}
    for row in calibration:
        if not isinstance(row, dict):
            raise ValueError("calibration 구간 형식이 올바르지 않습니다.")
        day = _nonnegative_count(row.get("elapsed_days"), "elapsed_days")
        by_day.setdefault(day, []).append(row)

    results: list[dict[str, Any]] = []
    seen_days: set[int] = set()
    for summary in summaries:
        if not isinstance(summary, dict):
            raise ValueError("시점별 평가 요약 형식이 올바르지 않습니다.")
        day = _nonnegative_count(summary.get("elapsed_days"), "elapsed_days")
        if day in seen_days:
            raise ValueError("시점별 평가 요약에 중복 시점이 있습니다.")
        seen_days.add(day)
        rows = by_day.get(day)
        if not rows:
            raise ValueError(f"{day}일 calibration 구간이 없습니다.")
        at_risk = _nonnegative_count(summary.get("at_risk_count"), "at_risk_count")
        known = _nonnegative_count(
            summary.get("outcome_known_count"), "outcome_known_count"
        )
        if not 0 < known <= at_risk:
            raise ValueError(f"{day}일 평가 표본 수가 올바르지 않습니다.")

        sample_sum = 0
        weight_share_sum = 0.0
        predicted = 0.0
        observed = 0.0
        ece = 0.0
        under_share = 0.0
        over_share = 0.0
        bins: list[dict[str, Any]] = []
        seen_bins: set[int] = set()
        for row in rows:
            index = _nonnegative_count(
                row.get("calibration_bin_index"), "calibration_bin_index"
            )
            if index in seen_bins:
                raise ValueError(f"{day}일 calibration 구간이 중복됐습니다.")
            seen_bins.add(index)
            count = _nonnegative_count(row.get("sample_count"), "sample_count")
            share = _finite_number(row.get("ipcw_weight_share"), "ipcw_weight_share")
            mean_pred = _finite_number(
                row.get("mean_predicted_probability"), "mean_predicted_probability"
            )
            event_rate = _finite_number(
                row.get("observed_event_rate"), "observed_event_rate"
            )
            if count == 0 or not 0 < share <= 1:
                raise ValueError(
                    f"{day}일 calibration 표본·가중치가 올바르지 않습니다."
                )
            if not 0 <= mean_pred <= 1 or not 0 <= event_rate <= 1:
                raise ValueError(f"{day}일 calibration 확률이 범위를 벗어났습니다.")
            gap = mean_pred - event_rate
            if not math.isclose(
                gap,
                _finite_number(row.get("calibration_gap"), "calibration_gap"),
                abs_tol=1e-10,
            ):
                raise ValueError(f"{day}일 calibration 오차가 원본과 다릅니다.")
            sample_sum += count
            weight_share_sum += share
            predicted += share * mean_pred
            observed += share * event_rate
            ece += share * abs(gap)
            if gap < 0:
                under_share += share
            elif gap > 0:
                over_share += share
            bins.append(
                {
                    "bin_index": index,
                    "sample_count": count,
                    "ipcw_weight_share": share,
                    "probability_gap": gap,
                }
            )
        expected_ece = _finite_number(
            summary.get("expected_calibration_error"), "expected_calibration_error"
        )
        if sample_sum != known or not math.isclose(weight_share_sum, 1, abs_tol=1e-9):
            raise ValueError(f"{day}일 calibration 표본·가중치 합계가 다릅니다.")
        if not math.isclose(ece, expected_ece, abs_tol=1e-9):
            raise ValueError(f"{day}일 ECE가 원본 평가와 다릅니다.")
        results.append(
            {
                "elapsed_days": day,
                "at_risk_count": at_risk,
                "outcome_known_count": known,
                "nonempty_bin_count": len(bins),
                "weighted_mean_predicted_probability": predicted,
                "weighted_observed_event_rate": observed,
                "weighted_probability_gap": predicted - observed,
                "expected_calibration_error": ece,
                "underprediction_ipcw_weight_share": under_share,
                "overprediction_ipcw_weight_share": over_share,
                "bins": sorted(bins, key=lambda row: row["bin_index"]),
            }
        )
    if set(by_day) != seen_days:
        raise ValueError("평가 요약에 없는 calibration 시점이 있습니다.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Validation 비교 JSON"
    )
    parser.add_argument("--output", type=Path, required=True, help="진단 결과 JSON")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("입력과 출력 경로는 달라야 합니다.")
    try:
        raw = args.input.read_bytes()
        report = json.loads(raw)
        if not isinstance(report, dict):
            raise ValueError("평가 결과의 최상위 형식이 올바르지 않습니다.")
        result = {
            "event": "repurchase_landmark_calibration_diagnostics",
            "input_sha256": hashlib.sha256(raw).hexdigest(),
            "test_evaluated": False,
            "summary": summarize_landmark_calibration(report),
        }
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"재구매 보정 진단 실패: {exc}\n")
    print(json.dumps({"output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
