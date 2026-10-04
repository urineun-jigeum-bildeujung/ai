"""기존 Validation/Test의 모집단 변화를 사후 집계합니다.

새 독립 평가가 아니며 이 결과로 모델 선택·보정을 수행하지 않습니다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.diagnose_frozen_service_test import _load_verified_inputs
from scripts.modeling.maturity_analysis import add_split_ipcw_weights
from scripts.run_frozen_service_test import _test_rows
from scripts.run_service_model_comparison import _file_sha256


def summarize_population(rows: pd.DataFrame) -> dict[str, object]:
    """전체 앵커 구성과 정답 확인 표본의 IPCW 관측률을 분리해 집계합니다."""
    required = {
        "user_id",
        "target_id",
        "history_interval_count",
        "user_prior_order_count",
        "ipcw_outcome_known",
        "ipcw_event_within_horizon",
        "ipcw_weight",
    }
    if required - set(rows):
        raise ValueError("모집단 집계에 필요한 열이 누락됐습니다.")
    if rows.empty or rows[["user_id", "target_id"]].isna().any().any():
        raise ValueError("모집단이 비었거나 대상 키가 누락됐습니다.")
    result: dict[str, object] = {
        "sample_count": len(rows),
        "user_count": int(rows["user_id"].nunique()),
        "product_group_count": int(rows["target_id"].nunique()),
        "outcome_known_count": int(rows["ipcw_outcome_known"].sum()),
        "segments": {},
    }
    known = rows.loc[rows["ipcw_outcome_known"]]
    if known.empty:
        raise ValueError("정답 확인 표본이 없습니다.")
    weights = known["ipcw_weight"].to_numpy(dtype="float64")
    outcomes = known["ipcw_event_within_horizon"].to_numpy(dtype="float64")
    if (
        not np.isfinite(weights).all()
        or (weights <= 0).any()
        or not np.isin(outcomes, [0.0, 1.0]).all()
    ):
        raise ValueError("IPCW 가중치나 정답이 유효하지 않습니다.")
    result["weighted_observed_rate"] = float(np.average(outcomes, weights=weights))
    for field, bins, labels in (
        ("history_interval_count", [-0.5, 0.5, 1.5, float("inf")], ["0", "1", "2+"]),
        (
            "user_prior_order_count",
            [-0.5, 0.5, 1.5, 3.5, float("inf")],
            ["0", "1", "2-3", "4+"],
        ),
    ):
        values = pd.to_numeric(rows[field], errors="coerce").to_numpy(dtype="float64")
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or not np.equal(values, np.floor(values)).all()
        ):
            raise ValueError(f"{field}는 0 이상의 정수여야 합니다.")
        bucket = pd.cut(rows[field], bins=bins, labels=labels)
        segments = []
        for label in labels:
            in_bucket = bucket.eq(label).to_numpy(dtype=bool)
            group = rows.loc[in_bucket]
            group_known = group.loc[group["ipcw_outcome_known"]]
            group_weights = group_known["ipcw_weight"].to_numpy(dtype="float64")
            group_outcomes = group_known["ipcw_event_within_horizon"].to_numpy(
                dtype="float64"
            )
            segments.append(
                {
                    "bucket": label,
                    "sample_count": len(group),
                    "sample_share": len(group) / len(rows),
                    "outcome_known_count": len(group_known),
                    "ipcw_weight_share": float(group_weights.sum() / weights.sum()),
                    "weighted_observed_rate": (
                        float(np.average(group_outcomes, weights=group_weights))
                        if len(group_known)
                        else None
                    ),
                }
            )
        if sum(item["sample_count"] for item in segments) != len(rows):
            raise ValueError(f"{field} 구간이 전체 표본을 분할하지 못했습니다.")
        result["segments"][field] = segments
    return result


def decompose_observed_rate_change(
    validation: dict[str, object], test: dict[str, object], field: str
) -> dict[str, float]:
    """Validation 구간 관측률을 기준으로 구성 변화와 구간 내 변화를 분해합니다."""
    before = {row["bucket"]: row for row in validation["segments"][field]}
    after = {row["bucket"]: row for row in test["segments"][field]}
    if set(before) != set(after) or any(
        row["weighted_observed_rate"] is None
        for row in [*before.values(), *after.values()]
    ):
        raise ValueError("두 기간의 관측률 구간을 일대일로 비교할 수 없습니다.")
    composition = sum(
        (after[key]["ipcw_weight_share"] - before[key]["ipcw_weight_share"])
        * before[key]["weighted_observed_rate"]
        for key in before
    )
    within = sum(
        after[key]["ipcw_weight_share"]
        * (after[key]["weighted_observed_rate"] - before[key]["weighted_observed_rate"])
        for key in before
    )
    total = test["weighted_observed_rate"] - validation["weighted_observed_rate"]
    if not np.isclose(composition + within, total, rtol=0, atol=1e-10):
        raise ValueError("관측률 변화 분해가 전체 차이와 일치하지 않습니다.")
    return {
        "total_rate_change": float(total),
        "composition_component": float(composition),
        "within_bucket_component": float(within),
    }


def diagnose_population_shift(
    frozen_dir: Path,
    final_result: Path,
    validation_result: Path,
    snapshot_dir: Path,
) -> dict[str, object]:
    """원래 두 평가의 표본·관측률을 재현한 경우에만 사후 비교를 반환합니다."""
    final, sources = _load_verified_inputs(frozen_dir, final_result, snapshot_dir)
    validation = json.loads(validation_result.read_text(encoding="utf-8"))
    record = json.loads((frozen_dir / "freeze-record.json").read_text(encoding="utf-8"))
    if (
        record.get("validation_result_sha256") != _file_sha256(validation_result)
        or validation.get("source_sha256") != final.get("source_sha256")
        or validation.get("validation_end_at") != final.get("validation_end_at")
        or validation.get("test_evaluated") is not False
    ):
        raise ValueError("Validation 결과가 고정 모델·최종 Test와 일치하지 않습니다.")
    train_end = pd.Timestamp(validation["train_end_at"])
    validation_end = pd.Timestamp(validation["validation_end_at"])
    observation_end = pd.Timestamp(final["observation_end_at"])
    if not train_end < validation_end < observation_end:
        raise ValueError("평가 기간 컷이 유효하지 않습니다.")
    populations = {}
    for name, start, end in (
        ("validation", train_end, validation_end),
        ("test", validation_end, observation_end),
    ):
        rows = _test_rows(sources, validation_end_at=start, observation_end_at=end)
        weighted = add_split_ipcw_weights(rows, horizon_days=int(final["horizon_days"]))
        populations[name] = summarize_population(weighted)
    validation_summary = validation["summary"][0]
    final_summary = final["model_metrics"][0]
    for name, summary, count_key in (
        ("validation", validation_summary, "validation_sample_count"),
        ("test", final_summary, "test_sample_count"),
    ):
        population = populations[name]
        if (
            population["sample_count"] != summary[count_key]
            or population["outcome_known_count"] != summary["outcome_known_count"]
            or not np.isclose(
                population["weighted_observed_rate"],
                summary["weighted_observed_event_rate"],
                rtol=0,
                atol=1e-10,
            )
        ):
            raise ValueError(f"{name} 재구성 결과가 저장된 평가와 다릅니다.")
    return {
        "analysis_type": "posthoc_population_shift_not_model_selection",
        "validation_result_sha256": _file_sha256(validation_result),
        "final_result_sha256": _file_sha256(final_result),
        "source_sha256": final["source_sha256"],
        "populations": populations,
        "descriptive_rate_decomposition": {
            field: decompose_observed_rate_change(
                populations["validation"], populations["test"], field
            )
            for field in ("history_interval_count", "user_prior_order_count")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", required=True, type=Path)
    parser.add_argument("--final-test-result", required=True, type=Path)
    parser.add_argument("--validation-result", required=True, type=Path)
    parser.add_argument("--snapshot-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("기존 결과를 덮어쓸 수 없습니다.")
    result = diagnose_population_shift(
        args.frozen_dir,
        args.final_test_result,
        args.validation_result,
        args.snapshot_directory,
    )
    with args.output.open("x", encoding="utf-8") as destination:
        destination.write(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps({"output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
