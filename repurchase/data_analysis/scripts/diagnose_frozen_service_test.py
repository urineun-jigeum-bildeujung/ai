"""봉인된 Test의 고정 예측을 재구성해 사후 오차 구간만 진단합니다.

모델 선택·보정·재학습을 수행하지 않습니다. 출력은 집계값뿐이며 새 독립 평가가
아닙니다. 기존 최종 Test 결과의 지표와 일치하지 않으면 결과를 저장하지 않습니다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.artifacts import load_model_artifact, predict_artifact_probability
from scripts.modeling.maturity_analysis import add_split_ipcw_weights
from scripts.modeling.service_model_comparison import (
    summarize_service_brier_attribution,
)
from scripts.run_frozen_service_test import _test_rows
from scripts.run_service_model_comparison import (
    _file_sha256,
    _read_sources,
    model_code_sha256,
)
from scripts.validate_service_final_evaluation import SOURCE_NAMES

RELEVANT_CODE_FILES = frozenset(
    {
        "artifacts.py",
        "current_prediction.py",
        "evaluation.py",
        "features.py",
        "inference_features.py",
        "lightgbm_baseline.py",
        "maturity_analysis.py",
        "operational_event_intervals.py",
        "operational_quantity_intervals.py",
        "operational_source_quarantine.py",
        "operational_status_intervals.py",
        "operational_temporal_split.py",
        "operational_training_samples.py",
        "operational_validity_intervals.py",
        "service_model_comparison.py",
        "xgboost_aft.py",
    }
)


def _load_verified_inputs(
    frozen_dir: Path, final_result: Path, snapshot_dir: Path
) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    """완료된 Test 기록, 고정 모델과 원천·평가 코드 지문을 확인합니다."""
    result = json.loads(final_result.read_text(encoding="utf-8"))
    state = json.loads(
        (final_result.parent / "execution-state.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (frozen_dir / "pretest-manifest.json").read_text(encoding="utf-8")
    )
    if (
        state.get("status") != "completed"
        or state.get("result_sha256") != _file_sha256(final_result)
        or result.get("test_evaluated") is not True
        or result.get("evaluator_sha256")
        != _file_sha256(Path(__file__).with_name("run_frozen_service_test.py"))
    ):
        raise ValueError("완료된 최종 Test 기록과 평가 코드가 일치하지 않습니다.")
    if (
        result.get("source_sha256") != manifest.get("source_sha256")
        or result.get("validation_end_at") != manifest.get("validation_end_at")
        or result.get("observation_end_at")
        != manifest.get("observation_end_at_assumption")
        or result.get("horizon_days")
        != manifest.get("evaluation", {}).get("horizon_days")
    ):
        raise ValueError("최종 Test와 고정 모델의 원천·시점·평가 기간 계약이 다릅니다.")
    current_code = model_code_sha256()
    recorded_code = manifest["code_sha256"]
    if any(
        current_code.get(name) != recorded_code.get(name)
        for name in RELEVANT_CODE_FILES
    ):
        raise ValueError("최종 Test 이후 평가 관련 모델 코드가 변경됐습니다.")
    paths = {}
    for name in SOURCE_NAMES:
        candidates = [snapshot_dir / f"{name}.csv"]
        if name == "order_items":
            candidates.append(snapshot_dir / "order_items_model.csv")
        matches = [
            path
            for path in candidates
            if path.is_file() and _file_sha256(path) == result["source_sha256"][name]
        ]
        if len(matches) != 1:
            raise ValueError(f"{name} 원천 파일의 지문이 최종 Test와 다릅니다.")
        paths[name] = matches[0]
    return result, _read_sources(paths)


def summarize_segment_error(rows: pd.DataFrame) -> list[dict[str, object]]:
    """같은 정답 확인 행과 IPCW 가중치로 구간별 확률·오차를 집계합니다."""
    required = {
        "user_id",
        "target_id",
        "history_interval_count",
        "user_prior_order_count",
        "ipcw_outcome_known",
        "ipcw_event_within_horizon",
        "ipcw_weight",
        "reference_predicted_event_probability",
        "candidate_predicted_event_probability",
    }
    if required - set(rows.columns):
        raise ValueError("구간 진단에 필요한 평가 열이 누락됐습니다.")
    known = rows.loc[rows["ipcw_outcome_known"]].copy()
    if known.empty:
        raise ValueError("정답 확인 행이 없습니다.")
    weight = known["ipcw_weight"].to_numpy(dtype="float64")
    outcome = known["ipcw_event_within_horizon"].to_numpy(dtype="float64")
    aft = known["reference_predicted_event_probability"].to_numpy(dtype="float64")
    lightgbm = known["candidate_predicted_event_probability"].to_numpy(dtype="float64")
    if (
        not np.isfinite(weight).all()
        or (weight <= 0).any()
        or not np.isfinite(aft).all()
        or not np.isfinite(lightgbm).all()
        or (aft < 0).any()
        or (aft > 1).any()
        or (lightgbm < 0).any()
        or (lightgbm > 1).any()
        or not np.isin(outcome, [0.0, 1.0]).all()
    ):
        raise ValueError("구간 진단의 정답·가중치·확률이 유효하지 않습니다.")
    known["weighted_outcome"] = weight * outcome
    known["weighted_aft_probability"] = weight * aft
    known["weighted_lightgbm_probability"] = weight * lightgbm
    known["aft_weighted_error"] = weight * (aft - outcome) ** 2
    known["lightgbm_weighted_error"] = weight * (lightgbm - outcome) ** 2
    known["history_bucket"] = pd.cut(
        known["history_interval_count"],
        bins=[-0.5, 0.5, 1.5, float("inf")],
        labels=["0", "1", "2+"],
    )
    known["user_order_bucket"] = pd.cut(
        known["user_prior_order_count"],
        bins=[-0.5, 0.5, 1.5, 3.5, float("inf")],
        labels=["0", "1", "2-3", "4+"],
    )
    total_weight = float(weight.sum())
    output: list[dict[str, object]] = []
    for segment_kind, column in (
        ("history_interval_count", "history_bucket"),
        ("user_prior_order_count", "user_order_bucket"),
        ("product_group", "target_id"),
    ):
        for segment_value, group in known.groupby(column, observed=True, sort=True):
            group_weight = float(group["ipcw_weight"].sum())
            aft_error = float(group["aft_weighted_error"].sum())
            lightgbm_error = float(group["lightgbm_weighted_error"].sum())
            output.append(
                {
                    "segment_kind": segment_kind,
                    "segment_value": str(segment_value),
                    "outcome_known_count": len(group),
                    "user_count": int(group["user_id"].nunique()),
                    "ipcw_weight_share": group_weight / total_weight,
                    "weighted_observed_rate": float(group["weighted_outcome"].sum())
                    / group_weight,
                    "weighted_aft_probability": float(
                        group["weighted_aft_probability"].sum()
                    )
                    / group_weight,
                    "weighted_lightgbm_probability": float(
                        group["weighted_lightgbm_probability"].sum()
                    )
                    / group_weight,
                    "aft_brier": aft_error / group_weight,
                    "lightgbm_brier": lightgbm_error / group_weight,
                    "global_brier_difference_contribution": (aft_error - lightgbm_error)
                    / total_weight,
                }
            )
    return output


def diagnose_frozen_test(
    frozen_dir: Path, final_result: Path, snapshot_dir: Path
) -> dict[str, object]:
    """최종 Test의 원래 지표를 재현한 뒤, 사후 구간 진단만 반환합니다."""
    result, sources = _load_verified_inputs(frozen_dir, final_result, snapshot_dir)
    test = _test_rows(
        sources,
        validation_end_at=pd.Timestamp(result["validation_end_at"]),
        observation_end_at=pd.Timestamp(result["observation_end_at"]),
    )
    weighted = add_split_ipcw_weights(test, horizon_days=int(result["horizon_days"]))
    if (
        len(weighted) != result["sample_count"]
        or int(weighted["ipcw_outcome_known"].sum()) != result["outcome_known_count"]
    ):
        raise ValueError("재구성한 최종 Test 표본 수가 저장된 결과와 다릅니다.")
    paired = weighted.copy()
    for family, column in (
        ("xgboost_aft", "reference_predicted_event_probability"),
        ("lightgbm", "candidate_predicted_event_probability"),
    ):
        artifact = load_model_artifact(frozen_dir / family)
        if artifact.artifact_id != result["artifact_ids"][family]:
            raise ValueError(f"{family} 고정 모델 지문이 다릅니다.")
        paired[column] = predict_artifact_probability(artifact, test)
    # 기존 기여도 함수가 입력 확률·가중치와 축별 합계 보존을 함께 검증합니다.
    summarize_service_brier_attribution(paired)
    segments = summarize_segment_error(paired)
    saved = {row["model"]: row for row in result["model_metrics"]}
    expected_difference = float(saved["xgboost_aft"]["ipcw_brier_score"]) - float(
        saved["lightgbm"]["ipcw_brier_score"]
    )
    for kind in ("history_interval_count", "user_prior_order_count", "product_group"):
        actual = sum(
            row["global_brier_difference_contribution"]
            for row in segments
            if row["segment_kind"] == kind
        )
        if not np.isclose(actual, expected_difference, rtol=0, atol=1e-10):
            raise ValueError("구간별 Brier 차이가 저장된 최종 Test 지표와 다릅니다.")
    return {
        "analysis_type": "posthoc_frozen_test_diagnostics_not_model_selection",
        "final_result_sha256": _file_sha256(final_result),
        "source_sha256": result["source_sha256"],
        "artifact_ids": result["artifact_ids"],
        "sample_count": len(test),
        "outcome_known_count": int(weighted["ipcw_outcome_known"].sum()),
        "aft_minus_lightgbm_brier": expected_difference,
        "segment_rows": segments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", required=True, type=Path)
    parser.add_argument("--final-test-result", required=True, type=Path)
    parser.add_argument("--snapshot-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("기존 진단 결과를 덮어쓸 수 없습니다.")
    result = diagnose_frozen_test(
        args.frozen_dir, args.final_test_result, args.snapshot_directory
    )
    with args.output.open("x", encoding="utf-8") as destination:
        destination.write(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    print(json.dumps({"output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
