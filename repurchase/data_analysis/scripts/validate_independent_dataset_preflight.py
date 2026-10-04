"""새 보정 개발·최종 평가용 목데이터의 파일·기간·주문 키 분리를 사전검사합니다.

이 검사는 정합성 감사나 모델 평가가 아닙니다. 원천 6종의 상세 시점·클레임
정합성은 별도 감사로 확인해야 하며, 통과해도 사용자 확률 공개는 승인되지 않습니다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.export_local_service_snapshot import verify_snapshot
from scripts.run_service_model_comparison import _file_sha256
from scripts.validate_service_final_evaluation import SOURCE_NAMES

ROLES = ("calibration_development", "final_evaluation")
REQUIRED_COLUMNS = {
    "orders": {
        "order_id",
        "user_id",
        "ordered_at",
        "paid_at",
        "order_status",
        "purchase_type",
    },
    "order_items": {
        "order_item_id",
        "order_id",
        "product_id",
        "product_group_id_snapshot",
        "category_code_snapshot",
        "is_replenishable_snapshot",
        "pet_id",
        "quantity",
        "item_status",
        "cancelled_quantity",
        "returned_quantity",
    },
    "histories": {"history_id", "order_id", "from_status", "to_status", "changed_at"},
    "claims": {
        "claim_id",
        "order_id",
        "claim_type",
        "claim_status",
        "requested_at",
        "completed_at",
    },
    "claim_items": {"claim_item_id", "claim_id", "order_item_id", "quantity"},
    "pets": {"pet_id", "user_id", "birth_date"},
}


def _timestamp(value: object, name: str) -> pd.Timestamp:
    if not isinstance(value, str):
        raise ValueError(f"{name}은 시간대가 있는 시각이어야 합니다.")
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}은 유효한 시각이어야 합니다.") from exc
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError(f"{name}은 시간대가 있는 시각이어야 합니다.")
    return result


def _plan(directory: Path, role: str) -> dict[str, object]:
    plan = json.loads((directory / "evaluation-plan.json").read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("schema_version") != 1:
        raise ValueError("평가 계획 버전이 유효하지 않습니다.")
    if plan.get("dataset_role") != role:
        raise ValueError("평가 데이터 역할이 기대한 역할과 다릅니다.")
    for key in ("dataset_run_id", "generator_version", "source_dataset_version"):
        if not isinstance(plan.get(key), str) or not plan[key].strip():
            raise ValueError(f"{key}가 필요합니다.")
    digest = plan.get("config_hash")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("config_hash는 소문자 SHA-256이어야 합니다.")
    if type(plan.get("random_seed")) is not int:
        raise ValueError("random_seed는 정수여야 합니다.")
    start = _timestamp(plan.get("evaluation_start_at"), "evaluation_start_at")
    end = _timestamp(plan.get("observation_end_at"), "observation_end_at")
    if start >= end:
        raise ValueError("평가 시작 시각은 관측 종료보다 빨라야 합니다.")
    return plan


def _order_ids_in_window(
    path: Path, start: pd.Timestamp, end: pd.Timestamp, horizon_days: int
) -> tuple[set[str], int]:
    orders = pd.read_csv(path, usecols=["order_id", "paid_at"], dtype="string")
    if orders["order_id"].isna().any() or orders["order_id"].duplicated().any():
        raise ValueError("주문 ID가 누락되거나 중복됐습니다.")
    parsed_paid = orders["paid_at"].map(
        lambda value: pd.NaT if pd.isna(value) else _timestamp(value, "paid_at")
    )
    paid = pd.to_datetime(parsed_paid, utc=True, errors="coerce")
    if orders["paid_at"].notna().sum() != paid.notna().sum():
        raise ValueError("결제 시각 형식이 유효하지 않습니다.")
    if paid.gt(end).any():
        raise ValueError("관측 종료 이후 결제 주문이 포함됐습니다.")
    eligible = paid.ge(start) & paid.le(end)
    mature = eligible & paid.le(end - pd.Timedelta(days=horizon_days))
    return set(orders.loc[eligible, "order_id"].astype(str)), int(mature.sum())


def validate_preflight(
    baseline_dir: Path,
    final_test_result: Path,
    development_dir: Path,
    evaluation_dir: Path,
) -> dict[str, object]:
    """파일 지문, 생성 실행과 평가 주문의 시간·키 분리를 검사합니다."""
    baseline_manifest = verify_snapshot(baseline_dir)
    final = json.loads(final_test_result.read_text(encoding="utf-8"))
    if not isinstance(final, dict) or final.get("test_evaluated") is not True:
        raise ValueError("완료된 기존 최종 Test 결과가 필요합니다.")
    expected = final.get("source_sha256")
    if not isinstance(expected, dict) or set(expected) != set(SOURCE_NAMES):
        raise ValueError("기존 최종 Test의 원천 지문이 완전하지 않습니다.")
    for name in SOURCE_NAMES:
        paths = [baseline_dir / f"{name}.csv"]
        if name == "order_items":
            paths.append(baseline_dir / "order_items_model.csv")
        if not any(
            path.is_file() and _file_sha256(path) == expected[name] for path in paths
        ):
            raise ValueError(f"기존 Test의 {name} 원천이 기준 스냅샷과 다릅니다.")
    if baseline_manifest["files"]["orders"]["rows"] < 1:
        raise ValueError("기준 스냅샷에 주문이 없습니다.")
    cutoff = _timestamp(final.get("observation_end_at"), "기존 Test 관측 종료")
    horizon = final.get("horizon_days")
    if type(horizon) is not int or horizon < 1:
        raise ValueError("기존 평가 기간이 유효하지 않습니다.")
    baseline_ids = set(
        pd.read_csv(baseline_dir / "orders.csv", usecols=["order_id"], dtype="string")[
            "order_id"
        ]
        .dropna()
        .astype(str)
    )
    plans = {}
    windows = {}
    for role, directory in zip(ROLES, (development_dir, evaluation_dir), strict=True):
        verify_snapshot(directory)
        for name, required in REQUIRED_COLUMNS.items():
            columns = set(pd.read_csv(directory / f"{name}.csv", nrows=0).columns)
            if required - columns:
                raise ValueError(f"{role}의 {name} 원천 컬럼이 누락됐습니다.")
        plan = _plan(directory, role)
        start = _timestamp(plan["evaluation_start_at"], "evaluation_start_at")
        end = _timestamp(plan["observation_end_at"], "observation_end_at")
        ids, mature_count = _order_ids_in_window(
            directory / "orders.csv", start, end, horizon
        )
        if not ids or mature_count < 1:
            raise ValueError(
                f"{role}에 평가 주문과 {horizon}일 관측 가능 주문이 필요합니다."
            )
        if ids & baseline_ids:
            raise ValueError(f"{role} 평가 주문 ID가 기존 데이터와 겹칩니다.")
        plans[role] = plan
        windows[role] = (start, end, ids, mature_count)
    development = windows["calibration_development"]
    evaluation = windows["final_evaluation"]
    if not cutoff < development[0] or not development[1] < evaluation[0]:
        raise ValueError(
            "기존 Test·보정 개발·최종 평가의 시간 구간이 분리되지 않았습니다."
        )
    if development[2] & evaluation[2]:
        raise ValueError("보정 개발과 최종 평가의 주문 ID가 겹칩니다.")
    dev_plan = plans["calibration_development"]
    eval_plan = plans["final_evaluation"]
    if dev_plan["dataset_run_id"] == eval_plan["dataset_run_id"] or (
        dev_plan["generator_version"],
        dev_plan["config_hash"],
        dev_plan["random_seed"],
    ) == (
        eval_plan["generator_version"],
        eval_plan["config_hash"],
        eval_plan["random_seed"],
    ):
        raise ValueError("두 평가 데이터의 생성 실행이 독립적으로 식별되지 않습니다.")
    return {
        "status": "preflight_passed_not_source_audit_or_model_approval",
        "baseline_test_result_sha256": _file_sha256(final_test_result),
        "baseline_observation_end_at": cutoff.isoformat(),
        "horizon_days": horizon,
        "datasets": {
            role: {
                "dataset_run_id": plans[role]["dataset_run_id"],
                "evaluation_start_at": windows[role][0].isoformat(),
                "observation_end_at": windows[role][1].isoformat(),
                "evaluation_order_count": len(windows[role][2]),
                "mature_order_count": windows[role][3],
            }
            for role in ROLES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-directory", type=Path, required=True)
    parser.add_argument("--final-test-result", type=Path, required=True)
    parser.add_argument("--development-directory", type=Path, required=True)
    parser.add_argument("--evaluation-directory", type=Path, required=True)
    args = parser.parse_args()
    result = validate_preflight(
        args.baseline_directory,
        args.final_test_result,
        args.development_directory,
        args.evaluation_directory,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
