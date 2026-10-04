"""새 생성 데이터의 사전 분리 검사 계약을 작은 가상 CSV로 확인합니다."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_independent_dataset_preflight import (
    REQUIRED_COLUMNS,
    validate_preflight,
)


def _snapshot(directory: Path, *, order_id: str, paid_at: str) -> dict[str, object]:
    directory.mkdir()
    files = {}
    for name, columns in REQUIRED_COLUMNS.items():
        header = ",".join(sorted(columns))
        if name == "orders":
            values = {column: "" for column in columns}
            values.update(
                {
                    "order_id": order_id,
                    "user_id": "7",
                    "ordered_at": paid_at,
                    "paid_at": paid_at,
                    "order_status": "PAID",
                    "purchase_type": "NORMAL",
                }
            )
            text = (
                header
                + "\n"
                + ",".join(values[column] for column in sorted(columns))
                + "\n"
            )
            rows = 1
        else:
            text = header + "\n"
            rows = 0
        path = directory / f"{name}.csv"
        path.write_text(text, encoding="utf-8")
        files[name] = {
            "filename": path.name,
            "rows": rows,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (directory / "manifest.json").write_text(
        json.dumps({"files": files}), encoding="utf-8"
    )
    return {name: entry["sha256"] for name, entry in files.items()}


def _plan(directory: Path, role: str, run_id: str, start: str, end: str) -> None:
    (directory / "evaluation-plan.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_role": role,
                "dataset_run_id": run_id,
                "generator_version": "test-v1",
                "source_dataset_version": "synthetic-v1",
                "config_hash": "a" * 64
                if role == "calibration_development"
                else "b" * 64,
                "random_seed": 1 if role == "calibration_development" else 2,
                "evaluation_start_at": start,
                "observation_end_at": end,
            }
        ),
        encoding="utf-8",
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    baseline = tmp_path / "baseline"
    development = tmp_path / "development"
    evaluation = tmp_path / "evaluation"
    hashes = _snapshot(baseline, order_id="1", paid_at="2025-12-01T00:00:00Z")
    _snapshot(development, order_id="2", paid_at="2026-02-01T00:00:00Z")
    _snapshot(evaluation, order_id="3", paid_at="2026-04-01T00:00:00Z")
    _plan(
        development,
        "calibration_development",
        "run-dev",
        "2026-02-01T00:00:00Z",
        "2026-03-10T00:00:00Z",
    )
    _plan(
        evaluation,
        "final_evaluation",
        "run-final",
        "2026-04-01T00:00:00Z",
        "2026-05-10T00:00:00Z",
    )
    final = tmp_path / "final.json"
    final.write_text(
        json.dumps(
            {
                "test_evaluated": True,
                "source_sha256": hashes,
                "observation_end_at": "2026-01-01T00:00:00Z",
                "horizon_days": 30,
            }
        ),
        encoding="utf-8",
    )
    return baseline, final, development, evaluation


def test_preflight_accepts_separate_generated_windows(tmp_path: Path) -> None:
    result = validate_preflight(*_inputs(tmp_path))
    assert result["status"] == "preflight_passed_not_source_audit_or_model_approval"
    assert result["datasets"]["final_evaluation"]["mature_order_count"] == 1


def test_preflight_rejects_overlapping_evaluation_order(tmp_path: Path) -> None:
    baseline, final, development, evaluation = _inputs(tmp_path)
    order_path = evaluation / "orders.csv"
    content = order_path.read_text(encoding="utf-8")
    order_path.write_text(content.replace("3,PAID,", "2,PAID,"), encoding="utf-8")
    # 내용 변경을 감지하는 것이 먼저이므로 manifest를 맞춘 뒤 키 분리를 검사한다.
    manifest_path = evaluation / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["orders"]["sha256"] = hashlib.sha256(
        order_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="주문 ID가 겹칩니다"):
        validate_preflight(baseline, final, development, evaluation)


def test_preflight_rejects_same_generation_run(tmp_path: Path) -> None:
    baseline, final, development, evaluation = _inputs(tmp_path)
    plan_path = evaluation / "evaluation-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["dataset_run_id"] = "run-dev"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="독립적으로 식별"):
        validate_preflight(baseline, final, development, evaluation)


def test_preflight_rejects_paid_at_without_timezone(tmp_path: Path) -> None:
    baseline, final, development, evaluation = _inputs(tmp_path)
    order_path = evaluation / "orders.csv"
    content = order_path.read_text(encoding="utf-8")
    order_path.write_text(
        content.replace(
            "2026-04-01T00:00:00Z,2026-04-01T00:00:00Z",
            "2026-04-01T00:00:00Z,2026-04-01T00:00:00",
        ),
        encoding="utf-8",
    )
    manifest_path = evaluation / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["orders"]["sha256"] = hashlib.sha256(
        order_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="paid_at은 시간대"):
        validate_preflight(baseline, final, development, evaluation)


def test_preflight_rejects_unmatured_orders(tmp_path: Path) -> None:
    baseline, final, development, evaluation = _inputs(tmp_path)
    plan_path = evaluation / "evaluation-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["observation_end_at"] = "2026-04-15T00:00:00Z"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="관측 가능 주문"):
        validate_preflight(baseline, final, development, evaluation)
