"""Exercise the frozen evaluator with disposable synthetic data, not sealed holdout."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from scripts.evaluate_independent_synthetic_calibration import (
    _apply_mapping,
    evaluate_frozen,
    validate_freeze,
)
from scripts.generate_independent_synthetic_sources import generate_pair
from scripts.run_service_probability_calibration import run_calibration
from scripts.validate_independent_dataset_preflight import REQUIRED_COLUMNS


def _baseline(tmp_path: Path) -> tuple[Path, Path]:
    directory = tmp_path / "baseline"
    directory.mkdir()
    files = {}
    for name, columns in REQUIRED_COLUMNS.items():
        path = directory / f"{name}.csv"
        path.write_text(",".join(sorted(columns)) + "\n", encoding="utf-8")
        files[name] = {
            "filename": path.name,
            "rows": 0,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    orders = directory / "orders.csv"
    orders.write_text(
        ",".join(sorted(REQUIRED_COLUMNS["orders"]))
        + "\n"
        + ",".join(
            {
                "order_id": "1",
                "user_id": "1",
                "ordered_at": "2026-01-01T00:00:00+00:00",
                "paid_at": "2026-01-01T00:00:00+00:00",
                "order_status": "PAID",
                "purchase_type": "ONE_TIME",
            }[column]
            for column in sorted(REQUIRED_COLUMNS["orders"])
        )
        + "\n",
        encoding="utf-8",
    )
    files["orders"]["rows"] = 1
    files["orders"]["sha256"] = hashlib.sha256(orders.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(
        json.dumps({"files": files}), encoding="utf-8"
    )
    final = tmp_path / "baseline-final.json"
    final.write_text(
        json.dumps(
            {
                "test_evaluated": True,
                "source_sha256": {
                    name: entry["sha256"] for name, entry in files.items()
                },
                "observation_end_at": "2026-10-03T00:00:00+00:00",
                "horizon_days": 30,
            }
        ),
        encoding="utf-8",
    )
    return directory, final


def test_mapping_clips_without_changing_row_order() -> None:
    probability = pd.Series([0.9, 0.1, 0.5], index=[8, 2, 5])
    calibrated = _apply_mapping(
        probability, {"x_thresholds": [0.2, 0.8], "y_thresholds": [0.3, 0.7]}
    )
    assert calibrated.index.equals(probability.index)
    assert calibrated.tolist() == pytest.approx([0.7, 0.3, 0.5])


def test_frozen_evaluator_runs_on_disposable_pair(tmp_path: Path) -> None:
    baseline, baseline_final = _baseline(tmp_path)
    root = tmp_path / "pair"
    generate_pair(root, start=datetime.fromisoformat("2026-10-04T00:00:00+00:00"))
    dev = root / "calibration_development"
    final = root / "final_evaluation"
    names = ("orders", "order_items", "histories", "claims", "claim_items", "pets")
    result = run_calibration(
        {name: dev / f"{name}.csv" for name in names},
        observation_end_at=pd.Timestamp("2028-03-27T00:00:00+00:00"),
        landmark_days=(0, 7, 14, 30),
        bootstrap_replicates=2,
    )
    result_path = tmp_path / "development-result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    freeze = {
        "schema_version": 1,
        "purpose": "synthetic_pipeline_evaluation_only",
        "development_result_sha256": hashlib.sha256(
            result_path.read_bytes()
        ).hexdigest(),
        "development_dataset_run_id": "calibration_development-4201-20261004",
        "final_dataset_run_id": "final_evaluation-4202-20280328",
        "development_manifest_sha256": hashlib.sha256(
            (dev / "manifest.json").read_bytes()
        ).hexdigest(),
        "final_manifest_sha256": hashlib.sha256(
            (final / "manifest.json").read_bytes()
        ).hexdigest(),
        "candidate": "ipcw_weighted_isotonic",
        "aft_num_boost_round": 20,
        "aft_loss_distribution_scale": 1.0,
        "horizon_days": 30,
        "landmark_days": [0, 7, 14, 30],
        "bootstrap_replicates": 1000,
        "bootstrap_random_seed": 42,
        "decision_rule": "development_validation_brier_lower_at_every_landmark",
        "final_evaluation_used_for_selection": False,
        "operational_probability_publication_approved": False,
    }
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    validate_freeze(freeze_path, result_path, dev, final)
    evaluated = evaluate_frozen(
        freeze_path=freeze_path,
        development_result_path=result_path,
        development_dir=dev,
        evaluation_dir=final,
        baseline_dir=baseline,
        baseline_test_result=baseline_final,
    )
    assert evaluated["test_evaluated"] is True
    assert evaluated["operational_probability_publication_approved"] is False
    assert len(evaluated["summary"]) == 8
    assert len(evaluated["aft_ipcw_concordance"]) == 4
    assert evaluated["calibration_bins"]
    assert evaluated["low_history_subgroups"]
    json.dumps(evaluated, allow_nan=False)
    freeze["candidate"] = "raw"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(ValueError, match="후보 선택"):
        validate_freeze(freeze_path, result_path, dev, final)
