"""Validation 결정만으로 최종 Test 이전의 두 모델 후보를 고정합니다."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd

from scripts.modeling.artifacts import load_model_artifact, save_model_artifact
from scripts.modeling.lightgbm_baseline import (
    build_lightgbm_training_data,
    train_lightgbm_classifier,
)
from scripts.modeling.maturity_analysis import add_split_ipcw_weights
from scripts.modeling.operational_aft_input import build_service_aft_training_rows
from scripts.modeling.operational_asof import _utc
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
from scripts.modeling.operational_temporal_split import _attach_evaluation_contract
from scripts.modeling.operational_training_samples import (
    build_temporal_service_training_samples,
)
from scripts.modeling.operational_validity_intervals import (
    build_valid_purchase_item_intervals,
)
from scripts.modeling.service_model_comparison import _canonical_service_train_order
from scripts.modeling.xgboost_aft import (
    build_xgboost_aft_training_data,
    train_xgboost_aft_model,
)
from scripts.run_service_model_comparison import (
    _file_sha256,
    _read_sources,
    model_code_sha256,
)
from scripts.validate_service_final_evaluation import (
    FROZEN_AFT,
    SOURCE_NAMES,
    validate_manifest,
)


def build_pretest_manifest(comparison: dict[str, object]) -> dict[str, object]:
    """실험 결과의 출처를 묶고 최종 후보·평가 계약은 독립 상수로 고정합니다."""
    copied = (
        "source_sha256",
        "code_sha256",
        "observation_end_at_assumption",
        "train_end_at",
        "validation_end_at",
        "train_fraction",
        "validation_fraction",
        "feature_generation_version",
        "runtime_versions",
        "model_configuration",
        "evaluation_population_policy",
    )
    manifest = {key: comparison[key] for key in copied}
    manifest.update(
        schema_version=1,
        aft_configuration=FROZEN_AFT.copy(),
        evaluation={
            "horizon_days": 30,
            "primary_metric": "ipcw_brier_score",
            "bootstrap_unit": "user",
            "bootstrap_replicates": 1000,
            "bootstrap_random_seed": 42,
        },
    )
    return manifest


def build_refit_rows(
    sources: dict[str, pd.DataFrame], *, validation_end_at: pd.Timestamp
) -> pd.DataFrame:
    """Validation 종료 컷에서 라벨과 시점 피처를 새로 생성합니다. Test는 만들지 않습니다."""
    sources = _eligible_training_sources(sources, validation_end_at=validation_end_at)
    quarantine = quarantine_unrestorable_orders(
        sources["orders"],
        sources["order_items"],
        sources["histories"],
        sources["claims"],
        sources["claim_items"],
        as_of_at=validation_end_at,
    )
    orders = quarantine.orders
    items = quarantine.order_items
    status = build_order_status_intervals(quarantine.status_histories)
    quantity = build_order_item_quantity_intervals(
        orders, items, quarantine.claims, quarantine.claim_items
    )
    valid = build_valid_purchase_item_intervals(status, quantity)
    events = build_operational_event_intervals(valid, orders, items, sources["pets"])
    rows = build_temporal_service_training_samples(
        events, orders, observation_end_at=validation_end_at
    )
    if rows.empty or rows["anchor_at"].gt(validation_end_at).any():
        raise ValueError("최종 재학습 표본이 비었거나 Validation 종료 컷을 넘었습니다.")
    return _canonical_service_train_order(
        _attach_evaluation_contract(
            rows, split_name="train", split_end_at=validation_end_at
        )
    )


def _eligible_training_sources(
    sources: dict[str, pd.DataFrame], *, validation_end_at: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    """컷 전 결제 주문과 연결 원천만 남기되 해당 주문의 이후 클레임은 보존합니다."""
    cutoff = _utc(validation_end_at, column="validation_end_at")
    orders = sources["orders"]
    paid_at = orders["paid_at"].map(
        lambda value: None if pd.isna(value) else _utc(value, column="paid_at")
    )
    eligible = orders.loc[
        paid_at.map(lambda value: value is not None and value <= cutoff)
    ].copy()
    items = (
        sources["order_items"]
        .loc[sources["order_items"]["order_id"].isin(eligible["order_id"])]
        .copy()
    )
    histories = (
        sources["histories"]
        .loc[sources["histories"]["order_id"].isin(eligible["order_id"])]
        .copy()
    )
    claims = (
        sources["claims"]
        .loc[sources["claims"]["order_id"].isin(eligible["order_id"])]
        .copy()
    )
    claim_items = (
        sources["claim_items"]
        .loc[sources["claim_items"]["claim_id"].isin(claims["claim_id"])]
        .copy()
    )
    return {
        **sources,
        "orders": eligible,
        "order_items": items,
        "histories": histories,
        "claims": claims,
        "claim_items": claim_items,
    }


def freeze_models(
    comparison_path: Path, sources: dict[str, Path], output_dir: Path
) -> dict[str, object]:
    """원천·설정 검증 후 두 후보를 새 디렉터리에 원자적으로 보존합니다."""
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if (
        not isinstance(comparison, dict)
        or comparison.get("test_evaluated") is not False
    ):
        raise ValueError("Test 평가가 없는 Validation 결과만 고정할 수 있습니다.")
    comparison_sha256 = _file_sha256(comparison_path)
    freeze_code_sha256 = _file_sha256(Path(__file__))
    manifest = build_pretest_manifest(comparison)
    validate_manifest(manifest, comparison, sources)
    if output_dir.exists():
        raise ValueError("기존 모델 고정 디렉터리는 덮어쓸 수 없습니다.")
    if not output_dir.parent.is_dir():
        raise ValueError("모델 고정 결과의 상위 디렉터리가 없습니다.")

    # 검증된 파일도 학습 도중 바뀔 수 있으므로 파싱 전후에 다시 확인합니다.
    data = _read_sources(sources)
    for name in SOURCE_NAMES:
        if _file_sha256(sources[name]) != manifest["source_sha256"][name]:
            raise ValueError(f"학습 중 {name} 원천 파일이 변경됐습니다.")
    validation_end = pd.Timestamp(manifest["validation_end_at"])
    rows = build_refit_rows(data, validation_end_at=validation_end)
    aft_data = build_xgboost_aft_training_data(build_service_aft_training_rows(rows))
    aft = train_xgboost_aft_model(
        aft_data,
        loss_distribution=FROZEN_AFT["loss_distribution"],
        loss_distribution_scale=FROZEN_AFT["loss_distribution_scale"],
        num_boost_round=FROZEN_AFT["num_boost_round"],
    )
    weighted = add_split_ipcw_weights(rows, horizon_days=30)
    lightgbm_data = build_lightgbm_training_data(weighted)
    lightgbm = train_lightgbm_classifier(lightgbm_data)

    with tempfile.TemporaryDirectory(
        prefix=".repurchase-freeze-", dir=output_dir.parent
    ) as temporary:
        temporary_path = Path(temporary)
        frozen = {}
        for family, model in (("xgboost_aft", aft), ("lightgbm", lightgbm)):
            artifact_dir = save_model_artifact(
                model, temporary_path / family, horizon_days=30
            )
            loaded = load_model_artifact(artifact_dir)
            frozen[family] = {
                "artifact_id": loaded.artifact_id,
                "manifest_sha256": _file_sha256(artifact_dir / "manifest.json"),
            }
        record = {
            "schema_version": 1,
            "test_evaluated": False,
            "training_cutoff_at": validation_end.isoformat(),
            "training_sample_count": len(rows),
            "aft_training_sample_count": len(aft_data.row_index),
            "lightgbm_training_sample_count": len(lightgbm_data.target),
            "validation_result_sha256": comparison_sha256,
            "freeze_code_sha256": freeze_code_sha256,
            "source_sha256": manifest["source_sha256"],
            "code_sha256": manifest["code_sha256"],
            "artifacts": frozen,
        }
        (temporary_path / "pretest-manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        (temporary_path / "freeze-record.json").write_text(
            json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        if _file_sha256(comparison_path) != comparison_sha256:
            raise ValueError("학습 중 Validation 결과 파일이 변경됐습니다.")
        validate_manifest(manifest, comparison, sources)
        if model_code_sha256() != manifest["code_sha256"]:
            raise ValueError("학습 중 모델 코드가 변경됐습니다.")
        if _file_sha256(Path(__file__)) != freeze_code_sha256:
            raise ValueError("학습 중 모델 고정 코드가 변경됐습니다.")
        temporary_path.rename(output_dir)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for name in SOURCE_NAMES:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = freeze_models(
            args.validation_result,
            {name: getattr(args, name) for name in SOURCE_NAMES},
            args.output_dir,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
