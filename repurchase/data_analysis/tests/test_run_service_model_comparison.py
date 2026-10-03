"""서비스 모델 비교 CLI의 원천 CSV 읽기 계약을 확인합니다."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pandas.api.types import is_bool_dtype

from scripts import run_service_model_comparison as runner
from scripts.modeling.service_model_comparison import (
    AFTRoundSelection,
    AFTScaleSelection,
    ProductGroupSmoothingSelection,
)
from scripts.run_service_model_comparison import (
    _file_sha256,
    _read_sources,
    _require_finite_c_index,
    _summarize_validation_population,
)


def test_read_sources_preserves_ids_and_postgres_boolean(tmp_path: Path) -> None:
    """PostgreSQL ID의 문자열과 t/f boolean이 변환 중 손실되지 않습니다."""
    source = tmp_path / "order-items.csv"
    source.write_text(
        "order_item_id,order_id,product_id,product_group_id_snapshot,pet_id,is_replenishable_snapshot\n"
        "001,010,123,456,,t\n"
        "002,011,124,457,999,f\n",
        encoding="utf-8",
    )

    rows = _read_sources({"order_items": source})["order_items"]

    assert rows["order_item_id"].tolist() == ["001", "002"]
    assert rows["order_id"].tolist() == ["010", "011"]
    assert rows["pet_id"].isna().tolist() == [True, False]
    assert is_bool_dtype(rows["is_replenishable_snapshot"])
    assert rows["is_replenishable_snapshot"].tolist() == [True, False]
    assert len(_file_sha256(source)) == 64


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_c_index_is_rejected(score: float) -> None:
    """비유한 C-index는 JSON의 결측값이나 문자열로 저장하지 않습니다."""
    with pytest.raises(ValueError, match="C-index"):
        _require_finite_c_index(pd.DataFrame({"ipcw_c_index": [score]}))


@pytest.mark.parametrize(
    ("observation_end", "replicates"),
    [("2026-09-29T15:44:00", "1000"), ("2026-09-29T15:44:00+09:00", "0")],
)
def test_cli_rejects_invalid_options_before_reading_csv(
    monkeypatch: pytest.MonkeyPatch, observation_end: str, replicates: str
) -> None:
    """시간대와 반복 횟수 오류는 원천 파일 접근 전에 종료됩니다."""
    arguments = ["run_service_model_comparison"]
    for name in (
        "orders",
        "order-items",
        "pets",
        "histories",
        "claims",
        "claim-items",
    ):
        arguments.extend((f"--{name}", "/not-a-real-source.csv"))
    arguments.extend(
        ("--observation-end-at", observation_end, "--bootstrap-replicates", replicates)
    )
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(SystemExit) as exc:
        runner.main()

    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("observation_end", "replicates"),
    [("2026-09-29T15:44:00", 1), ("2026-09-29T15:44:00+09:00", 0)],
)
def test_library_entry_rejects_invalid_options_before_reading_csv(
    observation_end: str, replicates: int
) -> None:
    """직접 함수 호출에서도 잘못된 설정은 같은 단계에서 거절됩니다."""
    with pytest.raises(ValueError):
        runner.run_comparison(
            {},
            observation_end_at=pd.Timestamp(observation_end),
            bootstrap_replicates=replicates,
        )


@pytest.mark.parametrize("days", [(), (7, 7), (-1,), (True,)])
def test_library_rejects_invalid_landmarks_before_reading_csv(
    days: tuple[object, ...],
) -> None:
    with pytest.raises(ValueError, match="조건부 평가 시점"):
        runner.run_comparison(
            {},
            observation_end_at=pd.Timestamp("2026-09-29T15:44:00+09:00"),
            conditional_landmark_days=days,
        )


def test_library_rejects_code_changed_after_import_before_source_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """모듈 로드 후 디스크 코드가 바뀌면 CSV를 읽기 전에 거절합니다."""
    changed = dict(runner.IMPORTED_MODEL_CODE_SHA256)
    changed["run_service_model_comparison.py"] = "0" * 64
    monkeypatch.setattr(runner, "model_code_sha256", lambda: changed)

    def fail_if_source_opened(_: dict[str, Path]) -> dict[str, pd.DataFrame]:
        raise AssertionError("코드 변경 후에는 CSV를 읽으면 안 됩니다.")

    monkeypatch.setattr(runner, "_read_sources", fail_if_source_opened)
    with pytest.raises(ValueError, match="실행 전에 코드 파일이 변경"):
        runner.run_comparison(
            {}, observation_end_at=pd.Timestamp("2026-09-29T15:44:00+09:00")
        )


def test_cli_does_not_write_nonfinite_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """비유한 지표가 들어오면 결과 파일을 만들기 전에 직렬화를 거절합니다."""
    arguments = ["run_service_model_comparison"]
    for name in (
        "orders",
        "order-items",
        "pets",
        "histories",
        "claims",
        "claim-items",
    ):
        arguments.extend((f"--{name}", "/unused.csv"))
    output = tmp_path / "invalid-result.json"
    arguments.extend(
        (
            "--observation-end-at",
            "2026-09-29T15:44:00+09:00",
            "--output",
            str(output),
        )
    )
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(
        runner,
        "run_comparison",
        lambda paths, **kwargs: {"summary": [{"ipcw_c_index": float("nan")}]},
    )

    with pytest.raises(ValueError, match="Out of range float"):
        runner.main()

    assert not output.exists()


@pytest.mark.parametrize(
    ("train_fraction", "validation_fraction"),
    [(0, 0.7), (0.7, 0.7), (0.8, 0.7), (0.7, 1), (float("nan"), 0.8)],
)
def test_library_rejects_invalid_time_cut_fractions_before_reading_csv(
    train_fraction: float, validation_fraction: float
) -> None:
    """시간 컷이 역전되거나 유효 범위를 벗어나면 원천 조회 전에 거절합니다."""
    with pytest.raises(ValueError, match="시간 컷 비율"):
        runner.run_comparison(
            {},
            observation_end_at=pd.Timestamp("2026-09-29T15:44:00+09:00"),
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )


def test_cli_rejects_invalid_time_cuts_before_reading_csv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI도 잘못된 시간 컷을 파일 접근 전에 종료 코드 2로 거절합니다."""
    arguments = ["run_service_model_comparison"]
    for name in ("orders", "order-items", "pets", "histories", "claims", "claim-items"):
        arguments.extend((f"--{name}", "/not-a-real-source.csv"))
    arguments.extend(
        (
            "--observation-end-at",
            "2026-09-29T15:44:00+09:00",
            "--train-fraction",
            "0.85",
            "--validation-fraction",
            "0.70",
        )
    )
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(SystemExit) as exc:
        runner.main()

    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("candidates", "inner_ratio"),
    [((5,), 0.8), ((0, 20), 0.8), ((5, 5), 0.8), ((5, 20), 1.0)],
)
def test_library_rejects_invalid_aft_selection_before_reading_csv(
    candidates: tuple[int, ...], inner_ratio: float
) -> None:
    with pytest.raises(ValueError, match="AFT"):
        runner.run_comparison(
            {},
            observation_end_at=pd.Timestamp("2026-09-29T15:44:00+09:00"),
            aft_round_candidates=candidates,
            inner_train_ratio=inner_ratio,
        )


def test_cli_rejects_invalid_aft_selection_before_reading_csv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = ["run_service_model_comparison"]
    for name in ("orders", "order-items", "pets", "histories", "claims", "claim-items"):
        arguments.extend((f"--{name}", "/not-a-real-source.csv"))
    arguments.extend(
        (
            "--observation-end-at",
            "2026-09-29T15:44:00+09:00",
            "--aft-round-candidates",
            "5",
            "5",
        )
    )
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(SystemExit) as exc:
        runner.main()

    assert exc.value.code == 2


@pytest.mark.parametrize(
    "scales",
    [("1.0",), ("0", "1.0"), ("1.0", "1.0"), ("nan", "1.0")],
)
def test_cli_rejects_invalid_aft_scales_before_reading_csv(
    monkeypatch: pytest.MonkeyPatch, scales: tuple[str, ...]
) -> None:
    arguments = ["run_service_model_comparison"]
    for name in ("orders", "order-items", "pets", "histories", "claims", "claim-items"):
        arguments.extend((f"--{name}", "/not-a-real-source.csv"))
    arguments.extend(
        (
            "--observation-end-at",
            "2026-09-29T15:44:00+09:00",
            "--aft-scale-candidates",
            *scales,
        )
    )
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2


def test_library_rejects_joint_aft_search_before_reading_csv() -> None:
    with pytest.raises(ValueError, match="별도 실험"):
        runner.run_comparison(
            {},
            observation_end_at=pd.Timestamp("2026-09-29T15:44:00+09:00"),
            aft_round_candidates=(5, 20),
            aft_scale_candidates=(0.5, 1.0),
        )


@pytest.mark.parametrize(
    "candidates", [(1.0,), (0.0, 1.0), (1.0, 1.0), (1.0, float("nan"))]
)
def test_library_rejects_invalid_group_smoothing_before_reading_csv(
    candidates: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="상품군 확률 수축 강도 후보"):
        runner.run_comparison(
            {},
            observation_end_at=pd.Timestamp("2026-09-29T15:44:00+09:00"),
            product_group_smoothing_candidates=candidates,
        )


@pytest.mark.parametrize(
    ("selector", "selection_options"),
    [
        ("select_service_aft_boost_rounds", {"aft_round_candidates": (5, 20)}),
        (
            "select_service_product_group_smoothing",
            {"product_group_smoothing_candidates": (1.0, 4.0)},
        ),
    ],
)
def test_inner_selection_precedes_outer_validation_label_generation(
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
    selection_options: dict[str, object],
) -> None:
    """외부 Validation 라벨을 만들기 전에 내부 시점의 표본을 다시 생성합니다."""
    first = pd.Timestamp("2024-01-01T00:00:00Z")
    end = pd.Timestamp("2026-01-01T00:00:00Z")
    sources = {
        name: pd.DataFrame()
        for name in (
            "orders",
            "order_items",
            "pets",
            "histories",
            "claims",
            "claim_items",
        )
    }
    monkeypatch.setattr(runner, "_read_sources", lambda paths: sources)
    monkeypatch.setattr(
        runner,
        "quarantine_unrestorable_orders",
        lambda orders, items, histories, claims, claim_items: SimpleNamespace(
            orders=orders,
            order_items=items,
            status_histories=histories,
            claims=claims,
            claim_items=claim_items,
            missing_history_order_count=0,
            missing_history_paid_order_count=0,
            missing_history_order_item_count=0,
            status_mismatch_order_count=0,
            status_mismatch_order_item_count=0,
        ),
    )
    for name in (
        "build_order_status_intervals",
        "build_order_item_quantity_intervals",
        "build_valid_purchase_item_intervals",
    ):
        monkeypatch.setattr(runner, name, lambda *args: None)
    monkeypatch.setattr(
        runner,
        "build_operational_event_intervals",
        lambda *args: SimpleNamespace(
            pet_targets=pd.DataFrame({"valid_from": [first]})
        ),
    )
    cuts: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def capture_split(*args: object, **kwargs: pd.Timestamp) -> object:
        cuts.append((kwargs["train_end_at"], kwargs["validation_end_at"]))
        return object()

    class SelectionReached(Exception):
        pass

    def stop_at_selection(*args: object, **kwargs: object) -> None:
        raise SelectionReached

    monkeypatch.setattr(runner, "build_service_train_validation_split", capture_split)
    monkeypatch.setattr(runner, selector, stop_at_selection)

    with pytest.raises(SelectionReached):
        runner.run_comparison(
            {},
            observation_end_at=end,
            train_fraction=0.4,
            validation_fraction=0.55,
            inner_train_ratio=0.8,
            **selection_options,
        )

    outer_train_end = first + (end - first) * 0.4
    assert cuts == [(first + (outer_train_end - first) * 0.8, outer_train_end)]


def test_selected_aft_setting_reaches_outer_training_and_fixed_path_stays_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """내부 선택값이 바깥 재학습에 전달되고 미선택 경로는 20회를 유지합니다."""
    first = pd.Timestamp("2024-01-01T00:00:00Z")
    end = pd.Timestamp("2026-01-01T00:00:00Z")
    sources = {
        name: pd.DataFrame()
        for name in (
            "orders",
            "order_items",
            "pets",
            "histories",
            "claims",
            "claim_items",
        )
    }
    monkeypatch.setattr(runner, "_read_sources", lambda paths: sources)
    monkeypatch.setattr(
        runner,
        "quarantine_unrestorable_orders",
        lambda orders, items, histories, claims, claim_items: SimpleNamespace(
            orders=orders,
            order_items=items,
            status_histories=histories,
            claims=claims,
            claim_items=claim_items,
            missing_history_order_count=0,
            missing_history_paid_order_count=0,
            missing_history_order_item_count=0,
            status_mismatch_order_count=0,
            status_mismatch_order_item_count=0,
        ),
    )
    # 모델 실행 경로만 검증하므로 CI 설치 패키지의 메타데이터에 의존하지 않습니다.
    monkeypatch.setattr(runner, "version", lambda package: "test-version")
    for name in (
        "build_order_status_intervals",
        "build_order_item_quantity_intervals",
        "build_valid_purchase_item_intervals",
    ):
        monkeypatch.setattr(runner, name, lambda *args: None)
    monkeypatch.setattr(
        runner,
        "build_operational_event_intervals",
        lambda *args: SimpleNamespace(
            pet_targets=pd.DataFrame({"valid_from": [first]})
        ),
    )
    validation = pd.DataFrame(
        {"user_id": ["u1"], "target_id": ["g1"], "history_interval_count": [0]}
    )
    monkeypatch.setattr(
        runner,
        "build_service_train_validation_split",
        lambda *args, **kwargs: SimpleNamespace(
            train=pd.DataFrame({"user_id": ["u1"]}), validation=validation
        ),
    )
    monkeypatch.setattr(
        runner,
        "select_service_aft_boost_rounds",
        lambda *args, **kwargs: AFTRoundSelection(
            selected_rounds=7,
            candidates=pd.DataFrame([{"num_boost_round": 7, "ipcw_brier_score": 0.1}]),
        ),
    )
    monkeypatch.setattr(
        runner,
        "select_service_aft_scale",
        lambda *args, **kwargs: AFTScaleSelection(
            selected_scale=2.0,
            candidates=pd.DataFrame(
                [{"loss_distribution_scale": 2.0, "ipcw_brier_score": 0.1}]
            ),
        ),
    )
    monkeypatch.setattr(
        runner,
        "select_service_product_group_smoothing",
        lambda *args, **kwargs: ProductGroupSmoothingSelection(
            selected_strength=4.0,
            candidates=pd.DataFrame(
                [
                    {
                        "product_group_smoothing_strength": 4.0,
                        "ipcw_brier_score": 0.1,
                    }
                ]
            ),
        ),
    )
    training_options: list[dict[str, object]] = []

    def record_outer_training(*args: object, **kwargs: object) -> SimpleNamespace:
        training_options.append(kwargs)
        return SimpleNamespace(
            summary=pd.DataFrame([{"ipcw_c_index": 0.6}]),
            calibration=pd.DataFrame(),
            brier_attribution=pd.DataFrame(),
            paired_bootstrap=SimpleNamespace(summary={}, trials=pd.DataFrame()),
            product_group_aft_bootstrap=SimpleNamespace(
                summary={"point_brier_improvement": 0.01},
                trials=pd.DataFrame([{"brier_improvement": 0.01}]),
            ),
            conditional_landmarks=None,
            conditional_calibration=None,
        )

    monkeypatch.setattr(runner, "compare_service_aft_lightgbm", record_outer_training)

    selected = runner.run_comparison(
        {}, observation_end_at=end, aft_round_candidates=(5, 7)
    )
    fixed = runner.run_comparison({}, observation_end_at=end)
    scale = runner.run_comparison(
        {}, observation_end_at=end, aft_scale_candidates=(1.0, 2.0)
    )
    live = runner.run_comparison(
        None,
        observation_end_at=end,
        sources=sources,
        source_metadata={"source_snapshots": {"order_extracted_at": end.isoformat()}},
    )
    baseline = runner.run_comparison(
        {}, observation_end_at=end, product_group_smoothing_candidates=(1.0, 4.0)
    )

    assert [item["aft_boost_rounds"] for item in training_options] == [
        7,
        20,
        20,
        20,
        20,
    ]
    assert "aft_loss_distribution_scale" not in training_options[0]
    assert "aft_loss_distribution_scale" not in training_options[1]
    assert training_options[2]["aft_loss_distribution_scale"] == 2.0
    assert training_options[4]["product_group_smoothing_strength"] == 4.0
    assert selected["aft_round_selection"]["selected_rounds"] == 7
    assert "aft_round_selection" not in fixed
    assert scale["aft_scale_selection"]["selected_scale"] == 2.0
    assert baseline["product_group_smoothing_selection"]["selected_strength"] == 4.0
    assert "product_group_aft_bootstrap" not in fixed
    assert baseline["product_group_aft_bootstrap"] == {
        "reference_model": "product_group_probability_baseline",
        "candidate_model": "xgboost_aft",
        "summary": {"point_brier_improvement": 0.01},
        "trials": [{"brier_improvement": 0.01}],
    }
    assert baseline["model_configuration"]["product_group_probability_baseline"] == {
        "product_group_smoothing_strength": 4.0,
        "horizon_days": 30,
        "source_key": "target_id",
        "prior": "train_global_ipcw_event_probability",
    }
    assert scale["evaluation_population_policy"] == (
        runner.service_evaluation_population_policy()
    )
    assert selected["model_configuration"]["xgboost_aft"]["num_boost_round"] == 7
    assert (
        fixed["model_configuration"]["xgboost_aft"]["parameters"][
            "aft_loss_distribution_scale"
        ]
        == 1.0
    )
    assert (
        scale["model_configuration"]["xgboost_aft"]["parameters"][
            "aft_loss_distribution_scale"
        ]
        == 2.0
    )
    assert scale["model_configuration"]["lightgbm"]["parameters"]["random_state"] == 42
    assert live["summary"] == fixed["summary"]
    assert live["source_snapshots"]["order_extracted_at"] == end.isoformat()
    assert "source_sha256" not in live
    assert selected["code_sha256"] == runner.model_code_sha256()
    assert live["code_sha256"] == selected["code_sha256"]


def test_memory_source_requires_snapshot_metadata() -> None:
    with pytest.raises(ValueError, match="메모리 원천"):
        runner.run_comparison(
            None,
            observation_end_at=pd.Timestamp("2026-09-29T15:44:00+09:00"),
            sources={},
        )


def test_validation_population_counts_each_history_bucket_once() -> None:
    """구매 간격 0·1·2회 이상 표본 수가 전체 Validation과 일치합니다."""
    rows = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u3"],
            "target_id": ["p1", "p1", "p2", "p3"],
            "history_interval_count": [0, 1, 2, 4],
        }
    )

    assert _summarize_validation_population(rows) == {
        "user_count": 3,
        "product_group_count": 3,
        "history_interval_count_0": 1,
        "history_interval_count_1": 1,
        "history_interval_count_2_or_more": 2,
    }


@pytest.mark.parametrize("invalid", [None, -1, 1.5, float("inf")])
def test_validation_population_rejects_invalid_history_count(invalid: object) -> None:
    """이력량 결측·음수·소수·무한대가 분포에서 조용히 빠지지 않습니다."""
    rows = pd.DataFrame(
        {
            "user_id": ["u1"],
            "target_id": ["p1"],
            "history_interval_count": [invalid],
        }
    )

    with pytest.raises(ValueError, match="과거 구매 간격 수"):
        _summarize_validation_population(rows)
