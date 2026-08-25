"""재구매 분석의 핵심 전처리 규칙을 검증합니다.

숫자형 식별자의 불필요한 소수점 제거와 같은 주문 안의 여러 상품 행을
하나의 구매 사건으로 합치는 로직을 검사해 잘못된 재구매 간격 생성을 막습니다.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from scripts import loaders
from scripts.dataset_registry import DATASET_FILES
from scripts.download_datasets import (
    DatasetIntegrityError,
    ManifestError,
    VerificationEvidence,
    VerifiedDatasetFile,
    load_manifest,
    merge_manifest_entries,
    verify_checksum,
    verify_zip_archive,
    write_manifest,
)
from scripts.loaders import (
    DatasetLoadError,
    _add_candidate_flags,
    _read_single_r_frame,
    _string_id,
    load_complete_journey,
    load_uci_online_retail_ii,
)
from scripts.profile_mock_generation import (
    ActivitySegment,
    ProfileConfigurationError,
    assign_activity_segments,
    build_mock_generation_profile,
    frequency_distribution,
    profile_cycle_heterogeneity,
    profile_order_composition,
    profile_product_switching,
    profile_quantity_interval_relationship,
    profile_temporal_behavior,
    profile_user_activity,
    spearman_rank_correlation,
)
from scripts.profile_repurchase import profile_repurchase_scope
from scripts.visualize_profiles import filter_complete_months


def test_checksum_accepts_exact_file_content(tmp_path: Path) -> None:
    """파일 내용이 등록된 SHA-256과 정확히 같으면 검증을 통과하는지 확인합니다."""
    path = tmp_path / "source.bin"
    path.write_bytes(b"hello")

    actual = verify_checksum(
        path,
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
    )

    assert actual == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_checksum_rejects_changed_file_content(tmp_path: Path) -> None:
    """파일 내용이 달라지면 크기와 관계없이 무결성 오류가 발생하는지 확인합니다."""
    path = tmp_path / "source.bin"
    path.write_bytes(b"changed")

    with pytest.raises(DatasetIntegrityError):
        verify_checksum(
            path,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
        )


def test_zip_rejects_unexpected_member(tmp_path: Path) -> None:
    """ZIP 내부 파일명이 등록값과 다르면 다른 배포 파일로 판단하는지 확인합니다."""
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as zip_file:
        zip_file.writestr("unexpected.xlsx", b"content")

    with pytest.raises(DatasetIntegrityError):
        verify_zip_archive(path, "expected.xlsx")


def test_partial_manifest_merge_preserves_only_present_files(tmp_path: Path) -> None:
    """부분 검증 시 존재하는 기록은 보존하고 사라진 파일 기록은 제거합니다."""
    retained_path = tmp_path / "retained.bin"
    retained_path.write_bytes(b"retained")
    existing = [
        {
            "source": {"filename": "retained.bin"},
            "local_file": {"actual_sha256": "old-retained"},
        },
        {
            "source": {"filename": "missing.bin"},
            "local_file": {"actual_sha256": "old-missing"},
        },
    ]

    spec = DATASET_FILES[0]
    (tmp_path / spec.filename).write_bytes(b"verified")
    verified = VerifiedDatasetFile(
        source=spec,
        bytes=8,
        evidence=VerificationEvidence("new-hash", ("sha256",)),
        verified_at="2026-08-24T00:00:00+00:00",
    )

    result = merge_manifest_entries(existing, [verified], tmp_path)

    filenames = [entry["source"]["filename"] for entry in result]
    assert filenames == sorted(["retained.bin", spec.filename])
    assert "missing.bin" not in filenames
    updated = next(
        entry for entry in result if entry["source"]["filename"] == spec.filename
    )
    assert updated["local_file"]["actual_sha256"] == "new-hash"


def test_manifest_round_trip_uses_current_schema_and_removes_part_file(
    tmp_path: Path,
) -> None:
    """명세를 원자적으로 저장하고 현재 스키마로 다시 읽을 수 있는지 확인합니다."""
    path = tmp_path / "manifest.json"
    entries = [
        {
            "source": {"filename": "source.bin"},
            "local_file": {"actual_sha256": "hash"},
        }
    ]

    write_manifest(path, entries)

    assert load_manifest(path) == entries
    assert not path.with_suffix(".json.part").exists()


def test_manifest_rejects_unknown_schema_version(tmp_path: Path) -> None:
    """코드와 다른 스키마의 명세를 자동 병합하지 않는지 확인합니다."""
    path = tmp_path / "manifest.json"
    path.write_text('{"manifest_schema_version":"0.9","files":[]}', encoding="utf-8")

    with pytest.raises(ManifestError):
        load_manifest(path)


def test_string_id_preserves_text_and_removes_numeric_suffix() -> None:
    """숫자형 ID의 '.0'만 제거되고 문자형 ID는 보존되는지 확인합니다."""
    values = pd.Series([13085.0, "C489434", pd.NA])

    result = _string_id(values)

    assert result.iloc[0] == "13085"
    assert result.iloc[1] == "C489434"
    assert pd.isna(result.iloc[2])


def test_uci_loader_rejects_two_input_sources(tmp_path: Path) -> None:
    """두 입력 경로를 동시에 받으면 어느 원천을 쓸지 모호하므로 중단합니다."""
    with pytest.raises(DatasetLoadError, match="동시에 지정"):
        load_uci_online_retail_ii(
            workbook_path=tmp_path / "source.xlsx",
            archive_path=tmp_path / "source.zip",
        )


def test_uci_loader_rejects_empty_workbook(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """읽힌 시트가 없는 워크북은 분석 가능한 원천으로 인정하지 않습니다."""
    monkeypatch.setattr(loaders.pd, "read_excel", lambda *args, **kwargs: {})

    with pytest.raises(DatasetLoadError, match="시트가 없습니다"):
        load_uci_online_retail_ii(workbook_path=tmp_path / "empty.xlsx")


def test_uci_loader_rejects_missing_source_columns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """필수 원천 컬럼이 빠지면 잘못된 분포를 만들기 전에 중단합니다."""
    monkeypatch.setattr(
        loaders.pd,
        "read_excel",
        lambda *args, **kwargs: {"Year 2009-2010": pd.DataFrame({"Invoice": []})},
    )

    with pytest.raises(DatasetLoadError, match="원천 컬럼이 누락"):
        load_uci_online_retail_ii(workbook_path=tmp_path / "invalid.xlsx")


def test_r_loader_rejects_non_dataframe_object(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """R 파일의 단일 객체가 표 형식이 아니면 명시적인 오류를 반환합니다."""
    monkeypatch.setattr(loaders.pyreadr, "read_r", lambda path: {None: "text"})

    with pytest.raises(DatasetLoadError, match="DataFrame이 아닙니다"):
        _read_single_r_frame(tmp_path / "invalid.rdata")


def test_complete_journey_loader_joins_product_category(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """거래와 상품 마스터를 연결하고 미매칭 상품은 결측으로 보존합니다."""
    transactions = pd.DataFrame(
        {
            "household_id": [1, 1],
            "basket_id": [10, 10],
            "product_id": [100, 200],
            "transaction_timestamp": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "quantity": [1, 2],
            "sales_value": [10.0, 8.0],
            "week": [1, 1],
        }
    )
    products = pd.DataFrame(
        {
            "product_id": [100],
            "product_category": ["DOG FOODS"],
        }
    )

    def fake_read(path: Path) -> pd.DataFrame:
        return transactions if path.name == "transactions.rds" else products

    monkeypatch.setattr(loaders, "_read_single_r_frame", fake_read)

    result = load_complete_journey(
        transactions_path=tmp_path / "transactions.rds",
        products_path=tmp_path / "products.rds",
    )

    assert result.loc[result["product_id"] == "100", "category_id"].item() == (
        "DOG FOODS"
    )
    assert result.loc[result["product_id"] == "200", "category_id"].isna().item()
    assert result["dataset"].eq("complete_journey").all()


def test_zero_amount_purchase_is_repurchase_but_not_monetary_candidate() -> None:
    """전액 할인 구매를 재구매 행동에는 포함하고 금액 분석에서는 제외합니다."""
    frame = pd.DataFrame(
        {
            "user_id": ["user-1"],
            "order_id": ["order-1"],
            "product_id": ["food-1"],
            "ordered_at": pd.to_datetime(["2026-01-01"]),
            "quantity": [1],
            "line_amount": [0.0],
            "is_explicit_cancellation": [False],
        }
    )

    result = _add_candidate_flags(frame)

    assert bool(result.loc[0, "is_valid_repurchase_candidate"])
    assert not bool(result.loc[0, "is_valid_monetary_candidate"])


def test_same_order_lines_are_one_repurchase_event() -> None:
    """같은 주문의 여러 행이 하나의 구매 사건으로 집계되는지 확인합니다."""
    frame = pd.DataFrame(
        {
            "user_id": ["user-1", "user-1", "user-1"],
            "product_id": ["food-1", "food-1", "food-1"],
            "order_id": ["order-1", "order-1", "order-2"],
            "ordered_at": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-11"]),
            "quantity": [1, 2, 1],
            "line_amount": [10.0, 20.0, 10.0],
        }
    )

    result = profile_repurchase_scope(frame, "product_id")

    assert result is not None
    assert result["event_count"] == 2
    assert result["observed_interval_count"] == 1
    assert result["positive_interval_days"]["p50"] == 10.0
    assert result["event_level_right_censoring_rate"] == 0.5


def test_partial_last_month_is_removed_from_visualization() -> None:
    """관측 종료일이 월 중간이면 불완전한 마지막 월이 제외되는지 확인합니다."""
    frame = pd.DataFrame(
        {
            "month": ["2026-01", "2026-02", "2026-03"],
            "order_count": [100, 120, 20],
            "user_count": [80, 90, 15],
        }
    )

    result = filter_complete_months(
        frame,
        {
            "start": "2026-01-01T00:00:00",
            "end": "2026-03-05T12:00:00",
        },
    )

    assert result["month"].tolist() == ["2026-01", "2026-02"]


def test_user_activity_profile_preserves_segment_shares() -> None:
    """사용자별 주문 수가 정의된 활동 구간의 건수와 비율로 집계됩니다."""
    order_counts = {"u1": 1, "u2": 2, "u3": 3, "u4": 4, "u5": 10, "u6": 11}
    rows = [
        {"user_id": user_id, "order_id": f"{user_id}-{order_index}"}
        for user_id, count in order_counts.items()
        for order_index in range(count)
    ]

    result = profile_user_activity(pd.DataFrame(rows))

    segments = {row["name"]: row for row in result["segments"]}
    assert segments["single"]["user_count"] == 1
    assert segments["low_frequency"]["user_count"] == 2
    assert segments["repeat"]["user_count"] == 2
    assert segments["high_frequency"]["user_count"] == 1
    assert segments["repeat"]["order_count_distribution"]["min"] == 4.0
    assert segments["repeat"]["order_count_distribution"]["max"] == 10.0
    assert sum(row["user_rate"] for row in result["segments"]) == pytest.approx(1.0)


def test_activity_segment_overlap_is_rejected() -> None:
    """활동 구간이 겹치면 같은 사용자가 이중 분류되지 않도록 중단합니다."""
    overlapping = (
        ActivitySegment("first", 1, 3),
        ActivitySegment("second", 3, None),
    )

    with pytest.raises(ProfileConfigurationError, match="겹칩니다"):
        assign_activity_segments(pd.Series([3]), segments=overlapping)


def test_frequency_distribution_preserves_exact_integer_probability() -> None:
    """정수 빈도표가 각 값의 관측 건수와 비율을 손실 없이 보존합니다."""
    result = frequency_distribution(pd.Series([1, 1, 2, pd.NA]))

    assert result["observed_count"] == 3
    assert result["missing_count"] == 1
    assert result["values"] == [
        {"value": 1, "count": 2, "rate": pytest.approx(2 / 3)},
        {"value": 2, "count": 1, "rate": pytest.approx(1 / 3)},
    ]


def test_order_composition_preserves_duplicate_lines_and_timestamp_conflicts() -> None:
    """중복 상품 행과 주문 내부 시각 불일치를 생성 품질 신호로 집계합니다."""
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2"],
            "order_id": ["o1", "o1", "o2", "o2"],
            "product_id": ["p1", "p1", "p2", "p3"],
            "source_row_id": ["r1", "r2", "r3", "r4"],
            "ordered_at": pd.to_datetime(
                [
                    "2026-01-01 10:00:00",
                    "2026-01-01 10:00:00",
                    "2026-01-02 10:00:00",
                    "2026-01-02 10:00:01",
                ]
            ),
        }
    )

    result = profile_order_composition(frame)

    assert result["order_count"] == 2
    assert result["duplicate_product_line_order_rate"] == 0.5
    assert result["inconsistent_timestamp_order_rate"] == 0.5
    assert result["timestamp_span_seconds"]["max"] == 1.0


def test_cycle_profile_separates_event_and_pair_weighting() -> None:
    """구매가 잦은 쌍이 개인 대표 주기 분포를 과도하게 지배하지 않습니다."""
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2"],
            "order_id": ["o1", "o2", "o3", "o4", "o5"],
            "product_id": ["p1", "p1", "p1", "p2", "p2"],
            "ordered_at": pd.to_datetime(
                ["2026-01-01", "2026-01-11", "2026-01-21", "2026-01-01", "2026-01-31"]
            ),
            "quantity": [1, 1, 1, 1, 1],
            "line_amount": [10.0, 10.0, 10.0, 20.0, 20.0],
        }
    )

    result = profile_cycle_heterogeneity(frame, "product_id")

    assert result["observed_interval_count"] == 3
    assert result["pair_count_with_observed_interval"] == 2
    assert result["event_weighted_interval_days"]["p50"] == 10.0
    assert result["pair_weighted_median_interval_days"]["p50"] == 20.0
    assert result["pair_interval_cv"]["count"] == 1
    assert result["pair_interval_cv"]["p50"] == 0.0


def test_temporal_profile_separates_user_order_cycle_and_calendar_pattern() -> None:
    """사용자 주문 간격과 달력 범주의 경험적 비율을 함께 보존합니다."""
    orders = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2"],
            "order_id": ["o1", "o2", "o3", "o4", "o5"],
            "ordered_at": pd.to_datetime(
                [
                    "2026-01-05 09:00:00",
                    "2026-01-15 09:00:00",
                    "2026-01-25 09:00:00",
                    "2026-02-02 09:00:00",
                    "2026-02-22 09:00:00",
                ]
            ),
        }
    )

    result = profile_temporal_behavior(orders)

    assert result["event_weighted_order_interval_days"]["count"] == 3
    assert result["event_weighted_order_interval_days"]["p50"] == pytest.approx(10.0)
    assert result["user_weighted_median_order_interval_days"]["p50"] == pytest.approx(
        15.0
    )
    month_rates = {
        row["value"]: row["rate"] for row in result["month_of_year_frequency"]["values"]
    }
    assert month_rates == {1: pytest.approx(0.6), 2: pytest.approx(0.4)}


def test_spearman_returns_none_when_rank_relationship_cannot_be_measured() -> None:
    """한쪽 값이 모두 같으면 순위 상관을 계산할 수 없음을 명시합니다."""
    result = spearman_rank_correlation(pd.Series([1, 1]), pd.Series([10, 20]))

    assert result is None


def test_quantity_profile_uses_previous_purchase_for_next_interval() -> None:
    """현재 간격에 현재 수량이 아니라 직전 구매 수량을 연결합니다."""
    frame = pd.DataFrame(
        {
            "user_id": ["u1"] * 4,
            "order_id": ["o1", "o2", "o3", "o4"],
            "product_id": ["p1"] * 4,
            "ordered_at": pd.to_datetime(
                ["2026-01-01", "2026-01-11", "2026-01-31", "2026-03-02"]
            ),
            "quantity": [1, 2, 3, 100],
            "line_amount": [10.0, 20.0, 30.0, 1_000.0],
        }
    )

    result = profile_quantity_interval_relationship(frame, "product_id")

    assert result["eligible_transition_count"] == 3
    assert result["previous_quantity"]["max"] == 3.0
    assert result["spearman_quantity_to_next_interval"] == pytest.approx(1.0)


def test_product_switch_profile_excludes_ambiguous_multi_product_orders() -> None:
    """동시구매를 전환으로 오인하지 않고 새 상품의 다음 주문 정착을 계산합니다."""
    frame = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2", "u2"],
            "order_id": ["o1", "o2", "o3", "o4", "o4", "o5"],
            "category_id": ["food"] * 6,
            "product_id": ["p1", "p2", "p2", "p1", "p2", "p3"],
            "ordered_at": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-06",
                    "2026-01-16",
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-11",
                ]
            ),
        }
    )

    result = profile_product_switching(frame)

    assert result["category_order_count"] == 5
    assert result["multi_product_category_order_rate"] == pytest.approx(0.2)
    assert result["eligible_transition_count"] == 2
    assert result["switch_count"] == 1
    assert result["switch_rate"] == pytest.approx(0.5)
    assert result["first_followup_switch_rate"] == pytest.approx(1.0)
    assert result["early_switch_threshold_days"] == pytest.approx(10.0)
    assert result["early_switch_rate"] == pytest.approx(1.0)
    assert result["evaluable_switch_followup_count"] == 1
    assert result["next_order_same_product_rate"] == pytest.approx(1.0)
    assert result["next_order_switch_back_rate"] == pytest.approx(0.0)


def test_mock_generation_profile_connects_analysis_layers() -> None:
    """생성 가이드 프로파일이 사용자·주문·주기·전환 분석을 한 구조로 묶습니다."""
    frame = pd.DataFrame(
        {
            "dataset": ["fixture", "fixture"],
            "source_row_id": ["r1", "r2"],
            "user_id": ["u1", "u1"],
            "order_id": ["o1", "o2"],
            "product_id": ["p1", "p1"],
            "category_id": ["food", "food"],
            "ordered_at": pd.to_datetime(["2026-01-01", "2026-01-11"]),
            "quantity": [1, 1],
            "line_amount": [10.0, 10.0],
            "is_valid_repurchase_candidate": [True, True],
        }
    )

    result = build_mock_generation_profile(frame, include_category_breakdown=True)

    assert result["dataset"] == "fixture"
    assert result["user_activity"]["user_count"] == 1
    assert result["order_composition"]["order_count"] == 2
    assert result["product_cycle"]["observed_interval_count"] == 1
    assert result["category_breakdown"][0]["category_id"] == "food"
