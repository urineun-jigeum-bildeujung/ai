"""구매 사건에서 동일 상품 재구매·우측검열 라벨을 만드는 규칙을 검증합니다.

테스트를 구현보다 먼저 작성해, 재구매가 관측된 사건과 아직 관측되지 않은
마지막 사건의 의미가 코드 작성 과정에서 달라지지 않도록 고정합니다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.preprocessing.labels import (
    RepurchaseLabelBuildError,
    build_same_product_repurchase_labels,
    validate_same_product_repurchase_labels,
)


def make_purchase_events() -> pd.DataFrame:
    """재구매·우측검열·동시각 후속 구매를 모두 포함한 사건을 만듭니다."""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u2", "u2"],
            "order_id": ["o1", "o2", "o3", "o4", "o5"],
            "product_id": ["p1", "p2", "p1", "p1", "p1"],
            "ordered_at": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-05",
                    "2026-01-11",
                    "2026-01-03",
                    "2026-01-03",
                ]
            ),
        }
    )


def test_build_labels_separates_observed_and_right_censored_events() -> None:
    """다음 동일 상품 구매는 관측 라벨, 마지막 구매는 우측검열로 만듭니다."""
    events = make_purchase_events()

    labels = build_same_product_repurchase_labels(
        events,
        observation_end_at=pd.Timestamp("2026-01-20"),
    ).set_index(["user_id", "order_id", "product_id"])

    first_p1 = labels.loc[("u1", "o1", "p1")]
    assert first_p1["next_same_product_at"] == pd.Timestamp("2026-01-11")
    assert first_p1["duration_days"] == 10
    assert bool(first_p1["event_observed"])
    assert not bool(first_p1["is_right_censored"])

    last_p1 = labels.loc[("u1", "o3", "p1")]
    assert pd.isna(last_p1["next_same_product_at"])
    assert last_p1["duration_days"] == 9
    assert not bool(last_p1["event_observed"])
    assert bool(last_p1["is_right_censored"])


def test_build_labels_preserves_zero_day_followup_instead_of_guessing() -> None:
    """다른 주문의 동시각 재구매 후보를 임의로 제거하지 않고 표시합니다."""
    labels = build_same_product_repurchase_labels(
        make_purchase_events(),
        observation_end_at=pd.Timestamp("2026-01-20"),
    ).set_index(["user_id", "order_id", "product_id"])

    first_u2_event = labels.loc[("u2", "o4", "p1")]
    assert first_u2_event["next_order_id"] == "o5"
    assert first_u2_event["duration_days"] == 0
    assert bool(first_u2_event["has_zero_day_followup"])


def test_validation_requires_one_censored_event_per_user_product_pair() -> None:
    """각 사용자·상품 시퀀스의 마지막 사건만 우측검열인지 검증합니다."""
    events = make_purchase_events()
    labels = build_same_product_repurchase_labels(
        events,
        observation_end_at=pd.Timestamp("2026-01-20"),
    )

    summary = validate_same_product_repurchase_labels(events, labels)

    assert summary["source_event_count"] == 5
    assert summary["user_product_pair_count"] == 3
    assert summary["observed_repurchase_count"] == 2
    assert summary["right_censored_count"] == 3
    assert summary["zero_day_followup_count"] == 1
    assert all(summary["invariants"].values())


def test_build_labels_rejects_observation_end_before_latest_event() -> None:
    """관측 종료일이 실제 마지막 구매보다 이르면 음수 검열 기간을 막습니다."""
    with pytest.raises(RepurchaseLabelBuildError, match="관측 종료 시각"):
        build_same_product_repurchase_labels(
            make_purchase_events(),
            observation_end_at=pd.Timestamp("2026-01-10"),
        )


def test_build_labels_rejects_missing_required_columns() -> None:
    """사건 키나 구매 시각이 없는 입력을 조용히 처리하지 않습니다."""
    with pytest.raises(RepurchaseLabelBuildError, match="필수 컬럼"):
        build_same_product_repurchase_labels(
            make_purchase_events().drop(columns="product_id"),
            observation_end_at=pd.Timestamp("2026-01-20"),
        )
