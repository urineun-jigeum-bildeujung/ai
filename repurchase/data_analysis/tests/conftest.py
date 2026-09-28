"""여러 재구매 E2E 테스트가 공유하는 작은 고정 입력을 제공합니다."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

CLOUD_CONTRACT_FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures" / "cloud_contract"


@pytest.fixture
def cloud_source_contract() -> dict[str, object]:
    """클라우드 원천 주문의 공식 고정 예제를 매 테스트마다 새로 읽습니다.

    파일을 매번 새로 읽어 한 테스트의 데이터 변경이 다른 테스트로 전파되지
    않게 합니다. JSON은 백엔드·클라우드 팀과도 같은 값을 확인할 수 있는
    언어 중립적인 계약 예제로 사용합니다.
    """
    path = CLOUD_CONTRACT_FIXTURE_DIRECTORY / "source_orders.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def prediction_publication_contract() -> dict[str, object]:
    """완료·실패·작성 중 배치를 함께 가진 결과 발행 고정 예제를 읽습니다."""
    path = CLOUD_CONTRACT_FIXTURE_DIRECTORY / "prediction_publications.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def uci_e2e_purchase_events() -> pd.DataFrame:
    """반복 구매와 단발 구매가 함께 있는 작은 구매 이력을 만듭니다."""
    rows: list[dict[str, object]] = []
    for user_id, product_id, start_at, interval_days, event_count in (
        ("u1", "p1", "2026-01-01", 5, 24),
        ("u2", "p2", "2026-01-03", 8, 16),
    ):
        for index in range(event_count):
            rows.append(
                {
                    "user_id": user_id,
                    "order_id": f"{user_id}-o{index:02d}",
                    "product_id": product_id,
                    "ordered_at": pd.Timestamp(start_at)
                    + pd.Timedelta(days=interval_days * index),
                }
            )
    # 관측 기간이 충분히 지난 단발 구매로 Train의 미재구매 정답을 만듭니다.
    rows.append(
        {
            "user_id": "u3",
            "order_id": "u3-o00",
            "product_id": "p3",
            "ordered_at": pd.Timestamp("2026-01-04"),
        }
    )
    return pd.DataFrame(rows)
