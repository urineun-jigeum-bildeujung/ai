"""Gateway 신뢰 경계와 추천 대상 Pet 소유권을 검증한다."""
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.api import main as api
from src.data_access import pet_repository as repository
from test_allergy_profile_contract import pet, synthetic_api

HEADERS = {"X-Internal-Secret": "synthetic-test-secret", "X-Member-Id": "2"}
ROUTES = [
    ("/recommend/home", {"pet_id": 1}),
    ("/recommend/substitute", {"pet_id": 1, "base_product_id": 9}),
]


@pytest.mark.parametrize("path,body", ROUTES)
def test_missing_auth_is_rejected_before_any_pet_lookup(monkeypatch, path, body):
    monkeypatch.setenv("INTERNAL_GATEWAY_SECRET", HEADERS["X-Internal-Secret"])
    lookup = Mock(return_value=None)
    monkeypatch.setattr(api, "get_owned_pet_by_id", lookup)
    with TestClient(api.app) as client:
        response = client.post(path, json=body)
    assert response.status_code == 401
    lookup.assert_not_called()


@pytest.mark.parametrize("path,body", ROUTES)
@pytest.mark.parametrize("headers", [
    {"X-Member-Id": "2"},
    {**HEADERS, "X-Internal-Secret": "wrong"},
    {"X-Internal-Secret": HEADERS["X-Internal-Secret"]},
    *[{**HEADERS, "X-Member-Id": value} for value in [
        "", "0", "-1", " 2", "2,3", "2' OR TRUE", "9223372036854775808", "9" * 30,
    ]],
    [*HEADERS.items(), ("X-Member-Id", "3")],
    [*HEADERS.items(), ("X-Internal-Secret", HEADERS["X-Internal-Secret"])],
])
def test_invalid_or_duplicate_headers_never_read_data(monkeypatch, path, body, headers):
    monkeypatch.setenv("INTERNAL_GATEWAY_SECRET", HEADERS["X-Internal-Secret"])
    lookup = Mock(side_effect=AssertionError("Pet must not be queried"))
    monkeypatch.setattr(api, "get_owned_pet_by_id", lookup)
    with TestClient(api.app) as client:
        response = client.post(path, json=body, headers=headers)
    assert response.status_code == 401
    assert response.json() == {"detail": "SERVICE_UNAUTHORIZED"}
    lookup.assert_not_called()


@pytest.mark.parametrize("path,body", ROUTES)
def test_missing_secret_fails_closed_but_health_remains_public(monkeypatch, path, body):
    monkeypatch.delenv("INTERNAL_GATEWAY_SECRET", raising=False)
    lookup = Mock()
    monkeypatch.setattr(api, "get_owned_pet_by_id", lookup)
    with TestClient(api.app) as client:
        response = client.post(path, json=body, headers=HEADERS)
        assert client.get("/health").status_code == 200
    assert response.status_code == 503
    assert response.json() == {"detail": "SERVICE_AUTH_NOT_CONFIGURED"}
    lookup.assert_not_called()


@pytest.mark.parametrize("path,body", ROUTES)
def test_unowned_missing_or_deleted_pet_stops_before_downstream(monkeypatch, path, body):
    monkeypatch.setenv("INTERNAL_GATEWAY_SECRET", HEADERS["X-Internal-Secret"])
    lookup = Mock(return_value=None)
    monkeypatch.setattr(api, "get_owned_pet_by_id", lookup)
    downstream = Mock(side_effect=AssertionError("No downstream access before ownership"))
    for name in ["_get_model", "get_product_by_id", "list_products", "load_reviews_with_reviewer_pet",
                 "get_purchased_product_ids_for_user", "get_product_embeddings"]:
        monkeypatch.setattr(api, name, downstream)
    with TestClient(api.app) as client:
        response = client.post(path, json={**body, "member_id": 999}, headers=HEADERS)
    assert response.status_code == 404
    lookup.assert_called_once_with(1, 2)
    downstream.assert_not_called()


@pytest.mark.parametrize("path,body", ROUTES)
def test_owned_pet_preserves_success_and_ignores_body_member_id(monkeypatch, synthetic_api, path, body):
    lookup = Mock(return_value=pet("KNOWN_NONE"))
    monkeypatch.setattr(api, "get_owned_pet_by_id", lookup)
    response = synthetic_api.post(path, json={**body, "member_id": 999})
    assert response.status_code == 200
    assert response.json()["items"][0]["allergy_status"] == "SAFE"
    lookup.assert_called_once_with(1, 2)


@pytest.mark.parametrize("row", [None, (1, 2, "DOG", "POODLE", None, "MALE", True, 10, 3, [], [], "KNOWN_NONE")])
def test_owned_lookup_filters_in_sql_and_closes(monkeypatch, row):
    conn = Mock()
    cursor = Mock()
    cursor.fetchone.return_value = row
    conn.cursor.return_value.__enter__ = Mock(return_value=cursor)
    conn.cursor.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(repository, "USE_DUMMY_DATA", False)
    monkeypatch.setattr(repository, "get_connection", lambda _: conn)
    result = repository.get_owned_pet_by_id(1, 2)
    sql, params = cursor.execute.call_args.args
    assert "p.id = %s AND p.member_id = %s AND p.deleted_at IS NULL" in sql
    assert params == (1, 2)
    assert (result is None) == (row is None)
    conn.close.assert_called_once()


@pytest.mark.parametrize("member_id", [None, True, "2", 0, -1, 9223372036854775808])
def test_invalid_owner_never_degrades_to_unscoped_lookup(monkeypatch, member_id):
    connect = Mock(side_effect=AssertionError("No connection"))
    monkeypatch.setattr(repository, "get_connection", connect)
    monkeypatch.setattr(repository, "USE_DUMMY_DATA", False)
    assert repository.get_owned_pet_by_id(1, member_id) is None
    connect.assert_not_called()


@pytest.mark.parametrize("record,expected", [
    (pet(), True), ({**pet(), "user_id": 3}, False),
    ({**pet(), "deleted_at": "2026-10-01"}, False), (None, False),
])
def test_dummy_lookup_also_enforces_owner_and_deletion(monkeypatch, record, expected):
    monkeypatch.setattr(repository, "USE_DUMMY_DATA", True)
    monkeypatch.setattr(repository, "get_pet_by_id", lambda _: record)
    assert (repository.get_owned_pet_by_id(1, 2) is not None) == expected
