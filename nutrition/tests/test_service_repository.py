"""DB double 검증. 운영 DB 연결 또는 실제 evidence 검증으로 보고하지 않는다."""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "scripts"),
               str(Path(__file__).resolve().parents[1] / "scripts/nutrition")]
import api_nutrition as api
import service_repository as repo

PATH = "/api/nutrition/analyze/by-service-id"
BODY = {"pet_id": 123, "product_id": 456}
HEADERS = {"X-Internal-Secret": "test-only-secret", "X-Member-Id": "42"}


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for key in ("MEMBER_DATABASE_URL", "PRODUCT_DATABASE_URL", "INTERNAL_GATEWAY_SECRET", "NUTRITION_RUNTIME_MODE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(repo, "_connect", lambda _: pytest.fail("unexpected DB connection"))


def configure(monkeypatch):
    for key in ("MEMBER_DATABASE_URL", "PRODUCT_DATABASE_URL"):
        monkeypatch.setenv(key, "postgresql://test-only:not-a-real-password@invalid/db")
    monkeypatch.setenv("INTERNAL_GATEWAY_SECRET", HEADERS["X-Internal-Secret"])


def connection(monkeypatch, row, relations):
    configure(monkeypatch)
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = row
    cur.fetchall.side_effect = relations
    monkeypatch.setattr(repo, "_connect", lambda _: conn)
    return conn, cur


@pytest.mark.parametrize("allergies,expected", [([], "UNKNOWN"), ([("CHICKEN",)], "KNOWN_LIST")])
def test_pet_ownership_sql_and_profile(monkeypatch, allergies, expected):
    conn, cur = connection(monkeypatch, (123, "DOG", 4, 0.5, 3, True, None), [allergies])
    result = repo.get_pet(123, 42)
    assert result["allergy_profile_status"] == expected and result["life_stage"] is None
    assert result["age"] == 4 and result["weight"] == 0.5
    assert result["bcs"] == 3 and result["is_neutered"] is True
    assert result["birth_date"] is None
    sql, params = cur.execute.call_args_list[0].args
    assert "member_id = %s" in sql and "deleted_at IS NULL" in sql and params == (123, 42)
    conn.set_session.assert_called_once_with(readonly=True, isolation_level="REPEATABLE READ", autocommit=False)
    conn.close.assert_called_once()
    conn.commit.assert_not_called()


@pytest.mark.parametrize("getter,code", [(lambda: repo.get_pet(123, 42), "PET_NOT_FOUND"),
                                       (lambda: repo.get_product(456), "PRODUCT_NOT_FOUND")])
def test_not_found_closes_and_never_reads_relations(monkeypatch, getter, code):
    conn, cur = connection(monkeypatch, None, [])
    with pytest.raises(repo.ServiceNotFound, match=code):
        getter()
    assert cur.execute.call_count == 1
    conn.close.assert_called_once()


def test_product_aggregate_preserves_source(monkeypatch):
    conn, cur = connection(monkeypatch, (456, "bad-sku", "test", "FOOD", "DRY_FOOD", "SENIOR"),
                           [[("DOG",)], [("SALMON",)], [("CHKN-MEAT",)], [("CHOCOLATE_CACAO",)]])
    result = repo.get_product(456)
    assert result["target_species"] == ["DOG"] and result["allergen_flags"] == ["SALMON"]
    assert result["ingredient_codes"] == ["CHKN-MEAT"] and result["target_age_group"] == "SENIOR"
    assert result["caution_codes"] == ["CHOCOLATE_CACAO"]
    assert "public.product_cautions" in cur.execute.call_args_list[-1].args[0]
    assert "aafco_life_stage" not in result and "nutrition_items" not in result
    assert "is_active = TRUE" in cur.execute.call_args_list[0].args[0]
    assert all(call.args[1] == (456,) for call in cur.execute.call_args_list)
    conn.close.assert_called_once()


def test_list_active_products_preserves_independent_caution_codes(monkeypatch):
    conn, cur = connection(monkeypatch, None, [
        [(456, "bad-sku", "test", "FOOD", "DRY_FOOD", "ADULT")],
        [(456, "DOG")], [(456, "SALMON")], [(456, "CHKN-MEAT")],
        [(456, "CHOCOLATE_CACAO"), (456, "HIGH_FAT")],
    ])
    result = repo.list_active_products()[0]
    assert result["caution_codes"] == ["CHOCOLATE_CACAO", "HIGH_FAT"]
    assert result["ingredient_codes"] == ["CHKN-MEAT"]
    assert result["allergen_flags"] == ["SALMON"]
    sql, params = cur.execute.call_args_list[-1].args
    assert "public.product_cautions" in sql and "ANY(%s)" in sql and params == ([456],)
    assert all(call.args[0].startswith("SELECT ") for call in cur.execute.call_args_list)
    conn.set_session.assert_called_once_with(readonly=True, isolation_level="REPEATABLE READ", autocommit=False)
    conn.commit.assert_not_called()
    conn.close.assert_called_once()


@pytest.mark.parametrize("stage", ["connect", "session", "query", "close"])
def test_errors_are_sanitized_and_cleanup(monkeypatch, stage):
    conn, cur = connection(monkeypatch, (1,), [])
    error = RuntimeError("secret-DSN-password")
    if stage == "connect":
        monkeypatch.setattr(repo, "_connect", MagicMock(side_effect=error))
    else:
        {"session": conn.set_session, "query": cur.execute, "close": conn.close}[stage].side_effect = error
    with pytest.raises(repo.ServiceUnavailable) as caught:
        with repo._cursor("MEMBER_DATABASE_URL") as cursor:
            cursor.execute("SELECT 1")
    assert str(caught.value) == "SERVICE_DB_UNAVAILABLE"
    if stage != "connect":
        conn.close.assert_called_once()


@pytest.mark.parametrize("headers", [{}, {"X-Internal-Secret": "wrong", "X-Member-Id": "42"},
    {"X-Internal-Secret": "test-only-secret"}, {**HEADERS, "X-Member-Id": "0"},
    {**HEADERS, "X-Member-Id": "-1"}, {**HEADERS, "X-Member-Id": " 42"},
    {**HEADERS, "X-Member-Id": "４２"}, {**HEADERS, "X-Member-Id": "1' OR TRUE"},
    {**HEADERS, "X-Member-Id": "9223372036854775808"}])
def test_auth_rejects_before_db(monkeypatch, headers):
    configure(monkeypatch)
    with TestClient(api.app) as client:
        wire_headers = [(key.encode(), value.encode()) for key, value in headers.items()]
        assert client.post(PATH, json=BODY, headers=wire_headers).status_code == 401


def test_missing_configuration_and_duplicate_header(monkeypatch):
    with TestClient(api.app) as client:
        assert client.post(PATH, json=BODY).json()["detail"] == "SERVICE_SOURCE_NOT_CONFIGURED"
        configure(monkeypatch)
        monkeypatch.delenv("INTERNAL_GATEWAY_SECRET")
        result = client.post(PATH, json=BODY)
        assert result.status_code == 503 and result.json()["detail"] == "SERVICE_AUTH_NOT_CONFIGURED"
        configure(monkeypatch)
        result = client.post(PATH, json=BODY, headers=[*HEADERS.items(), ("X-Member-Id", "43")])
        assert result.status_code == 401


@pytest.mark.parametrize("failure,status,code", [
    (repo.ServiceNotFound("PET_NOT_FOUND"), 404, "PET_NOT_FOUND"),
    (repo.ServiceNotFound("PRODUCT_NOT_FOUND"), 404, "PRODUCT_NOT_FOUND"),
    (repo.ServiceUnavailable("DSN-secret"), 503, "SERVICE_DB_UNAVAILABLE"),
])
def test_endpoint_safe_errors(monkeypatch, failure, status, code):
    configure(monkeypatch)
    monkeypatch.setattr(repo, "get_pet", MagicMock(side_effect=failure))
    with TestClient(api.app) as client:
        result = client.post(PATH, json=BODY, headers=HEADERS)
        assert result.status_code == status and result.json() == {"detail": code}
        metrics = client.get("/metrics").text
        assert "DSN-secret" not in result.text + metrics and HEADERS["X-Internal-Secret"] not in metrics


@pytest.mark.parametrize("flags,expected", [(["CHICKEN"], "SAFETY_BLOCKED"), ([], "SAFETY_DATA_INSUFFICIENT")])
def test_repository_to_http_real_engine(monkeypatch, flags, expected):
    configure(monkeypatch)
    monkeypatch.setattr(repo, "get_pet", lambda pid, mid: {"id": pid, "species": "DOG", "age": 4,
        "weight": 10, "life_stage": None, "allergies": ["CHICKEN"], "allergy_profile_status": "KNOWN_LIST"})
    monkeypatch.setattr(repo, "get_product", lambda pid: {"id": pid, "sku": None, "category_code": "FOOD",
        "target_species": ["DOG"], "target_age_group": "ADULT", "ingredient_codes": [], "allergen_flags": flags})
    with TestClient(api.app) as client:
        result = client.post(PATH, json=BODY, headers=HEADERS)
        assert result.status_code == 200
        body = result.json()
        assert body["safety_status"] == expected and body["excluded"] is True
        assert body["input_provenance"]["aafco_life_stage_evidence_status"] == "UNKNOWN"
        assert body["analysis_engine"] == "pipeline_p1c_v1"


def test_invalid_source_never_passes(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(repo, "get_pet", lambda *_: {})
    monkeypatch.setattr(repo, "get_product", lambda *_: {})
    with TestClient(api.app) as client:
        result = client.post(PATH, json=BODY, headers=HEADERS)
        assert result.status_code == 422 and result.json() == {"detail": "SERVICE_SOURCE_INVALID"}


@pytest.mark.parametrize("up,status", [(True, 200), (False, 503)])
def test_service_readiness_and_liveness(monkeypatch, up, status):
    configure(monkeypatch)
    monkeypatch.setattr(api, "check_dependencies", lambda _: {"artifact": {"required": True, "status": "UP"}})
    monkeypatch.setattr(repo, "probe", lambda _: {"required": True, "status": "UP" if up else "DOWN"})
    with TestClient(api.app) as client:
        result = client.get("/ready")
        assert result.status_code == status and result.json()["runtime_mode"] == "service"
        assert result.json()["dependencies"]["member_db"]["required"] is True
        assert result.json()["dependencies"]["product_db"]["required"] is True
        assert client.get("/health").status_code == 200


def test_service_mode_missing_config_is_not_local_ready(monkeypatch):
    monkeypatch.setenv("NUTRITION_RUNTIME_MODE", "service")
    with TestClient(api.app) as client:
        assert client.get("/ready").status_code == 503


def test_probe_select_only(monkeypatch):
    conn, cur = connection(monkeypatch, (1,), [])
    assert repo.probe("MEMBER_DATABASE_URL") == {"required": True, "status": "UP"}
    cur.execute.assert_called_once_with("SELECT 1")
    conn.close.assert_called_once()


def test_driver_has_bounded_read_only_options(monkeypatch):
    import psycopg2
    connect = MagicMock()
    monkeypatch.setattr(psycopg2, "connect", connect)
    # autouse fixture가 실제 DB 접근을 막으므로 원래 함수의 코드만 격리 실행한다.
    import importlib.util
    spec = importlib.util.spec_from_file_location("repository_driver_test", repo.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._connect("test-only-dsn")
    kwargs = connect.call_args.kwargs
    assert kwargs["connect_timeout"] == 3
    assert "statement_timeout=3000" in kwargs["options"]
    assert "default_transaction_read_only=on" in kwargs["options"]


def test_http_ownership_uses_verified_member_and_stops_before_product(monkeypatch):
    configure(monkeypatch)
    get_pet = MagicMock(side_effect=repo.ServiceNotFound("PET_NOT_FOUND"))
    get_product = MagicMock()
    monkeypatch.setattr(repo, "get_pet", get_pet)
    monkeypatch.setattr(repo, "get_product", get_product)
    with TestClient(api.app) as client:
        assert client.post(PATH, json=BODY, headers=HEADERS).status_code == 404
    get_pet.assert_called_once_with(123, 42)
    get_product.assert_not_called()


def test_product_missing_after_owned_pet(monkeypatch):
    configure(monkeypatch)
    monkeypatch.setattr(repo, "get_pet", lambda *_: {"id": 123})
    monkeypatch.setattr(repo, "get_product", MagicMock(side_effect=repo.ServiceNotFound("PRODUCT_NOT_FOUND")))
    with TestClient(api.app) as client:
        result = client.post(PATH, json=BODY, headers=HEADERS)
        assert result.status_code == 404 and result.json()["detail"] == "PRODUCT_NOT_FOUND"


@pytest.mark.parametrize("key", ["MEMBER_DATABASE_URL", "PRODUCT_DATABASE_URL"])
def test_partial_db_configuration_not_ready(monkeypatch, key):
    monkeypatch.setenv(key, "test-only-dsn")
    monkeypatch.setattr(repo, "probe", lambda _: {"required": True, "status": "DOWN"})
    with TestClient(api.app) as client:
        result = client.get("/ready")
        assert result.status_code == 503 and result.json()["runtime_mode"] == "service"
        assert client.post(PATH, json=BODY, headers=HEADERS).json()["detail"] == "SERVICE_SOURCE_NOT_CONFIGURED"


def test_added_pet_size_column_uses_actual_source(monkeypatch):
    """Verify the pet SELECT includes breed size and returns the stored value."""
    _, cur = connection(monkeypatch, (123,'DOG',4,8,3,True,None,'MEDIUM'), [[]])
    assert repo.get_pet(123,42)['target_breed_size'] == 'MEDIUM'
    assert 'birth_date, target_breed_size' in cur.execute.call_args_list[0].args[0]


def test_added_product_columns_shared_by_single_and_list(monkeypatch):
    """Verify single and bulk product reads return identical target and feeding metadata."""
    row = (456,'MOCK-0456','synthetic','FOOD','DRY_FOOD','ADULT','LARGE','성체','급여 표시 원문')
    _, cur = connection(monkeypatch,row,[[],[],[],[]])
    one = repo.get_product(456)
    for key in ('target_breed_size','feeding_target','feeding_method'):
        assert key in cur.execute.call_args_list[0].args[0]
    connection(monkeypatch,None,[[row],[],[],[],[]])
    listed = repo.list_active_products()[0]
    assert listed == one
    assert listed['target_breed_size'] == 'LARGE' and listed['feeding_method'] == '급여 표시 원문'
