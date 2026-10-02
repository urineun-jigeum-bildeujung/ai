"""로컬 HTTP 관측과 additive Safety 회귀. AWS E2E를 주장하지 않는다."""
import shutil
import sys
from itertools import product as combinations
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/nutrition")]
import api_nutrition as api
import observability as obs
import allergen_repository as repository
from allergen_service import evaluate_safety, _evaluate_safety
from test_runtime_e2e import dog_food, dog_pet

NEW_FIELDS = {"safety_reason_codes", "conflicting_allergens", "safety_message"}


@pytest.fixture
def client(monkeypatch):
    registry = obs.Metrics()
    monkeypatch.setattr(api.app.state, "nutrition_metrics", registry)
    # 기존 app에 설치한 middleware는 state가 아니라 동일 registry 객체를 보관한다.
    old_stack = api.app.middleware_stack
    middleware = next(m for m in api.app.user_middleware if m.cls is obs.ObservabilityMiddleware)
    old_registry = middleware.kwargs["metrics"]
    middleware.kwargs["metrics"] = registry
    api.app.middleware_stack = None
    with TestClient(api.app, raise_server_exceptions=False) as value:
        yield value
    middleware.kwargs["metrics"] = old_registry
    api.app.middleware_stack = old_stack


def payload():
    return {"pet": dog_pet(), "product": dog_food(nutrition_items=[])}


def test_existing_business_fields_and_priority_unchanged():
    # 3 profiles × 2 categories × 3 species × 4 stages × 3 ingredients = 216 branches.
    for profile, category, species, stage, ingredients in combinations(
        ["KNOWN_NONE", "KNOWN_LIST", "UNKNOWN"], ["food", "treat"], [None, "dog", "cat"],
        [None, "ADULT", "GROWTH", "ALL_LIFE_STAGES"], [[], ["chicken"], ["unmapped"]],
    ):
        pet = dog_pet(allergy_profile_status=profile, allergies=["chicken"] if profile == "KNOWN_LIST" else [])
        data = dog_food(category=category, target_species=species, aafco_life_stage=stage, ingredient_list=ingredients)
        actual = evaluate_safety(pet, data)
        assert {k: v for k, v in actual.items() if k not in NEW_FIELDS} == _evaluate_safety(pet, data)
        assert NEW_FIELDS <= actual.keys()


def test_product_cannot_clear_pet_allergy_and_conflict_projection_is_traceable():
    result = evaluate_safety(dog_pet(allergies=["chicken"], allergy_profile_status="KNOWN_LIST"),
                             dog_food(ingredient_list=["chicken"], allergies=[]))
    assert result["excluded"] and result["safety_status"] == "SAFETY_BLOCKED"
    assert result["safety_reason_codes"] == ["ALLERGY_CONFLICT"]
    assert result["conflicting_allergens"][0]["raw_ingredient"] == "chicken"


@pytest.mark.parametrize("trace", ["STALE_EVIDENCE", "LINEAGE_INTEGRITY_ERROR"])
def test_stale_evidence_reason_is_not_called_missing_ingredient(trace):
    result = evaluate_safety(dog_pet(allergies=["chicken"], allergy_profile_status="KNOWN_LIST"),
                             dog_food(product_allergen_refs=[], _evidence_trace=trace))
    assert result["safety_reason_codes"] == [trace] and result["excluded"]


def test_nutrition_only_insufficient_is_not_safety_exclusion(client):
    result = client.post("/api/nutrition/analyze", json=payload()).json()
    assert result["analysis_status"] == "INSUFFICIENT_DATA" and not result["excluded"]
    assert result["safety_status"] == "NOT_APPLICABLE"


def test_ready_real_local_dependencies_not_service_readiness(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["runtime_mode"] == "local_artifact"
    assert response.json()["dependencies"]["service_source"] == {"required": False, "status": "NOT_CONFIGURED"}
    assert client.get("/health").json() == api.health()


@pytest.mark.parametrize("relative", [p for p, _ in obs.JSON_DEPENDENCIES.values()] + [
    "data/processed/required_nutrient_matrix_v2.csv", "data/processed/allergen_evidence_catalog_p2.db"])
def test_missing_dependency_down_no_recreate(client, monkeypatch, tmp_path, relative):
    paths = [p for p, _ in obs.JSON_DEPENDENCIES.values()] + [
        "data/processed/required_nutrient_matrix_v2.csv", "data/processed/allergen_evidence_catalog_p2.db"]
    for path in paths:
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path != relative:
            shutil.copyfile(ROOT / path, destination)
    monkeypatch.setattr(api, "PROJECT_ROOT", tmp_path)
    result = client.get("/ready")
    assert result.status_code == 503 and not (tmp_path / relative).exists()
    assert str(tmp_path) not in result.text


@pytest.mark.parametrize("status", ["READY", "SAFETY_BLOCKED", "INSUFFICIENT_DATA", "SERVICE_DEGRADED", "FAILED"])
def test_http200_and_domain_are_separate(client, monkeypatch, status):
    body = {"analysis_status": status, "pet_id": "private-pet"}
    monkeypatch.setattr(api, "_analyze_direct", lambda req: body)
    response = client.post("/api/nutrition/analyze", json=payload())
    assert response.status_code == 200 and response.json() == body
    metrics = client.get("/metrics").text
    assert 'status_code="200"} 1' in metrics and f'status="{status}"}} 1' in metrics
    assert "private-pet" not in metrics


def test_safety_and_analysis_counted_separately(client, monkeypatch):
    monkeypatch.setattr(api, "_analyze_direct", lambda req: {"analysis_status": "INSUFFICIENT_DATA", "safety_status": "SAFETY_BLOCKED"})
    client.post("/api/nutrition/analyze", json=payload())
    text = client.get("/metrics").text
    assert 'axis="analysis_status",status="INSUFFICIENT_DATA"' in text
    assert 'axis="safety_status",status="SAFETY_BLOCKED"' in text


@pytest.mark.parametrize("request_id", ["", "x" * 1000, "user@example.com", "Bearer private-secret"])
def test_invalid_id_replaced_not_exported(client, request_id):
    result = client.get("/health", headers={"X-Request-ID": request_id})
    assert len(result.headers["x-request-id"]) == 32 and result.headers["x-request-id"] != request_id


def test_errors_and_unknown_paths_do_not_leak(client, monkeypatch):
    response = client.get("/health", headers={"X-Request-ID": "request_12345678"})
    assert response.headers["x-request-id"] == "request_12345678"
    assert client.post("/api/nutrition/analyze", json={}).status_code == 422
    assert client.post("/api/nutrition/compare").status_code == 422
    assert client.get("/private-person?token=private-secret").status_code == 404
    def fail(req):
        raise RuntimeError("private-secret")
    monkeypatch.setattr(api, "_analyze_direct", fail)
    response = client.post("/api/nutrition/analyze", json=payload())
    assert response.status_code == 500 and response.text == "Internal Server Error"
    assert "x-request-id" in response.headers
    text = client.get("/metrics").text
    assert "private-" not in text and 'status="FAILED"' in text
    assert 'route="__unmatched__",status_code="404"' in text


def test_concurrent_requests_no_context_cross_contamination(client, monkeypatch):
    monkeypatch.setattr(api, "_analyze_direct", lambda req: {"analysis_status": req.product.name})
    def request(index):
        body = payload()
        body["product"]["name"] = "READY" if index % 2 else "INSUFFICIENT_DATA"
        return client.post("/api/nutrition/analyze", json=body).status_code
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(request, range(20))) == [200] * 20
    text = client.get("/metrics").text
    assert 'status_code="200"} 20' in text and 'status="READY"} 10' in text and 'status="INSUFFICIENT_DATA"} 10' in text


def test_metrics_no_dependency_probe_no_self_count(client, monkeypatch):
    monkeypatch.setattr(api, "check_dependencies", lambda _: pytest.fail("metrics must not probe"))
    assert client.get("/metrics").text == client.get("/metrics").text


def test_catalog_query_does_not_create_missing_database(tmp_path):
    missing = tmp_path / "missing.db"
    assert repository.get_refs("absent", "version", "version", path=missing) == ([], "LINEAGE_INTEGRITY_ERROR")
    assert not missing.exists()


def test_unknown_domain_value_and_dependency_errors_are_redacted(client, monkeypatch):
    monkeypatch.setattr(api, "_analyze_direct", lambda req: {"analysis_status": "private-value"})
    client.post("/api/nutrition/analyze", json=payload())
    def fail(*args):
        raise OSError("private-credential")
    monkeypatch.setattr(obs, "_check_json", fail)
    response = client.get("/ready")
    assert response.status_code == 503 and "private-" not in response.text
    assert "private-" not in client.get("/metrics").text


@pytest.mark.parametrize("content", ['{"rows":[]}', 'invalid-json'])
def test_empty_or_corrupt_json_not_ready(tmp_path, content):
    path = tmp_path / "reference.json"
    path.write_text(content)
    if content.startswith("{"):
        assert not obs._check_json(path, "rows")
    else:
        with pytest.raises(ValueError):
            obs._check_json(path, "rows")


def test_duplicate_request_ids_are_not_reflected(client):
    response = client.get("/health", headers=[("X-Request-ID", "request_12345678"), ("X-Request-ID", "request_87654321")])
    assert len(response.headers["x-request-id"]) == 32
