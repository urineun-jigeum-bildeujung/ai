"""본문·개인정보를 수집하지 않는 로컬 runtime 관측과 읽기 전용 의존성 점검."""
from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections import Counter
from contextlib import closing
from contextvars import ContextVar
from threading import Lock
from uuid import uuid4

from starlette.responses import PlainTextResponse
from allergen_catalog_versions import DICTIONARY_VERSION, PIPELINE_VERSION

ROUTES = frozenset({
    "/health", "/ready", "/metrics", "/docs", "/openapi.json", "/redoc",
    "/docs/oauth2-redirect", "/api/nutrition/analyze", "/api/nutrition/safety",
    "/api/nutrition/report", "/api/nutrition/compare",
    "/api/nutrition/analyze/by-product-id", "/api/nutrition/analyze/by-service-id",
})
DOMAIN_STATUSES = frozenset({
    "READY", "PARTIAL", "INSUFFICIENT_DATA", "SAFETY_BLOCKED", "SAFETY_DATA_INSUFFICIENT",
    "SERVICE_DEGRADED", "FAILED", "UNKNOWN", "UNSUPPORTED", "NOT_APPLICABLE",
    "NO_CONFLICT_DETECTED", "CAUTION",
})
_observation = ContextVar("nutrition_observation", default=None)
_request_id = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{7,63}\Z", re.ASCII)


def observe_domain_result(result):
    observation = _observation.get()
    if observation is not None:
        for axis in ("analysis_status", "safety_status"):
            value = result.get(axis)
            if isinstance(value, str) and value in DOMAIN_STATUSES:
                observation[axis] = value
    return result


class Metrics:
    """프로세스별 집계. ID, 원문, 임의 경로, 오류 내용은 label로 허용하지 않는다."""
    def __init__(self):
        self._lock = Lock()
        self._http, self._domain = Counter(), Counter()

    def record(self, method, route, status, domain):
        method = method if method in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"} else "OTHER"
        route = route if route in ROUTES else "__unmatched__"
        status = status if isinstance(status, int) and 100 <= status <= 599 else 500
        with self._lock:
            self._http[method, route, status] += 1
            if route.startswith("/api/nutrition/"):
                for axis, value in domain.items():
                    if axis in {"analysis_status", "safety_status"} and value in DOMAIN_STATUSES:
                        self._domain[route, axis, value] += 1

    def render(self):
        with self._lock:
            http, domain = self._http.copy(), self._domain.copy()
        lines = ["# HELP nutrition_http_requests_total HTTP outcomes independent of domain decisions.",
                 "# TYPE nutrition_http_requests_total counter"]
        for (method, route, status), count in sorted(http.items()):
            lines.append(f'nutrition_http_requests_total{{method="{method}",route="{route}",status_code="{status}"}} {count}')
        lines += ["# HELP nutrition_domain_outcomes_total Observed domain decisions, not quality scores.",
                  "# TYPE nutrition_domain_outcomes_total counter"]
        for (route, axis, status), count in sorted(domain.items()):
            lines.append(f'nutrition_domain_outcomes_total{{route="{route}",axis="{axis}",status="{status}"}} {count}')
        return "\n".join(lines) + "\n"


class ObservabilityMiddleware:
    """ASGI body는 읽거나 재직렬화하지 않는다. 요청별 context는 반드시 해제한다."""
    def __init__(self, app, metrics):
        self.app, self.metrics = app, metrics

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        ids = [v for k, v in scope.get("headers", []) if k.lower() == b"x-request-id"]
        incoming = ids[0].decode("ascii", errors="replace") if len(ids) == 1 else ""
        request_id = incoming if _request_id.fullmatch(incoming) else uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        observation = {}
        token = _observation.set(observation)
        status, started = 500, False

        async def observed_send(message):
            nonlocal status, started
            if message["type"] == "http.response.start":
                status, started = message["status"], True
                message = {**message, "headers": [
                    (k, v) for k, v in message.get("headers", []) if k.lower() != b"x-request-id"
                ] + [(b"x-request-id", request_id.encode("ascii"))]}
            await send(message)

        try:
            await self.app(scope, receive, observed_send)
        except Exception:
            observation.clear()
            observation["analysis_status"] = "FAILED"
            if not started:
                await PlainTextResponse("Internal Server Error", status_code=500)(scope, receive, observed_send)
            raise
        finally:
            route = getattr(scope.get("route"), "path", "__unmatched__")
            if route != "/metrics":
                self.metrics.record(scope["method"], route, status, observation)
            _observation.reset(token)


JSON_DEPENDENCIES = {
    "feed_codes": ("data/raw/seed_feed_codes.json", "ingredient_matching_dictionary_v1"),
    "legacy_reference": ("data/raw/seed_nutrition_standard.json", "reference_tables"),
    "ingredient_normalizer": ("data/raw/seed_11_allergen_ingredient_map.json", "items"),
    "allergen_dictionary": ("data/raw/seed_11_allergen_ingredient_map_v3.json", "items"),
    "guaranteed_analysis": ("data/raw/seed_13_guaranteed_analysis.json", "items"),
    "products_opff": ("data/raw/seed_9_placeholder_feed_opff.json", "items"),
    "products_oem": ("data/raw/seed_9b_off_korean_oem.json", "products"),
    "products_global": ("data/raw/seed_9_global_brands_v2.json", "products"),
    "nias_reference": ("data/processed/nutrition_reference_nias_2024_parity_p0_v1.json", "rows"),
    "opff_cache": ("data/processed/opff_api_cache_v1.json", None),
}


def _check_json(path, key):
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get(key) if isinstance(value, dict) and key else value
    return isinstance(value, dict) and isinstance(rows, (dict, list)) and bool(rows)


def _check_matrix(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"species", "life_stage", "nutrient_code", "tier", "unit", "basis"}
        rows = list(reader)
        return bool(rows) and required.issubset(reader.fieldnames or []) and all(
            all(row.get(key) for key in required) for row in rows
        )


def _check_catalog(path):
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.1)) as db:
        if db.execute("PRAGMA quick_check").fetchone() != ("ok",):
            return False
        db.execute("""SELECT evidence_id, raw_ingredient_text, normalized_text, segmented_text,
                      matched_alias, mapping_method, source_dataset, usable_for_safety
                      FROM product_allergen_refs LIMIT 0""")
        if db.execute("SELECT 1 FROM product_allergen_refs LIMIT 1").fetchone() is None:
            return False
        invalid = db.execute("""
            SELECT 1 FROM product_allergen_refs r LEFT JOIN product_ingredient_components c
              ON r.component_occurrence_id=c.component_occurrence_id
            WHERE c.component_occurrence_id IS NULL OR c.processing_status IS NULL
               OR r.dictionary_version IS NULL OR r.pipeline_version IS NULL
               OR r.dictionary_version != ? OR r.pipeline_version != ? LIMIT 1
        """, (DICTIONARY_VERSION, PIPELINE_VERSION)).fetchone()
        return invalid is None


def check_dependencies(root):
    checks = {name: lambda p=path, k=key: _check_json(root / p, k)
              for name, (path, key) in JSON_DEPENDENCIES.items()}
    checks["coverage_matrix"] = lambda: _check_matrix(root / "data/processed/required_nutrient_matrix_v2.csv")
    checks["allergen_catalog"] = lambda: _check_catalog(root / "data/processed/allergen_evidence_catalog_p2.db")
    result = {}
    for name, check in checks.items():
        try:
            available = check()
        except Exception:
            available = False  # 오류 내용/경로/접속 문자열은 노출하지 않는다.
        result[name] = {"required": True, "status": "UP" if available else "DOWN"}
    result["service_source"] = {"required": False, "status": "NOT_CONFIGURED"}
    return result
