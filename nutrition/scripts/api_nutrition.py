"""FastAPI — 골라주개냥 영양성분 분석 API (8/20 시작, 4 endpoint)

- POST /api/nutrition/analyze   1사료 분석 (FR-AI-1-01~05)
- POST /api/nutrition/safety    안전 7원칙 P0 검증
- POST /api/nutrition/report    리포트 생성 (FR-AI-1-06)
- POST /api/nutrition/compare   Service-ID comparison
- GET  /health                  서비스 상태

실행:
    cd nutrition
    python -m uvicorn scripts.api_nutrition:app --reload --port 8002

테스트:
    curl -s http://localhost:8002/health
    curl -s -X POST http://localhost:8002/api/nutrition/analyze -H "Content-Type: application/json" -d '{ ... }'
"""
from __future__ import annotations

import re
import hmac
import os
import sys
from pathlib import Path
from typing import Any, Literal


# nutrition/ 패키지 경로
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NUTRITION_DIR = PROJECT_ROOT / "scripts" / "nutrition"
sys.path.insert(0, str(NUTRITION_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from pydantic import BaseModel, Field, field_validator  # noqa: E402
from starlette.responses import JSONResponse, PlainTextResponse  # noqa: E402
from observability import Metrics, ObservabilityMiddleware, check_dependencies, observe_domain_result  # noqa: E402

from match_v1 import load_seed  # type: ignore  # noqa: E402
from match_v1_1_category import match_by_category  # type: ignore  # noqa: E402
from normalizer import extract_allergens_from_text, normalize_ingredient  # type: ignore  # noqa: E402
from allergen_service import evaluate_safety  # type: ignore  # noqa: E402
from allergen_repository import get_refs  # type: ignore  # noqa: E402
from allergen_catalog_versions import DICTIONARY_VERSION, PIPELINE_VERSION  # type: ignore  # noqa: E402
from product_input_adapter import load_product_input  # type: ignore  # noqa: E402
from service_db_adapter import adapt_pet, adapt_product, evaluate_service_safety, ServiceInputError  # noqa: E402
from mock_integration_fixture import probe_fixture  # noqa: E402
from feeding import calculate_feeding  # noqa: E402
from mer_coefficient_policy_v2 import resolve_mer_coefficient  # noqa: E402
from mock_feeding_fixture import mock_energy  # noqa: E402
from service_compare import compare_service_analyses
from suitability import build_suitability
from presentation import build_nutrition_presentation
from product_target_contract import load_synthetic_target, resolve_product_target, evaluate_target_compatibility
import service_repository  # noqa: E402
from nutrition_readiness import (  # type: ignore  # noqa: E402
    evaluate_nutrition_coverage,
    evaluate_nutrition_readiness,
    resolve_reference_stage,
    resolve_product_target_stage,
)
from pipeline_p1c_v1 import (  # type: ignore  # noqa: E402
    align_units,
    classify_dry_wet,
    compare_nias,
    compute_nutrition_comparison_status,
    detect_outliers,
    load_raw,
    normalize,
    normalize_basis,
)


app = FastAPI(
    title="골라주개냥 영양성분 분석 API",
    version="1.0.0",
    description="사료 성분 분석 + 알레르기 hard filter + NIAS 2024 비교",
)

SEED_FEED_CODES = load_seed("seed_feed_codes.json")
app.state.nutrition_metrics = Metrics()
app.add_middleware(ObservabilityMiddleware, metrics=app.state.nutrition_metrics)


def _allergy_gate(pet: PetIn, product: ProductIn) -> dict[str, Any]:
    data = product.model_dump()
    refs, trace = get_refs(product.id, DICTIONARY_VERSION, PIPELINE_VERSION)
    if trace == "PRECOMPUTED": data["product_allergen_refs"] = refs
    elif trace in {"STALE_EVIDENCE", "LINEAGE_INTEGRITY_ERROR"}:
        # A stale or broken lineage must not fall back to ad-hoc ingredient parsing.
        # The shared safety service turns this into SAFETY_DATA_INSUFFICIENT when
        # the pet has a known allergen profile.
        data["product_allergen_refs"] = []
        data["_evidence_trace"] = trace
    return evaluate_safety(pet.model_dump(), data)


def _run_p0d(pet: PetIn, product: ProductIn) -> dict[str, Any]:
    """Run the production request through the same P0-D pipeline functions.

    This adapter is deliberately request-scoped: it reads only the versioned NIAS
    reference and never executes pipeline_p1c_v1.main(), which writes batch files.
    """
    stage_info = resolve_reference_stage(pet.model_dump())
    readiness = evaluate_nutrition_readiness(product.model_dump(), pet.species, stage_info)
    coverage = evaluate_nutrition_coverage(product.model_dump(), pet.species, stage_info)
    product_species = (product.target_species or "").upper()
    if product_species == "BOTH":
        product_species = pet.species.upper()
    raw_items = [
        {
            "product_id": product.id,
            "nutrient_code": item.nutrient_code,
            "value": item.value,
            "unit": item.unit,
            "basis": item.basis,
            "source": item.source,
        }
        for item in product.nutrition_items
    ]
    pipeline_items = normalize(raw_items)
    pipeline_items = align_units(pipeline_items)
    pipeline_items = detect_outliers(pipeline_items)
    pipeline_items = classify_dry_wet(pipeline_items)
    pipeline_items = normalize_basis(pipeline_items)
    if stage_info["status"] == "RESOLVED" and product_species in {"DOG", "CAT"}:
        references = load_raw()["nias_rows"]
        stage = stage_info["stage"]
        compared, _, _ = compare_nias(
            pipeline_items, references, {product.id: product_species}, {product.id: stage},
            {product.id: pet.life_stage_detail} if pet.life_stage_detail else None,
            {product.id: product.product_form} if product.product_form else None,
        )
        status = compute_nutrition_comparison_status(
            compared, {product.id: product_species}, {product.id: stage},
        ).get(product.id)
    else:
        stage = None
        compared = [{**item, "nias_compare_status": "NO_REF"} for item in pipeline_items]
        status = None
    # No items still has a valid P0-D result: every essential nutrient is missing.
    if status is None:
        essential_result = compute_nutrition_comparison_status(
            [{"product_id": product.id, "nutrient_code": "__MISSING__", "nias_compare_status": "NO_VALUE"}],
            {product.id: product_species}, {product.id: stage},
        )[product.id]
        status = essential_result

    return {
        "analysis_engine": "pipeline_p1c_v1",
        "p0_d_applied": True,
        "nutrition_comparison_status": status["nutrition_comparison_status"],
        "aafco_pass": status["aafco_pass"],
        "nutrition_comparison": status,
        "nutrition_items": compared,
        "input_readiness": readiness,
        # Matrix coverage is a separate descriptive axis. It never upgrades the
        # runtime decision and does not claim nutritional adequacy.
        "nutrition_coverage": coverage,
        "pet_reference_stage": stage_info,
        "product_target_stage": resolve_product_target_stage(product.aafco_life_stage),
        "warnings": (
            ["P0-D 판정에 필요한 구조화된 보증성분(nutrition_items)이 없습니다."]
            if not product.nutrition_items else []
        ),
    }


def _enforce_p0d_result_contract(result: dict[str, Any]) -> dict[str, Any]:
    """Fail-close when a P0-D adapter result omits its required status axes.

    ``_run_p0d`` owns these fields in normal runtime.  This guard also makes an
    incomplete adapter implementation (or a test double) explicit instead of
    allowing a KeyError or an accidental READY result at the API boundary.
    """
    readiness = result.get("input_readiness")
    if not isinstance(readiness, dict) or not readiness.get("input_readiness"):
        result["input_readiness"] = {
            "input_readiness": "INSUFFICIENT_DATA",
            "reason_codes": ["P0D_RESULT_CONTRACT_INVALID"],
        }

    coverage = result.get("nutrition_coverage")
    if not isinstance(coverage, dict) or not coverage.get("nutrition_coverage"):
        result["nutrition_coverage"] = {
            "nutrition_coverage": "UNKNOWN",
            "reason_codes": ["P0D_RESULT_CONTRACT_INVALID"],
        }
    return result


def _analyze_product(req: AnalyzeRequest, *, source_safety: dict[str, Any] | None = None) -> dict[str, Any]:
    """Single API path: food uses P0-D; legacy matching is limited to non-food categories."""
    product = req.product
    # 조회된 service ID로 로컬 catalog를 재조회하지 않는다. evidence ID는 adapter 소유다.
    safety = _allergy_gate(req.pet, product) if source_safety is None else source_safety

    if product.category == "food":
        result = _enforce_p0d_result_contract(_run_p0d(req.pet, product))
        result.update(safety)
        if (result["input_readiness"]["input_readiness"] != "READY"
                or safety["safety_status"] in {"SAFETY_BLOCKED", "SAFETY_DATA_INSUFFICIENT"}
                or result["nutrition_comparison_status"] == "UNKNOWN"):
            # Allergy safety and nutrition completeness are independent facts.
            # Never replace NO_CONFLICT_DETECTED with a nutrition-data result.
            result["analysis_status"] = "INSUFFICIENT_DATA"
        else:
            result["analysis_status"] = "READY"
        result["category_branch"] = "food"
        result["consumer_card"] = (
            "알레르기 관련 원료 정보를 충분히 확인할 수 없어 안전 판정을 보류했습니다."
            if safety["safety_status"] == "SAFETY_DATA_INSUFFICIENT"
            else "등록 알레르기와 충돌하는 원료가 확인되어 이 상품은 제외했습니다."
            if safety["safety_status"] == "SAFETY_BLOCKED"
            else "원료 또는 필수 보증성분 정보가 부족하여 영양 적합성을 판정할 수 없습니다."
            if result["nutrition_comparison_status"] == "UNKNOWN"
            else "구조화된 라벨 정보와 기준표를 비교한 결과입니다. 의료적 진단이나 처방이 아닙니다."
        )
    else:
        # P0-D is a complete-and-balanced food comparison rule. Other categories
        # retain their category logic, but the shared allergy gate still fail-closes.
        result = match_by_category(req.pet.model_dump(), product.model_dump())
        result.update(safety)
        result["analysis_engine"] = "match_v1_1_category"
        result["p0_d_applied"] = False
        result["input_readiness"] = {
            "input_readiness": "UNSUPPORTED",
            "reason_codes": ["CATEGORY_NOT_FOOD"],
        }
        result["nutrition_coverage"] = {
            "nutrition_coverage": "UNKNOWN",
            "reason_codes": ["CATEGORY_NOT_FOOD"],
        }
        result["nutrition_comparison_status"] = "NOT_APPLICABLE"
        result["aafco_pass"] = None
        result["analysis_status"] = (
            "INSUFFICIENT_DATA"
            if safety["safety_status"] in {"SAFETY_BLOCKED", "SAFETY_DATA_INSUFFICIENT"}
            else "READY"
        )

    if safety.get("conflicting_toxic_ingredients"):
        result["consumer_card"] = safety["safety_message"]

    result["ingredient_normalized"] = [
        normalize_ingredient(ingredient) for ingredient in product.ingredient_list
    ]
    result["pet_id"] = req.pet.id
    result["product_id"] = product.id
    return result


# ===== Pydantic 스키마 =====

class PetIn(BaseModel):
    id: str
    species: str = Field(..., pattern="^(dog|cat)$")
    age_years: float = Field(..., ge=0)
    weight_kg: float = Field(..., gt=0)
    allergies: list[str] = Field(default_factory=list)
    allergy_profile_status: str | None = Field(default=None, pattern="^(KNOWN_NONE|KNOWN_LIST|UNKNOWN)$")
    life_stage: str | None = None  # puppy/kitten/adult/senior
    # Optional raw-detail contract for NIAS rules that distinguish a broad
    # growth/reproduction stage. Omission is intentionally not inferred.
    life_stage_detail: str | None = None
    target_breed_size: Literal["SMALL", "MEDIUM", "LARGE"] | None = None
    product_target_stage: Literal["GROWTH", "ADULT", "SENIOR"] | None = None


class ProductIn(BaseModel):
    id: str
    name: str
    category: str = Field(default="food", pattern="^(food|treat|pad|litter|supplement)$")
    ingredient_list: list[str] = Field(default_factory=list)
    guaranteed_analysis: dict[str, float] = Field(default_factory=dict)
    # P0-D 입력 계약. 기존 guaranteed_analysis 는 nutrient code/unit/basis가 없어
    # P0-D의 기준 비교 입력으로 사용하지 않는다.
    nutrition_items: list["NutritionItemIn"] = Field(default_factory=list)
    target_species: str | None = Field(default=None, pattern="^(dog|cat|both)$")
    # This is a label claim supplied by the caller, not evidence of complete and
    # balanced nutritional adequacy.  Missing must remain missing.
    aafco_life_stage: str | None = None
    calorie_kcal_per_kg: float | None = None
    # Explicit label/source product type has priority over moisture inference
    # only for the documented DRY/WET forms. Unsupported strings remain unknown.
    product_form: str | None = None
    service_target_age_group: Literal["GROWTH", "ADULT", "SENIOR"] | None = None
    service_target_breed_size: Literal["SMALL", "MEDIUM", "LARGE"] | None = None
    feeding_target: str | None = None
    feeding_method: str | None = None
    product_attributes: dict[str, str] = Field(default_factory=dict)
    ingredient_source: str = "PRODUCT_LABEL"
    ingredient_source_version: str = "API_REQUEST"


class AnalyzeRequest(BaseModel):
    pet: PetIn
    product: ProductIn


class PersistedProductAnalyzeRequest(BaseModel):
    """Operational request: load immutable local product evidence by ID."""

    pet: PetIn
    product_id: str = Field(..., min_length=1)


class ServiceIdAnalyzeRequest(BaseModel):
    """서비스 준비용 계약. 인증 주체는 요청 본문에서 받지 않는다."""
    model_config = {"extra": "forbid"}
    pet_id: int = Field(..., gt=0, strict=True)
    product_id: int = Field(..., gt=0, strict=True)
    allergy_profile_status: Literal["UNKNOWN", "KNOWN_NONE", "KNOWN_LIST"] | None = None


class ServiceCompareRequest(BaseModel):
    model_config = {"extra": "forbid"}
    pet_id: int = Field(..., gt=0, strict=True)
    product_ids: list[int] = Field(..., min_length=2, max_length=2)
    allergy_profile_status: Literal["UNKNOWN", "KNOWN_NONE", "KNOWN_LIST"] | None = None

    @field_validator("product_ids", mode="before")
    @classmethod
    def validate_product_ids(cls, value):
        if not isinstance(value, list) or len(value) != 2 or any(type(v) is not int or v <= 0 for v in value):
            raise ValueError("COMPARE_EXACTLY_TWO_POSITIVE_PRODUCT_IDS_REQUIRED")
        if value[0] == value[1]:
            raise ValueError("COMPARE_PRODUCT_IDS_MUST_BE_DISTINCT")
        return value


def _attach_projections(result, pet, product, synthetic_fixture=None):
    target = resolve_product_target(product, synthetic_fixture)
    result["target_compatibility"] = evaluate_target_compatibility(pet, target)
    result["presentation"] = build_nutrition_presentation(
        result.get("nutrition_items", []), result["nutrition_comparison_status"])
    result["suitability"] = build_suitability(
        nutrition_comparison=result.get("nutrition_comparison"),
        target_compatibility=result["target_compatibility"],
        safety_status=result["safety_status"], excluded=result["excluded"])
    return target


def _analyze_direct(req):
    result = _analyze_product(req)
    pet = req.pet.model_dump()
    if pet.get("product_target_stage") is None:
        age = pet["age_years"]
        pet["product_target_stage"] = "GROWTH" if age < 1 else "ADULT" if age < 7 else "SENIOR"
    product = req.product.model_dump()
    _attach_projections(result, pet, product)
    result["product_label"] = {
        "feeding_target": product.get("feeding_target"), "feeding_method": product.get("feeding_method"),
        "target_age_group": product.get("service_target_age_group"),
        "target_breed_size": product.get("service_target_breed_size"), "source": "API_REQUEST",
    }
    return result


def analyze_service_records(pet_source: dict, product_source: dict) -> dict[str, Any]:
    """ownership 확인을 마친 source에만 사용할 내부 경계. HTTP 인증을 대신하지 않는다."""
    pet = adapt_pet(pet_source)
    loaded = adapt_product(product_source)
    source_meta = loaded.get("provenance", {}).get("nutrition_source") or {}
    if source_meta.get("type") == "MOCK_INTEGRATION_FIXTURE" and pet.get("life_stage") == "UNKNOWN":
        # Integration fixture only: expose the already-existing AGE_RULE in the
        # shared Rule Engine.  No life-stage value is invented, and the strict
        # production evidence path keeps its previous behavior.
        pet["life_stage"] = None
        loaded["provenance"]["pet_reference_stage_source"] = "AGE_RULE_ELIGIBLE"
    product = loaded["product"]
    req = AnalyzeRequest(pet=PetIn(**pet), product=ProductIn(**product))
    result = _analyze_product(req, source_safety=evaluate_service_safety(pet, product))
    target = _attach_projections(result, pet, product, load_synthetic_target(product_source))
    loaded["provenance"]["product_target"] = target
    # MER uses canonical calendar age and the Service neuter flag. The
    # Nutrition reference-stage compatibility adjustment cannot select a factor.
    coefficient = resolve_mer_coefficient({**pet, "is_neutered": pet_source.get("is_neutered")})
    feeding = calculate_feeding(
        weight_kg=pet["weight_kg"], species=pet["species"],
        energy=mock_energy(product_source, source_meta), coefficient=coefficient,
        allow_energy_requirement=True,
    )
    if result["excluded"]:
        feeding.update(status="BLOCKED", daily_serving_g=None, mer_kcal_per_day=None)
        feeding["reason_codes"].append("FEEDING_SAFETY_EXCLUDED")
    feeding.update(applicability=coefficient["applicability"],
                   disclaimer_code=coefficient["disclaimer_code"])
    result["feeding"] = feeding
    result["product_label"] = {
        "feeding_target": product.get("feeding_target"),
        "feeding_method": product.get("feeding_method"),
        "target_age_group": product.get("service_target_age_group"),
        "target_breed_size": product.get("service_target_breed_size"),
        "source": "SERVICE_DB",
    }
    if source_meta.get("type") == "MOCK_INTEGRATION_FIXTURE":
        loaded["provenance"]["integration_data"] = {
            "data_generation_type": "SCHEMA_DRIVEN_SYNTHETIC", "production_evidence": False,
            "schema_contract": "SERVICE_DB_COMPATIBLE",
            "fixture_version": source_meta.get("fixture_version"),
            "result_type": "DERIVED_RULE_RESULT",
            "input_evidence_type": "SCHEMA_DRIVEN_SYNTHETIC_DATA",
        }
    result["input_provenance"] = loaded["provenance"]
    return result


class NutritionItemIn(BaseModel):
    """라벨 기반 보증성분 1건. value=0은 placeholder로 처리되어 판정에 쓰지 않는다."""

    nutrient_code: str = Field(..., pattern="^[A-Z0-9_]+$")
    value: float | None = None
    unit: str = "PERCENT"
    basis: str = Field(default="AS_FED", pattern="^(AS_FED|DRY_MATTER)$")
    source: str = "API_REQUEST"


class SafetyRequest(BaseModel):
    pet: PetIn
    product: ProductIn
    consumer_card: str | None = None


# ===== 안전 7원칙 P0 (8/20 stub, 8/21 본격) =====

DIAGNOSIS_FORBIDDEN = ["진단", "처방", "치료", "diagnose", "prescribe", "treat"]
CAUSATION_FORBIDDEN = ["때문에", "원인", "because of", "caused by"]
UNSOURCED_NUMERIC_PATTERN = re.compile(r"\b\d{2,3}%\s*(안전|추천|도움|효과|개선)", re.IGNORECASE)


def safety_check_consumer_card(card: str | None) -> dict:
    """안전 7원칙 P0 자동 검증

    P0-1: 진단/처방/치료 단어 사용 금지
    P0-2: 구매행동 → 건강 인과 단정 금지
    P0-3: 출처 없는 수치 금지
    """
    if not card:
        return {"pass": True, "violations": [], "note": "consumer_card 없음 — skip"}
    violations: list[dict[str, Any]] = []

    for word in DIAGNOSIS_FORBIDDEN:
        if word in card:
            violations.append({"rule": "P0-1", "match": word, "severity": "critical", "action": "FAIL — 의료 확정 표현"})

    for word in CAUSATION_FORBIDDEN:
        if word in card:
            violations.append({"rule": "P0-2", "match": word, "severity": "high", "action": "REVIEW — 인과 단정 의심"})

    for m in UNSOURCED_NUMERIC_PATTERN.finditer(card):
        violations.append({"rule": "P0-3", "match": m.group(0), "severity": "high", "action": "REVIEW — 출처 없는 수치"})

    return {
        "pass": len(violations) == 0,
        "violations": violations,
        "note": f"{len(violations)}건 위반" if violations else "PASS",
    }


# ===== Endpoints =====

@app.get("/ready")
def ready() -> JSONResponse:
    dependencies = check_dependencies(PROJECT_ROOT)
    mode = "service" if service_repository.service_mode() else "local_artifact"
    if mode == "service":
        dependencies["member_db"] = service_repository.probe("MEMBER_DATABASE_URL")
        dependencies["product_db"] = service_repository.probe("PRODUCT_DATABASE_URL")
        dependencies["service_auth"] = {"required": True, "status": "UP" if os.getenv("INTERNAL_GATEWAY_SECRET") else "DOWN"}
        dependencies["service_source"] = {"required": True, "status": "UP" if service_repository.configured() else "DOWN"}
        dependencies["mock_integration_fixture"] = probe_fixture()
    if os.getenv("NUTRITION_RUNTIME_MODE", "local") not in {"local", "service"}:
        dependencies["runtime_config"] = {"required": True, "status": "DOWN"}
    available = all(v["status"] == "UP" for v in dependencies.values() if v["required"])
    return JSONResponse({"status": "ready" if available else "not_ready",
                         "runtime_mode": mode, "dependencies": dependencies},
                        status_code=200 if available else 503)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> PlainTextResponse:
    return PlainTextResponse(app.state.nutrition_metrics.render(),
                             media_type="text/plain; version=0.0.4")

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "nutrition",
        "version": app.version,
        # Keep ``endpoints`` as the backward-compatible list of endpoints that
        # can produce a current runtime response.  Future routes are reported
        # separately so an HTTP 501 route is never mistaken for an available
        # Nutrition capability.
        "endpoints": [
            "POST /api/nutrition/analyze",
            "POST /api/nutrition/analyze/by-product-id",
            "POST /api/nutrition/analyze/by-service-id",
            "POST /api/nutrition/safety",
            "POST /api/nutrition/report",
            "POST /api/nutrition/compare",
        ],
        "future_endpoints": [],
    }


@app.post("/api/nutrition/analyze")
def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    """1사료 분석 (FR-AI-1-01~05)

    Returns:
        {pet_id, product_id, category_branch, excluded, exclude_reasons,
         nutrient_score, nutrients_checked, aafco_pass, lifestage_match,
         warnings, consumer_card, details, ingredient_normalized[]}
    """
    return observe_domain_result(_analyze_direct(req))


@app.post("/api/nutrition/analyze/by-product-id")
def analyze_by_product_id(req: PersistedProductAnalyzeRequest) -> dict[str, Any]:
    """Analyze one persisted local product without caller-supplied nutrients.

    This is an operational demonstrator route.  It does not replace the pending
    canonical BE contract at ``/nutrition/analyze``.
    """
    try:
        loaded = load_product_input(req.product_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Persisted product not found") from exc
    try:
        product = ProductIn(**loaded["product"])
    except ValidationError as exc:
        # Raw artifact/schema failures are not server failures and must not
        # receive invented unit, basis, category, or nutrient values.
        return observe_domain_result({
            "product_id": req.product_id,
            "analysis_engine": None,
            "p0_d_applied": False,
            "input_readiness": {"input_readiness": "INSUFFICIENT_DATA", "reason_codes": ["SOURCE_SCHEMA_INVALID"]},
            "nutrition_coverage": {"nutrition_coverage": "UNKNOWN", "reason_codes": ["SOURCE_SCHEMA_INVALID"]},
            "analysis_status": "INSUFFICIENT_DATA",
            "nutrition_comparison_status": "UNKNOWN",
            "aafco_pass": None,
            "source_validation_status": "INVALID_SOURCE_DATA",
            "source_validation_errors": [item["type"] for item in exc.errors()],
            "input_provenance": loaded["provenance"],
        })
    result = _analyze_direct(AnalyzeRequest(pet=req.pet, product=product))
    result["input_provenance"] = loaded["provenance"]
    return observe_domain_result(result)


def _authenticated_member_id(request: Request) -> int:
    if not service_repository.configured():
        raise HTTPException(status_code=503, detail="SERVICE_SOURCE_NOT_CONFIGURED")
    expected = os.getenv("INTERNAL_GATEWAY_SECRET")
    if not expected:
        raise HTTPException(status_code=503, detail="SERVICE_AUTH_NOT_CONFIGURED")
    secrets = request.headers.getlist("X-Internal-Secret")
    members = request.headers.getlist("X-Member-Id")
    if len(secrets) != 1 or not hmac.compare_digest(secrets[0].encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="SERVICE_UNAUTHORIZED")
    if (len(members) != 1 or not members[0].isascii() or not members[0].isdecimal()
            or len(members[0]) > 19 or not 0 < int(members[0]) <= 9223372036854775807):
        raise HTTPException(status_code=401, detail="SERVICE_UNAUTHORIZED")
    return int(members[0])


def _service_pet_profile(pet: dict[str, Any], req: BaseModel) -> dict[str, Any]:
    if "allergy_profile_status" not in req.model_fields_set:
        return pet
    declared = req.allergy_profile_status or "UNKNOWN"
    stored = pet.get("allergy_profile_status", "UNKNOWN")
    if stored in {"KNOWN_NONE", "KNOWN_LIST"} and declared != stored:
        declared = "UNKNOWN"
    return {**pet, "allergy_profile_status": declared}


@app.post("/api/nutrition/analyze/by-service-id")
def analyze_by_service_id(req: ServiceIdAnalyzeRequest, request: Request) -> dict[str, Any]:
    """Gateway 인증과 SQL ownership을 모두 통과한 실제 source만 분석한다."""
    member_id = _authenticated_member_id(request)
    try:
        pet = service_repository.get_pet(req.pet_id, member_id)
        pet = _service_pet_profile(pet, req)
        product = service_repository.get_product(req.product_id)
        return observe_domain_result(analyze_service_records(pet, product))
    except service_repository.ServiceNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except service_repository.ServiceUnavailable:
        raise HTTPException(status_code=503, detail="SERVICE_DB_UNAVAILABLE") from None
    except (ServiceInputError, ValidationError):
        raise HTTPException(status_code=422, detail="SERVICE_SOURCE_INVALID") from None


@app.post("/api/nutrition/safety")
def safety(req: SafetyRequest) -> dict[str, Any]:
    """안전 7원칙 P0 자동 검증 (Phase 1 stub)"""
    card = req.consumer_card
    if not card:
        result = observe_domain_result(_analyze_product(AnalyzeRequest(pet=req.pet, product=req.product)))
        card = result.get("consumer_card")

    check = safety_check_consumer_card(card)
    return {
        "pet_id": req.pet.id,
        "product_id": req.product.id,
        "consumer_card_preview": card[:200] if card else None,
        **check,
    }


@app.post("/api/nutrition/report")
def report(req: AnalyzeRequest) -> dict[str, Any]:
    """리포트 생성 (FR-AI-1-06 + 안전 검증 결합)"""
    result = observe_domain_result(_analyze_product(req))
    card = result.get("consumer_card", "")

    safety = safety_check_consumer_card(card)

    return {
        "pet_id": req.pet.id,
        "product_id": req.product.id,
        "category": result.get("category_branch"),
        "excluded": result.get("excluded", False),
        "consumer_card": card,
        "safety_check": safety,
        "generated_at": _now_iso(),
    }


@app.post("/api/nutrition/compare")
def compare(req: ServiceCompareRequest, request: Request) -> dict[str, Any]:
    member_id = _authenticated_member_id(request)
    try:
        pet = service_repository.get_pet(req.pet_id, member_id)
        pet = _service_pet_profile(pet, req)
        products = [service_repository.get_product(pid) for pid in req.product_ids]
        analyses = [observe_domain_result(analyze_service_records(pet, product)) for product in products]
        return compare_service_analyses(req.pet_id, req.product_ids, analyses)
    except service_repository.ServiceNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except service_repository.ServiceUnavailable:
        raise HTTPException(status_code=503, detail="SERVICE_DB_UNAVAILABLE") from None
    except (ServiceInputError, ValidationError):
        raise HTTPException(status_code=422, detail="SERVICE_SOURCE_INVALID") from None


def _now_iso() -> str:
    from datetime import datetime, timezone, timedelta
    kst = timezone(timedelta(hours=9))
    return datetime.now(kst).isoformat()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
