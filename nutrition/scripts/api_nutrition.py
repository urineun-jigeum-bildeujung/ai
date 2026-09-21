"""FastAPI — 골라주개냥 영양성분 분석 API (8/20 시작, 4 endpoint)

- POST /api/nutrition/analyze   1사료 분석 (FR-AI-1-01~05)
- POST /api/nutrition/safety    안전 7원칙 P0 검증
- POST /api/nutrition/report    리포트 생성 (FR-AI-1-06)
- POST /api/nutrition/compare   LLM 비교 (Phase 2, 9월)
- GET  /health                  서비스 상태

실행:
    cd "/Users/aku/Documents/통합 프로젝트(우리는지금빌드중)/ai"
    uvicorn ai.scripts.api_nutrition:app --reload --port 8002

테스트:
    curl -s http://localhost:8002/health
    curl -s -X POST http://localhost:8002/api/nutrition/analyze -H "Content-Type: application/json" -d '{ ... }'
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any


# nutrition/ 패키지 경로
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NUTRITION_DIR = PROJECT_ROOT / "scripts" / "nutrition"
sys.path.insert(0, str(NUTRITION_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import ValidationError  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from match_v1 import load_seed  # type: ignore  # noqa: E402
from match_v1_1_category import match_by_category  # type: ignore  # noqa: E402
from normalizer import extract_allergens_from_text, normalize_ingredient  # type: ignore  # noqa: E402
from allergen_service import evaluate_safety  # type: ignore  # noqa: E402
from allergen_repository import get_refs  # type: ignore  # noqa: E402
from allergen_catalog_versions import DICTIONARY_VERSION, PIPELINE_VERSION  # type: ignore  # noqa: E402
from product_input_adapter import load_product_input  # type: ignore  # noqa: E402
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
    description="사료 성분 분석 + 알레르기 hard filter + NIAS 2024 비교 (8/20 시작)",
)

SEED_FEED_CODES = load_seed("seed_feed_codes.json")


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


def _analyze_product(req: AnalyzeRequest) -> dict[str, Any]:
    """Single API path: food uses P0-D; legacy matching is limited to non-food categories."""
    product = req.product
    safety = _allergy_gate(req.pet, product)

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
        result["nutrition_comparison_status"] = "NOT_APPLICABLE"
        result["aafco_pass"] = None

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

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "nutrition",
        "version": app.version,
        "endpoints": [
            "POST /api/nutrition/analyze",
            "POST /api/nutrition/analyze/by-product-id",
            "POST /api/nutrition/safety",
            "POST /api/nutrition/report",
            "POST /api/nutrition/compare (Phase 2)",
        ],
    }


@app.post("/api/nutrition/analyze")
def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    """1사료 분석 (FR-AI-1-01~05)

    Returns:
        {pet_id, product_id, category_branch, excluded, exclude_reasons,
         nutrient_score, nutrients_checked, aafco_pass, lifestage_match,
         warnings, consumer_card, details, ingredient_normalized[]}
    """
    return _analyze_product(req)


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
        return {
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
        }
    result = _analyze_product(AnalyzeRequest(pet=req.pet, product=product))
    result["input_provenance"] = loaded["provenance"]
    return result


@app.post("/api/nutrition/safety")
def safety(req: SafetyRequest) -> dict[str, Any]:
    """안전 7원칙 P0 자동 검증 (Phase 1 stub)"""
    card = req.consumer_card
    if not card:
        result = _analyze_product(AnalyzeRequest(pet=req.pet, product=req.product))
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
    result = _analyze_product(req)
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
def compare() -> dict[str, Any]:
    """LLM 비교 (Phase 2, 9월 구현 예정)"""
    raise HTTPException(status_code=501, detail="Phase 2 (9월) 구현 예정 — sentence-transformers + GPT-4o-mini")


def _now_iso() -> str:
    from datetime import datetime, timezone, timedelta
    kst = timezone(timedelta(hours=9))
    return datetime.now(kst).isoformat()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
