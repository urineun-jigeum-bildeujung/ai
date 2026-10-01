"""조회 완료된 Service source를 Canonical Input으로 변환한다. SQL/인증/저장은 하지 않는다.

target_species, ingredient_codes, allergen_flags는 관계 테이블을 조회한 호출자가
명시적으로 전달하는 배열이다. 이 집계 인터페이스를 실제 DB column으로 주장하지 않는다.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from decimal import Decimal

from allergen_service import DICTIONARY, V3_PATH, evaluate_safety
from allergen_repository import get_refs
from allergen_catalog_versions import DICTIONARY_VERSION, PIPELINE_VERSION
from gtin_validation import is_valid_gtin
from product_input_adapter import RAW, _canonical_gtin_from_product_id, load_product_input
from mock_integration_fixture import build_mock_fixture


class ServiceInputError(ValueError):
    """오류 코드는 고정 문자열만 사용하며 원본 개인정보를 포함하지 않는다."""


def _identifier(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ServiceInputError("SERVICE_IDENTIFIER_INVALID")
    text = str(value)
    if not text.isascii() or not text.isdecimal() or int(text) <= 0:
        raise ServiceInputError("SERVICE_IDENTIFIER_INVALID")
    return text


def _codes(value):
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value):
        raise ServiceInputError("SERVICE_CODE_LIST_INVALID")
    return sorted(set(value))


def _number(value, *, positive):
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ServiceInputError("PET_MEASUREMENT_MISSING_OR_INVALID")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise ServiceInputError("PET_MEASUREMENT_MISSING_OR_INVALID")
    return number


def adapt_pet(source):
    species = {"DOG": "dog", "CAT": "cat"}.get(source.get("species"))
    if species is None:
        raise ServiceInputError("PET_SPECIES_UNSUPPORTED")
    codes = _codes(source.get("allergies"))
    declared = source.get("allergy_profile_status")
    consistent = ((declared == "KNOWN_NONE" and not codes)
                  or (declared == "KNOWN_LIST" and bool(codes)))
    profile = declared if consistent else "UNKNOWN"
    # 동일 canonical 코드의 대소문자만 정렬한다. SALMON→fish 등의 alias 확장은 금지.
    allergies = [c.casefold() if c.casefold() in DICTIONARY["entries"] else "SERVICE_CODE:" + c for c in codes]
    return {
        "id": _identifier(source.get("id")), "species": species,
        "age_years": _number(source.get("age"), positive=False),
        "weight_kg": _number(source.get("weight"), positive=True),
        "allergies": allergies, "allergy_profile_status": profile,
        # 생애주기 누락을 age 기반 adult fallback으로 승격시키지 않는다.
        "life_stage": source.get("life_stage") or "UNKNOWN",
        "life_stage_detail": source.get("life_stage_detail"),
        "service_allergy_codes": codes,
    }


def local_identity_index():
    """명시적 source product ID만 인덱싱한다. 이름/브랜드는 조회하지 않는다."""
    index = defaultdict(list)
    for source, filename, key in (
        ("OPFF", "seed_9_placeholder_feed_opff.json", "items"),
        ("OEM", "seed_9b_off_korean_oem.json", "products"),
        ("GLOBAL", "seed_9_global_brands_v2.json", "products"),
    ):
        rows = json.loads((RAW / filename).read_text(encoding="utf-8"))[key]
        for row in rows:
            product_id = row.get("product_id")
            canonical = _canonical_gtin_from_product_id(product_id)
            if canonical:
                index[canonical].append({"product_id": product_id, "source_dataset": source})
    return dict(index)


def bridge_sku(sku, index):
    if not isinstance(sku, str) or not is_valid_gtin(sku):
        return {"status": "INVALID_SKU", "canonical_gtin": None, "matches": []}
    canonical = _canonical_gtin_from_product_id(sku)
    matches = index.get(canonical, [])
    return {"status": "MATCHED" if len(matches) == 1 else "AMBIGUOUS" if matches else "UNMATCHED",
            "canonical_gtin": canonical, "matches": matches}


def identity_coverage(products, index):
    counts = Counter(bridge_sku(p.get("sku"), index)["status"] for p in products)
    return {"products": len(products), "sku_present": sum(bool(p.get("sku")) for p in products),
            "valid_gtin": sum(counts[s] for s in ("MATCHED", "UNMATCHED", "AMBIGUOUS")),
            **{s: counts[s] for s in ("MATCHED", "UNMATCHED", "AMBIGUOUS", "INVALID_SKU")}}


def structured_refs(product_id, codes):
    raw = json.loads(V3_PATH.read_text(encoding="utf-8"))
    index = defaultdict(set)
    for item in raw["items"]:
        for code in item.get("ingredient_codes", []):
            index[code].add(item["allergen_code"])
    refs = []
    for code in codes:
        matches = index.get(code, set())
        resolved = len(matches) == 1
        refs.append({"product_id": product_id, "allergen_code": next(iter(matches)) if resolved else None,
                     "raw_text": code, "normalized_text": code, "matched_text": code if resolved else None,
                     "mapping_method": "STRUCTURED_SOURCE" if resolved else "UNRESOLVED",
                     "source": "SERVICE_INGREDIENT_CODE", "source_version": "SERVICE_INPUT",
                     "dictionary_version": DICTIONARY["version"]})
    return refs


def adapt_product(source, *, index=None):
    category = {"FOOD": "food", "TREAT": "treat", "SUPPLEMENT": "supplement"}.get(source.get("category_code"))
    if category is None:
        raise ServiceInputError("PRODUCT_CATEGORY_MISSING_OR_UNSUPPORTED")
    product_id = _identifier(source.get("id"))
    species_codes = _codes(source.get("target_species"))
    species = {("DOG",): "dog", ("CAT",): "cat", ("CAT", "DOG"): "both"}.get(tuple(species_codes))
    ingredient_codes = _codes(source.get("ingredient_codes"))
    flags = _codes(source.get("allergen_flags"))
    mock_fixture = build_mock_fixture(source)
    bridge = (mock_fixture["identifier"] if mock_fixture is not None
              else bridge_sku(source.get("sku"), local_identity_index() if index is None else index))
    product = {
        "id": product_id, "name": source.get("product_name") or product_id,
        "category": category, "target_species": species,
        # 서비스 타깃 연령은 AAFCO label evidence가 아니다.
        "aafco_life_stage": None,
        "service_target_age_group": source.get("target_age_group"),
        "product_form": source.get("subcategory_code"),
        "ingredient_list": [], "nutrition_items": [],
        "ingredient_source": "SERVICE_INGREDIENT_CODE", "ingredient_source_version": "SERVICE_INPUT",
        "service_ingredient_codes": ingredient_codes, "service_allergen_flags": flags,
        "product_allergen_refs": structured_refs(product_id, ingredient_codes),
        "_evidence_trace": "STRUCTURED_SOURCE",
    }
    provenance = {"identity_bridge": bridge, "nutrition_source": None,
                  "operating_identity_verified": False,
                  "integration_identity_verified": False,
                  "service_target_age_group": source.get("target_age_group"),
                  "aafco_life_stage_evidence_status": "UNKNOWN"}
    if mock_fixture is not None:
        # Integration fixtures are a separate evidence namespace.  They never
        # enter ``load_product_input`` and never masquerade as GTIN evidence.
        product["nutrition_items"] = mock_fixture["nutrition_items"]
        product["aafco_life_stage"] = mock_fixture["aafco_life_stage"]
        provenance["nutrition_source"] = mock_fixture["nutrition_source"]
        provenance["integration_identity_verified"] = True
        provenance["fixture_status"] = mock_fixture["fixture_status"]
        provenance["fixture_profile"] = mock_fixture["fixture_profile"]
        provenance["aafco_life_stage_evidence_status"] = (
            "MOCK_INTEGRATION_FIXTURE" if mock_fixture["aafco_life_stage"] else "UNKNOWN"
        )
        return {"product": product, "provenance": provenance}
    if bridge["status"] == "MATCHED":
        evidence_id = bridge["matches"][0]["product_id"]
        loaded = load_product_input(evidence_id)  # 기존 Gold gate도 이 경로에서 재검증한다.
        local = loaded["product"]
        # 종/분류 충돌은 차단한다. 마케팅 연령과 AAFCO 단계는 비교하지 않는다.
        # exact identity만으로 local stage의 AAFCO 근거까지 검증되지는 않는다.
        conflicts = [key for key in ("category", "target_species")
                     if product.get(key) and local.get(key)
                     and product[key] != local[key]]
        if conflicts:
            provenance["identity_bridge"] = {**bridge, "status": "AMBIGUOUS", "conflicting_fields": conflicts}
        else:
            product["nutrition_items"] = local["nutrition_items"]
            provenance["nutrition_source"] = loaded["provenance"]
            if not ingredient_codes:
                refs, trace = get_refs(evidence_id, DICTIONARY_VERSION, PIPELINE_VERSION)
                product["_evidence_trace"] = trace
                if trace == "PRECOMPUTED":
                    product["product_allergen_refs"] = refs
                elif trace == "NO_EVIDENCE":
                    product.pop("product_allergen_refs")
                    product["ingredient_list"] = local["ingredient_list"]
                    product["ingredient_source"] = local["ingredient_source"]
                    product["ingredient_source_version"] = local["ingredient_source_version"]
    return {"product": product, "provenance": provenance}


def evaluate_service_safety(pet, product):
    """기존 gate 실행 후 같은 service namespace의 명시적 충돌만 추가 차단한다."""
    result = evaluate_safety(pet, product)
    conflicts = sorted(set(pet["service_allergy_codes"]) & set(product["service_allergen_flags"]))
    if not conflicts:
        return result  # 교집합 부재는 안전 근거가 아니다.
    return {**result, "safety_status": "SAFETY_BLOCKED", "allergy_check_status": "CONFLICT",
            "excluded": True, "exclude_reasons": sorted(set(result["exclude_reasons"]) | {"SERVICE_ALLERGEN_CONFLICT"}),
            "safety_reason_codes": list(dict.fromkeys(["ALLERGY_CONFLICT", *result["safety_reason_codes"]])),
            "conflicting_allergens": [*result["conflicting_allergens"], *[
                {"allergen_code": code, "evidence_source": "SERVICE_ALLERGEN_FLAG", "namespace": "SERVICE"}
                for code in conflicts]],
            "warnings": list(dict.fromkeys([*result["warnings"], "서비스에 등록된 알레르겐 코드가 일치합니다."])),
            "safety_message": "등록한 알레르기 코드와 상품의 서비스 알레르겐 코드가 일치하여 제외했습니다."}
