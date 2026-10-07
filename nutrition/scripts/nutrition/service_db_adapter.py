"""조회 완료된 Service source를 Canonical Input으로 변환한다. SQL/인증/저장은 하지 않는다.

target_species, ingredient_codes, allergen_flags, caution_codes는 관계 테이블을 조회한 호출자가
명시적으로 전달하는 배열이다. 이 집계 인터페이스를 실제 DB column으로 주장하지 않는다.
"""
from __future__ import annotations

import json
import math
from calendar import monthrange
from collections import Counter, defaultdict
from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from allergen_service import DICTIONARY, V3_PATH, evaluate_safety
from allergen_repository import get_refs
from allergen_catalog_versions import DICTIONARY_VERSION, PIPELINE_VERSION
from gtin_validation import is_valid_gtin
from product_input_adapter import RAW, _canonical_gtin_from_product_id, load_product_input
from mock_integration_fixture import build_mock_fixture
from service_caution_policy import evaluate_cautions, EVIDENCE_SOURCE, CAUTION_POLICY


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


def _today():
    return datetime.now(ZoneInfo("Asia/Seoul")).date()


def _birth_age(value, today):
    # PostgreSQL DATE and exact ISO dates only; no timestamps or date guessing.
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            raise ServiceInputError("PET_BIRTH_DATE_INVALID") from None
        if parsed.isoformat() != value:
            raise ServiceInputError("PET_BIRTH_DATE_INVALID")
        value = parsed
    if type(value) is not date or value > today:
        raise ServiceInputError("PET_BIRTH_DATE_INVALID")
    months = (today.year - value.year) * 12 + today.month - value.month
    if today.day < min(value.day, monthrange(today.year, today.month)[1]):
        months -= 1
    stage = "GROWTH_REPRODUCTION" if months < 12 else "ADULT_MAINTENANCE"
    return months / 12, stage


def _target_enum(value, allowed, reason):
    if value is not None and (not isinstance(value, str) or value not in allowed):
        raise ServiceInputError(reason)
    return value


def adapt_pet(source):
    species = {"DOG": "dog", "CAT": "cat"}.get(source.get("species"))
    if species is None:
        raise ServiceInputError("PET_SPECIES_UNSUPPORTED")
    codes = _codes(source.get("allergies"))
    declared = source.get("allergy_profile_status")
    consistent = ((declared == "KNOWN_NONE" and not codes)
                  or (declared == "KNOWN_LIST" and bool(codes)))
    profile = declared if consistent else "UNKNOWN"
    allergies = []
    for code in codes:
        for canonical in service_allergen_codes(code):
            if canonical not in allergies:
                allergies.append(canonical)
    birth_date = source.get("birth_date")
    if birth_date is not None:
        age, stage = _birth_age(birth_date, _today())
    else:
        age = _number(source.get("age"), positive=False)
        stage = source.get("life_stage") or "UNKNOWN"
    return {
        "id": _identifier(source.get("id")), "species": species,
        "age_years": age,
        "weight_kg": _number(source.get("weight"), positive=True),
        "allergies": allergies, "allergy_profile_status": profile,
        # Missing birth_date keeps the existing explicit-stage/fail-close policy.
        # Birth date establishes age only, never pregnancy or lactation.
        "life_stage": stage,
        "life_stage_detail": None if birth_date is not None else source.get("life_stage_detail"),
        "service_allergy_codes": codes,
        "service_target_breed_size": _target_enum(source.get("target_breed_size"),
            {"SMALL", "MEDIUM", "LARGE"}, "PET_TARGET_BREED_SIZE_INVALID"),
        "product_target_stage": "GROWTH" if age < 1 else "ADULT" if age < 7 else "SENIOR",
        "product_target_stage_policy_version": "pet_product_target_stage_v1",
    }


# Source enum spellings whose meaning is explicitly present in dictionary v3.
# Group codes expand to every supported constituent; unsupported enums remain
# namespaced evidence so the existing profile gate fails closed.
SERVICE_ALLERGEN_ALIASES = {
    **{code: (code.casefold(),) for code in (
        "CHICKEN", "BEEF", "PORK", "LAMB", "FISH", "DAIRY",
        "EGG", "WHEY", "CORN", "RICE", "SOY", "POTATO", "YEAST", "TAPIOCA",
    )},
    "CHEESE": ("치즈",),
    "WHEAT_GLUTEN": ("wheat",),
    "OAT_BARLEY": ("oat", "barley"),
    "SWEET_POTATO": ("sweet potato",),
    "CRUSTACEAN": ("crustacean",),
}


# Specific service enums require a same-specific canonical entry in v3.
# The dictionary's ingredient alias "salmon" -> "fish" is not such evidence.
SPECIFIC_FISH_CODES = {
    "SALMON", "TUNA", "BONITO", "ANCHOVY", "MACKEREL", "HERRING", "SARDINE", "WHITEFISH",
}


def service_allergen_codes(code):
    from allergen_service import _norm

    if code in CAUTION_POLICY:
        return ["SERVICE_CODE:" + code]
    if code in SPECIFIC_FISH_CODES:
        canonical = code.casefold()
        entry = DICTIONARY["entries"].get(canonical)
        if entry is not None and entry["allergen_code"] == canonical:
            return [canonical]
        return ["SERVICE_CODE:" + code]
    values = SERVICE_ALLERGEN_ALIASES.get(code)
    if values is None:
        return ["SERVICE_CODE:" + code]
    mapped = [DICTIONARY["aliases"].get(_norm(value)) for value in values]
    if any(value is None or value not in DICTIONARY["entries"] for value in mapped):
        return ["SERVICE_CODE:" + code]
    return sorted(set(mapped))


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
    from allergen_service import _norm

    raw = json.loads(V3_PATH.read_text(encoding="utf-8"))
    index = defaultdict(set)
    for item in raw["items"]:
        for code in item.get("ingredient_codes", []):
            index[code].add(item["allergen_code"])
    refs = []
    for code in codes:
        matches = index.get(code, set())
        if code in CAUTION_POLICY:
            matches = set()
        elif code in SPECIFIC_FISH_CODES:
            specific = service_allergen_codes(code)
            matches = set(specific) if specific[0] in DICTIONARY["entries"] else set()
        normalized = code
        method = "STRUCTURED_SOURCE"
        if not matches and code not in SPECIFIC_FISH_CODES and code not in CAUTION_POLICY:
            normalized = _norm(code)
            canonical = DICTIONARY["aliases"].get(normalized)
            entry = DICTIONARY["entries"].get(canonical)
            if (normalized not in DICTIONARY["conflicts"] and canonical is not None
                    and entry is not None and entry["allergen_code"] == canonical):
                matches = {canonical}
                method = "CANONICAL_ALIAS"
        resolved = len(matches) == 1
        refs.append({"product_id": product_id, "allergen_code": next(iter(matches)) if resolved else None,
                     "raw_text": code, "normalized_text": normalized, "matched_text": normalized if resolved else None,
                     "mapping_method": method if resolved else "UNRESOLVED",
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
    caution_codes = _codes(source.get("caution_codes"))
    mock_fixture = build_mock_fixture(source)
    bridge = (mock_fixture["identifier"] if mock_fixture is not None
              else bridge_sku(source.get("sku"), local_identity_index() if index is None else index))
    product = {
        "id": product_id, "service_sku": source.get("sku"), "name": source.get("product_name") or product_id,
        "category": category, "target_species": species,
        # 서비스 타깃 연령은 AAFCO label evidence가 아니다.
        "aafco_life_stage": None,
        "service_target_age_group": _target_enum(source.get("target_age_group"),
            {"GROWTH", "ADULT", "SENIOR"}, "PRODUCT_TARGET_AGE_GROUP_INVALID"),
        "service_target_breed_size": _target_enum(source.get("target_breed_size"),
            {"SMALL", "MEDIUM", "LARGE"}, "PRODUCT_TARGET_BREED_SIZE_INVALID"),
        "feeding_target": source.get("feeding_target"),
        "feeding_method": source.get("feeding_method"),
        "product_form": source.get("subcategory_code"),
        "ingredient_list": [], "nutrition_items": [],
        "ingredient_source": "SERVICE_INGREDIENT_CODE", "ingredient_source_version": "SERVICE_INPUT",
        "service_ingredient_codes": ingredient_codes, "service_allergen_flags": flags,
        "service_caution_codes": caution_codes,
        "product_allergen_refs": structured_refs(product_id, ingredient_codes),
        "_evidence_trace": "STRUCTURED_SOURCE",
    }
    provenance = {"identity_bridge": bridge, "nutrition_source": None,
                  "operating_identity_verified": False,
                  "integration_identity_verified": False,
                  "service_target_age_group": source.get("target_age_group"),
                  "aafco_life_stage_evidence_status": "UNKNOWN"}
    if caution_codes:
        provenance["service_cautions"] = {"codes": caution_codes, "evidence_source": EVIDENCE_SOURCE}
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


def _evaluate_service_allergen_safety(pet, product):
    """기존 gate 실행 후 같은 service namespace의 명시적 충돌만 추가 차단한다."""
    result = evaluate_safety(pet, product)
    allergy_flags = set(product["service_allergen_flags"]) - set(CAUTION_POLICY)
    conflicts = sorted(set(pet["service_allergy_codes"]) & allergy_flags)
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


def evaluate_service_safety(pet, product):
    """Preserve the existing allergen gate, then independently apply TOXIC."""
    result = _evaluate_service_allergen_safety(pet, product)
    codes = product.get("service_caution_codes", [])
    if not codes:
        return result
    evaluations = evaluate_cautions(pet["species"], codes)
    toxic = [item for item in evaluations if item["policy_status"] == "BLOCKED"]
    result = {**result, "caution_evaluations": evaluations, "conflicting_toxic_ingredients": toxic}
    if not toxic:
        return result
    message = "해당 반려동물 종에 독성으로 정의된 주의 원료가 등록되어 이 상품을 제외했습니다."
    return {**result, "safety_status": "SAFETY_BLOCKED", "excluded": True,
            "safety_reason_codes": list(dict.fromkeys([*result["safety_reason_codes"], "TOXIC_INGREDIENT"])),
            "exclude_reasons": sorted(set(result["exclude_reasons"]) | {"TOXIC_INGREDIENT"}),
            "warnings": list(dict.fromkeys([*result["warnings"], message])),
            "safety_message": message}
