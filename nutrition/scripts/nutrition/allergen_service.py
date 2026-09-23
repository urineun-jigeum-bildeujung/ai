"""Canonical allergen domain service (P0).

The v3 dictionary remains the persisted source of truth.  Older dictionaries are
compatibility inputs only; callers must use this module for profile, ingredient,
and safety decisions.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
V3_PATH = ROOT / "data" / "raw" / "seed_11_allergen_ingredient_map_v3.json"
SAFE_METHODS = {"CANONICAL_EXACT", "CANONICAL_ALIAS", "STRUCTURED_SOURCE"}


def _norm(value: object) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"\s+", " ", value).strip()


def load_dictionary() -> dict[str, Any]:
    """Load v3 and expose one normalized contract plus explicit alias conflicts."""
    raw = json.loads(V3_PATH.read_text(encoding="utf-8"))
    aliases: dict[str, str] = {}
    conflicts: dict[str, set[str]] = {}
    entries: dict[str, dict[str, Any]] = {}
    for item in raw["items"]:
        code = item["allergen_code"]
        values = [code, item.get("allergen_name_ko", ""), *item.get("matched_keywords", [])]
        entries[code] = {
            "allergen_code": code,
            "canonical_name": item.get("allergen_name_ko", code),
            "aliases": sorted({_norm(v) for v in values if v}),
            "language": "multilingual",
            "source": "; ".join(raw.get("_source", {}).get("primary", "").split(" + ")),
            "source_version": raw.get("version", "3.0"),
            "status": "ACTIVE",
        }
    for code, entry in entries.items():
        for alias in entry["aliases"]:
            previous = aliases.get(alias)
            if previous and previous != code:
                conflicts.setdefault(alias, {previous}).add(code)
            else:
                aliases[alias] = code
    for alias in conflicts:
        aliases.pop(alias, None)  # never pick a conflicting alias automatically
    return {"version": "allergen_sot_v1", "entries": entries, "aliases": aliases,
            "conflicts": {k: sorted(v) for k, v in conflicts.items()}}


DICTIONARY = load_dictionary()


def canonicalize_profile(allergies: list[str], profile_status: str) -> dict[str, Any]:
    """Canonicalize only declared aliases; no fuzzy matching is performed."""
    if profile_status == "UNKNOWN":
        return {"status": "UNKNOWN", "codes": [], "unresolved": []}
    if profile_status == "KNOWN_NONE":
        return {"status": "KNOWN_NONE", "codes": [], "unresolved": []}
    codes, unresolved = [], []
    for raw in allergies:
        code = DICTIONARY["aliases"].get(_norm(raw))
        if code is None:
            unresolved.append(raw)
        elif code not in codes:
            codes.append(code)
    return {"status": "KNOWN_LIST", "codes": codes, "unresolved": unresolved}


def product_allergen_refs(product_id: str, ingredients: list[str], *, source: str = "PRODUCT_LABEL",
                          source_version: str = "API_REQUEST") -> list[dict[str, Any]]:
    """Boundary-aware phrase matching.  OCR/fuzzy guesses are deliberately absent."""
    refs: list[dict[str, Any]] = []
    for raw in ingredients:
        normalized = _norm(raw)
        # Alias must be a whole Unicode word/phrase; this is not substring matching.
        matched = False
        for alias, code in DICTIONARY["aliases"].items():
            if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", normalized):
                refs.append({"product_id": product_id, "allergen_code": code, "raw_text": raw,
                             "normalized_text": normalized, "matched_text": alias,
                             "mapping_method": "CANONICAL_EXACT" if normalized == alias else "CANONICAL_ALIAS",
                             "confidence": 1.0, "source": source, "source_version": source_version,
                             "dictionary_version": DICTIONARY["version"]})
                matched = True
        # Turkey is not an active v3 allergen entry. Preserve it as unresolved evidence,
        # never as a direct safety clearance/block decision.
        if re.search(r"(?<!\w)dinde(?!\w)", normalized):
            refs.append({"product_id": product_id, "allergen_code": "turkey", "raw_text": raw,
                         "normalized_text": normalized, "matched_text": "dinde",
                         "mapping_method": "UNRESOLVED", "confidence": 0.0, "source": source,
                         "source_version": source_version, "dictionary_version": DICTIONARY["version"]})
        if not matched and not any(r["raw_text"] == raw and r["mapping_method"] == "UNRESOLVED" for r in refs):
            refs.append({"product_id": product_id, "allergen_code": None, "raw_text": raw,
                         "normalized_text": normalized, "matched_text": None, "mapping_method": "UNRESOLVED",
                         "confidence": 0.0, "source": source, "source_version": source_version,
                         "dictionary_version": DICTIONARY["version"]})
    return refs


def _pet_stage(pet: dict[str, Any]) -> str | None:
    stage = (pet.get("life_stage") or "").casefold()
    if stage in {"puppy", "kitten", "growth", "growth_reproduction"}: return "GROWTH_REPRODUCTION"
    if stage in {"adult", "maintenance", "adult_maintenance"}: return "ADULT_MAINTENANCE"
    # An explicitly supplied but unsupported stage (including senior, until its
    # reference policy is defined) must not silently become adult maintenance.
    if stage: return None
    age = pet.get("age_years")
    if isinstance(age, (int, float)) and age >= 0:
        return "GROWTH_REPRODUCTION" if age < 1 else "ADULT_MAINTENANCE"
    return None


def _product_stage(value: str | None) -> str | None:
    value = (value or "").upper()
    if value in {"ADULT", "MAINTENANCE", "ADULT_MAINTENANCE"}: return "ADULT_MAINTENANCE"
    if value in {"GROWTH", "GROWTH_REPRODUCTION"}: return "GROWTH_REPRODUCTION"
    if value == "ALL_LIFE_STAGES": return value
    return None


def evaluate_safety(pet: dict[str, Any], product: dict[str, Any]) -> dict[str, Any]:
    """Shared fail-closed safety result for API, direct matchers, and batch callers."""
    category = product.get("category", "food")
    # Allergy profiles belong to the pet.  Product-provided values are
    # untrusted product metadata and must not override the user's profile.
    raw_allergies = pet.get("allergies") or []
    profile_status = pet.get("allergy_profile_status") or ("KNOWN_LIST" if raw_allergies else "KNOWN_NONE")
    profile = canonicalize_profile(raw_allergies, profile_status)
    if profile["status"] == "UNKNOWN" or profile["unresolved"]:
        return {"safety_status": "SAFETY_DATA_INSUFFICIENT", "allergy_check_status": "INSUFFICIENT_DATA",
                "excluded": True, "exclude_reasons": ["SAFETY_DATA_INSUFFICIENT"], "product_allergen_refs": [],
                "warnings": ["알레르기 프로필을 확인할 수 없습니다."], "life_stage_status": "UNKNOWN"}
    if category == "food":
        target = product.get("target_species")
        if not target:
            return {"safety_status": "SAFETY_DATA_INSUFFICIENT", "allergy_check_status": "NOT_EVALUATED",
                    "excluded": True, "exclude_reasons": ["PRODUCT_SPECIES_UNKNOWN"], "product_allergen_refs": [],
                    "warnings": ["상품 대상 종 정보가 없습니다."], "life_stage_status": "UNKNOWN"}
        if target != "both" and target != pet.get("species"):
            return {"safety_status": "SAFETY_BLOCKED", "allergy_check_status": "NOT_EVALUATED",
                    "excluded": True, "exclude_reasons": ["SPECIES_MISMATCH"], "product_allergen_refs": [],
                    "warnings": ["상품 대상 종과 반려동물 종이 다릅니다."], "life_stage_status": "NOT_APPLICABLE"}
        product_stage = _product_stage(product.get("aafco_life_stage"))
        pet_stage = _pet_stage(pet)
        if pet_stage is None:
            return {"safety_status": "SAFETY_DATA_INSUFFICIENT", "allergy_check_status": "NOT_EVALUATED",
                    "excluded": True, "exclude_reasons": ["PET_LIFE_STAGE_UNSUPPORTED"], "product_allergen_refs": [],
                    "warnings": ["반려동물 생애주기 기준을 확인할 수 없습니다."], "life_stage_status": "UNKNOWN"}
        if product_stage is None:
            return {"safety_status": "SAFETY_DATA_INSUFFICIENT", "allergy_check_status": "NOT_EVALUATED",
                    "excluded": True, "exclude_reasons": ["PRODUCT_LIFE_STAGE_UNKNOWN"], "product_allergen_refs": [],
                    "warnings": ["상품 생애주기 정보가 없습니다."], "life_stage_status": "UNKNOWN"}
        if product_stage != "ALL_LIFE_STAGES" and product_stage != pet_stage:
            return {"safety_status": "SAFETY_BLOCKED", "allergy_check_status": "NOT_EVALUATED",
                    "excluded": True, "exclude_reasons": ["LIFE_STAGE_MISMATCH"], "product_allergen_refs": [],
                    "warnings": ["상품 생애주기와 반려동물 생애주기가 다릅니다."], "life_stage_status": "LIFE_STAGE_MISMATCH"}
    refs = product.get("product_allergen_refs")
    evidence_trace = product.get("_evidence_trace") or ("PRECOMPUTED" if refs is not None else "RUNTIME_FALLBACK")
    if refs is None:
        refs = product_allergen_refs(product.get("id", "unknown"), product.get("ingredient_list", []),
                                     source=product.get("ingredient_source", "PRODUCT_LABEL"),
                                     source_version=product.get("ingredient_source_version", "API_REQUEST"))
    else:
        # Batch schema preserves raw_ingredient_text; normalize it to the domain
        # field used by the shared gate without changing the persisted evidence.
        refs = [{**r, "raw_text": r.get("raw_text", r.get("raw_ingredient_text", ""))} for r in refs]
    if profile["status"] == "KNOWN_NONE":
        return {"safety_status": "NOT_APPLICABLE", "allergy_check_status": "NOT_APPLICABLE", "evidence_trace": evidence_trace, "excluded": False,
                "exclude_reasons": [], "warnings": [], "product_allergen_refs": refs, "unmapped_ingredients": [],
                "life_stage_status": "MATCHED" if category == "food" else "NOT_APPLICABLE"}
    unresolved = [r for r in refs if r["mapping_method"] not in SAFE_METHODS]
    if not refs:
        unresolved = [{"raw_text": "INGREDIENT_LIST_MISSING", "mapping_method": "UNRESOLVED"}]
    if unresolved:
        return {"safety_status": "SAFETY_DATA_INSUFFICIENT", "allergy_check_status": "INSUFFICIENT_DATA", "evidence_trace": evidence_trace,
                "excluded": True, "exclude_reasons": ["SAFETY_DATA_INSUFFICIENT"], "product_allergen_refs": refs,
                "unmapped_ingredients": sorted({r["raw_text"] for r in unresolved}),
                "warnings": ["안전 판정 불가 원료가 있습니다."], "life_stage_status": "MATCHED" if category == "food" else "NOT_APPLICABLE"}
    conflicts = [r for r in refs if r["allergen_code"] in profile["codes"]]
    if conflicts:
        return {"safety_status": "SAFETY_BLOCKED", "allergy_check_status": "CONFLICT", "evidence_trace": evidence_trace, "excluded": True,
                "exclude_reasons": sorted({r["allergen_code"] for r in conflicts}), "warnings": ["등록 알레르기와 충돌합니다."],
                "product_allergen_refs": refs, "unmapped_ingredients": [], "life_stage_status": "MATCHED" if category == "food" else "NOT_APPLICABLE"}
    return {"safety_status": "NO_CONFLICT_DETECTED", "allergy_check_status": "NO_CONFLICT_DETECTED", "evidence_trace": evidence_trace, "excluded": False,
            "exclude_reasons": [], "warnings": [], "product_allergen_refs": refs, "unmapped_ingredients": [],
            "life_stage_status": "MATCHED" if category == "food" else "NOT_APPLICABLE"}
