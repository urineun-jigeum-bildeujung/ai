"""Service enum and ingredient evidence mapping, including fail-close gaps."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/nutrition")]
import service_db_adapter as adapter


@pytest.mark.parametrize("code,expected", [
    ("CHICKEN", ["chicken"]), ("SALMON", ["SERVICE_CODE:SALMON"]), ("TUNA", ["SERVICE_CODE:TUNA"]),
    ("BONITO", ["SERVICE_CODE:BONITO"]), ("ANCHOVY", ["SERVICE_CODE:ANCHOVY"]),
    ("CHEESE", ["dairy"]), ("WHEY", ["dairy"]), ("CRUSTACEAN", ["shellfish"]),
    ("WHEAT_GLUTEN", ["wheat"]), ("OAT_BARLEY", ["barley", "oat"]),
    ("SWEET_POTATO", ["potato"]), ("TAPIOCA", ["potato"]),
    ("DUCK", ["SERVICE_CODE:DUCK"]), ("OTHER", ["SERVICE_CODE:OTHER"]),
    ("ONION", ["SERVICE_CODE:ONION"]),
    ("poultry", ["SERVICE_CODE:poultry"]),
    ("CHICKEN ", ["SERVICE_CODE:CHICKEN "]),
])
def test_service_enum_mapping(code, expected):
    pet = adapter.adapt_pet({"id": 1, "species": "DOG", "age": 3, "weight": 10,
        "allergies": [code], "allergy_profile_status": "KNOWN_LIST", "life_stage": "adult"})
    assert pet["allergies"] == expected
    assert pet["service_allergy_codes"] == [code]


def test_korean_aliases_resolve_without_guessing_other_ingredients():
    refs = adapter.structured_refs("1", ["닭고기", "연어", "쌀", "계란", "오리고기", "당근", "칠면조"])
    assert [r["allergen_code"] for r in refs] == ["chicken", "fish", "rice", "egg", None, None, None]
    assert [r["mapping_method"] for r in refs] == ["CANONICAL_ALIAS"] * 4 + ["UNRESOLVED"] * 3
    assert [r["raw_text"] for r in refs] == ["닭고기", "연어", "쌀", "계란", "오리고기", "당근", "칠면조"]


@pytest.mark.parametrize("ingredient", ["닭고기", "닭가슴살", "chicken"])
def test_mapped_ingredient_still_blocks_declared_allergy(ingredient):
    pet = adapter.adapt_pet({"id": 1, "species": "DOG", "age": 3, "weight": 10,
        "allergies": ["CHICKEN"], "allergy_profile_status": "KNOWN_LIST", "life_stage": "adult"})
    product = {"id": "2", "category": "food", "target_species": "dog", "aafco_life_stage": "ADULT",
        "ingredient_list": [], "service_allergen_flags": [],
        "product_allergen_refs": adapter.structured_refs("2", [ingredient])}
    result = adapter.evaluate_service_safety(pet, product)
    assert result["excluded"] and result["safety_status"] == "SAFETY_BLOCKED"


def test_unknown_ingredient_is_not_cleared_after_alias_mapping():
    pet = adapter.adapt_pet({"id": 1, "species": "DOG", "age": 3, "weight": 10,
        "allergies": ["CHICKEN"], "allergy_profile_status": "KNOWN_LIST", "life_stage": "adult"})
    product = {"id": "2", "category": "food", "target_species": "dog", "aafco_life_stage": "ADULT",
        "ingredient_list": [], "service_allergen_flags": [],
        "product_allergen_refs": adapter.structured_refs("2", ["당근"])}
    result = adapter.evaluate_service_safety(pet, product)
    assert result["excluded"] and "UNMAPPED_INGREDIENT" in result["safety_reason_codes"]
