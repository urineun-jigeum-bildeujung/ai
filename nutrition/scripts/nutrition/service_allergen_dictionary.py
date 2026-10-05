"""Exact Service enum identities; v3 clinical/label evidence remains unchanged."""
from copy import deepcopy

from allergen_service import DICTIONARY, _norm

SOURCE_REVISION = "c23e7a1ab1b1a31ff32a84e19a50eab7c28bf5c9"
SOURCE_PATH = "modules/common-core/src/main/java/com/golajugaenyang/common/core/domain/AllergenCode.java"
VERSION = "service_allergen_identity_v1"
SERVICE_TOXIC_CODES = {"CHOCOLATE", "GRAPE_RAISIN", "ONION", "GARLIC"}

# Identity only: null allergen references in this source are not safety clearance.
INGREDIENT_IDENTITIES = {"당근": "CARROT", "비트": "BEET"}
INGREDIENT_IDENTITY_SOURCE = "seed_15_ingredient_synonym_v2.json"
INGREDIENT_IDENTITY_SOURCE_SHA256 = "1b87866d1b6369bb93bc2c3e2bce70a30507fe8572142955201e2393de993393"

# These are exact Backend display names and enum spellings, not cross-reactivity.
SERVICE_IDENTITIES = {
    "DUCK": ("duck", "오리고기"), "TURKEY": ("turkey", "칠면조"),
    "RABBIT": ("rabbit", "토끼고기"), "VENISON": ("venison", "사슴고기"),
    "GOAT": ("goat", "염소고기"), "INSECT": ("insect", "곤충 단백질"),
    "KANGAROO": ("kangaroo", "캥거루"),
    "SALMON": ("salmon", "연어"), "TUNA": ("tuna", "참치"),
    "ANCHOVY": ("anchovy", "멸치·앤초비"), "BONITO": ("bonito", "가다랑어"),
    "LENTIL": ("lentil", "렌틸콩"), "PEA": ("pea", "완두콩"),
    "CHICKPEA": ("chickpea", "병아리콩"),
}


def service_dictionary():
    dictionary = deepcopy(DICTIONARY)
    dictionary["version"] = VERSION
    for code, (canonical, display_name) in SERVICE_IDENTITIES.items():
        dictionary["entries"][canonical] = {
            "allergen_code": canonical, "canonical_name": display_name,
            "source": "BACKEND_AllergenCode", "source_version": SOURCE_REVISION,
            "evidence_scope": "SERVICE_ENUM_IDENTITY",
        }
        for alias in (code, canonical, display_name):
            dictionary["aliases"][_norm(alias)] = canonical
    return dictionary
