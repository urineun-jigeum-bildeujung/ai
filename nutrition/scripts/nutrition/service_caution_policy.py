"""Backend CautionIngredientCode snapshot; only TOXIC has a P0 action policy.

Source: urineun-jigeum-bildeujung/sever at the revision below, file
modules/common-core/src/main/java/com/golajugaenyang/common/core/domain/CautionIngredientCode.java.
Species are the Backend Species enum literals, never an invented BOTH value.
CONDITIONAL/NUTRITION are preserved for parity, not automatic blocking rules.
"""

BACKEND_REVISION = "eb48f6c20ccd132f436113cc59861f45dcd885b1"
BACKEND_ENUM_SHA256 = "8baea45d54231c0398d716193d4460fcb1044cb64c5c86a49749000f64bfa411"
POLICY_SOURCE = "BACKEND_CautionIngredientCode"
EVIDENCE_SOURCE = "public.product_cautions.caution_code"

# code -> (cautionLevel, applicableSpecies); exact 1:1 Backend enum parity.
CAUTION_POLICY = {
    "XYLITOL": ("TOXIC", ("DOG",)),
    "CHOCOLATE_CACAO": ("TOXIC", ("DOG", "CAT")),
    "GRAPE_RAISIN": ("TOXIC", ("DOG", "CAT")),
    "ONION": ("TOXIC", ("DOG", "CAT")),
    "GARLIC": ("TOXIC", ("DOG", "CAT")),
    "ALLIUM": ("TOXIC", ("DOG", "CAT")),
    "MACADAMIA": ("TOXIC", ("DOG",)),
    "ALCOHOL": ("TOXIC", ("DOG", "CAT")),
    "CAFFEINE": ("TOXIC", ("DOG", "CAT")),
    "AVOCADO": ("TOXIC", ("DOG",)),
    "FRUIT_PITS": ("TOXIC", ("DOG",)),
    "NUTMEG_SPICE": ("TOXIC", ("DOG",)),
    "RAW_YEAST_DOUGH": ("TOXIC", ("DOG", "CAT")),
    "CITRUS": ("TOXIC", ("CAT",)),
    "HIGH_FAT": ("CONDITIONAL", ("DOG", "CAT")),
    "HIGH_SODIUM": ("CONDITIONAL", ("DOG", "CAT")),
    "LACTOSE_DAIRY": ("CONDITIONAL", ("DOG", "CAT")),
    "EXCESS_FISH": ("CONDITIONAL", ("CAT",)),
    "RAW_FISH": ("CONDITIONAL", ("CAT",)),
    "RAW_EGG_WHITE": ("CONDITIONAL", ("CAT",)),
    "EXCESS_LIVER": ("CONDITIONAL", ("CAT",)),
    "DOG_FOOD": ("CONDITIONAL", ("CAT",)),
    "TAURINE_DEFICIENCY": ("NUTRITION", ("CAT",)),
}


def evaluate_cautions(species, codes):
    evaluations = []
    for code in codes:
        level, applicable = CAUTION_POLICY.get(code, (None, ()))
        status = ("POLICY_NOT_DEFINED" if level != "TOXIC" else
                  "BLOCKED" if species.upper() in applicable else "NOT_APPLICABLE")
        evaluations.append({
            "caution_code": code, "caution_level": level,
            "applicable_species": list(applicable), "policy_status": status,
            "evidence_source": EVIDENCE_SOURCE, "policy_source": POLICY_SOURCE,
            "policy_version": BACKEND_REVISION,
        })
    return evaluations
