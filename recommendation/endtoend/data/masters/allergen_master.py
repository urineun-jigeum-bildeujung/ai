# -*- coding: utf-8 -*-
"""
알러지 마스터 코드 (전체 120개, 확정된 리스트 기준).
allergen_ingredient_map.allergy_code / pet_profile.allergy_codes / product_master.allergen_flags
전부 이 코드 체계를 공유한다.
"""

ALLERGEN_MASTER = {
    "protein_meat": [
        "chicken", "duck", "beef", "lamb", "pork", "turkey",
        "quail", "rabbit", "venison", "kangaroo", "goat_meat", "ostrich",
    ],
    "protein_seafood": [
        "cod", "salmon", "tuna", "seabass", "mackerel", "flounder",
        "anchovy", "sardine", "herring", "halibut", "trout", "eel",
        "shrimp", "crab", "lobster", "octopus", "squid", "clam",
        "mussel", "oyster", "scallop", "kelp",
    ],
    "dairy_egg": [
        "egg_yolk", "egg_white", "quail_egg", "milk", "goat_milk",
        "cheddar_cheese", "yogurt", "butter", "casein", "beta_lactoglobulin",
    ],
    "grain_legume": [
        "corn", "wheat", "rice", "barley", "oat", "quinoa",
        "mung_bean", "buckwheat", "rye", "soybean", "kidney_bean",
        "pea", "lentil", "green_bean",
    ],
    "fruit": [
        "apple", "banana", "strawberry", "blueberry", "cranberry", "watermelon",
        "pear", "mango", "pineapple", "grapefruit", "lemon", "coconut",
        "orange", "kiwi", "peach", "plum", "melon", "olive",
    ],
    "vegetable": [
        "carrot", "sweet_potato", "potato", "pumpkin", "zucchini", "broccoli",
        "cauliflower", "spinach", "beet", "bell_pepper", "cucumber", "tomato",
        "lettuce", "napa_cabbage", "cabbage", "kale", "parsley", "chicory",
        "celery", "radish", "chili_pepper", "eggplant", "mushroom", "aloe_vera", "ginger",
    ],
    "nut_seed": [
        "peanut", "chestnut", "walnut", "almond", "pine_nut", "pistachio",
        "cashew", "hazelnut", "brazil_nut", "macadamia", "sesame",
        "sunflower_seed", "flaxseed",
    ],
    "other": [
        "honey", "gluten", "brewers_yeast", "baking_yeast", "sugar", "cinnamon",
    ],
}

# 코드만 평탄화한 전체 리스트 (120개) — feature 벡터 순서 고정용
ALLERGEN_VOCAB = [code for codes in ALLERGEN_MASTER.values() for code in codes]