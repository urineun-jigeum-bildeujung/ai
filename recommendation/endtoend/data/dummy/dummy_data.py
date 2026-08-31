# -*- coding: utf-8 -*-
"""
더미 pet_profile / product_master.
확정 스키마 반영: sex(MALE/FEMALE) + neutered(boolean) 분리, concerns(관심 건강정보) 포함.
실제 목데이터/합성데이터가 오면 이 파일만 교체하면 나머지 파이프라인은 그대로 동작해야 한다.
"""

# -----------------------------
# pet_profile
# -----------------------------
PET_PROFILES = [
    {
        "pet_id": "pet_001",
        "user_id": "user_001",
        "species": "DOG",
        "breed": "말티즈",
        "birth_date": "2022-03-01",
        "sex": "FEMALE",
        "neutered": True,
        "weight": 3.2,
        "bcs": 3,
        "allergy_codes": ["chicken"],
        "concerns": ["tartar", "tear_stain"],
    },
    {
        "pet_id": "pet_002",
        "user_id": "user_002",
        "species": "CAT",
        "breed": "코리안숏헤어",
        "birth_date": "2019-06-15",
        "sex": "MALE",
        "neutered": True,
        "weight": 4.5,
        "bcs": 4,
        "allergy_codes": [],
        "concerns": ["hairball", "obesity"],
    },
    {
        "pet_id": "pet_003",
        "user_id": "user_003",
        "species": "DOG",
        "breed": "골든리트리버",
        "birth_date": "2024-01-10",
        "sex": "MALE",
        "neutered": False,
        "weight": 18.0,
        "bcs": 3,
        "allergy_codes": ["beef", "corn"],
        "concerns": ["reduced_activity"],
    },
]

# -----------------------------
# product_master
# -----------------------------
PRODUCTS = [
    {
        "product_id": "prod_001",
        "product_name": "그레인프리 닭가슴살 사료",
        "brand_name": "브랜드A",
        "category_code": "FOOD",
        "subcategory_code": "DRY_FOOD",
        "ingredients": ["chicken", "sweet_potato", "carrot"],
        "allergen_flags": ["chicken"],
        "target_species": ["DOG", "CAT"],
        "target_breed_size": "SMALL",
        "target_age_group": "ADULT",
        "price": 32000,
    },
    {
        "product_id": "prod_002",
        "product_name": "소고기 성견 사료",
        "brand_name": "브랜드B",
        "category_code": "FOOD",
        "subcategory_code": "DRY_FOOD",
        "ingredients": ["beef", "rice", "corn"],
        "allergen_flags": ["beef", "corn"],
        "target_species": ["DOG"],
        "target_breed_size": "LARGE",
        "target_age_group": "ADULT",
        "price": 45000,
    },
    {
        "product_id": "prod_003",
        "product_name": "연어 관절 영양제",
        "brand_name": "브랜드C",
        "category_code": "SUPPLEMENT",
        "subcategory_code": "CHEWABLE_SUPPLEMENT",
        "ingredients": ["salmon", "green_bean"],
        "allergen_flags": ["salmon"],
        "target_species": ["DOG", "CAT"],
        "target_breed_size": None,
        "target_age_group": "SENIOR",
        "price": 28000,
    },
    {
        "product_id": "prod_004",
        "product_name": "오리고기 저알러지 사료",
        "brand_name": "브랜드A",
        "category_code": "FOOD",
        "subcategory_code": "DRY_FOOD",
        "ingredients": ["duck", "pumpkin", "blueberry"],
        "allergen_flags": ["duck"],
        "target_species": ["DOG", "CAT"],
        "target_breed_size": "MEDIUM",
        "target_age_group": "ADULT",
        "price": 38000,
    },
    {
        "product_id": "prod_005",
        "product_name": "참치 습식 간식",
        "brand_name": "브랜드D",
        "category_code": "TREAT",
        "subcategory_code": "WET_TREAT",
        "ingredients": ["tuna"],
        "allergen_flags": ["tuna"],
        "target_species": ["CAT"],
        "target_breed_size": None,
        "target_age_group": None,
        "price": 3500,
    },
]