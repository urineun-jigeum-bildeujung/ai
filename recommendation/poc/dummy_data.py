# -*- coding: utf-8 -*-
"""
POC용 더미 데이터.
실제 스키마(pet_profile, product_master, reviews)의 필드명·타입을 그대로 따른다.
목데이터/합성데이터가 들어오면 이 파일만 교체하면 파이프라인은 그대로 동작해야 한다.
"""

# -----------------------------
# pet_profile (온보딩 정보)
# -----------------------------
PET_PROFILES = [
    {
        "pet_id": "pet_001",
        "user_id": "user_001",
        "species": "DOG",
        "breed": "말티즈",
        "birth_date": "2022-03-01",   # 월령 계산용
        "weight": 3.2,
        "allergy_codes": ["chicken"],
        "bcs": 3,
        "neutered": True,
    },
    {
        "pet_id": "pet_002",
        "user_id": "user_002",
        "species": "CAT",
        "breed": "코리안숏헤어",
        "birth_date": "2019-06-15",
        "weight": 4.5,
        "allergy_codes": [],
        "bcs": 4,
        "neutered": True,
    },
    {
        "pet_id": "pet_003",
        "user_id": "user_003",
        "species": "DOG",
        "breed": "골든리트리버",
        "birth_date": "2024-01-10",
        "weight": 18.0,
        "allergy_codes": ["beef", "corn"],
        "bcs": 3,
        "neutered": False,
    },
]

# -----------------------------
# product_master (상품)
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

# -----------------------------
# reviews (리뷰 원본) — 별점 있는 것만 사용 (전처리 규칙 반영)
# -----------------------------
REVIEWS = [
    {
        "review_id": "rev_001",
        "product_id": "prod_001",
        "rating": 5,
        "review_text": "말티즈가 정말 잘 먹어요. 대변 상태도 좋아졌고 털에 윤기가 나요.",
    },
    {
        "review_id": "rev_002",
        "product_id": "prod_001",
        "rating": 2,
        "review_text": "며칠 먹다가 안 먹어요. 가격도 비싼 편이에요.",
    },
    {
        "review_id": "rev_003",
        "product_id": "prod_002",
        "rating": 4,
        "review_text": "대형견인데 잘 먹고 활력이 넘쳐요. 가성비도 좋아요.",
    },
    {
        "review_id": "rev_004",
        "product_id": "prod_002",
        "rating": 1,
        "review_text": "먹고 나서 설사했어요. 가려워하기도 하고 저희 아이한텐 안 맞아요.",
    },
    {
        "review_id": "rev_005",
        "product_id": "prod_003",
        "rating": 5,
        "review_text": "노령견 관절에 도움되는 것 같고 이상 반응 없이 잘 먹어요.",
    },
    {
        "review_id": "rev_006",
        "product_id": "prod_004",
        "rating": 5,
        "review_text": "저알러지 사료라 예민한 아이인데 잘 맞아요. 피부도 좋아졌어요.",
    },
    {
        "review_id": "rev_007",
        "product_id": "prod_004",
        "rating": 3,
        "review_text": "그냥 무난해요. 가격은 비싸다고 느껴져요.",
    },
    {
        "review_id": "rev_008",
        "product_id": "prod_005",
        "rating": 5,
        "review_text": "고양이가 그릇을 싹 비워요. 기호성 최고예요.",
    },
]