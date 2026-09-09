# -*- coding: utf-8 -*-
"""
더미 리뷰 데이터 + reviewer_pet 프로필 + 정형 aspect 평점(1~3점).

[변경 이력]
리뷰 작성 화면에서 사용자가 5개 aspect(기호성/소화·배변/피부·모질/체중·활력/알러지반응)를
1~3점으로 직접 선택하는 정형 입력 방식으로 확정되어, 텍스트 기반 aspect 추출 대신
이 필드들을 직접 채워둔다. review_text는 이제 추천 로직에서 사용하지 않고
참고용(사용자에게 그대로 노출)으로만 남겨둔다.
평가하지 않은 항목은 None으로 둔다 (중립 3점과 구분하기 위함).

필드명은 rating_converter.ASPECT_FIELD_TO_CODE 기준:
  palatability_rating, digestion_rating, skin_coat_rating,
  vitality_weight_rating, allergic_reaction_rating (전부 1~3 또는 None)
"""

DUMMY_REVIEWS = [
    {
        "review_id": "rev_001",
        "product_id": "prod_001",
        "rating": 4,
        "review_text": "고양이가 소화 잘 시켜요. 이 정도면 합리적이에요.",
        "palatability_rating": 2, "digestion_rating": 3, "skin_coat_rating": None,
        "vitality_weight_rating": None, "allergic_reaction_rating": None,
        "reviewer_pet": {"species": "CAT", "birth_date": "2020-05-01", "weight": 4.0, "allergy_codes": []},
    },
    {
        "review_id": "rev_002",
        "product_id": "prod_001",
        "rating": 4,
        "review_text": "노령견인데 잘 먹어요.",
        "palatability_rating": 3, "digestion_rating": None, "skin_coat_rating": None,
        "vitality_weight_rating": None, "allergic_reaction_rating": None,
        "reviewer_pet": {"species": "DOG", "birth_date": "2014-03-01", "weight": 18.0, "allergy_codes": []},
    },
    {
        "review_id": "rev_003",
        "product_id": "prod_001",
        "rating": 4,
        "review_text": "대형견인데 알러지 반응 없었어요.",
        "palatability_rating": None, "digestion_rating": None, "skin_coat_rating": None,
        "vitality_weight_rating": None, "allergic_reaction_rating": 3,
        "reviewer_pet": {"species": "DOG", "birth_date": "2021-01-01", "weight": 32.0, "allergy_codes": []},
    },
    {
        "review_id": "rev_004",
        "product_id": "prod_001",
        "rating": 4,
        "review_text": "저희 냥이는 밥 시간을 기다려요. 변 냄새 줄었어요.",
        "palatability_rating": 3, "digestion_rating": 3, "skin_coat_rating": None,
        "vitality_weight_rating": None, "allergic_reaction_rating": None,
        "reviewer_pet": {"species": "CAT", "birth_date": "2022-08-01", "weight": 3.5, "allergy_codes": ["chicken"]},
    },
    {
        "review_id": "rev_005",
        "product_id": "prod_001",
        "rating": 5,
        "review_text": "소형견이라 밥 시간을 기다려요. 가성비 좋아요.",
        "palatability_rating": 3, "digestion_rating": None, "skin_coat_rating": None,
        "vitality_weight_rating": None, "allergic_reaction_rating": None,
        "reviewer_pet": {"species": "DOG", "birth_date": "2023-02-01", "weight": 3.2, "allergy_codes": ["chicken"]},
    },
    {
        "review_id": "rev_006",
        "product_id": "prod_001",
        "rating": 2,
        "review_text": "고양이가 가성비가 아쉬워요.",
        "palatability_rating": 2, "digestion_rating": None, "skin_coat_rating": None,
        "vitality_weight_rating": None, "allergic_reaction_rating": None,
        "reviewer_pet": {"species": "CAT", "birth_date": "2019-01-01", "weight": 5.0, "allergy_codes": []},
    },
    {
        "review_id": "rev_007",
        "product_id": "prod_001",
        "rating": 4,
        "review_text": "노령견인데 가격 대비 만족해요. 이상 반응 없이 잘 먹어요.",
        "palatability_rating": 3, "digestion_rating": None, "skin_coat_rating": None,
        "vitality_weight_rating": None, "allergic_reaction_rating": 3,
        "reviewer_pet": {"species": "DOG", "birth_date": "2013-06-01", "weight": 20.0, "allergy_codes": []},
    },
    {
        "review_id": "rev_008",
        "product_id": "prod_001",
        "rating": 4,
        "review_text": "노령견인데 털에 윤기가 나요.",
        "palatability_rating": None, "digestion_rating": None, "skin_coat_rating": 3,
        "vitality_weight_rating": None, "allergic_reaction_rating": None,
        "reviewer_pet": {"species": "DOG", "birth_date": "2015-09-01", "weight": 17.0, "allergy_codes": []},
    },
    {
        "review_id": "rev_009",
        "product_id": "prod_001",
        "rating": 4,
        "review_text": "소형견이라 가성비 좋아요. 가려워하지 않아요.",
        "palatability_rating": None, "digestion_rating": None, "skin_coat_rating": 3,
        "vitality_weight_rating": None, "allergic_reaction_rating": None,
        "reviewer_pet": {"species": "DOG", "birth_date": "2023-04-01", "weight": 3.0, "allergy_codes": ["chicken"]},
    },
    {
        "review_id": "rev_010",
        "product_id": "prod_001",
        "rating": 3,
        "review_text": "저희 강아지는 예민한 아이한텐 안 맞아요.",
        "palatability_rating": None, "digestion_rating": None, "skin_coat_rating": None,
        "vitality_weight_rating": None, "allergic_reaction_rating": 1,
        "reviewer_pet": {"species": "DOG", "birth_date": "2022-01-01", "weight": 6.0, "allergy_codes": ["beef"]},
    },
]