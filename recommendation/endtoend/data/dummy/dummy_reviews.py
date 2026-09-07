# -*- coding: utf-8 -*-
"""
더미 리뷰 데이터 (KcELECTRA 파인튜닝 검증용) + reviewer_pet 프로필 추가.

[변경 사항]
reviews.pet_id가 아직 스키마에 반영되지 않은 상태라, "리뷰 작성자 프로필 유사도" 로직을
미리 개발·검증하기 위해 각 더미 리뷰에 가상의 reviewer_pet(작성자의 pet_profile)을 붙였다.
실제로 reviews.pet_id가 반영되면, 이 필드는 DB 조인 결과로 대체하면 된다
(review_id -> reviews.pet_id -> pet_profile 조회).

- 별점 4,5 = positive / 1,2,3 = negative (라벨링 가이드라인 반영)
- 실제 성능 검증용이 아니라, 파인튜닝 코드/파이프라인이 정상 동작하는지
  확인하는 목적의 소규모 합성 데이터. 백엔드 합성 리뷰 오면 이 파일만 교체.
"""

DUMMY_REVIEWS = [
    {
        "review_id": "rev_001",
        "product_id": "prod_001",
        "rating": 4,
        "review_text": "고양이가 소화 잘 시켜요. 이 정도면 합리적이에요.",
        "reviewer_pet": {"species": "CAT", "birth_date": "2020-05-01", "weight": 4.0, "allergy_codes": []},
    },
    {
        "review_id": "rev_002",
        "product_id": "prod_001",
        "rating": 4,
        "review_text": "노령견인데 잘 먹어요.",
        "reviewer_pet": {"species": "DOG", "birth_date": "2014-03-01", "weight": 18.0, "allergy_codes": []},
    },
    {
        "review_id": "rev_003",
        "product_id": "prod_001",
        "rating": 4,
        "review_text": "대형견인데 알러지 반응 없었어요.",
        "reviewer_pet": {"species": "DOG", "birth_date": "2021-01-01", "weight": 32.0, "allergy_codes": []},
    },
    {
        "review_id": "rev_004",
        "product_id": "prod_001",
        "rating": 4,
        "review_text": "저희 냥이는 밥 시간을 기다려요. 변 냄새 줄었어요.",
        "reviewer_pet": {"species": "CAT", "birth_date": "2022-08-01", "weight": 3.5, "allergy_codes": ["chicken"]},
    },
    {
        "review_id": "rev_005",
        "product_id": "prod_001",
        "rating": 5,
        "review_text": "소형견이라 밥 시간을 기다려요. 가성비 좋아요.",
        "reviewer_pet": {"species": "DOG", "birth_date": "2023-02-01", "weight": 3.2, "allergy_codes": ["chicken"]},
    },
    {
        "review_id": "rev_006",
        "product_id": "prod_001",
        "rating": 2,
        "review_text": "고양이가 가성비가 아쉬워요.",
        "reviewer_pet": {"species": "CAT", "birth_date": "2019-01-01", "weight": 5.0, "allergy_codes": []},
    },
    {
        "review_id": "rev_007",
        "product_id": "prod_001",
        "rating": 4,
        "review_text": "노령견인데 가격 대비 만족해요. 이상 반응 없이 잘 먹어요.",
        "reviewer_pet": {"species": "DOG", "birth_date": "2013-06-01", "weight": 20.0, "allergy_codes": []},
    },
    {
        "review_id": "rev_008",
        "product_id": "prod_001",
        "rating": 4,
        "review_text": "노령견인데 털에 윤기가 나요.",
        "reviewer_pet": {"species": "DOG", "birth_date": "2015-09-01", "weight": 17.0, "allergy_codes": []},
    },
    {
        "review_id": "rev_009",
        "product_id": "prod_001",
        "rating": 4,
        "review_text": "소형견이라 가성비 좋아요. 가려워하지 않아요.",
        "reviewer_pet": {"species": "DOG", "birth_date": "2023-04-01", "weight": 3.0, "allergy_codes": ["chicken"]},
    },
    {
        "review_id": "rev_010",
        "product_id": "prod_001",
        "rating": 3,
        "review_text": "저희 강아지는 예민한 아이한텐 안 맞아요.",
        "reviewer_pet": {"species": "DOG", "birth_date": "2022-01-01", "weight": 6.0, "allergy_codes": ["beef"]},
    },
]