# -*- coding: utf-8 -*-
"""
더미 리뷰 데이터 (KcELECTRA 파인튜닝 검증용).

- 별점 4,5 = positive / 1,2,3 = negative (라벨링 가이드라인 반영)
- 실제 성능 검증용이 아니라, 파인튜닝 코드/파이프라인이 정상 동작하는지
  확인하는 목적의 소규모 합성 데이터. 백엔드 합성 리뷰 오면 이 파일만 교체.
- 별점 없는 리뷰는 수집 대상에서 제외한다는 전처리 규칙에 따라,
  이 더미셋도 전부 rating이 채워진 리뷰만 포함한다.
"""

DUMMY_REVIEWS = [
    {
        "review_id": 'rev_001',
        "product_id": 'prod_001',
        "rating": 4,
        "review_text": '고양이가 소화 잘 시켜요. 이 정도면 합리적이에요.',
    },
    {
        "review_id": 'rev_002',
        "product_id": 'prod_001',
        "rating": 4,
        "review_text": '노령견인데 잘 먹어요.',
    },
    {
        "review_id": 'rev_003',
        "product_id": 'prod_001',
        "rating": 4,
        "review_text": '대형견인데 알러지 반응 없었어요.',
    },
    {
        "review_id": 'rev_004',
        "product_id": 'prod_001',
        "rating": 4,
        "review_text": '저희 냥이는 밥 시간을 기다려요. 변 냄새 줄었어요.',
    },
    {
        "review_id": 'rev_005',
        "product_id": 'prod_001',
        "rating": 5,
        "review_text": '소형견이라 밥 시간을 기다려요. 가성비 좋아요.',
    },
    {
        "review_id": 'rev_006',
        "product_id": 'prod_001',
        "rating": 2,
        "review_text": '고양이가 가성비가 아쉬워요.',
    },
    {
        "review_id": 'rev_007',
        "product_id": 'prod_001',
        "rating": 4,
        "review_text": '노령견인데 가격 대비 만족해요. 이상 반응 없이 잘 먹어요.',
    },
    {
        "review_id": 'rev_008',
        "product_id": 'prod_001',
        "rating": 4,
        "review_text": '노령견인데 털에 윤기가 나요.',
    },
    {
        "review_id": 'rev_009',
        "product_id": 'prod_001',
        "rating": 4,
        "review_text": '소형견이라 가성비 좋아요. 가려워하지 않아요.',
    },
    {
        "review_id": 'rev_010',
        "product_id": 'prod_001',
        "rating": 3,
        "review_text": '저희 강아지는 예민한 아이한텐 안 맞아요.',
    },
    {
        "review_id": 'rev_011',
        "product_id": 'prod_002',
        "rating": 4,
        "review_text": '말티즈인데 체중 관리에 도움돼요. 가려워하지 않아요.',
    },
    {
        "review_id": 'rev_012',
        "product_id": 'prod_002',
        "rating": 1,
        "review_text": '소형견이라 살이 쪘어요. 털이 푸석해요.',
    },
    {
        "review_id": 'rev_013',
        "product_id": 'prod_002',
        "rating": 1,
        "review_text": '저희 강아지는 가격이 비싸요. 기운이 없어 보여요.',
    },
    {
        "review_id": 'rev_014',
        "product_id": 'prod_002',
        "rating": 2,
        "review_text": '대형견인데 가성비가 아쉬워요. 예민한 아이한텐 안 맞아요.',
    },
    {
        "review_id": 'rev_015',
        "product_id": 'prod_002',
        "rating": 4,
        "review_text": '고양이가 예민한 아이인데 잘 맞아요.',
    },
    {
        "review_id": 'rev_016',
        "product_id": 'prod_002',
        "rating": 3,
        "review_text": '대형견인데 가격이 비싸요.',
    },
    {
        "review_id": 'rev_017',
        "product_id": 'prod_002',
        "rating": 4,
        "review_text": '말티즈인데 이상 반응 없이 잘 먹어요. 적정 체중 유지돼요.',
    },
    {
        "review_id": 'rev_018',
        "product_id": 'prod_002',
        "rating": 1,
        "review_text": '대형견인데 가격이 비싸요. 피부 트러블 생겼어요.',
    },
    {
        "review_id": 'rev_019',
        "product_id": 'prod_002',
        "rating": 4,
        "review_text": '저희 냥이는 이상 반응 없이 잘 먹어요. 대변 상태 좋아졌어요.',
    },
    {
        "review_id": 'rev_020',
        "product_id": 'prod_002',
        "rating": 5,
        "review_text": '말티즈인데 이상 반응 없이 잘 먹어요.',
    },
    {
        "review_id": 'rev_021',
        "product_id": 'prod_003',
        "rating": 4,
        "review_text": '고양이가 털에 윤기가 나요. 소화 잘 시켜요.',
    },
    {
        "review_id": 'rev_022',
        "product_id": 'prod_003',
        "rating": 3,
        "review_text": '저희 강아지는 알러지 반응 있었어요.',
    },
    {
        "review_id": 'rev_023',
        "product_id": 'prod_003',
        "rating": 4,
        "review_text": '노령견인데 털빠짐이 줄었어요.',
    },
    {
        "review_id": 'rev_024',
        "product_id": 'prod_003',
        "rating": 1,
        "review_text": '고양이가 가성비가 아쉬워요. 털빠짐이 심해졌어요.',
    },
    {
        "review_id": 'rev_025',
        "product_id": 'prod_003',
        "rating": 4,
        "review_text": '노령견인데 기호성 좋아요. 이상 반응 없이 잘 먹어요.',
    },
    {
        "review_id": 'rev_026',
        "product_id": 'prod_003',
        "rating": 5,
        "review_text": '말티즈인데 대변 상태 좋아졌어요.',
    },
    {
        "review_id": 'rev_027',
        "product_id": 'prod_003',
        "rating": 4,
        "review_text": '노령견인데 털빠짐이 줄었어요.',
    },
    {
        "review_id": 'rev_028',
        "product_id": 'prod_003',
        "rating": 1,
        "review_text": '고양이가 속이 안 좋아 보여요. 살이 너무 빠졌어요.',
    },
    {
        "review_id": 'rev_029',
        "product_id": 'prod_003',
        "rating": 3,
        "review_text": '말티즈인데 무기력해 보여요. 이 가격이면 다른 걸 사겠어요.',
    },
    {
        "review_id": 'rev_030',
        "product_id": 'prod_004',
        "rating": 1,
        "review_text": '소형견이라 입맛에 안 맞나봐요.',
    },
    {
        "review_id": 'rev_031',
        "product_id": 'prod_004',
        "rating": 5,
        "review_text": '저희 강아지는 예민한 아이인데 잘 맞아요.',
    },
    {
        "review_id": 'rev_032',
        "product_id": 'prod_004',
        "rating": 4,
        "review_text": '말티즈인데 배변량 적당해요. 잘 먹어요.',
    },
    {
        "review_id": 'rev_033',
        "product_id": 'prod_004',
        "rating": 2,
        "review_text": '속이 안 좋아 보여요.',
    },
    {
        "review_id": 'rev_034',
        "product_id": 'prod_004',
        "rating": 5,
        "review_text": '대형견인데 밥 시간을 기다려요. 대변 상태 좋아졌어요.',
    },
    {
        "review_id": 'rev_035',
        "product_id": 'prod_004',
        "rating": 2,
        "review_text": '노령견인데 털빠짐이 심해졌어요. 살이 너무 빠졌어요.',
    },
    {
        "review_id": 'rev_036',
        "product_id": 'prod_004',
        "rating": 5,
        "review_text": '말티즈인데 이상 반응 없이 잘 먹어요.',
    },
    {
        "review_id": 'rev_037',
        "product_id": 'prod_004',
        "rating": 4,
        "review_text": '고양이가 알러지 반응 없었어요. 가성비 좋아요.',
    },
    {
        "review_id": 'rev_038',
        "product_id": 'prod_004',
        "rating": 4,
        "review_text": '대형견인데 가성비 좋아요.',
    },
    {
        "review_id": 'rev_039',
        "product_id": 'prod_004',
        "rating": 3,
        "review_text": '말티즈인데 예민한 아이한텐 안 맞아요.',
    },
    {
        "review_id": 'rev_040',
        "product_id": 'prod_005',
        "rating": 5,
        "review_text": '저희 냥이는 가려워하지 않아요. 소화 잘 시켜요.',
    },
    {
        "review_id": 'rev_041',
        "product_id": 'prod_005',
        "rating": 4,
        "review_text": '활력이 넘쳐요. 털에 윤기가 나요.',
    },
    {
        "review_id": 'rev_042',
        "product_id": 'prod_005',
        "rating": 5,
        "review_text": '저희 강아지는 예민한 아이인데 잘 맞아요.',
    },
    {
        "review_id": 'rev_043',
        "product_id": 'prod_005',
        "rating": 2,
        "review_text": '저희 강아지는 피부 트러블 생겼어요.',
    },
    {
        "review_id": 'rev_044',
        "product_id": 'prod_005',
        "rating": 3,
        "review_text": '저희 냥이는 알러지 반응 있었어요. 이 가격이면 다른 걸 사겠어요.',
    },
    {
        "review_id": 'rev_045',
        "product_id": 'prod_005',
        "rating": 4,
        "review_text": '고양이가 털에 윤기가 나요.',
    },
    {
        "review_id": 'rev_046',
        "product_id": 'prod_005',
        "rating": 5,
        "review_text": '노령견인데 가격 대비 만족해요.',
    },
    {
        "review_id": 'rev_047',
        "product_id": 'prod_005',
        "rating": 4,
        "review_text": '말티즈인데 밥 시간을 기다려요. 가격 대비 만족해요.',
    },
]