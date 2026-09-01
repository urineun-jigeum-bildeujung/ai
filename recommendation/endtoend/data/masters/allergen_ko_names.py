# -*- coding: utf-8 -*-
"""
알러지 코드 -> 한국어 표시명 매핑.
allergen_master.py의 ALLERGEN_MASTER는 영문 코드만 가지고 있어서,
리뷰 텍스트에 자연어로 삽입하려면 한국어 이름이 필요해 별도로 관리한다.
원본 확정 리스트(한국어) 기준으로 작성.
"""

ALLERGEN_KO_NAMES = {
    # 육류
    "chicken": "닭", "duck": "오리", "beef": "소", "lamb": "양", "pork": "돼지",
    "turkey": "칠면조", "quail": "메추라기", "rabbit": "토끼", "venison": "사슴",
    "kangaroo": "캥거루", "goat_meat": "염소", "ostrich": "타조",
    # 해산물
    "cod": "대구", "salmon": "연어", "tuna": "참치", "seabass": "농어",
    "mackerel": "고등어", "flounder": "가자미", "anchovy": "멸치", "sardine": "정어리",
    "herring": "청어", "halibut": "광어", "trout": "송어", "eel": "장어",
    "shrimp": "새우", "crab": "게", "lobster": "바닷가재", "octopus": "문어",
    "squid": "오징어", "clam": "조개", "mussel": "홍합", "oyster": "굴",
    "scallop": "가리비", "kelp": "다시마",
    # 유제품류 및 난류
    "egg_yolk": "계란 노른자", "egg_white": "계란 흰자", "quail_egg": "메추리알",
    "milk": "우유", "goat_milk": "산양유", "cheddar_cheese": "체다치즈",
    "yogurt": "요거트", "butter": "버터", "casein": "카제인",
    "beta_lactoglobulin": "베타락토글로불린",
    # 곡류 및 콩류
    "corn": "옥수수", "wheat": "밀", "rice": "쌀", "barley": "보리", "oat": "귀리",
    "quinoa": "퀴노아", "mung_bean": "녹두", "buckwheat": "메밀", "rye": "호밀",
    "soybean": "대두", "kidney_bean": "강낭콩", "pea": "완두콩", "lentil": "렌틸콩",
    "green_bean": "깍지콩",
    # 과일류
    "apple": "사과", "banana": "바나나", "strawberry": "딸기", "blueberry": "블루베리",
    "cranberry": "크랜베리", "watermelon": "수박", "pear": "배", "mango": "망고",
    "pineapple": "파인애플", "grapefruit": "자몽", "lemon": "레몬", "coconut": "코코넛",
    "orange": "오렌지", "kiwi": "키위", "peach": "복숭아", "plum": "자두",
    "melon": "멜론", "olive": "올리브",
    # 채소류
    "carrot": "당근", "sweet_potato": "고구마", "potato": "감자", "pumpkin": "호박",
    "zucchini": "주키니호박", "broccoli": "브로콜리", "cauliflower": "콜리플라워",
    "spinach": "시금치", "beet": "비트", "bell_pepper": "파프리카", "cucumber": "오이",
    "tomato": "토마토", "lettuce": "상추", "napa_cabbage": "배추", "cabbage": "양배추",
    "kale": "케일", "parsley": "파슬리", "chicory": "치커리", "celery": "샐러리",
    "radish": "무", "chili_pepper": "고추", "eggplant": "가지", "mushroom": "버섯",
    "aloe_vera": "알로에 베라", "ginger": "생강",
    # 견과류 및 종실류
    "peanut": "땅콩", "chestnut": "밤", "walnut": "호두", "almond": "아몬드",
    "pine_nut": "잣", "pistachio": "피스타치오", "cashew": "캐슈넛",
    "hazelnut": "헤이즐넛", "brazil_nut": "브라질넛", "macadamia": "마카다미아",
    "sesame": "참깨", "sunflower_seed": "해바라기씨", "flaxseed": "아마씨",
    # 효모류 및 기타
    "honey": "꿀", "gluten": "글루텐", "brewers_yeast": "맥주 효모",
    "baking_yeast": "빵 효모", "sugar": "설탕", "cinnamon": "계피",
}