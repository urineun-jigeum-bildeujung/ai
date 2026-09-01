# -*- coding: utf-8 -*-
"""
관심 건강정보 코드 -> 한국어 표시명 매핑.
concern_master.py의 CONCERN_MASTER는 영문 코드만 가지고 있어서,
리뷰 텍스트에 자연어로 삽입하려면 한국어 이름이 필요해 별도로 관리한다.
원본 확정 리스트(한국어) 기준으로 작성.
"""

CONCERN_KO_NAMES = {
    # 피부·모질
    "itching": "가려움", "flaking": "각질", "hair_loss": "탈모",
    "redness_rash": "붉은기/발진", "dull_coat": "광택 저하", "dandruff": "비듬",
    "hot_spot": "핫스팟", "hyperpigmentation": "색소 침착",
    "seborrheic_dermatitis": "지루성 피부염", "fungal_dermatitis": "곰팡이성 피부염",
    "acne": "여드름",
    # 눈
    "tear_stain": "눈물자국", "eye_discharge": "눈곱", "redness_eye": "충혈",
    "conjunctivitis": "결막염", "keratitis": "각막염", "cataract": "백내장",
    "glaucoma": "녹내장", "cherry_eye": "체리아이",
    # 귀
    "ear_wax": "귀지", "ear_odor": "귀 냄새", "ear_itching": "귀 가려움",
    "otitis_externa": "외이염", "ear_mites": "귀 진드기",
    # 구강·치아
    "tartar": "치석", "plaque": "치태", "gum_disease": "잇몸질환",
    "bad_breath": "구취", "tooth_looseness_fracture": "치아 흔들림/파절",
    "stomatitis": "구내염",
    # 호흡기
    "cough": "기침", "sneezing": "재채기", "nasal_congestion": "코막힘",
    "runny_nose": "콧물", "kennel_cough": "켄넬코프", "asthma": "천식",
    "brachycephalic_syndrome": "단두종 호흡기증후군",
    # 관절·이동성
    "arthritis": "관절염", "gait_abnormality_limping": "보행 이상/절음",
    "patellar_luxation": "슬개골 탈구", "hip_dysplasia": "고관절 이형성",
    "reduced_activity": "활동량 감소", "stiffness": "뻣뻣함",
    "age_related_muscle_weakness": "노령성 근력저하",
    # 소화기
    "diarrhea": "설사", "vomiting": "구토", "constipation": "변비",
    "bloating": "가스참", "loss_of_appetite": "식욕부진", "hairball": "헤어볼",
    "pancreatitis": "췌장염", "enteritis": "장염", "food_intolerance": "식이불내성",
    # 비뇨기·신장
    "frequent_urination": "빈뇨/잦은 배뇨", "dysuria": "배뇨곤란", "hematuria": "혈뇨",
    "urinary_stones": "요로결석", "cystitis": "방광염",
    "chronic_kidney_disease": "만성신부전", "water_intake_change": "음수량 변화",
    # 체중·대사
    "obesity": "비만", "underweight": "저체중", "rapid_weight_change": "급격한 체중변화",
    "diabetes": "당뇨", "thyroid_dysfunction": "갑상선기능이상",
    "appetite_change": "식욕 변화",
    # 알레르기·민감성
    "food_allergy": "식이알레르기", "atopic_dermatitis": "아토피성 피부염",
    "seasonal_allergy": "계절성 알레르기", "contact_allergy": "접촉성 알레르기",
    "flea_allergy": "벼룩 알레르기",
    # 심장
    "cardiac_cough": "심장성 기침", "dyspnea": "호흡곤란",
    "exercise_intolerance": "운동 불내성", "heart_murmur": "심잡음", "syncope": "실신",
    # 면역·감염
    "frequent_infection": "잦은 감염", "delayed_wound_healing": "상처 회복 지연",
    "weakened_immunity": "면역력 저하", "parasites": "기생충(내·외부)",
    # 스트레스·행동
    "separation_anxiety": "분리불안", "increased_barking_howling": "짖음/하울링 증가",
    "over_grooming": "과잉그루밍", "aggression_change": "공격성 변화",
    "stress_defecation": "배변 실수(스트레스성)", "withdrawal_appetite_loss": "은둔/식욕저하",
    # 노령 케어
    "cognitive_dysfunction": "인지기능장애", "decreased_activity": "활동성 감소",
    "muscle_loss": "근육량 감소", "joint_organ_comprehensive_care": "관절·장기 복합관리",
    "hearing_vision_decline": "청력/시력 저하",
    # 임신·번식
    "estrus": "발정", "pregnancy": "임신", "mastitis": "유선염",
    "pyometra": "자궁축농증",
}