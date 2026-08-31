# -*- coding: utf-8 -*-
"""
관심 건강정보 마스터 코드 (확정된 15개 카테고리 기준).
pet_profile.concerns 필드가 이 코드 체계를 참조한다.
"""

CONCERN_MASTER = {
    "skin_coat": [
        "itching", "flaking", "hair_loss", "redness_rash", "dull_coat",
        "dandruff", "hot_spot", "hyperpigmentation",
        "seborrheic_dermatitis", "fungal_dermatitis", "acne",
    ],
    "eye": [
        "tear_stain", "eye_discharge", "redness_eye", "conjunctivitis",
        "keratitis", "cataract", "glaucoma", "cherry_eye",
    ],
    "ear": [
        "ear_wax", "ear_odor", "ear_itching", "otitis_externa", "ear_mites",
    ],
    "dental": [
        "tartar", "plaque", "gum_disease", "bad_breath",
        "tooth_looseness_fracture", "stomatitis",
    ],
    "respiratory": [
        "cough", "sneezing", "nasal_congestion", "runny_nose",
        "kennel_cough", "asthma", "brachycephalic_syndrome",
    ],
    "joint_mobility": [
        "arthritis", "gait_abnormality_limping", "patellar_luxation",
        "hip_dysplasia", "reduced_activity", "stiffness", "age_related_muscle_weakness",
    ],
    "digestive": [
        "diarrhea", "vomiting", "constipation", "bloating",
        "loss_of_appetite", "hairball", "pancreatitis", "enteritis", "food_intolerance",
    ],
    "urinary_kidney": [
        "frequent_urination", "dysuria", "hematuria", "urinary_stones",
        "cystitis", "chronic_kidney_disease", "water_intake_change",
    ],
    "weight_metabolism": [
        "obesity", "underweight", "rapid_weight_change",
        "diabetes", "thyroid_dysfunction", "appetite_change",
    ],
    "allergy_sensitivity": [
        "food_allergy", "atopic_dermatitis", "seasonal_allergy",
        "contact_allergy", "flea_allergy",
    ],
    "heart": [
        "cardiac_cough", "dyspnea", "exercise_intolerance", "heart_murmur", "syncope",
    ],
    "immune_infection": [
        "frequent_infection", "delayed_wound_healing", "weakened_immunity", "parasites",
    ],
    "stress_behavior": [
        "separation_anxiety", "increased_barking_howling", "over_grooming",
        "aggression_change", "stress_defecation", "withdrawal_appetite_loss",
    ],
    "senior_care": [
        "cognitive_dysfunction", "decreased_activity", "muscle_loss",
        "joint_organ_comprehensive_care", "hearing_vision_decline",
    ],
    "reproduction": [
        "estrus", "pregnancy", "mastitis", "pyometra",
    ],
}

# 코드만 평탄화한 전체 리스트 — feature 벡터 순서 고정용
CONCERN_VOCAB = [code for codes in CONCERN_MASTER.values() for code in codes]