"""Create a Korean reading aid for the frozen v2 human audit without changing labels."""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/eval/allergen_mapping_precision_audit_p2_v2.csv"
OUT = ROOT / "data/eval/allergen_mapping_precision_audit_p2_v2_review_ko.csv"
QUESTION = "이 원재료 원문이 allergen_code에 표시된 알레르겐을 실제로 의미하는가?"
TRANSLATIONS = {
 1:"닭고기와 칠면조의 건조 단백질", 2:"우유 및 우유 부산물", 3:"닭고기와 칠면조의 건조 단백질", 4:"옥수수 글루텐 가루", 5:"옥수수 글루텐 박", 6:"옥수수 글루텐", 7:"옥수수 글루텐",
 8:"[번역 불확실] 금잔화 추출물·루테인·첨가물·신선한 뼈 없는 양고기 등이 섞인 손상/비정형 문구", 9:"건조 닭고기(4%)", 10:"건조 닭 간(4%)", 11:"기름 및 지방(생선기름 1%)", 12:"효모", 13:"소고기", 14:"닭 육수", 15:"닭 지방", 16:"닭 육수", 17:"연어유(DHA 공급원)", 18:"닭고기 가루 13%", 19:"닭 육수", 20:"신선한 닭 내장(간", 21:"고구마", 22:"고구마", 23:"천연 닭고기 향", 24:"가금류 단백질(12%", 25:"천연 닭고기 향", 26:"닭고기 가루 23%", 27:"건조 닭 연골", 28:"신선한 닭고기(24%)", 29:"가금류 단백질(10%", 30:"육류(심장 기반 닭고기 15%", 31:"건조 돼지고기(4%)", 32:"건조 닭고기(4%)", 33:"닭고기 기름 및 지방", 34:"건조 닭 간(4%)", 35:"건조 닭 간(4%)", 36:"심장 기반 닭고기 40%", 37:"소고기", 38:"오트밀", 39:"소고기", 40:"닭고기", 41:"구성: 건조 닭고기와 칠면조(31%)", 42:"밀 글루텐* 쌀, 옥수수 글루텐", 43:"옥수수 가루", 44:"곡물(압출 옥수수 가루 60%", 45:"옥수수 글루텐 가루", 46:"달걀 및 닭고기 향", 47:"밀 글루텐", 48:"옥수수 글루텐", 49:"글루텐, 식물성 섬유", 50:"돼지 간 향", 51:"닭 간 향", 52:"닭 간 향", 53:"옥수수 글루텐 박", 54:"옥수수 글루텐 박", 55:"옥수수", 56:"당류. 닭고기와 완두콩 맛. 육류 및 동물성 부산물(닭고기 4% 포함)", 57:"생선 및 생선 제품(연어 4%)", 58:"[번역 불확실] 제품 내 닭고기 최소 4% 및 재수화 채소 관련 이탈리아어 문구가 잘려 있음", 59:"한 입과 전체 제품에 소고기 최소 4%. 첨가물 및 영양 첨가물 문구", 60:"연어",
}

def main():
    with SOURCE.open(encoding="utf-8", newline="") as handle: rows = list(csv.DictReader(handle))
    fields = ["row_number", "product_id", "source_dataset", "raw_ingredient_text", "review_translation_ko", "segmented_text", "matched_text", "matched_alias", "allergen_code", "review_question_ko", "review_label", "reviewer_note", "mapping_method", "confidence", "dictionary_version"]
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for number, row in enumerate(rows, 1):
            writer.writerow({"row_number": number, "product_id": row["product_id"], "source_dataset": row["source_dataset"], "raw_ingredient_text": row["raw_ingredient_text"], "review_translation_ko": TRANSLATIONS[number], "segmented_text": row["segmented_text"], "matched_text": row["matched_text"], "matched_alias": row["matched_alias"], "allergen_code": row["allergen_code"], "review_question_ko": QUESTION, "review_label": "", "reviewer_note": "", "mapping_method": row["mapping_method"], "confidence": row["confidence"], "dictionary_version": row["dictionary_version"]})
    print({"rows": len(rows), "translated": sum(not text.startswith("[번역 불확실]") for text in TRANSLATIONS.values()), "uncertain": sum(text.startswith("[번역 불확실]") for text in TRANSLATIONS.values())})

if __name__ == "__main__": main()
