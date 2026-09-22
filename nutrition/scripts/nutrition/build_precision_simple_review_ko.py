"""Create a Korean convenience view; no runtime or human-label data is changed."""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/eval/allergen_mapping_precision_audit_p2_v2_review_ko.csv"
OUT = ROOT / "data/eval/allergen_mapping_precision_audit_p2_v2_simple_review_ko.csv"
DISPLAY = {"chicken":"닭", "dairy":"유제품", "corn":"옥수수", "wheat":"밀", "lamb":"양", "fish":"생선", "yeast":"효모", "beef":"소고기", "potato":"감자", "pork":"돼지고기", "oat":"귀리", "egg":"달걀"}
MATCHED = {"poulet":"닭", "pollo":"닭", "volaille":"가금류", "poultry":"가금류", "maïs":"옥수수", "mais":"옥수수", "bœuf":"소고기", "porc":"돼지고기", "fisk":"생선", "agneau":"양고기", "poisson":"생선", "chicken":"닭", "corn":"옥수수", "egg":"달걀", "gluten":"글루텐", "milk":"우유", "oatmeal":"귀리", "pork":"돼지고기", "salmon":"연어", "sweet potato":"고구마", "yeast":"효모", "beef":"소고기"}

def main():
    with SOURCE.open(encoding="utf-8", newline="") as handle: rows = list(csv.DictReader(handle))
    fields = ["번호", "원재료_한국어_뜻", "매칭_단어_한국어_뜻", "시스템_판정_한국어_뜻", "사람_검토", "검토_메모", "원재료_원문", "매칭_원문", "allergen_code"]
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            matched = row["matched_text"].strip().lower()
            writer.writerow({"번호": row["row_number"], "원재료_한국어_뜻": row["review_translation_ko"], "매칭_단어_한국어_뜻": MATCHED.get(matched, f"[번역 불확실] {row['matched_text']}"), "시스템_판정_한국어_뜻": DISPLAY.get(row["allergen_code"], f"[번역 불확실] {row['allergen_code']}"), "사람_검토": "", "검토_메모": "", "원재료_원문": row["raw_ingredient_text"], "매칭_원문": row["matched_text"], "allergen_code": row["allergen_code"]})
    print({"rows": len(rows), "uncertain_matched": sum(row["matched_text"].strip().lower() not in MATCHED for row in rows)})

if __name__ == "__main__": main()
