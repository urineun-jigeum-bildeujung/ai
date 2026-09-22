import fs from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const root = fileURLToPath(new URL("../../", import.meta.url));
const input = `${root}data/eval/allergen_mapping_precision_audit_p2_v2.csv`;
const output = `${root}data/eval/allergen_human_review_p2_v2.xlsx`;

function parseCsv(text) {
  const lines = text.trimEnd().split(/\r?\n/);
  const header = lines.shift().split(",");
  return lines.map((line) => {
    const values = []; let value = "", quoted = false;
    for (let i = 0; i < line.length; i += 1) {
      const char = line[i];
      if (char === '"' && line[i + 1] === '"') { value += '"'; i += 1; }
      else if (char === '"') quoted = !quoted;
      else if (char === "," && !quoted) { values.push(value); value = ""; }
      else value += char;
    }
    values.push(value);
    return Object.fromEntries(header.map((key, index) => [key, values[index] ?? ""]));
  });
}

const rows = parseCsv(await fs.readFile(input, "utf8"));
const workbook = Workbook.create();
const review = workbook.worksheets.add("검토 입력");
const reference = workbook.worksheets.add("참조 정보");
review.showGridLines = false;

review.getRange("A1:I1").merge();
review.getRange("A1").values = [["알레르기 매핑 Human Review 입력표 (P2 v2)"]];
review.getRange("A2:I2").merge();
review.getRange("A2").values = [["판단은 원문과 분할 문맥을 먼저 확인합니다. confidence와 mapping_method는 참조 정보 시트에서 필요할 때만 확인합니다."]];
review.getRange("A4:I4").values = [["row_id", "product_id", "raw_ingredient_text", "segmented_text", "matched_text", "matched_alias", "allergen_code", "review_label", "reviewer_note"]];
// U+200B preserves identifier display as text in spreadsheet engines that infer digits as numbers.
const displayId = (value) => `\u200B${value}`;
const reviewRows = rows.map((r, i) => [i + 1, displayId(r.product_id), r.raw_ingredient_text, r.segmented_text, r.matched_text, r.matched_alias, r.allergen_code, "", ""]);
review.getRange(`B5:B${4 + reviewRows.length}`).setNumberFormat("@");
review.getRange(`A5:I${4 + reviewRows.length}`).values = reviewRows;
review.getRange(`H5:H${4 + reviewRows.length}`).dataValidation = { rule: { type: "list", values: ["CORRECT", "INCORRECT", "AMBIGUOUS"] } };
review.getRange("A1:I1").format = { font: { bold: true, size: 15, color: "#1F1F1F" }, horizontalAlignment: "left" };
review.getRange("A2:I2").format = { font: { italic: true, color: "#555555" }, wrapText: true };
review.getRange("A4:I4").format = { fill: "#1F4E78", font: { bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true };
review.getRange(`A5:I${4 + reviewRows.length}`).format.wrapText = true;
review.getRange(`H5:I${4 + reviewRows.length}`).format.fill = "#FFF2CC";
review.getRange(`A4:I${4 + reviewRows.length}`).format.borders = { preset: "outside", style: "thin", color: "#B7C9D6" };
review.freezePanes.freezeRows(4);
for (const [column, width] of [["A:A", 8], ["B:B", 22], ["C:C", 38], ["D:D", 34], ["E:G", 20], ["H:H", 18], ["I:I", 38]]) review.getRange(column).format.columnWidth = width;

reference.getRange("A1:J1").values = [["row_id", "product_id", "source_dataset", "mapping_method", "confidence", "dictionary_version", "raw_ingredient_text", "matched_text", "matched_alias", "allergen_code"]];
reference.getRange(`B2:B${rows.length + 1}`).setNumberFormat("@");
reference.getRange(`A2:J${rows.length + 1}`).values = rows.map((r, i) => [i + 1, displayId(r.product_id), r.source_dataset, r.mapping_method, Number(r.confidence), r.dictionary_version, r.raw_ingredient_text, r.matched_text, r.matched_alias, r.allergen_code]);
reference.getRange("A1:J1").format = { fill: "#1F4E78", font: { bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", wrapText: true };
reference.getRange(`A2:J${rows.length + 1}`).format.wrapText = true;
reference.freezePanes.freezeRows(1);
for (const [column, width] of [["A:A", 8], ["B:B", 22], ["C:F", 20], ["G:G", 40], ["H:J", 20]]) reference.getRange(column).format.columnWidth = width;

const inspected = await workbook.inspect({ kind: "table", range: "검토 입력!A1:I10", include: "values", tableMaxRows: 10, tableMaxCols: 9 });
if (!inspected.ndjson.includes("raw_ingredient_text")) throw new Error("reviewer view verification failed");
const preview = await workbook.render({ sheetName: "검토 입력", range: "A1:I12", scale: 1.5, format: "png" });
await fs.writeFile("/tmp/allergen_human_review_p2_v2_preview.png", new Uint8Array(await preview.arrayBuffer()));
const file = await SpreadsheetFile.exportXlsx(workbook);
await file.save(output);
