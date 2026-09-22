import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";
import { fileURLToPath } from "node:url";

const path = fileURLToPath(new URL("../../data/eval/allergen_human_review_p2_v2.xlsx", import.meta.url));
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(path));
const result = await workbook.inspect({ kind: "table", range: "검토 입력!A4:I64", include: "values", tableMaxRows: 61, tableMaxCols: 9 });
console.log(result.ndjson);
