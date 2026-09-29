"""로컬 pg_dump의 products SKU만 집계한다. DB 접속·파일 생성·원본 행 출력 없음."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from service_db_adapter import identity_coverage, local_identity_index


def product_identifiers(sql):
    rows, columns, found = [], None, False
    for line in sql.splitlines():
        if line.startswith("COPY public.products ("):
            if found:
                raise ValueError("DUPLICATE_PRODUCTS_COPY")
            match = re.fullmatch(r"COPY public\.products \(([^)]+)\) FROM stdin;", line)
            if not match:
                raise ValueError("UNSUPPORTED_PRODUCTS_COPY")
            columns, found = match[1].split(", "), True
            if not {"id", "sku"}.issubset(columns):
                raise ValueError("PRODUCT_ID_SKU_COLUMNS_MISSING")
        elif columns is not None:
            if line == "\\.":
                columns = None
            else:
                values = line.split("\t")
                if len(values) != len(columns):
                    raise ValueError("PRODUCT_COPY_COLUMN_COUNT_INVALID")
                row = dict(zip(columns, values))
                rows.append({"id": row["id"], "sku": None if row["sku"] == r"\N" else row["sku"]})
    if not found or columns is not None:
        raise ValueError("PRODUCTS_COPY_MISSING_OR_INCOMPLETE")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("DUPLICATE_SERVICE_PRODUCT_ID")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--pg-restore", default="pg_restore")
    args = parser.parse_args()
    process = subprocess.run([args.pg_restore, "--data-only", "--table=products", "--file=-", str(args.dump)],
                             capture_output=True, text=True, timeout=30)
    if process.returncode:
        raise SystemExit("LOCAL_DUMP_READ_FAILED")
    products = product_identifiers(process.stdout)
    index = local_identity_index()
    print(json.dumps({"scope": "LOCAL_DUMP_SNAPSHOT_NOT_LIVE_DB",
                      "dump_sha256": hashlib.sha256(args.dump.read_bytes()).hexdigest(),
                      "local_gtin_clusters": len(index),
                      "local_ambiguous_clusters": sum(len(v) > 1 for v in index.values()),
                      **identity_coverage(products, index)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
