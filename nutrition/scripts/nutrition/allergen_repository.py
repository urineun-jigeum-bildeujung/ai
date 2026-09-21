"""SQLite catalog with deterministic evidence lineage and fail-closed reads."""
from __future__ import annotations
import hashlib
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data/processed/allergen_evidence_catalog_p2.db"
TERMINAL_COMPONENT_STATUSES = {"RESOLVED", "PARTIALLY_RESOLVED", "UNRESOLVED", "NON_ALLERGEN", "FILTERED_NON_INGREDIENT", "INVALID_COMPONENT"}
EVIDENCE_IDENTITY_FIELDS = ("component_occurrence_id", "product_id", "allergen_code", "raw_ingredient_text", "normalized_text", "matched_alias", "mapping_method", "ingredient_source", "source_dataset", "source_version", "dictionary_version", "pipeline_version")
REF_FIELDS = ("component_occurrence_id", "product_id", "allergen_code", "raw_ingredient_text", "normalized_text", "segmented_text", "matched_text", "matched_alias", "mapping_method", "confidence", "evidence_status", "usable_for_safety", "ingredient_source", "source_version", "dictionary_version", "pipeline_version", "created_at", "source_row_id", "evidence_scope", "source_dataset")
SCHEMA = """
CREATE TABLE IF NOT EXISTS product_ingredient_components (
 component_occurrence_id TEXT PRIMARY KEY, source_dataset TEXT, source_version TEXT, source_record_id TEXT, product_id TEXT, field_name TEXT, raw_ingredient_text TEXT, component_index INTEGER, raw_component TEXT, normalized_component TEXT, parser_version TEXT, pipeline_version TEXT, processing_status TEXT NOT NULL, processing_reason TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS product_allergen_refs (
 evidence_id TEXT PRIMARY KEY, component_occurrence_id TEXT NOT NULL, product_id TEXT NOT NULL, allergen_code TEXT NOT NULL DEFAULT '', raw_ingredient_text TEXT NOT NULL, normalized_text TEXT NOT NULL, segmented_text TEXT NOT NULL, matched_text TEXT, matched_alias TEXT NOT NULL DEFAULT '', mapping_method TEXT NOT NULL, confidence REAL NOT NULL, evidence_status TEXT NOT NULL, usable_for_safety INTEGER NOT NULL, ingredient_source TEXT NOT NULL, source_version TEXT NOT NULL, dictionary_version TEXT NOT NULL, pipeline_version TEXT NOT NULL, created_at TEXT NOT NULL, source_row_id INTEGER, evidence_scope TEXT, source_dataset TEXT NOT NULL,
 UNIQUE(component_occurrence_id, product_id, allergen_code, raw_ingredient_text, normalized_text, matched_alias, mapping_method, ingredient_source, source_dataset, source_version, dictionary_version, pipeline_version));
CREATE INDEX IF NOT EXISTS idx_allergen_refs_product ON product_allergen_refs(product_id);
CREATE INDEX IF NOT EXISTS idx_allergen_refs_component ON product_allergen_refs(component_occurrence_id);
"""

def _value(row, field):
    value = row.get(field, "")
    return "" if value is None else str(value)

def identity(row):
    """Stable semantic ID; deliberately excludes run timestamps such as created_at."""
    return hashlib.sha256("|".join(_value(row, field) for field in EVIDENCE_IDENTITY_FIELDS).encode()).hexdigest()

def connect(path=DB):
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection

def _ref_values(row):
    values = []
    required_strings = {"component_occurrence_id", "product_id", "allergen_code", "raw_ingredient_text", "normalized_text", "matched_alias", "mapping_method", "ingredient_source", "source_version", "dictionary_version", "pipeline_version", "created_at", "source_dataset"}
    for field in REF_FIELDS:
        value = row.get(field)
        if field == "segmented_text": value = json.dumps(value if value is not None else [], ensure_ascii=False)
        elif field == "usable_for_safety": value = int(bool(value))
        elif field in required_strings: value = "" if value is None else value
        values.append(value)
    return values

def _validate_catalog(components, refs):
    component_ids = set()
    for component in components:
        cid = component.get("component_occurrence_id")
        if not cid or cid in component_ids or component.get("processing_status") not in TERMINAL_COMPONENT_STATUSES:
            raise ValueError("invalid component lineage input")
        component_ids.add(cid)
    for ref in refs:
        if not ref.get("component_occurrence_id") or ref["component_occurrence_id"] not in component_ids:
            raise ValueError("orphan evidence input")
        if any(ref.get(field) is None for field in ("raw_ingredient_text", "normalized_text", "mapping_method", "dictionary_version", "pipeline_version")):
            raise ValueError("missing mandatory evidence input")

def _insert_refs(connection, refs):
    columns = ("evidence_id",) + REF_FIELDS
    updates = ",".join(f"{field}=excluded.{field}" for field in REF_FIELDS)
    sql = f"INSERT INTO product_allergen_refs ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)}) ON CONFLICT(evidence_id) DO UPDATE SET {updates}"
    connection.executemany(sql, [(identity(row), *_ref_values(row)) for row in refs])

def upsert_refs(refs, path=DB):
    """Incremental write: never removes evidence belonging to unrelated products."""
    connection = connect(path)
    try:
        with connection: _insert_refs(connection, refs)
        return connection.execute("select count(*) from product_allergen_refs").fetchone()[0]
    finally: connection.close()

def replace_all_refs(refs, path=DB):
    """Atomic full evidence refresh; use only with a complete reference set."""
    connection = connect(path)
    try:
        with connection:
            connection.execute("delete from product_allergen_refs")
            _insert_refs(connection, refs)
        return connection.execute("select count(*) from product_allergen_refs").fetchone()[0]
    finally: connection.close()

def replace_all_catalog(components, refs, path=DB, reset_schema=False):
    """Atomic fresh rebuild after validation, preserving a prior catalog on failure."""
    _validate_catalog(components, refs)
    connection = connect(path)
    fields = ("component_occurrence_id", "source_dataset", "source_version", "source_record_id", "product_id", "field_name", "raw_ingredient_text", "component_index", "raw_component", "normalized_component", "parser_version", "pipeline_version", "processing_status", "processing_reason", "created_at")
    try:
        with connection:
            if reset_schema:
                connection.execute("drop table if exists product_allergen_refs")
                connection.execute("drop table if exists product_ingredient_components")
                connection.executescript(SCHEMA)
            connection.execute("delete from product_allergen_refs")
            connection.execute("delete from product_ingredient_components")
            connection.executemany(f"insert into product_ingredient_components ({','.join(fields)}) values ({','.join('?' for _ in fields)})", [tuple(row.get(field) for field in fields) for row in components])
            _insert_refs(connection, refs)
        return {"components": len(components), "evidence": connection.execute("select count(*) from product_allergen_refs").fetchone()[0]}
    finally: connection.close()

# Backward-compatible name; it is intentionally incremental, not a full rebuild.
upsert = upsert_refs

def get_refs(product_id, dictionary_version, pipeline_version, path=DB):
    """Return precomputed evidence only when parent lineage and serialization are valid."""
    try:
        connection = connect(path)
        rows = [dict(row) for row in connection.execute("select r.*, c.processing_status as parent_processing_status from product_allergen_refs r left join product_ingredient_components c on r.component_occurrence_id=c.component_occurrence_id where r.product_id=?", (product_id,))]
        connection.close()
    except sqlite3.Error:
        return [], "LINEAGE_INTEGRITY_ERROR"
    if not rows: return [], "NO_EVIDENCE"
    if any(row["dictionary_version"] != dictionary_version or row["pipeline_version"] != pipeline_version for row in rows): return [], "STALE_EVIDENCE"
    required = ("evidence_id", "component_occurrence_id", "product_id", "raw_ingredient_text", "normalized_text", "mapping_method", "source_dataset")
    if any(any(row.get(field) in (None, "") for field in required) or row.get("parent_processing_status") not in TERMINAL_COMPONENT_STATUSES for row in rows): return [], "LINEAGE_INTEGRITY_ERROR"
    try:
        for row in rows:
            row["usable_for_safety"] = bool(row["usable_for_safety"])
            row["segmented_text"] = json.loads(row["segmented_text"])
            if not isinstance(row["segmented_text"], list): raise ValueError("segmented_text must be a list")
            row.pop("parent_processing_status", None)
    except (TypeError, ValueError, json.JSONDecodeError):
        return [], "LINEAGE_INTEGRITY_ERROR"
    return rows, "PRECOMPUTED"
