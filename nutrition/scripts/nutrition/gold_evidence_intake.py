"""Strict, fail-close plumbing for human-reviewed Gold evidence intake.

This module has no network or raw-data side effects.  A human intake CSV is
only an *assertion* until every row passes the typed checks below and is then
written explicitly as a separate operational artifact.  In particular, raw
candidate metadata remains local context: it is never promoted to
authoritative manufacturer identity, life-stage, or product-form evidence.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

try:  # Support both ``nutrition.*`` package imports and existing script imports.
    from .gold_evidence_gate import (
        IDENTITY_BINDING_FIELDS,
        IDENTITY_SOURCE_TYPES,
        NUTRIENT_SOURCE_TYPES,
        REQUIRED,
        VALID_BASES,
        VALID_DOG_GROWTH_DETAILS,
        VALID_LIFE_STAGES,
        VALID_PRODUCT_FORMS,
        evaluate_runtime_eligibility,
    )
    from .gtin_validation import is_valid_gtin
except ImportError:  # pragma: no cover - exercised by standalone scripts
    from gold_evidence_gate import (
        IDENTITY_BINDING_FIELDS,
        IDENTITY_SOURCE_TYPES,
        NUTRIENT_SOURCE_TYPES,
        REQUIRED,
        VALID_BASES,
        VALID_DOG_GROWTH_DETAILS,
        VALID_LIFE_STAGES,
        VALID_PRODUCT_FORMS,
        evaluate_runtime_eligibility,
    )
    from gtin_validation import is_valid_gtin


INTAKE_SCHEMA_VERSION = "gold_evidence_intake_v1"
OPERATIONAL_ARTIFACT_VERSION = "verified_evidence_gold_v1"

# This extends the P1 manifest contract already emitted by
# ``close_gold_fallback_evidence_p1.py``.  The explicit metadata values prevent
# a reviewer from turning raw candidate text into a verified fact merely by
# setting an evidence-state flag.
INTAKE_FIELDS = (
    "candidate_id", "product_id", "canonical_gtin", "manufacturer",
    "manufacturer_product_name", "species", "life_stage", "life_stage_detail",
    "product_form", "product_variant", "package_size", "market_region",
    "formula_version", "effective_date", "identity_source_type",
    "identity_source_url_or_document", "identity_source_authority", "nutrient_code",
    "value", "unit", "basis", "value_qualifier", "guarantee_type",
    "nutrient_source_type", "nutrient_source_url_or_document",
    "nutrient_source_authority", "observed_at", "retrieved_at",
    "verification_method", "identity_state", "life_stage_evidence_state",
    "life_stage_detail_evidence_state", "product_form_evidence_state",
    "evidence_state", "verified", "runtime_eligible", "review_note",
)

# Keep the historical verified_evidence_gold_v1 columns first, then preserve
# the full identity/gate provenance needed by the runtime adapter.
OPERATIONAL_EVIDENCE_FIELDS = (
    "product_id", "canonical_gtin", "source_url", "market_region",
    "product_variant_size", "formula_version", "nutrient_code", "value",
    "unit", "basis", "guarantee_type", "observed_at", "retrieved_at",
    "verified", "verification_method", "runtime_eligible", "candidate_id",
    "manufacturer", "manufacturer_product_name", "species", "product_variant",
    "package_size", "effective_date", "identity_source_type",
    "identity_source_url_or_document", "identity_source_authority",
    "identity_state", "identity_conflict", "gtin_valid", "life_stage",
    "life_stage_detail", "life_stage_evidence_state",
    "life_stage_detail_evidence_state", "product_form",
    "product_form_evidence_state", "nutrient_source_type",
    "nutrient_source_url_or_document", "nutrient_source_authority",
    "value_qualifier", "evidence_state", "evidence_type",
    "identity_binding_verified", "validation_status", "validation_version",
    "validated_at", "gate_reason_codes", "missing_nutrients",
    "intake_row_number", "review_note", "candidate_context_state",
)

_AUTHORITATIVE_METADATA_STATE = "VERIFIED_AUTHORITATIVE"
_PROVENANCE_DISALLOWED_TOKENS = (
    "INFERRED", "KEYWORD", "LOCAL_METADATA", "MISSING", "CONFLICT",
)


@dataclass(frozen=True)
class ParsedIntake:
    """A CSV parse result.  A missing optional intake is deliberately empty."""

    source_path: str | None
    rows: tuple[dict[str, str], ...]
    input_missing: bool
    schema_errors: tuple[str, ...]


def _text(value: object) -> str:
    return str(value or "").strip()


def _as_bool(value: object) -> bool | None:
    text = _text(value).casefold()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _iso_value(value: str) -> bool:
    if not value:
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _numeric_value(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _has_disallowed_provenance_token(*values: object) -> bool:
    return any(
        token in _text(value).upper()
        for value in values
        for token in _PROVENANCE_DISALLOWED_TOKENS
    )


def read_human_gold_evidence_intake(path: str | Path | None) -> ParsedIntake:
    """Read the P1 human intake CSV without converting missing input to data.

    ``None`` or a non-existent path is an empty human intake, not an exception.
    A present file with a bad header is represented as a schema error so that no
    row can be accepted accidentally.
    """
    if path is None:
        return ParsedIntake(None, (), True, ())
    source = Path(path)
    if not source.exists():
        return ParsedIntake(str(source), (), True, ())
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = tuple(field for field in INTAKE_FIELDS if field not in fieldnames)
        rows = []
        for row_number, raw in enumerate(reader, start=2):
            row = {field: _text(raw.get(field)) for field in INTAKE_FIELDS}
            row["_row_number"] = str(row_number)
            rows.append(row)
    errors = tuple(f"INPUT_SCHEMA_MISSING_COLUMN:{field}" for field in missing)
    return ParsedIntake(str(source), tuple(rows), False, errors)


# A short alias makes the module easy to use from the P1 orchestrator while
# retaining the explicit human-evidence name above for callers and reports.
read_intake_csv = read_human_gold_evidence_intake


def _candidate_index(candidates: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    values = candidates.values() if isinstance(candidates, Mapping) else candidates
    indexed: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for raw in values:
        candidate = dict(raw)
        candidate_id = _text(candidate.get("candidate_id"))
        if not candidate_id:
            continue
        if candidate_id in indexed:
            duplicates.add(candidate_id)
        indexed[candidate_id] = candidate
    return indexed, duplicates


def _allowed_product_ids(candidate: Mapping[str, Any]) -> set[str]:
    """Return local product IDs that a GTIN-cluster candidate may bind.

    ``source_records`` is the existing P1 representation (``SOURCE:product``
    entries separated by semicolons).  The explicit list is preferred when an
    orchestrator has already parsed it.  No fuzzy product-name identity is used.
    """
    values: set[str] = set()
    explicit = candidate.get("allowed_product_ids") or candidate.get("source_product_ids")
    if isinstance(explicit, (list, tuple, set, frozenset)):
        values.update(_text(value) for value in explicit if _text(value))
    elif _text(explicit):
        values.add(_text(explicit))
    for entry in _text(candidate.get("source_records")).split(";"):
        if ":" in entry:
            _, product_id = entry.split(":", 1)
            if _text(product_id):
                values.add(_text(product_id))
    return values


def _candidate_context_issues(candidate: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    gtin = _text(candidate.get("canonical_gtin"))
    if not is_valid_gtin(gtin):
        reasons.append("CANDIDATE_INVALID_GTIN")
    if not _allowed_product_ids(candidate):
        reasons.append("CANDIDATE_PRODUCT_MEMBERSHIP_UNAVAILABLE")
    species = _text(candidate.get("species")).upper()
    if species not in REQUIRED:
        reasons.append("CANDIDATE_SPECIES_UNRESOLVED")
    if _text(candidate.get("life_stage")) not in VALID_LIFE_STAGES:
        reasons.append("CANDIDATE_LIFE_STAGE_UNRESOLVED")
    if _text(candidate.get("product_form")) not in VALID_PRODUCT_FORMS:
        reasons.append("CANDIDATE_PRODUCT_FORM_UNRESOLVED")
    # A raw cluster may know the broad growth stage without containing the
    # required early/late detail.  A later authoritative human intake may
    # supply that detail; the raw absence itself is not an immutable failure.
    if _text(candidate.get("identity_conflict")) == "HARD_IDENTITY_CONFLICT":
        reasons.append("HARD_IDENTITY_CONFLICT")
    return reasons


def _row_issues(row: Mapping[str, str], candidate: Mapping[str, Any]) -> tuple[list[str], float | None]:
    reasons = _candidate_context_issues(candidate)
    product_id = _text(row.get("product_id"))
    row_gtin = _text(row.get("canonical_gtin"))
    candidate_gtin = _text(candidate.get("canonical_gtin"))
    if not is_valid_gtin(row_gtin):
        reasons.append("INVALID_GTIN")
    if row_gtin != candidate_gtin:
        reasons.append("CANONICAL_GTIN_MISMATCH")
    if product_id not in _allowed_product_ids(candidate):
        reasons.append("PRODUCT_ID_NOT_IN_CANDIDATE_CLUSTER")
    if _text(row.get("species")).upper() != _text(candidate.get("species")).upper():
        reasons.append("SPECIES_MISMATCH")

    # All identity fields are supplied by the human-reviewed document.  They
    # must be nonempty now and will be checked for exact consistency per group.
    for field in IDENTITY_BINDING_FIELDS:
        if not _text(row.get(field)):
            reasons.append(f"IDENTITY_BINDING_FIELD_MISSING:{field}")

    if _text(row.get("identity_state")) != "VERIFIED":
        reasons.append("IDENTITY_NOT_VERIFIED")
    if _text(row.get("identity_source_type")) not in IDENTITY_SOURCE_TYPES:
        reasons.append("IDENTITY_SOURCE_NOT_GTIN_BINDING")
    if not _text(row.get("identity_source_url_or_document")):
        reasons.append("IDENTITY_SOURCE_DOCUMENT_MISSING")
    if not _text(row.get("identity_source_authority")):
        reasons.append("IDENTITY_SOURCE_AUTHORITY_MISSING")

    # Local product text is only context.  A reviewer must explicitly repeat
    # the authoritative value and provenance; a state flag alone cannot
    # promote the raw candidate value.
    if _text(row.get("life_stage")) != _text(candidate.get("life_stage")):
        reasons.append("LIFE_STAGE_MISMATCH_OR_MISSING")
    if _text(row.get("product_form")) != _text(candidate.get("product_form")):
        reasons.append("PRODUCT_FORM_MISMATCH_OR_MISSING")
    if (
        _text(candidate.get("species")).upper() == "DOG"
        and _text(candidate.get("life_stage")) == "GROWTH_REPRODUCTION"
    ):
        row_detail = _text(row.get("life_stage_detail"))
        candidate_detail = _text(candidate.get("life_stage_detail"))
        if row_detail not in VALID_DOG_GROWTH_DETAILS:
            reasons.append("DOG_GROWTH_LIFE_STAGE_DETAIL_MISMATCH_OR_MISSING")
        elif candidate_detail and row_detail != candidate_detail:
            reasons.append("DOG_GROWTH_LIFE_STAGE_DETAIL_MISMATCH_OR_MISSING")
    if _text(row.get("life_stage_evidence_state")) != _AUTHORITATIVE_METADATA_STATE:
        reasons.append("LIFE_STAGE_PROVENANCE_UNVERIFIED")
    if _text(row.get("product_form_evidence_state")) != _AUTHORITATIVE_METADATA_STATE:
        reasons.append("PRODUCT_FORM_PROVENANCE_UNVERIFIED")
    if (
        _text(candidate.get("species")).upper() == "DOG"
        and _text(candidate.get("life_stage")) == "GROWTH_REPRODUCTION"
        and _text(row.get("life_stage_detail_evidence_state")) != _AUTHORITATIVE_METADATA_STATE
    ):
        reasons.append("DOG_GROWTH_LIFE_STAGE_DETAIL_PROVENANCE_UNVERIFIED")

    nutrient_code = _text(row.get("nutrient_code"))
    if nutrient_code not in REQUIRED.get(_text(candidate.get("species")).upper(), set()):
        reasons.append("UNSUPPORTED_NUTRIENT_CODE")
    numeric = _numeric_value(_text(row.get("value")))
    if numeric is None:
        reasons.append("NUTRIENT_VALUE_INVALID")
    if _text(row.get("unit")) != "PERCENT":
        reasons.append("NUTRIENT_UNIT_UNSUPPORTED")
    if _text(row.get("basis")) not in VALID_BASES:
        reasons.append("NUTRIENT_BASIS_UNSUPPORTED")
    if _text(row.get("guarantee_type")) != "GUARANTEED":
        reasons.append("GUARANTEE_TYPE_UNSUPPORTED")
    if _text(row.get("nutrient_source_type")) not in NUTRIENT_SOURCE_TYPES:
        reasons.append("NUTRIENT_SOURCE_UNSUPPORTED")
    for field in (
        "nutrient_source_url_or_document", "nutrient_source_authority",
        "observed_at", "retrieved_at", "verification_method",
    ):
        if not _text(row.get(field)):
            reasons.append(f"NUTRIENT_PROVENANCE_FIELD_MISSING:{field}")
    for field in ("observed_at", "retrieved_at"):
        if _text(row.get(field)) and not _iso_value(_text(row.get(field))):
            reasons.append(f"NUTRIENT_PROVENANCE_TIMESTAMP_INVALID:{field}")
    if _as_bool(row.get("verified")) is not True:
        reasons.append("NUTRIENT_EVIDENCE_NOT_HUMAN_VERIFIED")
    evidence_state = _text(row.get("evidence_state"))
    if evidence_state and evidence_state != "VERIFIED":
        reasons.append("EVIDENCE_STATE_NOT_VERIFIED")
    if _has_disallowed_provenance_token(
        row.get("verification_method"), row.get("identity_source_type"),
        row.get("nutrient_source_type"), evidence_state,
        row.get("life_stage_evidence_state"), row.get("product_form_evidence_state"),
        row.get("life_stage_detail_evidence_state"),
    ):
        reasons.append("INFERRED_OR_LOCAL_PROVENANCE_NOT_ALLOWED")
    return sorted(set(reasons)), numeric


def _runtime_candidate(candidate: Mapping[str, Any], row: Mapping[str, str]) -> dict[str, Any]:
    """Combine local context with *validated* human evidence, never vice versa."""
    result = dict(candidate)
    for field in IDENTITY_BINDING_FIELDS:
        result[field] = _text(row.get(field))
    result.update({
        "gtin_valid": True,
        "identity_state": "VERIFIED",
        "identity_source_type": _text(row.get("identity_source_type")),
        "identity_source_url_or_document": _text(row.get("identity_source_url_or_document")),
        "identity_source_authority": _text(row.get("identity_source_authority")),
        "species": _text(candidate.get("species")).upper(),
        "life_stage": _text(row.get("life_stage")),
        "life_stage_detail": _text(row.get("life_stage_detail")),
        "life_stage_evidence_state": _text(row.get("life_stage_evidence_state")),
        "life_stage_detail_evidence_state": _text(row.get("life_stage_detail_evidence_state")),
        "product_form": _text(row.get("product_form")),
        "product_form_evidence_state": _text(row.get("product_form_evidence_state")),
    })
    return result


def _runtime_evidence(candidate: Mapping[str, Any], row: Mapping[str, str], numeric: float) -> dict[str, Any]:
    evidence = {field: candidate[field] for field in IDENTITY_BINDING_FIELDS}
    evidence.update({
        "verified": True,
        "nutrient_code": _text(row.get("nutrient_code")),
        "value": numeric,
        "unit": _text(row.get("unit")),
        "basis": _text(row.get("basis")),
        "guarantee_type": _text(row.get("guarantee_type")),
        "nutrient_source_type": _text(row.get("nutrient_source_type")),
        "nutrient_source_url_or_document": _text(row.get("nutrient_source_url_or_document")),
        "nutrient_source_authority": _text(row.get("nutrient_source_authority")),
        "observed_at": _text(row.get("observed_at")),
        "retrieved_at": _text(row.get("retrieved_at")),
        "verification_method": _text(row.get("verification_method")),
    })
    return evidence


def _rejection(row: Mapping[str, str], reasons: Iterable[str]) -> dict[str, Any]:
    return {
        "intake_row_number": int(_text(row.get("_row_number")) or 0),
        "candidate_id": _text(row.get("candidate_id")),
        "product_id": _text(row.get("product_id")),
        "canonical_gtin": _text(row.get("canonical_gtin")),
        "nutrient_code": _text(row.get("nutrient_code")),
        "validation_status": "REJECTED",
        "reason_codes": sorted(set(reasons)),
    }


def _binding_conflicts(items: list[dict[str, Any]]) -> set[str]:
    return {
        field
        for field in IDENTITY_BINDING_FIELDS
        if len({_text(item["row"].get(field)) for item in items}) != 1
    }


def _candidate_context(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "canonical_gtin": _text(candidate.get("canonical_gtin")),
        "allowed_product_ids": sorted(_allowed_product_ids(candidate)),
        "species": _text(candidate.get("species")).upper(),
        "life_stage": _text(candidate.get("life_stage")),
        "life_stage_detail": _text(candidate.get("life_stage_detail")),
        "product_form": _text(candidate.get("product_form")),
        "identity_conflict": _text(candidate.get("identity_conflict")),
        "context_state": "LOCAL_METADATA_CONTEXT_ONLY",
    }


def validate_human_gold_evidence_intake(
    intake: ParsedIntake,
    candidates: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    *,
    validated_at: str | None = None,
) -> dict[str, Any]:
    """Validate typed intake rows and re-run the strict runtime gate per candidate.

    An accepted row is a valid human-reviewed nutrient evidence row.  It is not
    automatically runtime-eligible: that flag is calculated only after all
    required nutrient rows for the exact candidate identity pass the existing
    ``evaluate_runtime_eligibility`` gate.
    """
    when = validated_at or datetime.now(timezone.utc).isoformat()
    if not _iso_value(when):
        raise ValueError("validated_at must be an ISO-8601 timestamp")
    indexed, duplicate_ids = _candidate_index(candidates)
    rejected: list[dict[str, Any]] = []
    provisional: list[dict[str, Any]] = []

    if intake.schema_errors:
        for row in intake.rows:
            rejected.append(_rejection(row, intake.schema_errors))
    else:
        for row in intake.rows:
            candidate_id = _text(row.get("candidate_id"))
            if candidate_id in duplicate_ids:
                rejected.append(_rejection(row, ["DUPLICATE_CANDIDATE_ID"]))
                continue
            candidate = indexed.get(candidate_id)
            if candidate is None:
                rejected.append(_rejection(row, ["CANDIDATE_NOT_FOUND"]))
                continue
            reasons, numeric = _row_issues(row, candidate)
            if reasons:
                rejected.append(_rejection(row, reasons))
                continue
            provisional.append({"row": row, "candidate": candidate, "numeric": numeric})

    # Any malformed/incompatible row in a candidate group blocks its otherwise
    # valid rows too.  Selecting only the convenient subset would be a silent
    # conflict-resolution policy, which this P1 flow deliberately does not have.
    rejected_candidate_ids = {record["candidate_id"] for record in rejected if record["candidate_id"]}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in provisional:
        grouped[_text(item["row"].get("candidate_id"))].append(item)

    accepted: list[dict[str, Any]] = []
    operational_candidates: list[dict[str, Any]] = []
    for candidate_id, items in sorted(grouped.items()):
        if candidate_id in rejected_candidate_ids:
            for item in items:
                rejected.append(_rejection(item["row"], ["CANDIDATE_GROUP_CONTAINS_REJECTED_ROW"]))
            continue
        conflicts = _binding_conflicts(items)
        if conflicts:
            for item in items:
                rejected.append(_rejection(item["row"], [f"INTERNAL_BINDING_CONFLICT:{field}" for field in conflicts]))
            continue
        duplicate_conflicts: set[str] = set()
        by_nutrient: dict[str, set[tuple[float | None, str, str, str]]] = defaultdict(set)
        for item in items:
            row = item["row"]
            by_nutrient[_text(row.get("nutrient_code"))].add((
                item["numeric"], _text(row.get("unit")), _text(row.get("basis")), _text(row.get("guarantee_type")),
            ))
        duplicate_conflicts = {code for code, values in by_nutrient.items() if len(values) > 1}
        if duplicate_conflicts:
            for item in items:
                if _text(item["row"].get("nutrient_code")) in duplicate_conflicts:
                    rejected.append(_rejection(item["row"], ["DUPLICATE_NUTRIENT_VALUE_CONFLICT"]))
                else:
                    rejected.append(_rejection(item["row"], ["CANDIDATE_GROUP_CONTAINS_CONFLICTING_NUTRIENT"]))
            continue

        candidate = _runtime_candidate(items[0]["candidate"], items[0]["row"])
        evidence = [_runtime_evidence(candidate, item["row"], item["numeric"]) for item in items]
        gate = evaluate_runtime_eligibility(candidate, evidence)
        context = _candidate_context(items[0]["candidate"])
        operational_candidates.append({
            "candidate_id": candidate_id,
            "runtime_candidate": candidate,
            "local_candidate_context": context,
            "evidence_row_count": len(evidence),
            "runtime_eligible": gate["runtime_eligible"],
            "reason_codes": gate["reason_codes"],
            "missing_nutrients": gate["missing_nutrients"],
        })
        for item, runtime_row in zip(items, evidence):
            raw = item["row"]
            accepted.append({
                **candidate,
                **runtime_row,
                "source_url": runtime_row["nutrient_source_url_or_document"],
                "product_variant_size": candidate["package_size"],
                "value_qualifier": _text(raw.get("value_qualifier")),
                "evidence_state": "VERIFIED",
                "evidence_type": "NUTRITION_GUARANTEE",
                "identity_binding_verified": True,
                "validation_status": "ACCEPTED",
                "validation_version": INTAKE_SCHEMA_VERSION,
                "validated_at": when,
                "runtime_eligible": gate["runtime_eligible"],
                "gate_reason_codes": gate["reason_codes"],
                "missing_nutrients": gate["missing_nutrients"],
                "intake_row_number": int(_text(raw.get("_row_number")) or 0),
                "review_note": _text(raw.get("review_note")),
                "candidate_context_state": "LOCAL_METADATA_CONTEXT_ONLY",
            })

    reason_counts = Counter(reason for row in rejected for reason in row["reason_codes"])
    return {
        "validation_version": INTAKE_SCHEMA_VERSION,
        "validated_at": when,
        "source_path": intake.source_path,
        "input_missing": intake.input_missing,
        "schema_errors": list(intake.schema_errors),
        "human_intake_rows": len(intake.rows),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "runtime_eligible_count": sum(record["runtime_eligible"] for record in operational_candidates),
        "accepted_evidence": accepted,
        "rejected_rows": rejected,
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "operational_candidates": operational_candidates,
    }


# A concise alias for an orchestrator that already knows it is processing P1.
validate_intake_rows = validate_human_gold_evidence_intake


def build_operational_evidence_artifact(validation: Mapping[str, Any]) -> dict[str, Any]:
    """Build a serializable, separate artifact; this function writes nothing."""
    return {
        "artifact_version": OPERATIONAL_ARTIFACT_VERSION,
        "validation_version": validation["validation_version"],
        "validated_at": validation["validated_at"],
        "source_path": validation["source_path"],
        "input_missing": validation["input_missing"],
        "schema_errors": list(validation["schema_errors"]),
        "human_intake_rows": validation["human_intake_rows"],
        "accepted": validation["accepted"],
        "rejected": validation["rejected"],
        "runtime_eligible_count": validation["runtime_eligible_count"],
        "rejection_reason_counts": dict(validation["rejection_reason_counts"]),
        "operational_evidence": list(validation["accepted_evidence"]),
        "operational_candidates": list(validation["operational_candidates"]),
        "rejected_rows": list(validation["rejected_rows"]),
    }


def _csv_value(value: object) -> object:
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value


def write_operational_evidence_artifact(
    artifact: Mapping[str, Any],
    *,
    csv_path: str | Path,
    json_path: str | Path,
) -> None:
    """Explicitly write the runtime artifact.  Callers choose all paths."""
    csv_target, json_target = Path(csv_path), Path(json_path)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    with csv_target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OPERATIONAL_EVIDENCE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in artifact["operational_evidence"]:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in OPERATIONAL_EVIDENCE_FIELDS})
    json_target.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
